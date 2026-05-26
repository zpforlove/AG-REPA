import argparse
import faulthandler
import logging
import math
import os
import sys
import traceback
from pathlib import Path
import json
import random

import deepspeed
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from beats.BEATs import BEATs, BEATsConfig
from data_loader import AudioSetDataset
from models import AudioSetTokenizer
from utils import load_config

# Global variable declaration
global trainer_global_object

# --- Logging setup ---
global logger


def setup_logger(rank=-1):
    """Set up the global logger based on the process rank"""
    current_logger = logging.getLogger(__name__)
    if current_logger.hasHandlers():
        current_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    log_format = f'%(asctime)s - RANK {rank} - %(name)s - %(levelname)s - %(message)s' if rank != -1 else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    current_logger.setLevel(logging.INFO if rank <= 0 else logging.WARNING)
    current_logger.propagate = False

    global logger
    logger = current_logger
    return logger


class LLMTrainer:
    def __init__(self, config_path, cmd_args):
        """
        Initialize the LLM trainer, adapted to DeepSpeed and the new AudioSet data pipeline
        """
        self.cmd_args = cmd_args
        self.config = load_config(config_path)
        global logger
        if logger is None:
            logger = setup_logger(self.cmd_args.local_rank)

        print(f"Rank {self.cmd_args.local_rank}: Starting trainer initialization...")

        # Initialize components
        self.audioset_tokenizer = None
        self.beats_feature_extractor = None
        self.llm_tokenizer = None
        self.model_engine = None
        self.optimizer = None
        self.pytorch_optimizer = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.device = None

        # Metadata storage
        self.audioset_descriptions = {}

        # Style and coarse→fine related members
        self.style_num_tokens = int(self.config['hyperparameters']['llm'].get('style_num_tokens', 8))
        self.style_strength = float(self.config['hyperparameters']['llm'].get('style_strength', 0.5))
        self.style_dropout_p = float(self.config['hyperparameters']['llm'].get('style_dropout_p', 0.2))

        self.coarse_num_clusters = int(self.config['hyperparameters']['llm'].get('coarse_num_clusters', 128))
        self.coarse_code_to_cluster_path = self.config['hyperparameters']['llm'].get(
            'coarse_code_to_cluster_path', None
        )
        self.as_vocab_size = int(self.config['hyperparameters']['llm']['as_vocab_size'])

        # Mapping of [AS_k] and [AUDIO_i] to tokenizer ids
        self.as_token_ids = []  # len = vocab_size, corresponds to the tokenizer ids of [AS_0..AS_{V-1}]
        self.id_to_as_code = {}  # tokenizer id -> k (AS_k)
        self.audio_token_ids = []  # len = style_num_tokens, corresponds to the tokenizer ids of [AUDIO_0..K-1]

        # Mapping from coarse codeword to cluster id
        self.as_code_to_cluster = None  # list[int] of length as_vocab_size

        self._setup_environment()
        if self.cmd_args.local_rank <= 0:
            self._setup_wandb()
        self._setup_model_and_deepspeed()

        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / 'llm'
        if self.cmd_args.local_rank <= 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            print(f"Checkpoint directory created/confirmed: {self.checkpoint_dir}")

        print(f"Rank {self.cmd_args.local_rank}: LLM trainer initialized successfully")

    def _load_metadata(self):
        print("Loading AudioSet metadata...")
        base_dir = self.config['data']['audioset']['file_dir']
        print(f"Loading metadata using the absolute base directory: {base_dir}")
        ontology_path = os.path.join(base_dir, "ontology.json")
        try:
            with open(ontology_path, 'r', encoding='utf-8') as f:
                ontology_data = json.load(f)
            self.ontology_map = {item['id']: {'name': item['name'], 'description': item['description']} for item in
                                 ontology_data}
            print(f"Ontology file loaded successfully: {ontology_path}")
        except Exception as e:
            logger.error(f"Failed to load ontology file: {ontology_path}, error: {e}")
            raise

        def load_segments_df(csv_path):
            try:
                df = pd.read_csv(csv_path, header=None, comment='#', quotechar='"', skipinitialspace=True,
                                 names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'])
                df.set_index('YTID', inplace=True)
                print(f"CSV segments file loaded successfully: {csv_path}")
                return df
            except Exception as e:
                logger.error(f"Failed to load CSV file: {csv_path}, error: {e}")
                raise

        # Load the training and validation segments data
        self.train_segments_df = load_segments_df(os.path.join(base_dir, "unbalanced_train_segments.csv"))
        self.eval_segments_df = load_segments_df(os.path.join(base_dir, "eval_segments.csv"))

    def _setup_environment(self):
        rank = self.cmd_args.local_rank
        self.device = f'cuda:{rank}' if torch.cuda.is_available() and rank != -1 else 'cpu'

        try:
            # 1. Load the AudioSet descriptions
            print("Loading the AudioSet description file...")
            audioset_desc_path = self.config['data']['audioset']['desc_path']
            with open(audioset_desc_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line)
                    self.audioset_descriptions[item['audio_filename']] = {
                        'event': item['event'],
                        'description': item['description']
                    }
            print(f"Loaded {len(self.audioset_descriptions)} AudioSet descriptions.")

            # --- 2. Load the BEATs Feature Extractor and the AudioSetTokenizer ---
            print(f"Rank {rank}: Loading the BEATs Feature Extractor...")
            beats_checkpoint_path = self.config['paths']['beats_feature_extractor_checkpoint']
            if not os.path.exists(beats_checkpoint_path):
                raise FileNotFoundError(f"BEATs feature extractor checkpoint not found at: {beats_checkpoint_path}")

            checkpoint = torch.load(beats_checkpoint_path, map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.beats_feature_extractor = BEATs(cfg)
            self.beats_feature_extractor.load_state_dict(checkpoint['model'])
            self.beats_feature_extractor.eval()
            print(f"Rank {rank}: BEATs Feature Extractor loaded successfully from {beats_checkpoint_path}.")

            # Automatically find and load the best AudioSetTokenizer
            ast_checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / 'ast'
            if not ast_checkpoint_dir.exists() or not any(ast_checkpoint_dir.iterdir()):
                raise FileNotFoundError(
                    f"AudioSetTokenizer checkpoint directory is empty or not found: {ast_checkpoint_dir}")

            best_loss = float('inf')
            best_checkpoint_path = None
            for pth_file in ast_checkpoint_dir.glob("best_epoch_*_loss_*.pth"):
                try:
                    loss_str = pth_file.stem.split('_loss_')[-1]
                    loss_val = float(loss_str)
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_checkpoint_path = pth_file
                except (ValueError, IndexError):
                    logger.warning(
                        f"Rank {rank}: Could not parse loss from filename: {pth_file.name}")
                    continue

            if best_checkpoint_path is None:
                raise FileNotFoundError(f"No valid 'best_epoch_*_loss_*.pth' checkpoints found in {ast_checkpoint_dir}")

            logger.info(f"Rank {rank}: Found best AST checkpoint: {best_checkpoint_path} (loss {best_loss:.4f})")

            ast_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            ast_config = self.config['hyperparameters']['ast']
            self.audioset_tokenizer = AudioSetTokenizer(
                input_dim=ast_config['input_dim'],
                hidden_dim=ast_config['hidden_dim'],
                vocab_size=ast_config['vocab_size']
            )
            self.audioset_tokenizer.load_state_dict(ast_checkpoint['model_state_dict'])
            self.audioset_tokenizer.eval()
            logger.info(f"Rank {rank}: AudioSet Tokenizer loaded from best checkpoint.")

            # Record the BEATs embedding dimension (style input dimension)
            self.beats_embed_dim = getattr(cfg, 'encoder_embed_dim', 768)

        except Exception as e:
            logger.error(f"Rank {rank}: Environment setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _setup_wandb(self):
        if self.config['logging']['wandb']['enabled']:
            try:
                wandb.init(
                    project=self.config['logging']['wandb']['project'],
                    name=self.config['logging']['wandb']['name'] + "_LLM",
                    config={**self.config, **vars(self.cmd_args)}
                )
                print("WandB initialized successfully")
            except Exception as e:
                logger.error(f"WandB initialization failed: {e}")
                self.config['logging']['wandb']['enabled'] = False

    def _setup_model_and_deepspeed(self):
        print(f"Loading LLM tokenizer: {self.config['hyperparameters']['llm']['model_name']}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.config['hyperparameters']['llm']['model_name'],
            trust_remote_code=True
        )
        # Extend the vocabulary (including [TAG][DES][AT][END][PAD], [AS_*], [AUDIO_i])
        self._extend_tokenizer_vocab()

        print(f"Loading LLM model: {self.config['hyperparameters']['llm']['model_name']}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config['hyperparameters']['llm']['model_name'],
            trust_remote_code=True
        )
        print("LLM model loaded in full precision (FP32).")
        model.resize_token_embeddings(len(self.llm_tokenizer))

        # ---- pad / eos fix ----
        if getattr(model.config, "pad_token_id", None) is None and self.llm_tokenizer.pad_token_id is not None:
            model.config.pad_token_id = self.llm_tokenizer.pad_token_id

        end_id = self.llm_tokenizer.convert_tokens_to_ids('[END]')
        base_eos = getattr(model.config, "eos_token_id", None)
        if base_eos is None:
            model.config.eos_token_id = end_id
        else:
            if isinstance(base_eos, (list, tuple, set)):
                eos_ids = sorted(set(list(base_eos) + [end_id]))
            else:
                eos_ids = [base_eos] if base_eos != end_id else [base_eos]
            model.config.eos_token_id = eos_ids
        try:
            model.generation_config.eos_token_id = model.config.eos_token_id
        except Exception:
            pass
        if getattr(self.llm_tokenizer, "eos_token_id", None) is None:
            try:
                self.llm_tokenizer.eos_token = '[END]'
            except Exception:
                pass

        if self.config['hyperparameters']['llm'].get('gradient_checkpointing', False):
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False

        # ============= Add the style and coarse→fine modules (register parameters before DeepSpeed initialization) =============
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            # Compatibility with some models' field names
            hidden_size = getattr(model.config, "n_embd", None) or getattr(model.config, "dim", None)
            if hidden_size is None:
                raise ValueError("Could not infer hidden_size from the LLM config.")

        # 1) Style projection layer: BEATs -> LLM hidden dimension
        model.style_proj = nn.Linear(self.beats_embed_dim, hidden_size)
        # Injection strength and Dropout probability that can be tuned during training (saved on the model for distributed consistency)
        model.style_strength = self.style_strength
        model.style_dropout_p = self.style_dropout_p
        model.style_num_tokens = self.style_num_tokens

        # 2) Coarse head: hidden -> num_clusters
        model.coarse_head = nn.Linear(hidden_size, self.coarse_num_clusters)

        # ============= DeepSpeed initialization =============
        self.pytorch_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(self.config['hyperparameters']['llm']['optimizer']['scheduler']['max_lr']),
            weight_decay=float(self.config['hyperparameters']['llm']['optimizer']['weight_decay']),
            betas=eval(str(self.config['hyperparameters']['llm']['optimizer']['betas']))
        )

        print(f"Rank {self.cmd_args.local_rank}: Initializing the DeepSpeed engine...")
        self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
            args=self.cmd_args,
            model=model,
            optimizer=self.pytorch_optimizer
        )
        self.device = self.model_engine.device
        print(f"DeepSpeed engine initialization complete. Model on device: {self.device}")

        # Move the BEATs feature extractor and the AudioSet tokenizer to the target device
        self.beats_feature_extractor.to(self.device)
        self.audioset_tokenizer.to(self.device)

        # coarse: build the code->cluster mapping
        self._build_coarse_mapping()

    def _extend_tokenizer_vocab(self):
        """
        Extend the vocabulary:
          - Special tokens: [TAG], [DES], [AT], [END], [PAD]
          - AudioSet vocabulary tokens: [AS_0]..[AS_{V-1}]
          - Style placeholder tokens: [AUDIO_0]..[AUDIO_{K-1}]
        """
        print("Extending the tokenizer vocabulary...")

        as_vocab_size = int(self.config['hyperparameters']['llm']['as_vocab_size'])
        style_num_tokens = self.style_num_tokens

        special_tokens = ['[TAG]', '[DES]', '[AT]', '[END]', '[PAD]']
        as_target_tokens = [f'[AS_{j}]' for j in range(as_vocab_size)]
        audio_style_tokens = [f'[AUDIO_{j}]' for j in range(style_num_tokens)]

        tokens_to_add = special_tokens + as_target_tokens + audio_style_tokens
        num_added = self.llm_tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        print(f"Added {num_added} tokens.")

        # Ensure the tokenizer has a pad_token
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Record commonly used IDs
        self.special_token_ids = {
            'tag': self.llm_tokenizer.convert_tokens_to_ids('[TAG]'),
            'des': self.llm_tokenizer.convert_tokens_to_ids('[DES]'),
            'at': self.llm_tokenizer.convert_tokens_to_ids('[AT]'),
            'end': self.llm_tokenizer.convert_tokens_to_ids('[END]'),
            'pad': self.llm_tokenizer.convert_tokens_to_ids('[PAD]'),
        }

        # Record [AS_k] -> id, and id -> k
        self.as_token_ids = []
        self.id_to_as_code = {}
        for j in range(as_vocab_size):
            tid = self.llm_tokenizer.convert_tokens_to_ids(f'[AS_{j}]')
            self.as_token_ids.append(tid)
            self.id_to_as_code[tid] = j

        # Record the ids of [AUDIO_i]
        self.audio_token_ids = [self.llm_tokenizer.convert_tokens_to_ids(f'[AUDIO_{i}]')
                                for i in range(style_num_tokens)]

        print(f"Vocabulary size after extension: {len(self.llm_tokenizer)}")

    def _build_coarse_mapping(self):
        """Build the mapping from codeword -> coarse cluster id"""
        V = self.as_vocab_size
        C = self.coarse_num_clusters
        mapping = None

        if self.coarse_code_to_cluster_path and os.path.exists(self.coarse_code_to_cluster_path):
            try:
                with open(self.coarse_code_to_cluster_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Allow either a dict {"0":12,...} or a list[cluster_id]*V
                if isinstance(data, dict):
                    mapping = [int(data[str(i)]) for i in range(V)]
                elif isinstance(data, list) and len(data) == V:
                    mapping = [int(x) for x in data]
                else:
                    logger.warning("The format of coarse_code_to_cluster_path is not as expected; falling back to uniform bucketing.")
            except Exception as e:
                logger.warning(f"Failed to read the coarse mapping; falling back to uniform bucketing: {e}")

        if mapping is None:
            # Uniform bucketing (coarse but usable)
            bucket = max(V // C, 1)
            mapping = [min(i // bucket, C - 1) for i in range(V)]

        self.as_code_to_cluster = torch.tensor(mapping, dtype=torch.long, device=self.device)  # [V]

    def _setup_dataloaders(self, is_train=True):
        is_distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        if is_train:
            print("Initializing the training data loader (AudioSet)...")
            dataset = AudioSetDataset(self.config['data']['audioset']['train_root'], self.config)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=True) if is_distributed else None

            self.train_loader = DataLoader(
                dataset, batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                sampler=sampler, shuffle=(sampler is None),
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory'],
                persistent_workers=True,
                drop_last=True
            )
            print(f"Training data loader initialized, containing {len(self.train_loader)} batches.")
        else:
            print("Initializing the validation data loader (AudioSet)...")
            dataset = AudioSetDataset(self.config['data']['audioset']['val_root'], self.config)
            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=False) if is_distributed else None

            self.val_loader = DataLoader(
                dataset, batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                sampler=sampler, shuffle=False, persistent_workers=True,
                num_workers=self.config['data']['num_workers'], pin_memory=self.config['data']['pin_memory']
            )
            print(f"Validation data loader initialized, containing {len(self.val_loader)} batches.")

    def setup_lr_scheduler(self):
        """Set up the OneCycleLR scheduler based on the configuration file."""
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: The optimizer or training data loader is not initialized; cannot create the LR scheduler.")
            return
        print(f"Rank {self.cmd_args.local_rank}: Setting up the LR scheduler...")
        try:
            scheduler_config = self.config['hyperparameters']['llm']['optimizer']['scheduler']
            num_epochs = self.config['hyperparameters']['llm']['num_epochs']
            gradient_accumulation_steps = self.model_engine.gradient_accumulation_steps()

            total_micro_batches = len(self.train_loader) * num_epochs
            total_steps = math.ceil(total_micro_batches / gradient_accumulation_steps)

            if total_steps <= 0:
                logger.error(
                    f"Rank {self.cmd_args.local_rank}: The computed total number of steps ({total_steps}) is invalid; the LR scheduler was not created.")
                return

            print(f"Rank {self.cmd_args.local_rank}: Total number of steps for the LR scheduler: {total_steps}")
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.pytorch_optimizer,
                max_lr=float(scheduler_config['max_lr']),
                total_steps=total_steps,
                pct_start=float(scheduler_config['pct_start']),
                div_factor=float(scheduler_config['div_factor']),
                final_div_factor=float(scheduler_config['final_div_factor'])
            )
            print(f"Rank {self.cmd_args.local_rank}: OneCycleLR scheduler created successfully.")
        except KeyError as e:
            logger.error(f"Rank {self.cmd_args.local_rank}: Failed to create the LR scheduler. Missing key: {e}.")
            raise
        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Failed to set up the LR scheduler: {e}\n{traceback.format_exc()}")
            raise

    @torch.no_grad()
    def _extract_as_and_style(self, waveform, K):
        """
        Extract from the audio waveform:
          - AST target tokens (used as ground-truth)
          - K BEATs style chunks (evenly split over time) -> [K, D_beats]
        """
        # BEATs features [B, T, D]
        beats_features, _ = self.beats_feature_extractor.extract_features(waveform)  # [1, T, D]
        T, D = beats_features.shape[1], beats_features.shape[2]

        # Prepare [B, D, T] for AST tokenize
        input_feats_tok = beats_features.permute(0, 2, 1)
        as_tokens = self.audioset_tokenizer.tokenize(input_feats_tok).view(-1)  # [L_as]

        # Compute the K style chunks: evenly split over time and take the mean
        # If T < K, allow repeated boundaries to guarantee K chunks
        idx_edges = torch.linspace(0, T, steps=K + 1, device=beats_features.device).long()
        style_chunks = []
        for i in range(K):
            a, b = idx_edges[i].item(), idx_edges[i + 1].item()
            if b <= a:
                b = min(a + 1, T)
            seg = beats_features[0, a:b, :]  # [len, D]
            style_chunks.append(seg.mean(dim=0, keepdim=True))  # [1, D]
        style_BKD = torch.cat(style_chunks, dim=0)  # [K, D]

        return as_tokens, style_BKD  # [L_as], [K, D]

    def _prepare_batch_data(self, batch, is_train):
        """
        Prepare a single batch of data for the LLM, adding the style prefix:
        [TAG] events [DES] desc [AUDIO_0]...[AUDIO_{K-1}] [AT] -> [AS_tokens] [END]
        """
        file_paths = batch['file_path']
        B = len(file_paths)

        full_sequences_text = []
        prompt_text_only = []
        style_list = []  # list of [K, D_beats], later stack -> [B, K, D_beats]

        K = self.style_num_tokens

        for i in range(B):
            fpath = Path(file_paths[i])
            filename = fpath.name
            desc_data = self.audioset_descriptions.get(filename)
            if not desc_data:
                full_sequences_text.append("[PAD]")
                prompt_text_only.append("[PAD]")
                # A style chunk filled with 0 as a placeholder
                style_list.append(torch.zeros(K, self.beats_embed_dim, device=self.device))
                continue

            event = desc_data['event'].strip()
            description = desc_data['description'].strip()

            waveform = batch['waveform'][i:i + 1].to(self.device, non_blocking=True)

            with torch.no_grad():
                # Extract the AST tokens and the K style chunks
                as_tokens, style_BKD = self._extract_as_and_style(waveform, K)

            # Target token string
            as_tokens_str = "".join([f'[AS_{tok.item()}]' for tok in as_tokens])

            # Style placeholder string
            audio_placeholders = "".join([f'[AUDIO_{j}]' for j in range(K)])

            # prompt and target
            prompt_part = f"[TAG] {event} [DES] {description} {audio_placeholders} [AT]"
            target_part = f"{as_tokens_str}[END]"

            full_sequences_text.append(prompt_part + " " + target_part)  # Note the space
            prompt_text_only.append(prompt_part)
            style_list.append(style_BKD)

        # Tokenize the whole batch
        tokenized_full = self.llm_tokenizer(
            full_sequences_text, padding='longest', truncation=True,
            max_length=self.config['hyperparameters']['llm']['max_sequence_length'],
            return_tensors='pt'
        ).to(self.device)
        input_ids = tokenized_full['input_ids']
        attention_mask = tokenized_full['attention_mask']

        # prompt length (used for the labels mask)
        tokenized_prompts = self.llm_tokenizer(prompt_text_only, padding='longest', return_tensors='pt').to(self.device)
        prompt_lengths = tokenized_prompts.attention_mask.sum(dim=1)

        # Build labels: mask out the prompt and pad
        labels = input_ids.clone()
        pad_id = self.llm_tokenizer.pad_token_id
        for b in range(labels.size(0)):
            prompt_len = int(prompt_lengths[b].item())
            labels[b, :prompt_len] = -100
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Stack the style chunks
        style_feats = torch.stack(style_list, dim=0)  # [B, K, D_beats]

        # Filter out completely invalid samples
        valid_indices = [idx for idx, label_row in enumerate(labels) if (label_row != -100).any()]
        if len(valid_indices) < B:
            logger.warning(f"Dropping {B - len(valid_indices)} invalid samples from the batch.")
            input_ids = input_ids[valid_indices]
            attention_mask = attention_mask[valid_indices]
            labels = labels[valid_indices]
            style_feats = style_feats[valid_indices]

        return input_ids, attention_mask, labels, style_feats  # Also return style_feats

    def _inject_style_inputs_embeds(self, input_ids, style_feats, is_train: bool):
        """
        Build inputs_embeds and inject the style into the [AUDIO_i] positions (linear interpolation):
        emb'[pos] = (1-α)*emb[pos] + α*Proj(style[i])
        where α can be subject to Style Dropout during training (CFG no-style path).
        """
        # Get the embedding layer (go through module for compatibility with the DeepSpeed wrapper)
        embed_layer = self.model_engine.module.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # [B, T, H]

        B, T, H = inputs_embeds.shape
        K = self.style_num_tokens
        assert style_feats.shape[:2] == (B, K), "style_feats has the wrong dimensions, [B,K,D_beats]"

        # Project to the hidden dimension
        style_proj = self.model_engine.module.style_proj  # Linear(D_beats->H)
        style_H = style_proj(style_feats)  # [B, K, H]

        # Style Dropout: set α=0 with probability p during training; not done during validation
        p = self.model_engine.module.style_dropout_p if is_train else 0.0
        alpha_base = float(self.model_engine.module.style_strength)
        # Generate α for each sample
        alphas = []
        for b in range(B):
            if is_train and random.random() < p:
                alphas.append(0.0)
            else:
                alphas.append(alpha_base)
        alphas = torch.tensor(alphas, device=inputs_embeds.device).view(B, 1, 1)  # [B,1,1]

        # Find the position of each [AUDIO_i] and interpolate
        for i in range(K):
            tok_id = self.audio_token_ids[i]
            pos = (input_ids == tok_id)  # [B, T] boolean
            if not pos.any():
                continue
            # Original embeddings
            orig = inputs_embeds[pos]  # [M, H]
            # Style embeddings (broadcast per sample)
            # Build a [B, T, H] style tensor, but only take values at the pos positions
            styl_full = torch.zeros_like(inputs_embeds)
            styl_full[:, :, :] = 0.0
            styl_full[:, :, :] = 0.0  # placeholder
            # Expand the i-th style chunk to [B,1,H] and then broadcast over T
            styl_expand = style_H[:, i, :].unsqueeze(1).expand(B, T, H)
            # Linear interpolation
            mixed = (1.0 - alphas) * inputs_embeds + alphas * styl_expand
            # Replace only at the pos positions
            inputs_embeds[pos] = mixed[pos]

        return inputs_embeds  # [B, T, H]

    def _compute_coarse_loss(self, hidden_states, labels):
        """
        Compute the auxiliary loss of the coarse head:
          - Supervise the coarse cluster id only at positions where labels != -100 and the label is an [AS_k]
        """
        # hidden_states: [B, T, H] (last layer)
        B, T, H = hidden_states.shape
        # Find the valid positions
        valid_mask = (labels != -100)  # [B, T]
        # Keep only the labels that are [AS_*]
        label_ids = labels.clone()
        # Build the id -> as_code map (set to -1 if absent)
        id_to_as = self.id_to_as_code
        as_code_map = torch.full_like(label_ids, fill_value=-1)
        # Iterate over the batch (to avoid building a huge lookup tensor)
        for b in range(B):
            for t in torch.nonzero(valid_mask[b], as_tuple=False).view(-1):
                tok_id = int(label_ids[b, t].item())
                if tok_id in id_to_as:
                    as_code = id_to_as[tok_id]
                    as_code_map[b, t] = as_code

        pos_idx = torch.nonzero(as_code_map >= 0, as_tuple=False)  # [N,2]
        if pos_idx.numel() == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        feats = hidden_states[pos_idx[:, 0], pos_idx[:, 1], :]  # [N, H]
        coarse_logits = self.model_engine.module.coarse_head(feats)  # [N, C]

        # Target coarse id
        as_codes = as_code_map[pos_idx[:, 0], pos_idx[:, 1]]  # [N]
        coarse_targets = self.as_code_to_cluster[as_codes]  # [N]

        loss_coarse = F.cross_entropy(coarse_logits, coarse_targets)
        return loss_coarse

    def train(self):
        print("Starting LLM model training...")
        self._setup_dataloaders(is_train=True)
        self.setup_lr_scheduler()

        num_epochs = self.config['hyperparameters']['llm']['num_epochs']
        rank = self.cmd_args.local_rank

        # Auxiliary loss weight (can add hyperparameters.llm.coarse_loss_weight in the config, default 0.5)
        coarse_w = float(self.config['hyperparameters']['llm'].get('coarse_loss_weight', 0.5))

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model_engine.train()
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            pbar = self.train_loader
            if rank <= 0:
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=150)

            for step, batch in enumerate(pbar):
                input_ids, attention_mask, labels, style_feats = self._prepare_batch_data(batch, is_train=True)

                if input_ids.size(0) == 0:
                    logger.warning(f"Rank {rank}: Skipping an invalid batch during training (all samples were filtered out).")
                    continue

                # Build inputs_embeds with style injection
                inputs_embeds = self._inject_style_inputs_embeds(input_ids, style_feats, is_train=True)

                outputs = self.model_engine(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                    use_cache=False
                )
                loss_lm = outputs.loss

                # Take the last layer's hidden states to compute the coarse auxiliary loss
                last_h = outputs.hidden_states[-1]  # [B,T,H]
                loss_coarse = self._compute_coarse_loss(last_h, labels)

                loss = loss_lm + coarse_w * loss_coarse

                self.model_engine.backward(loss)
                self.model_engine.step()

                if self.lr_scheduler and self.model_engine.is_gradient_accumulation_boundary():
                    self.lr_scheduler.step()

                self.global_step += 1

                if rank <= 0:
                    current_lr = self.pytorch_optimizer.param_groups[0]['lr']
                    ppl = torch.exp(loss_lm.detach()).item()
                    pbar.set_postfix(
                        {'loss': f"{loss.item():.4f}", 'lm': f"{loss_lm.item():.4f}",
                         'coarse': f"{loss_coarse.item():.4f}", 'ppl': f"{ppl:.2f}",
                         'lr': f"{current_lr:.2e}"})
                    if self.config['logging']['wandb']['enabled']:
                        wandb.log({
                            'train/loss_total': loss.item(),
                            'train/loss_lm': loss_lm.item(),
                            'train/loss_coarse': loss_coarse.item(),
                            'train/perplexity_lm': ppl,
                            'train/learning_rate': current_lr,
                            'global_step': self.global_step
                        })

            self.validate()

    def validate(self):
        rank = self.cmd_args.local_rank
        if rank <= 0:
            print("Starting validation...")

        if self.val_loader is None:
            self._setup_dataloaders(is_train=False)

        self.model_engine.eval()
        total_val_loss = 0.0
        total_batches = 0
        total_val_lm = 0.0
        total_val_coarse = 0.0

        coarse_w = float(self.config['hyperparameters']['llm'].get('coarse_loss_weight', 0.5))

        pbar = self.val_loader
        if rank <= 0:
            pbar = tqdm(self.val_loader, desc="Validating", ncols=120)

        with torch.no_grad():
            for batch in pbar:
                input_ids, attention_mask, labels, style_feats = self._prepare_batch_data(batch, is_train=False)

                if input_ids.size(0) == 0:
                    logger.warning(f"Rank {rank}: Skipping an invalid batch during validation.")
                    continue

                # No Style Dropout during validation (α=style_strength)
                inputs_embeds = self._inject_style_inputs_embeds(input_ids, style_feats, is_train=False)

                outputs = self.model_engine(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                    use_cache=False
                )
                loss_lm = outputs.loss
                last_h = outputs.hidden_states[-1]
                loss_coarse = self._compute_coarse_loss(last_h, labels)

                loss = loss_lm + coarse_w * loss_coarse

                total_val_loss += loss.item()
                total_val_lm += loss_lm.item()
                total_val_coarse += loss_coarse.item()
                total_batches += 1

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if world_size > 1:
            # Aggregate
            def _allreduce_sum(x):
                t = torch.tensor(x, device=self.device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                return t.item()

            total_val_loss = _allreduce_sum(total_val_loss)
            total_val_lm = _allreduce_sum(total_val_lm)
            total_val_coarse = _allreduce_sum(total_val_coarse)
            total_batches_t = torch.tensor(total_batches, device=self.device)
            torch.distributed.all_reduce(total_batches_t, op=torch.distributed.ReduceOp.SUM)
            total_batches = int(total_batches_t.item())

        avg_val_loss = total_val_loss / total_batches if total_batches > 0 else float('inf')
        avg_val_lm = total_val_lm / total_batches if total_batches > 0 else float('inf')
        avg_val_coarse = total_val_coarse / total_batches if total_batches > 0 else float('inf')

        is_best_model = avg_val_loss < self.best_val_loss

        if rank <= 0:
            avg_perplexity = np.exp(avg_val_lm) if avg_val_lm < 20 else float('inf')
            print(
                f"Validation complete | Total loss: {avg_val_loss:.4f} | LM: {avg_val_lm:.4f} | Coarse: {avg_val_coarse:.4f} | PPL(LM): {avg_perplexity:.2f}")

            if self.config['logging']['wandb']['enabled']:
                wandb.log({
                    'val/loss_total': avg_val_loss,
                    'val/loss_lm': avg_val_lm,
                    'val/loss_coarse': avg_val_coarse,
                    'val/ppl_lm': avg_perplexity,
                    'epoch': self.current_epoch + 1
                })

        if is_best_model:
            self.best_val_loss = avg_val_loss
            if rank <= 0:
                print(f"Found a new best model, validation loss: {self.best_val_loss:.4f}. Saving...")
            filename = f"best_model_loss_{self.best_val_loss:.4f}_epoch_{self.current_epoch + 1}.pth"
            self.save_checkpoint(filename)

    def save_checkpoint(self, filename):
        """
        Save the best model checkpoint as a single .pth file.
        """
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.cmd_args.local_rank <= 0:
            try:
                checkpoint_file_path = self.checkpoint_dir / filename
                print(f"Preparing to save the checkpoint to {checkpoint_file_path}")

                # Extract the unwrapped model state dict from the DeepSpeed engine
                model_state_dict = self.model_engine.module.state_dict()

                # Collect training state information
                client_state = {
                    'epoch': self.current_epoch + 1,
                    'global_step': self.global_step,
                    'best_val_loss': self.best_val_loss,
                    'config_yaml': self.config,
                }

                content_to_save = {
                    'model_state_dict': model_state_dict,
                    'pytorch_optimizer_state_dict': self.pytorch_optimizer.state_dict() if self.pytorch_optimizer else None,
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                    'client_state': client_state,
                }

                # Save as a single file
                torch.save(content_to_save, checkpoint_file_path)
                print(f"Best model checkpoint saved to: {checkpoint_file_path}")

            except Exception as e:
                logger.error(f"Failed to save the checkpoint: {e}")
                logger.error(traceback.format_exc())

        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def main():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Train an LLM on AudioSet using DeepSpeed (with style prefix & Coarse head)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the configuration file")
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed by DeepSpeed")
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    setup_logger(args.local_rank)
    deepspeed.init_distributed()

    global trainer_global_object
    try:
        trainer_global_object = LLMTrainer(args.config, args)
        trainer_global_object.train()
    except Exception as e:
        logger.error("A critical error occurred during training!")
        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        print("Training finished.")


if __name__ == "__main__":
    faulthandler.enable()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    seed = 666
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    logger = setup_logger()
    print("Deterministic settings enabled (cudnn.benchmark=False, cudnn.deterministic=True).")

    main()
