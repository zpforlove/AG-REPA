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
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from beats.BEATs import BEATs, BEATsConfig
from beats.Tokenizers import Tokenizers, TokenizersConfig
from data_loader import LibriSpeechDataset, AudioSetDataset
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


class BalancedDistributedSampler(DistributedSampler):
    """
    A distributed sampler designed to balance data from different subsets within a ConcatDataset.
    By oversampling the smaller datasets, it makes the number of samples provided by each subset
    equal to that of the largest subset in each epoch.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("The dataset must be of type ConcatDataset")

        # Call the parent constructor, but we override part of the computation logic
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.dataset_lengths = [len(d) for d in dataset.datasets]
        self.max_len = max(self.dataset_lengths)

        # First compute the unaligned total number of samples obtained via upsampling
        unpadded_total_size = self.max_len * len(dataset.datasets)

        # Correctly compute the per-GPU sample count and the aligned total sample count based on drop_last
        if self.drop_last:
            # If extra samples are dropped, round down (using integer division) to ensure the total is divisible
            self.num_samples = unpadded_total_size // self.num_replicas
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If not dropping, round up (using math.ceil) and pad to fill in the samples
            self.num_samples = math.ceil(unpadded_total_size / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        # Generate indices for each sub-dataset
        for i, dataset_len in enumerate(self.dataset_lengths):
            # Upsample the indices of the current dataset so that it reaches max_len
            sub_indices = torch.randint(high=dataset_len, size=(self.max_len,), generator=g).tolist()

            # Add the offset to convert to global indices within the ConcatDataset
            offset = self.dataset.cumulative_sizes[i - 1] if i > 0 else 0
            indices.extend([idx + offset for idx in sub_indices])

        if self.shuffle:
            # Shuffle across all generated global indices
            shuffled_order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffled_order]

        # --- Adjust the length of the index list based on drop_last ---
        if not self.drop_last:
            # Pad the indices to ensure all ranks have the same number of samples
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            # Truncate the extra indices
            indices = indices[:self.total_size]

        assert len(
            indices) == self.total_size, f"Index list length ({len(indices)}) does not match the expected total size ({self.total_size})!"

        # Assign the subset for the current rank
        subset_indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(
            subset_indices) == self.num_samples, f"Subset index length ({len(subset_indices)}) does not match the expected sample count ({self.num_samples})!"

        return iter(subset_indices)

    def __len__(self):
        return self.num_samples


class LLMTrainer:
    def __init__(self, config_path, cmd_args):
        """
        Initialize the LLM trainer, adapted to DeepSpeed and the new BEATs-only data pipeline
        """
        self.cmd_args = cmd_args
        self.config = load_config(config_path)
        global logger
        if logger is None:
            logger = setup_logger(self.cmd_args.local_rank)

        print(f"Rank {self.cmd_args.local_rank}: Starting trainer initialization...")

        # Initialize components
        self.beats_feature_extractor = None
        self.beats_tokenizer = None
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
        self.librispeech_train_trans = {}
        self.librispeech_val_trans = {}
        self.audioset_descriptions = {}

        # --- Members related to style and coarse->fine ---
        self.style_num_tokens = int(self.config['hyperparameters']['llm'].get('style_num_tokens', 16))
        self.style_strength = float(self.config['hyperparameters']['llm'].get('style_strength', 0.5))
        self.style_dropout_p = float(self.config['hyperparameters']['llm'].get('style_dropout_p', 0.2))

        # Coarse now targets the BEATs Token
        self.coarse_num_clusters = int(self.config['hyperparameters']['llm'].get('coarse_num_clusters', 128))
        self.coarse_code_to_cluster_path = self.config['hyperparameters']['llm'].get(
            'coarse_code_to_cluster_path', None
        )
        self.beats_vocab_size = int(self.config['hyperparameters']['llm']['beats_vocab_size'])

        # tokenizer ID mapping
        self.beats_token_ids = []  # tokenizer ids for [BEATs_0..BEATs_{V-1}]
        self.id_to_beats_code = {}  # tokenizer id -> k
        self.audio_token_ids = []  # tokenizer ids for [AUDIO_0..K-1]

        # Mapping from coarse codeword to cluster id (now maps the BEATs codebook)
        self.beats_code_to_cluster = None

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
        # Keep the original metadata-loading logic
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
            logger.error(f"Failed to load the ontology file: {ontology_path}, error: {e}")
            raise

        def load_segments_df(csv_path):
            try:
                df = pd.read_csv(csv_path, header=None, comment='#', quotechar='"', skipinitialspace=True,
                                 names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'])
                df.set_index('YTID', inplace=True)
                print(f"CSV segments file loaded successfully: {csv_path}")
                return df
            except Exception as e:
                logger.error(f"Failed to load the CSV file: {csv_path}, error: {e}")
                raise

        self.train_segments_df = load_segments_df(os.path.join(base_dir, "unbalanced_train_segments.csv"))
        self.eval_segments_df = load_segments_df(os.path.join(base_dir, "eval_segments.csv"))

    def _setup_environment(self):
        rank = self.cmd_args.local_rank
        self.device = f'cuda:{rank}' if torch.cuda.is_available() and rank != -1 else 'cpu'

        def _build_librispeech_trans_map(root_path):
            trans_map = {}
            trans_files = list(Path(root_path).rglob("*.trans.txt"))
            print(f"Found {len(trans_files)} transcription files in {root_path}.")
            for trans_file in trans_files:
                with open(trans_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split(' ', 1)
                        if len(parts) == 2:
                            file_id, text = parts
                            trans_map[file_id] = text.lower()
            return trans_map

        try:
            # 1. Load the LibriSpeech transcriptions
            print("Building the LibriSpeech training-set transcription map...")
            self.librispeech_train_trans = _build_librispeech_trans_map(
                self.config['data']['librispeech']['train_root'])
            print("Building the LibriSpeech validation-set transcription map...")
            self.librispeech_val_trans = _build_librispeech_trans_map(self.config['data']['librispeech']['val_root'])

            # 2. Load the AudioSet descriptions
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

            # --- 3. Load the BEATs Feature Extractor (for Style) ---
            print(f"Rank {rank}: Loading the BEATs Feature Extractor...")
            beats_checkpoint_path = self.config['paths']['beats_feature_extractor_checkpoint']
            if not os.path.exists(beats_checkpoint_path):
                raise FileNotFoundError(f"BEATs feature extractor checkpoint not found at: {beats_checkpoint_path}")

            checkpoint = torch.load(beats_checkpoint_path, map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.beats_feature_extractor = BEATs(cfg)
            self.beats_feature_extractor.load_state_dict(checkpoint['model'])
            self.beats_feature_extractor.eval()

            # Record the BEATs embedding dimension (the style input dimension)
            self.beats_embed_dim = getattr(cfg, 'encoder_embed_dim', 768)
            print(
                f"Rank {rank}: BEATs Feature Extractor loaded successfully from {beats_checkpoint_path}. Embed Dim: {self.beats_embed_dim}")

            # --- 4. Load the BEATs Tokenizer (for generating targets) ---
            beats_tokenizer_path = self.config['paths']['beats_tokenizer']
            if not os.path.exists(beats_tokenizer_path):
                raise FileNotFoundError(f"BEATs Tokenizer checkpoint not found at: {beats_tokenizer_path}")

            logger.info(f"Rank {rank}: Loading the BEATs Tokenizer from {beats_tokenizer_path}...")
            tokenizer_checkpoint = torch.load(beats_tokenizer_path, map_location='cpu')
            tokenizer_cfg = TokenizersConfig(tokenizer_checkpoint['cfg'])
            self.beats_tokenizer = Tokenizers(tokenizer_cfg)
            self.beats_tokenizer.load_state_dict(tokenizer_checkpoint['model'])
            self.beats_tokenizer.eval()
            logger.info(f"Rank {rank}: BEATs Tokenizer loaded successfully and set to eval mode.")

        except Exception as e:
            logger.error(f"Rank {rank}: Environment setup failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def _setup_wandb(self):
        if self.config['logging']['wandb']['enabled']:
            try:
                wandb.init(
                    project=self.config['logging']['wandb']['project'],
                    name=self.config['logging']['wandb']['name'] + "_LLM_BEATsOnly",
                    config={**self.config, **vars(self.cmd_args)}
                )
                print("WandB initialized successfully")
            except Exception as e:
                logger.error(f"WandB initialization failed: {e}")
                self.config['logging']['wandb']['enabled'] = False

    def _setup_model_and_deepspeed(self):
        print(f"Loading LLM Tokenizer: {self.config['hyperparameters']['llm']['model_name']}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.config['hyperparameters']['llm']['model_name'],
            trust_remote_code=True
        )
        # Extend the vocabulary
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

        # ============= Add the style and coarse->fine modules =============
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None) or getattr(model.config, "dim", None)
            if hidden_size is None:
                raise ValueError("Could not infer hidden_size from the LLM config.")

        # 1) style projection layer: BEATs -> LLM hidden dim
        model.style_proj = nn.Linear(self.beats_embed_dim, hidden_size)
        # Training-time tunable injection strength and dropout probability (saved on the model for distributed consistency)
        model.style_strength = self.style_strength
        model.style_dropout_p = self.style_dropout_p
        model.style_num_tokens = self.style_num_tokens

        # 2) coarse head: hidden -> num_clusters
        # Note: this is now used to predict the cluster that a BEATs Token belongs to
        model.coarse_head = nn.Linear(hidden_size, self.coarse_num_clusters)

        # Optimizer
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

        # Move the BEATs feature extractor and BEATs tokenizer to the target device
        self.beats_feature_extractor.to(self.device)
        self.beats_tokenizer.to(self.device)

        # coarse: build the BEATs code->cluster mapping
        self._build_coarse_mapping()

    def _extend_tokenizer_vocab(self):
        """
        Extend the vocabulary:
          - Special tokens: [TEXT], [ST], [TAG], [DES], [AT], [END], [PAD]
          - BEATs vocabulary tokens: [BEATs_0]..[BEATs_{V-1}]
          - Style placeholder tokens: [AUDIO_0]..[AUDIO_{K-1}]
        * Note: S3 and AS Tokens have been removed
        """
        print("Extending the tokenizer vocabulary...")

        beats_vocab_size = int(self.config['hyperparameters']['llm']['beats_vocab_size'])
        style_num_tokens = self.style_num_tokens

        # Define all the required tokens
        special_tokens = ['[TEXT]', '[ST]', '[TAG]', '[DES]', '[AT]', '[END]', '[PAD]']
        beats_target_tokens = [f'[BEATs_{j}]' for j in range(beats_vocab_size)]
        audio_style_tokens = [f'[AUDIO_{j}]' for j in range(style_num_tokens)]

        # Add all tokens at once
        num_added = self.llm_tokenizer.add_tokens(
            special_tokens + beats_target_tokens + audio_style_tokens,
            special_tokens=True
        )

        print(f"Added {num_added} tokens.")

        # Ensure the tokenizer has a pad_token
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Record commonly used IDs
        self.special_token_ids = {
            'text': self.llm_tokenizer.convert_tokens_to_ids('[TEXT]'),
            'st': self.llm_tokenizer.convert_tokens_to_ids('[ST]'),
            'tag': self.llm_tokenizer.convert_tokens_to_ids('[TAG]'),
            'des': self.llm_tokenizer.convert_tokens_to_ids('[DES]'),
            'at': self.llm_tokenizer.convert_tokens_to_ids('[AT]'),
            'end': self.llm_tokenizer.convert_tokens_to_ids('[END]'),
            'pad': self.llm_tokenizer.convert_tokens_to_ids('[PAD]'),
        }

        # Record [BEATs_k] -> id, and id -> k (used for Coarse Loss)
        self.beats_token_ids = []
        self.id_to_beats_code = {}
        for j in range(beats_vocab_size):
            tid = self.llm_tokenizer.convert_tokens_to_ids(f'[BEATs_{j}]')
            self.beats_token_ids.append(tid)
            self.id_to_beats_code[tid] = j

        # Record the ids of [AUDIO_i] (used for Style Injection)
        self.audio_token_ids = [self.llm_tokenizer.convert_tokens_to_ids(f'[AUDIO_{i}]')
                                for i in range(style_num_tokens)]

        print(f"Vocabulary size after extension: {len(self.llm_tokenizer)}")

    def _build_coarse_mapping(self):
        """Build the mapping from BEATs codeword -> coarse cluster id"""
        V = self.beats_vocab_size
        C = self.coarse_num_clusters
        mapping = None

        if self.coarse_code_to_cluster_path and os.path.exists(self.coarse_code_to_cluster_path):
            try:
                with open(self.coarse_code_to_cluster_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    mapping = [int(data[str(i)]) for i in range(V)]
                elif isinstance(data, list) and len(data) == V:
                    mapping = [int(x) for x in data]
                else:
                    logger.warning("The format of coarse_code_to_cluster_path is unexpected; falling back to uniform bucketing.")
            except Exception as e:
                logger.warning(f"Failed to read the coarse mapping; falling back to uniform bucketing: {e}")

        if mapping is None:
            bucket = max(V // C, 1)
            mapping = [min(i // bucket, C - 1) for i in range(V)]

        self.beats_code_to_cluster = torch.tensor(mapping, dtype=torch.long, device=self.device)

    def _setup_dataloaders(self, is_train=True):
        is_distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        if is_train:
            print("Initializing the training data loader (LibriSpeech + AudioSet)...")
            ls_train_ds = LibriSpeechDataset(self.config['data']['librispeech']['train_root'], self.config)
            as_train_ds = AudioSetDataset(self.config['data']['audioset']['train_root'], self.config)
            dataset = ConcatDataset([ls_train_ds, as_train_ds])

            if is_distributed:
                sampler = BalancedDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            else:
                logger.warning("In non-distributed mode, the balanced sampler is not used.")
                sampler = None

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
            print("Initializing the validation data loader (LibriSpeech + AudioSet)...")
            ls_val_ds = LibriSpeechDataset(self.config['data']['librispeech']['val_root'], self.config)
            as_val_ds = AudioSetDataset(self.config['data']['audioset']['val_root'], self.config)
            dataset = ConcatDataset([ls_val_ds, as_val_ds])

            sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                         shuffle=False) if is_distributed else None
            self.val_loader = DataLoader(
                dataset, batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                sampler=sampler, shuffle=False, persistent_workers=True,
                num_workers=self.config['data']['num_workers'], pin_memory=self.config['data']['pin_memory']
            )
            print(f"Validation data loader initialized, containing {len(self.val_loader)} batches.")

    def setup_lr_scheduler(self):
        """Set up the OneCycleLR scheduler based on the config file."""
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error(f"Rank {self.cmd_args.local_rank}: Optimizer or training data loader not initialized; cannot create the LR scheduler.")
            return
        print(f"Rank {self.cmd_args.local_rank}: Setting up the LR scheduler...")
        try:
            scheduler_config = self.config['hyperparameters']['llm']['optimizer']['scheduler']
            num_epochs = self.config['hyperparameters']['llm']['num_epochs']
            gradient_accumulation_steps = self.model_engine.gradient_accumulation_steps()

            total_micro_batches = len(self.train_loader) * num_epochs
            total_steps = math.ceil(total_micro_batches / gradient_accumulation_steps)

            if total_steps <= 0:
                logger.error(f"Rank {self.cmd_args.local_rank}: The computed total number of steps ({total_steps}) is invalid; the LR scheduler was not created.")
                return

            print(f"Rank {self.cmd_args.local_rank}: Total steps for the LR scheduler: {total_steps}")
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
            logger.error(f"Rank {self.cmd_args.local_rank}: Failed to create the LR scheduler. Missing key in config: {e}.")
            raise
        except Exception as e:
            logger.error(f"Rank {self.cmd_args.local_rank}: Failed to set up the LR scheduler: {e}\n{traceback.format_exc()}")
            raise

    @torch.no_grad()
    def _extract_style_feats(self, waveform, K):
        """
        Extract BEATs features from the waveform, split them evenly into K segments, and take the mean of each segment as a style vector
        Return: [K, D_beats]
        """
        beats_features, _ = self.beats_feature_extractor.extract_features(waveform)  # [1, T, D]
        T, D = beats_features.shape[1], beats_features.shape[2]

        idx_edges = torch.linspace(0, T, steps=K + 1, device=beats_features.device).long()
        style_chunks = []
        for i in range(K):
            a, b = idx_edges[i].item(), idx_edges[i + 1].item()
            if b <= a:
                b = min(a + 1, T)
            seg = beats_features[0, a:b, :]  # [len, D]
            style_chunks.append(seg.mean(dim=0, keepdim=True))  # [1, D]

        style_BKD = torch.cat(style_chunks, dim=0)  # [K, D]
        return style_BKD

    def _prepare_batch_data(self, batch, is_train):
        """
        Prepare a single batch of data for the LLM, modified to generate only BEATs Tokens:
        1. TTS (LibriSpeech): [TEXT] text [AUDIO_0]..[AUDIO_K] [ST] -> [BEATs]...
        2. TTA (AudioSet): [TAG] events [DES] desc [AUDIO_0]..[AUDIO_K] [AT] -> [BEATs]...
        """
        file_paths = batch['file_path']
        dataset_types = batch['dataset_type']
        B = len(file_paths)

        full_sequences_text = []
        prompt_text_only = []
        style_list = []  # list of [K, D_beats]

        trans_map = self.librispeech_train_trans if is_train else self.librispeech_val_trans
        K = self.style_num_tokens
        # Style placeholder string
        audio_placeholders = "".join([f'[AUDIO_{j}]' for j in range(K)])

        for i in range(B):
            dtype = dataset_types[i]
            fpath = Path(file_paths[i])

            # Obtain the waveform uniformly
            waveform = batch['waveform'][i:i + 1].to(self.device, non_blocking=True)

            # --- Extract style features ---
            try:
                style_BKD = self._extract_style_feats(waveform, K)
            except Exception as e:
                full_sequences_text.append("[PAD]")
                prompt_text_only.append("[PAD]")
                style_list.append(torch.zeros(K, self.beats_embed_dim, device=self.device))
                continue

            # --- Extract the target Token (BEATs) ---
            # Regardless of the dataset, the target is BEATs Tokens
            with torch.no_grad():
                beats_indices = self.beats_tokenizer.extract_labels(waveform).view(-1).tolist()

            # Construct the target string
            beats_tokens_str = [f'[BEATs_{idx}]' for idx in beats_indices]
            target_part = "".join(beats_tokens_str) + "[END]"

            if dtype == 'librispeech':
                file_id = fpath.stem
                speech_text = trans_map.get(file_id, "").strip()
                if not speech_text or not beats_indices:
                    full_sequences_text.append("[PAD]")
                    prompt_text_only.append("[PAD]")
                    style_list.append(style_BKD)
                    continue

                # Construct the TTS prompt: TEXT + style + ST
                prompt_part = f"[TEXT] {speech_text} {audio_placeholders} [ST]"

            elif dtype == 'audioset':
                filename = fpath.name
                desc_data = self.audioset_descriptions.get(filename)
                if not desc_data or not beats_indices:
                    full_sequences_text.append("[PAD]")
                    prompt_text_only.append("[PAD]")
                    style_list.append(style_BKD)
                    continue

                event = desc_data['event'].strip()
                description = desc_data['description'].strip()

                # Construct the TTA prompt: TAG/DES + style + AT
                prompt_part = f"[TAG] {event} [DES] {description} {audio_placeholders} [AT]"

            else:
                full_sequences_text.append("[PAD]")
                prompt_text_only.append("[PAD]")
                style_list.append(style_BKD)
                continue

            # Add to the lists normally
            full_sequences_text.append(prompt_part + " " + target_part)
            prompt_text_only.append(prompt_part)
            style_list.append(style_BKD)

        # Tokenize
        tokenized_full = self.llm_tokenizer(
            full_sequences_text, padding='longest', truncation=True,
            max_length=self.config['hyperparameters']['llm']['max_sequence_length'],
            return_tensors='pt'
        ).to(self.device)
        input_ids = tokenized_full['input_ids']
        attention_mask = tokenized_full['attention_mask']

        # Prompt Mask
        tokenized_prompts = self.llm_tokenizer(prompt_text_only, padding='longest', return_tensors='pt').to(self.device)
        prompt_lengths = tokenized_prompts.attention_mask.sum(dim=1)

        labels = input_ids.clone()
        pad_id = self.llm_tokenizer.pad_token_id
        for b in range(labels.size(0)):
            prompt_len = int(prompt_lengths[b].item())
            labels[b, :prompt_len] = -100
        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Stack the style features
        style_feats = torch.stack(style_list, dim=0)  # [B, K, D]

        # Filter out invalid samples
        valid_indices = [idx for idx, label_row in enumerate(labels) if (label_row != -100).any()]
        if len(valid_indices) < B:
            logger.warning(f"Dropping {B - len(valid_indices)} invalid samples from the batch.")
            input_ids = input_ids[valid_indices]
            attention_mask = attention_mask[valid_indices]
            labels = labels[valid_indices]
            style_feats = style_feats[valid_indices]

        return input_ids, attention_mask, labels, style_feats

    def _inject_style_inputs_embeds(self, input_ids, style_feats, is_train: bool):
        """
        Construct inputs_embeds and inject the style into the [AUDIO_i] positions (linear interpolation):
        emb'[pos] = (1-alpha)*emb[pos] + alpha*Proj(style[i])
        """
        embed_layer = self.model_engine.module.get_input_embeddings()
        inputs_embeds = embed_layer(input_ids)  # [B, T, H]

        B, T, H = inputs_embeds.shape
        K = self.style_num_tokens

        # Projection
        style_proj = self.model_engine.module.style_proj  # [B, K, H]
        style_H = style_proj(style_feats)

        # Style Dropout
        p = self.model_engine.module.style_dropout_p if is_train else 0.0
        alpha_base = float(self.model_engine.module.style_strength)

        alphas = []
        for b in range(B):
            if is_train and random.random() < p:
                alphas.append(0.0)
            else:
                alphas.append(alpha_base)
        alphas = torch.tensor(alphas, device=inputs_embeds.device).view(B, 1, 1)

        # Injection
        for i in range(K):
            tok_id = self.audio_token_ids[i]
            pos = (input_ids == tok_id)
            if not pos.any():
                continue

            styl_expand = style_H[:, i, :].unsqueeze(1).expand(B, T, H)
            mixed = (1.0 - alphas) * inputs_embeds + alphas * styl_expand
            inputs_embeds[pos] = mixed[pos]

        return inputs_embeds

    def _compute_coarse_loss(self, hidden_states, labels):
        """
        Compute the auxiliary loss of the coarse head:
        supervise the coarse cluster id only at the positions where labels are [BEATs_k].
        """
        B, T, H = hidden_states.shape
        valid_mask = (labels != -100)

        label_ids = labels.clone()
        id_to_beats = self.id_to_beats_code
        beats_code_map = torch.full_like(label_ids, fill_value=-1)

        # Find all [BEATs_k] tokens in the batch
        for b in range(B):
            # Optimization: iterate only over the valid region
            valid_indices = torch.nonzero(valid_mask[b], as_tuple=False).view(-1)
            for t in valid_indices:
                tok_id = int(label_ids[b, t].item())
                if tok_id in id_to_beats:
                    beats_code_map[b, t] = id_to_beats[tok_id]

        pos_idx = torch.nonzero(beats_code_map >= 0, as_tuple=False)  # [N, 2]
        if pos_idx.numel() == 0:
            return torch.tensor(0.0, device=hidden_states.device)

        feats = hidden_states[pos_idx[:, 0], pos_idx[:, 1], :]  # [N, H]
        coarse_logits = self.model_engine.module.coarse_head(feats)  # [N, C]

        beats_codes = beats_code_map[pos_idx[:, 0], pos_idx[:, 1]]  # [N]
        coarse_targets = self.beats_code_to_cluster[beats_codes]  # [N]

        loss_coarse = F.cross_entropy(coarse_logits, coarse_targets)
        return loss_coarse

    def train(self):
        print("Starting LLM model training (BEATs-only mode)...")
        self._setup_dataloaders(is_train=True)
        self.setup_lr_scheduler()

        num_epochs = self.config['hyperparameters']['llm']['num_epochs']
        rank = self.cmd_args.local_rank
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
                # Prepare the data, including style_feats
                input_ids, attention_mask, labels, style_feats = self._prepare_batch_data(batch, is_train=True)

                if input_ids.size(0) == 0:
                    logger.warning(f"Rank {rank}: Skipping an invalid batch during training (all samples were filtered out).")
                    continue

                # Inject the style
                inputs_embeds = self._inject_style_inputs_embeds(input_ids, style_feats, is_train=True)

                outputs = self.model_engine(
                    inputs_embeds=inputs_embeds,  # Replaces the original input_ids
                    attention_mask=attention_mask,
                    labels=labels,
                    output_hidden_states=True,
                    use_cache=False
                )
                loss_lm = outputs.loss

                # Compute the Coarse auxiliary loss (now targeting the BEATs Token)
                last_h = outputs.hidden_states[-1]
                loss_coarse = self._compute_coarse_loss(last_h, labels)

                loss = loss_lm + coarse_w * loss_coarse

                self.model_engine.backward(loss)
                self.model_engine.step()

                if self.lr_scheduler and self.model_engine.is_gradient_accumulation_boundary():
                    self.lr_scheduler.step()

                self.global_step += 1

                if rank <= 0:
                    current_lr = self.pytorch_optimizer.param_groups[0]['lr']
                    perplexity = torch.exp(loss_lm.detach()).item()
                    pbar.set_postfix(
                        {'loss': f"{loss.item():.4f}", 'lm': f"{loss_lm.item():.4f}",
                         'coarse': f"{loss_coarse.item():.4f}", 'ppl': f"{perplexity:.2f}",
                         'lr': f"{current_lr:.2e}"})
                    if self.config['logging']['wandb']['enabled']:
                        wandb.log({
                            'train/loss_total': loss.item(),
                            'train/loss_lm': loss_lm.item(),
                            'train/loss_coarse': loss_coarse.item(),
                            'train/perplexity': perplexity,
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
        total_val_lm = 0.0
        total_val_coarse = 0.0
        total_batches = 0

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
            # Aggregate all losses
            def _allreduce_avg(val, count):
                t_sum = torch.tensor(val, device=self.device)
                t_cnt = torch.tensor(count, device=self.device)
                torch.distributed.all_reduce(t_sum, op=torch.distributed.ReduceOp.SUM)
                torch.distributed.all_reduce(t_cnt, op=torch.distributed.ReduceOp.SUM)
                return (t_sum / t_cnt).item() if t_cnt.item() > 0 else float('inf')

            avg_val_loss = _allreduce_avg(total_val_loss, total_batches)
            avg_val_lm = _allreduce_avg(total_val_lm, total_batches)
            avg_val_coarse = _allreduce_avg(total_val_coarse, total_batches)
        else:
            avg_val_loss = total_val_loss / total_batches if total_batches > 0 else float('inf')
            avg_val_lm = total_val_lm / total_batches if total_batches > 0 else float('inf')
            avg_val_coarse = total_val_coarse / total_batches if total_batches > 0 else float('inf')

        is_best_model = avg_val_loss < self.best_val_loss

        if rank <= 0:
            avg_perplexity = np.exp(avg_val_lm) if avg_val_lm < 20 else float('inf')
            print(
                f"Validation complete | Total loss: {avg_val_loss:.4f} | LM: {avg_val_lm:.4f} | Coarse: {avg_val_coarse:.4f} | PPL: {avg_perplexity:.2f}")

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

                # Collect training-state information
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

    parser = argparse.ArgumentParser(description="Train the LLM with DeepSpeed (BEATs-only generation mode, with Style and Coarse)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
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
        logger.error("A fatal error occurred during training!")
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
