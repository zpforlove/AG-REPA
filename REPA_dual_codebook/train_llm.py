import argparse
import faulthandler
import logging
import math
import os
import sys
import traceback
from pathlib import Path
import json

import deepspeed
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import wandb
from beats.BEATs import BEATs, BEATsConfig
from beats.Tokenizers import Tokenizers, TokenizersConfig
from data_loader import LibriSpeechDataset, AudioSetDataset
from models import AudioSetTokenizer
from utils import load_config

# Global variable declaration
global trainer_global_object

# --- Logging setup ---
global logger


def setup_logger(rank=-1):
    """Set up the global logger according to the process rank"""
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
    A distributed sampler designed to balance data from the different subsets within a ConcatDataset.
    By oversampling the smaller datasets, it ensures that in each epoch the number of samples
    provided by each subset is equal to the number of samples in the largest subset.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False):
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("The dataset must be of type ConcatDataset")

        # Call the parent constructor, but we will override part of the computation logic
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.dataset_lengths = [len(d) for d in dataset.datasets]
        self.max_len = max(self.dataset_lengths)

        # First compute the unaligned total number of samples obtained through upsampling
        unpadded_total_size = self.max_len * len(dataset.datasets)

        # Correctly compute the number of samples per GPU and the aligned total number of samples based on the drop_last parameter
        if self.drop_last:
            # If extra samples are to be dropped, round down (using integer division) to ensure the total is divisible
            self.num_samples = unpadded_total_size // self.num_replicas
            self.total_size = self.num_samples * self.num_replicas
        else:
            # If not dropping, round up (using math.ceil) and pad to complete the samples
            self.num_samples = math.ceil(unpadded_total_size / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        # Generate indices for each sub-dataset
        for i, dataset_len in enumerate(self.dataset_lengths):
            # Upsample the indices of the current dataset until they reach max_len
            sub_indices = torch.randint(high=dataset_len, size=(self.max_len,), generator=g).tolist()

            # Add the offset to convert to the global index within the ConcatDataset
            offset = self.dataset.cumulative_sizes[i - 1] if i > 0 else 0
            indices.extend([idx + offset for idx in sub_indices])

        if self.shuffle:
            # Shuffle across all generated global indices
            shuffled_order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffled_order]

        # --- Adjust the length of the index list based on the drop_last parameter ---
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

        # Assign a subset to the current rank
        subset_indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(
            subset_indices) == self.num_samples, f"Subset index length ({len(subset_indices)}) does not match the expected number of samples ({self.num_samples})!"

        return iter(subset_indices)

    def __len__(self):
        return self.num_samples


class LLMTrainer:
    def __init__(self, config_path, cmd_args):
        """
        Initialize the LLM trainer, adapted to DeepSpeed and the new AudioSet data pipeline

        Args:
            config_path: Path to the configuration file
            cmd_args: Command-line arguments, including the deepspeed configuration
        """
        self.cmd_args = cmd_args
        self.config = load_config(config_path)
        global logger
        if logger is None:
            logger = setup_logger(self.cmd_args.local_rank)

        print(f"Rank {self.cmd_args.local_rank}: Starting trainer initialization...")

        # Initialize the components
        self.audioset_tokenizer = None
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
        print(f"Loading metadata using absolute base directory: {base_dir}")
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

        # Load the training and validation segment data (note: if the training data changes, the CSV file names need to be modified)
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
                            # Convert all uppercase text to lowercase
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

            # --- 3. Load the AudioSet Tokenizer and the BEATs Feature Extractor ---
            # 3.1 Load the BEATs Feature Extractor
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

            # 3.2 Automatically find and load the best AudioSetTokenizer
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

            # --- 4. Load the BEATs Tokenizer ---
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
                    name=self.config['logging']['wandb']['name'] + "_LLM",
                    config={**self.config, **vars(self.cmd_args)}
                )
                print("WandB initialized successfully")
            except Exception as e:
                logger.error(f"WandB initialization failed: {e}")
                self.config['logging']['wandb']['enabled'] = False

    def _setup_model_and_deepspeed(self):
        print(f"Loading the LLM Tokenizer: {self.config['hyperparameters']['llm']['model_name']}...")
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            self.config['hyperparameters']['llm']['model_name'],
            trust_remote_code=True
        )
        # Extend the vocabulary
        self._extend_tokenizer_vocab()

        print(f"Loading the LLM model: {self.config['hyperparameters']['llm']['model_name']}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.config['hyperparameters']['llm']['model_name'],
            trust_remote_code=True
        )
        print("The LLM model has been loaded in full precision (FP32).")
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
        print(f"DeepSpeed engine initialized. Model on device: {self.device}")

        # Move the BEATs feature extractor and the AudioSet tokenizer to the target device
        self.beats_feature_extractor.to(self.device)
        self.audioset_tokenizer.to(self.device)
        self.beats_tokenizer.to(self.device)

    def _extend_tokenizer_vocab(self):
        """
        Extend the vocabulary to include the tokens required for the TTS and TTA tasks:
          - Special tokens: [TEXT], [ST], [TAG], [DES], [AT], [END], [PAD]
          - S3 vocabulary tokens: [S3_0]..[S3_{V-1}]
          - AudioSet vocabulary tokens: [AS_0]..[AS_{V-1}]
          - BEATs vocabulary tokens: [BEATs_0]..[BEATs_{V-1}]
        """
        print("Extending the tokenizer vocabulary...")

        s3_vocab_size = int(self.config['hyperparameters']['llm']['s3_vocab_size'])
        as_vocab_size = int(self.config['hyperparameters']['llm']['as_vocab_size'])
        beats_vocab_size = int(self.config['hyperparameters']['llm']['beats_vocab_size'])

        # Define all the required tokens
        special_tokens = ['[TEXT]', '[ST]', '[TAG]', '[DES]', '[AT]', '[END]', '[PAD]']
        s3_target_tokens = [f'[S3_{j}]' for j in range(s3_vocab_size)]
        as_target_tokens = [f'[AS_{j}]' for j in range(as_vocab_size)]
        beats_target_tokens = [f'[BEATs_{j}]' for j in range(beats_vocab_size)]

        # Add all the tokens at once
        num_added = self.llm_tokenizer.add_tokens(
            special_tokens + s3_target_tokens + as_target_tokens + beats_target_tokens,
            special_tokens=True
        )

        print(f"Added {num_added} new tokens.")

        # Ensure the tokenizer has a pad_token
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        # Record the commonly used IDs
        self.special_token_ids = {
            'text': self.llm_tokenizer.convert_tokens_to_ids('[TEXT]'),
            'st': self.llm_tokenizer.convert_tokens_to_ids('[ST]'),
            'tag': self.llm_tokenizer.convert_tokens_to_ids('[TAG]'),
            'des': self.llm_tokenizer.convert_tokens_to_ids('[DES]'),
            'at': self.llm_tokenizer.convert_tokens_to_ids('[AT]'),
            'end': self.llm_tokenizer.convert_tokens_to_ids('[END]'),
            'pad': self.llm_tokenizer.convert_tokens_to_ids('[PAD]'),
        }
        print(f"Vocabulary size after extension: {len(self.llm_tokenizer)}")

    def _setup_dataloaders(self, is_train=True):
        is_distributed = torch.distributed.is_initialized()
        rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        if is_train:
            print("Initializing the training DataLoader (LibriSpeech + AudioSet)...")
            ls_train_ds = LibriSpeechDataset(self.config['data']['librispeech']['train_root'], self.config)
            as_train_ds = AudioSetDataset(self.config['data']['audioset']['train_root'], self.config)
            dataset = ConcatDataset([ls_train_ds, as_train_ds])

            # --- Use the new balanced sampler ---
            if is_distributed:
                sampler = BalancedDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
            else:
                # In non-distributed mode, one could implement a non-distributed balanced sampler, or simply continue with random shuffling
                logger.warning("In non-distributed mode, the balanced sampler is not used.")
                sampler = None

            self.train_loader = DataLoader(
                dataset, batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                sampler=sampler, shuffle=(sampler is None),
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory'],
                persistent_workers=True,
                drop_last=True  # Ensure the number of batches is the same across all GPUs
            )
            print(f"Training DataLoader initialized, containing {len(self.train_loader)} batches.")
        else:
            print("Initializing the validation DataLoader (LibriSpeech + AudioSet)...")
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
            print(f"Validation DataLoader initialized, containing {len(self.val_loader)} batches.")

    def setup_lr_scheduler(self):
        """Set up the OneCycleLR scheduler based on the configuration file."""
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: The optimizer or training DataLoader is not initialized; cannot create the LR scheduler.")
            return
        print(f"Rank {self.cmd_args.local_rank}: Setting up the LR scheduler...")
        try:
            scheduler_config = self.config['hyperparameters']['llm']['optimizer']['scheduler']
            num_epochs = self.config['hyperparameters']['llm']['num_epochs']
            gradient_accumulation_steps = self.model_engine.gradient_accumulation_steps()

            # Use a more precise computation of the total number of steps
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
            logger.error(f"Rank {self.cmd_args.local_rank}: Failed to create the LR scheduler. "
                         f"Missing key in the config: {e}.")
            raise
        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Failed to set up the LR scheduler: {e}\n{traceback.format_exc()}")
            raise

    def _prepare_batch_data(self, batch, is_train):
        """
        Prepare a single batch of data for the LLM, supporting two formats:
        1. TTS (LibriSpeech): [TEXT] text [ST] -> [S3_0][BEATs_0][S3_1][BEATs_1]...[END]
        2. TTA (AudioSet): [TAG] events [DES] desc [AT] -> [AS_0][BEATs_0][AS_1][BEATs_1]...[END]
        """
        file_paths = batch['file_path']
        dataset_types = batch['dataset_type']
        B = len(file_paths)

        full_sequences_text = []
        prompt_text_only = []

        trans_map = self.librispeech_train_trans if is_train else self.librispeech_val_trans

        for i in range(B):
            dtype = dataset_types[i]
            fpath = Path(file_paths[i])

            # Get the waveform uniformly, since both branches need it
            waveform = batch['waveform'][i:i + 1].to(self.device, non_blocking=True)

            if dtype == 'librispeech':
                file_id = fpath.stem
                speech_text = trans_map.get(file_id, "").strip()
                if not speech_text:
                    full_sequences_text.append("[PAD]")
                    prompt_text_only.append("[PAD]")
                    continue

                # Get the S3 tokens
                s3_tokens = batch['s3_token'][i]
                s3_tokens_filtered = [tok.item() for tok in s3_tokens if tok != 0]

                # Get the BEATs tokens
                with torch.no_grad():
                    beats_indices = self.beats_tokenizer.extract_labels(waveform).view(-1).tolist()

                # Interleave the S3 and BEATs tokens 1:1
                target_tokens_str = []
                min_len = min(len(s3_tokens_filtered), len(beats_indices))
                for j in range(min_len):
                    target_tokens_str.append(f'[S3_{s3_tokens_filtered[j]}]')
                    target_tokens_str.append(f'[BEATs_{beats_indices[j]}]')

                # If there are no valid interleaved tokens, skip this sample
                if not target_tokens_str:
                    full_sequences_text.append("[PAD]")
                    prompt_text_only.append("[PAD]")
                    continue

                prompt_part = f"[TEXT] {speech_text} [ST]"
                target_part = "".join(target_tokens_str) + "[END]"

            elif dtype == 'audioset':
                filename = fpath.name
                desc_data = self.audioset_descriptions.get(filename)
                if not desc_data:
                    full_sequences_text.append("[PAD]")
                    prompt_text_only.append("[PAD]")
                    continue

                event = desc_data['event'].strip()
                description = desc_data['description'].strip()

                with torch.no_grad():
                    # 1. Extract BEATs features to obtain the AS tokens
                    beats_features, _ = self.beats_feature_extractor.extract_features(waveform)
                    input_feats_tok = beats_features.permute(0, 2, 1)
                    as_tokens = self.audioset_tokenizer.tokenize(input_feats_tok).view(-1).tolist()

                    # 2. Get the BEATs tokens directly from the waveform
                    beats_indices = self.beats_tokenizer.extract_labels(waveform).view(-1).tolist()

                # Interleave the AS and BEATs tokens 1:1
                target_tokens_str = []
                min_len = min(len(as_tokens), len(beats_indices))
                for j in range(min_len):
                    target_tokens_str.append(f'[AS_{as_tokens[j]}]')
                    target_tokens_str.append(f'[BEATs_{beats_indices[j]}]')

                # If there are no valid interleaved tokens, skip this sample
                if not target_tokens_str:
                    full_sequences_text.append("[PAD]")
                    prompt_text_only.append("[PAD]")
                    continue

                prompt_part = f"[TAG] {event} [DES] {description} [AT]"
                target_part = "".join(target_tokens_str) + "[END]"

            else:
                full_sequences_text.append("[PAD]")
                prompt_text_only.append("[PAD]")
                continue

            # Add a space between the prompt and the target so the tokenizer handles them correctly
            full_sequences_text.append(prompt_part + " " + target_part)
            prompt_text_only.append(prompt_part)

        # Tokenize the whole batch uniformly
        tokenized_full = self.llm_tokenizer(
            full_sequences_text, padding='longest', truncation=True,
            max_length=self.config['hyperparameters']['llm']['max_sequence_length'],
            return_tensors='pt'
        ).to(self.device)
        input_ids = tokenized_full['input_ids']
        attention_mask = tokenized_full['attention_mask']

        # Compute the length of the prompt part, used for the subsequent mask
        tokenized_prompts = self.llm_tokenizer(prompt_text_only, padding='longest', return_tensors='pt')
        prompt_lengths = tokenized_prompts.attention_mask.sum(dim=1)

        # Build the labels, setting the prompt and padding parts to -100
        labels = input_ids.clone()
        pad_id = self.llm_tokenizer.pad_token_id
        for b in range(labels.size(0)):
            prompt_len = int(prompt_lengths[b].item())
            labels[b, :prompt_len] = -100

        if pad_id is not None:
            labels[labels == pad_id] = -100

        # Filter out the samples that are fully masked due to lookup failure or text being too short
        valid_indices = [idx for idx, label_row in enumerate(labels) if (label_row != -100).any()]
        if len(valid_indices) < B:
            logger.warning(f"Dropping {B - len(valid_indices)} invalid samples from the batch.")
            input_ids = input_ids[valid_indices]
            attention_mask = attention_mask[valid_indices]
            labels = labels[valid_indices]

        return input_ids, attention_mask, labels

    def train(self):
        print("Starting LLM model training...")
        self._setup_dataloaders(is_train=True)
        self.setup_lr_scheduler()

        num_epochs = self.config['hyperparameters']['llm']['num_epochs']
        rank = self.cmd_args.local_rank

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.model_engine.train()
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            pbar = self.train_loader
            if rank <= 0:
                pbar = tqdm(self.train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", ncols=150)

            for step, batch in enumerate(pbar):
                input_ids, attention_mask, labels = self._prepare_batch_data(batch, is_train=True)

                # If all samples in the batch are invalid, skip it
                if input_ids.size(0) == 0:
                    logger.warning(f"Rank {rank}: Skipping an invalid batch during training (all samples were filtered out).")
                    continue

                outputs = self.model_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss

                self.model_engine.backward(loss)
                self.model_engine.step()

                if self.lr_scheduler and self.model_engine.is_gradient_accumulation_boundary():
                    self.lr_scheduler.step()

                self.global_step += 1

                if rank <= 0:
                    current_lr = self.pytorch_optimizer.param_groups[0]['lr']
                    perplexity = torch.exp(loss).item()
                    pbar.set_postfix(
                        {'loss': f"{loss.item():.4f}", 'ppl': f"{perplexity:.2f}", 'lr': f"{current_lr:.2e}"})
                    if self.config['logging']['wandb']['enabled']:
                        wandb.log({
                            'train/loss': loss.item(),
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
        total_batches = 0

        pbar = self.val_loader
        if rank <= 0:
            pbar = tqdm(self.val_loader, desc="Validating", ncols=120)

        with torch.no_grad():
            for batch in pbar:
                input_ids, attention_mask, labels = self._prepare_batch_data(batch, is_train=False)

                if input_ids.size(0) == 0:
                    logger.warning(f"Rank {rank}: Skipping an invalid batch during validation.")
                    continue

                outputs = self.model_engine(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                total_val_loss += loss.item()
                total_batches += 1

        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if world_size > 1:
            total_val_loss_tensor = torch.tensor(total_val_loss, device=self.device)
            total_batches_tensor = torch.tensor(total_batches, device=self.device)
            torch.distributed.all_reduce(total_val_loss_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(total_batches_tensor, op=torch.distributed.ReduceOp.SUM)
            avg_val_loss = (total_val_loss_tensor / total_batches_tensor).item() if total_batches_tensor > 0 else float(
                'inf')
        else:
            avg_val_loss = total_val_loss / total_batches if total_batches > 0 else float('inf')

        is_best_model = avg_val_loss < self.best_val_loss

        if rank <= 0:
            avg_perplexity = np.exp(avg_val_loss)
            print(f"Validation complete | Average loss: {avg_val_loss:.4f} | Average perplexity: {avg_perplexity:.2f}")

            if self.config['logging']['wandb']['enabled']:
                wandb.log({
                    'val/avg_loss': avg_val_loss,
                    'val/avg_perplexity': avg_perplexity,
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
        - Only rank 0 performs the file save operation.
        - The saved content includes the model state, optimizer state, scheduler state, and client state.
        """
        # The barriers at the start and end of the function remain unchanged, since all processes now call this function
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.cmd_args.local_rank <= 0:
            try:
                checkpoint_file_path = self.checkpoint_dir / filename
                print(f"Preparing to save the checkpoint to {checkpoint_file_path}")

                # Extract the unwrapped model state dict from the DeepSpeed engine
                model_state_dict = self.model_engine.module.state_dict()

                # Collect the training state information
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
    # --- Add the line below here to disable tokenizers parallelism ---
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser(description="Train an LLM on AudioSet using DeepSpeed")
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
