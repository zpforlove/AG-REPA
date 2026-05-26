import os
import argparse
import faulthandler
import logging
import math
import multiprocessing

import sys
import traceback
from pathlib import Path

import deepspeed
import matplotlib
import torch
import torch.nn.functional as F
import torchaudio
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, ConcatDataset, DistributedSampler
from vocos import Vocos

import wandb
from beats.BEATs import BEATs, BEATsConfig
from beats.Tokenizers import Tokenizers, TokenizersConfig
from data_loader import LibriSpeechDataset, AudioSetDataset
from models import AudioSetTokenizer
from models import MelSpectrogramExtractor, FlowMatchingModel
from utils import load_config, peak_norm

# Global variable declaration
global logger


def setup_logger_ds(rank=-1, config=None, level_override=None):
    current_logger = logging.getLogger("train_cfm_script")
    if current_logger.hasHandlers():
        current_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    log_format = f'%(asctime)s - RANK {rank} - %(module)s - %(levelname)s - %(message)s' if rank != -1 \
        else '%(asctime)s - %(module)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    current_logger.addHandler(handler)
    log_level_str = 'INFO'
    if config and 'logging' in config and \
            isinstance(config['logging'], dict) and 'console' in config['logging'] and \
            isinstance(config['logging']['console'], dict) and 'level' in config['logging']['console']:
        log_level_str = config['logging']['console']['level'].upper()
    if level_override:
        log_level_str = level_override.upper()
    log_level_resolved = getattr(logging, log_level_str, logging.INFO)
    current_logger.setLevel(log_level_resolved if rank <= 0 else logging.WARNING)
    current_logger.propagate = False
    return current_logger


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


class FlowMatchingTrainer:
    def __init__(self, config_dict, cmd_args):
        self.cmd_args = cmd_args
        self.config = config_dict
        self._is_shutdown = False
        global logger
        log_level_from_config = self.config.get('logging', {}).get('console', {}).get('level', 'INFO')
        logger = setup_logger_ds(self.cmd_args.local_rank, self.config, level_override=log_level_from_config)
        logger.info(f"Rank {self.cmd_args.local_rank}: Starting Unified Flow Matching Trainer initialization...")

        self.mel_extractor = None
        self.flow_model = None
        self.pytorch_optimizer = None
        self.optimizer = None
        self.model_engine = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.wandb_initialized = False
        self.device = None

        self.beats_feature_extractor = None
        self.beats_tokenizer = None

        # Prepare the teacher models for REPA distillation
        self.whisper_processor = None
        self.whisper_model = None

        self._setup_device()
        self._setup_external_models()
        self.setup_models()

        if self.cmd_args.local_rank <= 0:
            self.setup_wandb()
        self._init_data_loaders()

        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir']) / 'flow'
        if self.cmd_args.local_rank <= 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_flow_loss = float('inf')
        self.current_epoch = 0

    def _setup_device(self):
        if self.model_engine:
            self.device = self.model_engine.device
        elif torch.cuda.is_available() and self.cmd_args.local_rank != -1:
            self.device = torch.device(f'cuda:{self.cmd_args.local_rank}')
        elif torch.cuda.is_available():
            gpu_id_to_use = self.config.get('device', {}).get('device_ids', [0])[0]
            self.device = torch.device(f'cuda:{gpu_id_to_use}')
        else:
            self.device = torch.device('cpu')
        logger.info(f"Rank {self.cmd_args.local_rank}: Device determined: {self.device}")

    def _setup_external_models(self):
        """
        - Automatically find and load the best AudioSetTokenizer model.
        - Load the BEATs Tokenizer.
        - Load the BEATs Feature Extractor (used for the AST input and the REPA Audio teacher).
        - Load Whisper-large-v3 (used for the REPA Speech teacher).
        - Infer the teacher dimensions.
        """
        try:
            # ====== AudioSetTokenizer ======
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
                        f"Rank {self.cmd_args.local_rank}: Could not parse loss from filename: {pth_file.name}")
                    continue
            if best_checkpoint_path is None:
                raise FileNotFoundError(f"No valid 'best_epoch_*_loss_*.pth' checkpoints found in {ast_checkpoint_dir}")
            logger.info(
                f"Rank {self.cmd_args.local_rank}: Found best AST checkpoint: {best_checkpoint_path} (loss {best_loss:.4f})")

            ast_checkpoint = torch.load(best_checkpoint_path, map_location='cpu')
            ast_config = self.config['hyperparameters']['ast']
            self.audioset_tokenizer = AudioSetTokenizer(
                input_dim=ast_config['input_dim'],
                hidden_dim=ast_config['hidden_dim'],
                vocab_size=ast_config['vocab_size']
            )
            self.audioset_tokenizer.load_state_dict(ast_checkpoint['model_state_dict'])
            self.audioset_tokenizer.to(self.device).eval()
            logger.info(f"Rank {self.cmd_args.local_rank}: AudioSet Tokenizer loaded from best checkpoint.")

            # ====== BEATs Tokenizer ======
            beats_tokenizer_path = self.config['paths']['beats_tokenizer']
            if not os.path.exists(beats_tokenizer_path):
                raise FileNotFoundError(f"BEATs Tokenizer checkpoint not found at: {beats_tokenizer_path}")
            logger.info(f"Rank {self.cmd_args.local_rank}: Loading BEATs Tokenizer from {beats_tokenizer_path}...")
            tokenizer_checkpoint = torch.load(beats_tokenizer_path, map_location='cpu')
            tokenizer_cfg = TokenizersConfig(tokenizer_checkpoint['cfg'])
            self.beats_tokenizer = Tokenizers(tokenizer_cfg)
            self.beats_tokenizer.load_state_dict(tokenizer_checkpoint['model'])
            self.beats_tokenizer.to(self.device).eval()
            logger.info(f"Rank {self.cmd_args.local_rank}: BEATs Tokenizer loaded and set to eval mode.")

            # ====== BEATs Feature Extractor (used for the AST input and the Audio teacher) ======
            beats_extractor_path = self.config['paths']['beats_feature_extractor_checkpoint']
            if not os.path.exists(beats_extractor_path):
                raise FileNotFoundError(f"BEATs Feature Extractor checkpoint not found at: {beats_extractor_path}")

            extractor_checkpoint = torch.load(beats_extractor_path, map_location='cpu')
            extractor_cfg = BEATsConfig(extractor_checkpoint['cfg'])
            self.beats_feature_extractor = BEATs(extractor_cfg)
            self.beats_feature_extractor.load_state_dict(extractor_checkpoint['model'])
            self.beats_feature_extractor.to(self.device).eval()
            logger.info(
                f"Rank {self.cmd_args.local_rank}: BEATs Feature Extractor loaded (for AST input & Audio Teacher).")

            # Infer the dimension of the Audio teacher vector
            with torch.no_grad():
                dummy = torch.zeros(1, int(self.config['audio']['sample_rate']), device=self.device)
                feats, _ = self.beats_feature_extractor.extract_features(dummy)  # [1, T, C]
                self.teacher_dim_audio = int(feats.shape[-1])
            logger.info(f"Rank {self.cmd_args.local_rank}: teacher_dim_audio = {self.teacher_dim_audio}")

            # ====== Whisper-large-v3 (Speech teacher, encoder only) ======
            self.whisper_processor, self.whisper_model = None, None
            self.teacher_dim_speech = 0
            try:
                from transformers import WhisperProcessor, WhisperModel
                whisper_name_or_path = self.config['paths'].get('whisper_large_v3', 'openai/whisper-large-v3')
                self.whisper_processor = WhisperProcessor.from_pretrained(whisper_name_or_path)
                self.whisper_model = WhisperModel.from_pretrained(whisper_name_or_path)
                self.whisper_model.to(self.device).eval()
                # Read directly from config, or fall back to running a dummy once to get the dimension
                self.teacher_dim_speech = int(getattr(self.whisper_model.config, 'd_model', 0))
                if self.teacher_dim_speech <= 0:
                    with torch.no_grad():
                        # 1 second of 16k silence
                        sr16 = 16000
                        zeros = torch.zeros(1, sr16)
                        inputs = self.whisper_processor(zeros.numpy(), sampling_rate=sr16, return_tensors="pt")
                        enc_out = self.whisper_model.encoder(input_features=inputs.input_features.to(self.device))
                        self.teacher_dim_speech = int(enc_out.last_hidden_state.shape[-1])
                self.whisper_target_sr = 16000
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Whisper-large-v3 loaded. teacher_dim_speech = {self.teacher_dim_speech}")
            except Exception as e_w:
                logger.error(
                    f"Rank {self.cmd_args.local_rank}: Failed to load Whisper-large-v3: {e_w}. Speech teacher disabled.")
                self.whisper_processor, self.whisper_model = None, None
                self.teacher_dim_speech = 0

        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: External models setup failed: {e}\n{traceback.format_exc()}")
            raise

    def setup_wandb(self):
        if self.cmd_args.local_rank != 0:
            self.wandb_initialized = False
            return
        try:
            if self.config['logging']['wandb']['enabled']:
                project = self.config['logging']['wandb']['project']
                name_prefix = self.config['logging']['wandb']['name']
                exp_name_suffix = self.config.get('meta', {}).get('exp_name', 'CFM')
                name = f"{name_prefix}_{exp_name_suffix}"
                wandb.init(project=project, name=name, config={**self.config, **vars(self.cmd_args)})
                logger.info(f"Rank 0: WandB initialized - Project: {project}, Run: {name}")
                self.wandb_initialized = True
            else:
                logger.info("Rank 0: WandB logging disabled in config.")
                self.wandb_initialized = False
        except Exception as e:
            logger.error(f"Rank 0: WandB setup failed: {str(e)}")
            self.wandb_initialized = False

    def setup_models(self):
        logger.info(f"Rank {self.cmd_args.local_rank}: Initializing models and optimizer...")
        try:
            # Mel extractor
            self.mel_extractor = MelSpectrogramExtractor(self.config, target_device=self.device)
            logger.info(f"Rank {self.cmd_args.local_rank}: Mel Spectrogram Extractor Initialized.")

            # Pass in the teacher dimensions
            t_s = int(getattr(self, 'teacher_dim_speech', 0))
            t_a = int(getattr(self, 'teacher_dim_audio', 0))
            self.flow_model = FlowMatchingModel(self.config,
                                                teacher_dim_speech=t_s,
                                                teacher_dim_audio=t_a)
            logger.info(f"Rank {self.cmd_args.local_rank}: Flow Matching Model instantiated "
                        f"(teacher_dim_speech={t_s}, teacher_dim_audio={t_a}).")

            # Optimizer
            opt_cfg = self.config['hyperparameters']['flow']['optimizer']
            self.pytorch_optimizer = torch.optim.AdamW(
                self.flow_model.parameters(),
                lr=float(opt_cfg['scheduler']['max_lr']),
                weight_decay=float(opt_cfg['weight_decay']),
                betas=eval(str(opt_cfg['betas']))
            )

            # DeepSpeed initialization
            self.model_engine, _, _, _ = deepspeed.initialize(
                args=self.cmd_args,
                model=self.flow_model,
                optimizer=self.pytorch_optimizer
            )
            logger.info(
                f"Rank {self.cmd_args.local_rank}: DeepSpeed engine initialized on device: {self.model_engine.device}")
            self.device = self.model_engine.device

            # Vocos
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)
            logger.info(f"Rank {self.cmd_args.local_rank}: Vocos loaded to {self.device}.")

        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Model/DeepSpeed init failed: {str(e)}\n{traceback.format_exc()}")
            raise

    def setup_lr_scheduler(self):
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Optimizer or Train Loader not initialized. Cannot create LR scheduler.")
            return
        logger.info(f"Rank {self.cmd_args.local_rank}: Setting up LR scheduler...")
        scheduler_config = self.config['hyperparameters']['flow']['optimizer']['scheduler']
        num_epochs = self.config['hyperparameters']['flow']['num_epochs']
        gradient_accumulation_steps = self.model_engine.gradient_accumulation_steps()
        num_optimizer_steps_per_epoch = math.ceil(len(self.train_loader) / gradient_accumulation_steps)
        total_steps = num_optimizer_steps_per_epoch * num_epochs
        if total_steps <= 0:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Calculated total_steps ({total_steps}) is invalid. LR scheduler not created.")
            return
        logger.info(f"Rank {self.cmd_args.local_rank}: Total steps for LR scheduler: {total_steps}")
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.pytorch_optimizer, max_lr=float(scheduler_config['max_lr']),
            total_steps=total_steps, pct_start=float(scheduler_config['pct_start']),
            div_factor=float(scheduler_config['div_factor']),
            final_div_factor=float(scheduler_config['final_div_factor'])
        )
        logger.info(f"Rank {self.cmd_args.local_rank}: OneCycleLR scheduler created.")

    def _init_data_loaders(self):
        """Mix LibriSpeech and AudioSet using a ConcatDataset, with a balanced sampling strategy"""
        try:
            is_dist = torch.distributed.is_initialized()
            rank, world_size = (torch.distributed.get_rank(), torch.distributed.get_world_size()) if is_dist else (
                0, 1)
            batch_size = self.model_engine.train_micro_batch_size_per_gpu()

            ls_train_ds = LibriSpeechDataset(self.config['data']['librispeech']['train_root'],
                                             self.config)

            as_train_ds = AudioSetDataset(self.config['data']['audioset']['train_root'], self.config,
                                          str(self.device))
            train_dataset = ConcatDataset([ls_train_ds, as_train_ds])

            # --- Use the new balanced sampler ---
            if is_dist:
                train_sampler = BalancedDistributedSampler(train_dataset, num_replicas=world_size, rank=rank,
                                                           shuffle=True)
            else:
                # In non-distributed mode, one could implement a non-distributed balanced sampler, or simply continue with random shuffling
                logger.warning("In non-distributed mode, the balanced sampler is not used.")
                train_sampler = None

            self.train_loader = DataLoader(
                train_dataset, batch_size=batch_size, sampler=train_sampler, shuffle=(train_sampler is None),
                drop_last=True, num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory'],
                persistent_workers=True
            )

            ls_val_ds = LibriSpeechDataset(self.config['data']['librispeech']['val_root'],
                                           self.config)

            as_val_ds = AudioSetDataset(self.config['data']['audioset']['val_root'], self.config, str(self.device))
            val_dataset = ConcatDataset([ls_val_ds, as_val_ds])

            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank,
                                             shuffle=False) if is_dist else None
            self.val_loader = DataLoader(
                val_dataset, batch_size=batch_size, sampler=val_sampler, shuffle=False,
                num_workers=self.config['data']['num_workers'], pin_memory=self.config['data']['pin_memory'],
                persistent_workers=True
            )

            logger.info(f"Rank {rank}: Combined DataLoaders created with BalancedDistributedSampler for training.")
            self.setup_lr_scheduler()  # Create the LR Scheduler after the DataLoader is initialized
        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Failed to set up data loaders: {e}\n{traceback.format_exc()}")
            raise

    def prepare_batch(self, batch):
        """
        Prepare the batch data, adapted to dual-token (main token + BEATs token) input.
        - Extract the REPA teacher vectors.
        """
        try:
            device = self.device

            def _interleave_tokens(tokens1, tokens2):
                """Merge two token lists by 1:1 interleaving"""
                t1_list, t2_list = list(tokens1), list(tokens2)
                min_len = min(len(t1_list), len(t2_list))
                interleaved = [val for pair in zip(t1_list[:min_len], t2_list[:min_len]) for val in pair]
                return interleaved

            # -------------------- 0) Raw batch --------------------
            waveforms = batch['waveform'].to(device, non_blocking=True)  # [B, T_audio]
            dataset_types = batch['dataset_type']  # list[str]
            mel_lengths_batch = batch['mel_lengths']  # [B]
            B, T_audio = waveforms.shape

            ls_idx = [i for i, dt in enumerate(dataset_types) if dt == 'librispeech']
            as_idx = [i for i, dt in enumerate(dataset_types) if dt == 'audioset']

            # -------------------- 1) Target mel (computed directly from the raw waveform) --------------------
            full_mel_specs = self.mel_extractor(waveforms)
            _, n_mels, T_full = full_mel_specs.shape

            # -------------------- 2) Reference mel (prefix <=30%) --------------------
            ref_mel_parts, actual_ref_lengths = [], []
            for i in range(B):
                current_mel_len = mel_lengths_batch[i].item()
                max_ref_len_abs = max(1, int(current_mel_len * 0.3))
                ref_len = torch.randint(1, max_ref_len_abs + 1, (1,)).item() if max_ref_len_abs > 0 else 0
                actual_ref_lengths.append(ref_len)
                if ref_len > 0:
                    ref_part = full_mel_specs[i, :, :min(ref_len, T_full)]
                else:
                    ref_part = torch.empty((n_mels, 0), device=self.device, dtype=full_mel_specs.dtype)
                ref_mel_parts.append(ref_part)
            ref_mel_for_cond = torch.stack([F.pad(p, (0, T_full - p.shape[1])) for p in ref_mel_parts])

            # -------------------- 3) Conditional fused_embed (dual-token fusion) --------------------
            D_tok = self.model_engine.module.max_fused_embed_dim
            fused_embed = torch.zeros(B, D_tok, T_full, device=device, dtype=full_mel_specs.dtype)

            # Process the AudioSet samples
            if len(as_idx) > 0:
                as_wave = waveforms[torch.as_tensor(as_idx, device=device, dtype=torch.long)]
                with torch.no_grad():
                    # 1. Get the AS tokens
                    beats_features, _ = self.beats_feature_extractor.extract_features(as_wave)
                    input_feats_tok = beats_features.permute(0, 2, 1)
                    as_tokens = self.audioset_tokenizer.tokenize(input_feats_tok)  # [N_as, T_as]

                    # 2. Get the BEATs tokens
                    beats_tokens_raw = self.beats_tokenizer.extract_labels(as_wave)
                    if beats_tokens_raw.dim() == 0:
                        beats_tokens = beats_tokens_raw.view(1, 1)
                    elif beats_tokens_raw.dim() == 1:
                        beats_tokens = beats_tokens_raw.view(as_wave.shape[0], -1)
                    else:
                        beats_tokens = beats_tokens_raw  # [N_as, T_beats]

                    # 3. Interleave
                    interleaved_tokens_list = [_interleave_tokens(as_tok, b_tok) for as_tok, b_tok in
                                               zip(as_tokens, beats_tokens)]
                    max_len = max(len(t) for t in interleaved_tokens_list) if interleaved_tokens_list else 0
                    if max_len == 0: max_len = 1; interleaved_tokens_list = [[0] for _ in interleaved_tokens_list]

                    padded_tokens = torch.stack(
                        [F.pad(torch.tensor(t, device=device, dtype=torch.long), (0, max_len - len(t)), value=0) for t
                         in interleaved_tokens_list]
                    )

                    # 4. Embed and fuse-project (returns only one embedding)
                    as_shared = self.model_engine.module.embed_and_project_tokens('as', padded_tokens.to(device))

                if as_shared.shape[-1] > 0:
                    as_shared = F.interpolate(as_shared, size=T_full, mode='linear', align_corners=False)
                    as_shared = F.normalize(as_shared, p=2, dim=1)
                fused_embed[as_idx, :, :] = as_shared

            # Process the LibriSpeech samples
            if len(ls_idx) > 0:
                # 1. Get the S3 tokens
                s3_field = batch.get('s3_token', None)
                if s3_field is None:
                    raise RuntimeError("The 's3_token' field is missing from the batch; please confirm that LibriSpeechDataset correctly returns this key.")
                ls_tokens_list = []
                if isinstance(s3_field, (list, tuple)):
                    for i in ls_idx:
                        t = s3_field[i]
                        if not torch.is_tensor(t): t = torch.as_tensor(t, dtype=torch.long)
                        ls_tokens_list.append(t)
                elif torch.is_tensor(s3_field):
                    for i in ls_idx:
                        t = s3_field[i]
                        if t.dim() > 1: t = t.view(-1)
                        ls_tokens_list.append(t.to(torch.long))
                else:
                    raise RuntimeError(f"Unsupported 's3_token' type: {type(s3_field)}")

                # 2. Get the BEATs tokens
                ls_wave = waveforms[torch.as_tensor(ls_idx, device=device, dtype=torch.long)]
                with torch.no_grad():
                    beats_tokens_ls_raw = self.beats_tokenizer.extract_labels(ls_wave)
                    if beats_tokens_ls_raw.dim() == 0:
                        beats_tokens_ls = beats_tokens_ls_raw.view(1, 1)
                    elif beats_tokens_ls_raw.dim() == 1:
                        beats_tokens_ls = beats_tokens_ls_raw.view(ls_wave.shape[0], -1)
                    else:
                        beats_tokens_ls = beats_tokens_ls_raw  # [N_ls, T_beats]

                # 3. Interleave
                interleaved_ls_tokens_list = []
                for i in range(len(ls_idx)):
                    s3_toks = [t.item() for t in ls_tokens_list[i] if t != 0]  # S3 token
                    beats_toks = beats_tokens_ls[i]  # BEATs token
                    interleaved_ls_tokens_list.append(_interleave_tokens(s3_toks, beats_toks))

                max_token_len = max(len(t) for t in interleaved_ls_tokens_list) if interleaved_ls_tokens_list else 0
                if max_token_len == 0:
                    max_token_len = 1
                    interleaved_ls_tokens_list = [[0] for _ in interleaved_ls_tokens_list]

                padded_tokens = torch.stack([
                    F.pad(torch.tensor(t, device=device, dtype=torch.long), (0, max_token_len - len(t)), value=0) for t
                    in interleaved_ls_tokens_list
                ], dim=0)

                # 4. Embed and fuse-project
                with torch.no_grad():
                    ls_shared = self.model_engine.module.embed_and_project_tokens('s3', padded_tokens)

                if ls_shared.shape[-1] > 0:
                    ls_shared = F.interpolate(ls_shared, size=T_full, mode='linear', align_corners=False)
                    ls_shared = F.normalize(ls_shared, p=2, dim=1)
                fused_embed[ls_idx, :, :] = ls_shared

            # -------------------- 4) Teacher vectors --------------------
            teacher_audio_global = None
            if len(as_idx) > 0:
                as_wave = waveforms[torch.as_tensor(as_idx, device=device, dtype=torch.long)]
                with torch.no_grad():
                    beats_features, _ = self.beats_feature_extractor.extract_features(as_wave)
                    ta_local = beats_features.mean(dim=1)
                Cb = ta_local.shape[-1]
                ta_list = [torch.zeros(Cb, device=device, dtype=ta_local.dtype) for _ in range(B)]
                for k, i in enumerate(as_idx):
                    ta_list[i] = ta_local[k]
                teacher_audio_global = torch.stack(ta_list, dim=0)

            teacher_speech_global = None
            if len(ls_idx) > 0 and (self.whisper_model is not None and self.whisper_processor is not None):
                ls_wave = waveforms[torch.as_tensor(ls_idx, device=device, dtype=torch.long)]
                with torch.no_grad():
                    sr_conf = int(self.config['audio']['sample_rate'])
                    if sr_conf != self.whisper_target_sr:
                        ls_wave_16k = torchaudio.functional.resample(ls_wave, orig_freq=sr_conf,
                                                                     new_freq=self.whisper_target_sr)
                    else:
                        ls_wave_16k = ls_wave
                    ls_inputs = self.whisper_processor(ls_wave_16k.detach().cpu().numpy(),
                                                       sampling_rate=self.whisper_target_sr, return_tensors="pt")
                    input_features = ls_inputs.input_features.to(self.device)
                    enc_out = self.whisper_model.encoder(input_features=input_features)
                    ts_local = enc_out.last_hidden_state.mean(dim=1)
                Cw = ts_local.shape[-1]
                ts_list = [torch.zeros(Cw, device=device, dtype=ts_local.dtype) for _ in range(B)]
                for k, i in enumerate(ls_idx):
                    ts_list[i] = ts_local[k]
                teacher_speech_global = torch.stack(ts_list, dim=0)

            # -------------------- 5) Domain labels and return --------------------
            domain_ids = torch.tensor([0 if dt == 'librispeech' else 1 for dt in dataset_types], device=device,
                                      dtype=torch.long)
            ret = {
                'full_mel_specs': full_mel_specs,
                'cond_embed_dict': {
                    'fused_embed': fused_embed,
                    'ref_mel_for_cond': ref_mel_for_cond,
                    'domain_ids': domain_ids,
                    'teacher_speech_global': teacher_speech_global,
                    'teacher_audio_global': teacher_audio_global,
                },
                'ref_mel_parts_for_viz': ref_mel_parts,
                'actual_ref_lengths': actual_ref_lengths,
                'full_mel_lengths_for_viz': mel_lengths_batch
            }
            return ret

        except Exception as e:
            logger.error(f"Rank {self.cmd_args.local_rank}: Batch preparation failed: {e}\n{traceback.format_exc()}")
            return None

    def train_step(self, batch):
        """
        Single training step: standard FM forward/backward (calls the `forward` function that includes the REPA loss)
        """
        self.model_engine.train()
        data = self.prepare_batch(batch)
        if data is None:
            return {'flow_loss': float('inf'), 'is_valid_step': False}

        x = data['full_mel_specs']  # [B, n_mels, T]
        cond_embed_dict = data['cond_embed_dict']  # Condition dict
        B = x.size(0)
        t = torch.rand(B, device=x.device)

        # --- Main task: per-sample loss (model_engine.forward now returns fm_loss + repa_loss) ---
        per_sample_total_loss = self.model_engine(x=x, t=t, cond_embed_dict=cond_embed_dict)
        total_loss = per_sample_total_loss.mean()

        is_loss_valid = True
        if not torch.isfinite(total_loss):
            logger.error(
                f"Rank {self.cmd_args.local_rank}: NaN/Inf detected in training loss: {total_loss.item()}. Skipping step.")
            scaled_total_loss_item = 0.0
            is_loss_valid = False
            if self.optimizer:
                self.optimizer.zero_grad()
        else:
            scaled_total_loss_item = total_loss.item()

        if is_loss_valid:
            self.model_engine.backward(total_loss)
            self.model_engine.step()

        if self.lr_scheduler and is_loss_valid and self.model_engine.is_gradient_accumulation_boundary():
            self.lr_scheduler.step()

        return {'flow_loss': scaled_total_loss_item, 'is_valid_step': is_loss_valid}

    def train(self):
        logger.info(f"Rank {self.cmd_args.local_rank}: Starting Conditional Flow Matching training...")
        num_epochs = self.config['hyperparameters']['flow']['num_epochs']
        is_main_process = self.cmd_args.local_rank <= 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.model_engine.train()
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_flow_loss_sum = 0.0
            num_valid_batches_epoch = 0

            pbar_desc = f"Epoch {epoch + 1}/{num_epochs} [Training...]"
            pbar = tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc=pbar_desc, ncols=150, disable=not is_main_process
            )

            for step, batch in pbar:
                loss_info = self.train_step(batch)

                if loss_info['is_valid_step']:
                    epoch_flow_loss_sum += loss_info['flow_loss']
                    num_valid_batches_epoch += 1
                self.global_step += 1

                if is_main_process:
                    current_lr = self.pytorch_optimizer.param_groups[0]['lr'] if self.pytorch_optimizer else float(
                        'nan')
                    avg_epoch_loss_so_far = epoch_flow_loss_sum / num_valid_batches_epoch if num_valid_batches_epoch > 0 else 0.0
                    pbar.set_postfix({
                        'flow_loss': f"{loss_info['flow_loss']:.4f}",
                        'avg_flow': f"{avg_epoch_loss_so_far:.4f}",
                        'lr': f"{current_lr:.2e}"
                    })
                    if self.wandb_initialized and loss_info['is_valid_step']:
                        try:
                            wandb.log(
                                {'train/step_flow_loss': loss_info['flow_loss'], 'train/learning_rate': current_lr},
                                step=self.global_step)
                        except Exception as e:
                            logger.error(
                                f"Rank 0: WandB step logging error at global_step {self.global_step}: {e}\n{traceback.format_exc()}")

            # --- Processing logic after the epoch ends ---
            epoch_stats_tensor = torch.tensor([epoch_flow_loss_sum, float(num_valid_batches_epoch)],
                                              dtype=torch.float32, device=self.device)
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(epoch_stats_tensor, op=torch.distributed.ReduceOp.SUM)

            global_epoch_flow_loss_sum = epoch_stats_tensor[0].item()
            global_num_valid_batches_epoch = epoch_stats_tensor[1].item()
            avg_epoch_flow_loss = global_epoch_flow_loss_sum / global_num_valid_batches_epoch if global_num_valid_batches_epoch > 0 else 0.0

            if is_main_process:
                logger.info(
                    f"Rank 0: Epoch {epoch + 1} completed. Average Training Flow Loss (Global): {avg_epoch_flow_loss:.4f}")

            logger.info(
                f"Rank {self.cmd_args.local_rank}: Epoch {epoch + 1} training completed. Starting validation...")

            val_metrics = self.validate(epoch)
            val_flow_loss = val_metrics['val_flow_loss']

            should_save_this_epoch = torch.tensor([0], dtype=torch.int, device=self.device)
            if is_main_process:
                logger.info(f"Rank 0: Epoch {epoch + 1} Validation Flow Loss (Global): {val_flow_loss:.4f}")
                if val_flow_loss < self.best_val_flow_loss:
                    self.best_val_flow_loss = val_flow_loss
                    logger.info(
                        f"Rank 0: New best validation flow loss: {self.best_val_flow_loss:.4f}. Marking for checkpoint save.")
                    should_save_this_epoch[0] = 1

                if self.wandb_initialized:
                    try:
                        wandb.log({'epoch/epoch_num': epoch + 1,
                                   'epoch/train_avg_flow_loss': avg_epoch_flow_loss,
                                   'epoch/val_avg_flow_loss': val_flow_loss,
                                   'epoch/best_val_flow_loss': self.best_val_flow_loss},
                                  step=self.global_step)
                    except Exception as e:
                        logger.error(
                            f"Rank 0: WandB epoch logging error for epoch {epoch + 1}: {e}\n{traceback.format_exc()}")

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(should_save_this_epoch, src=0)

            if should_save_this_epoch.item() == 1:
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Received signal to save checkpoint for epoch {epoch + 1}.")
                tag_to_save = ""
                client_state_to_save = {}
                if is_main_process:
                    tag_to_save = f"best_ep{epoch + 1}_val_loss_{self.best_val_flow_loss:.4f}_step{self.global_step}"
                    client_state_to_save = {'epoch': epoch + 1, 'global_step': self.global_step,
                                            'best_val_metric': self.best_val_flow_loss,
                                            'config_snapshot': self.config}
                    logger.info(f"Rank 0: Checkpoint tag: {tag_to_save}")
                self.save_checkpoint(tag=tag_to_save, client_state=client_state_to_save)

            if torch.distributed.is_initialized():
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Reached end of epoch {epoch + 1}. Synchronizing all processes.")
                torch.distributed.barrier()
                logger.info(f"Rank {self.cmd_args.local_rank}: Passed end of epoch {epoch + 1} barrier.")

        if is_main_process:
            logger.info(f"Training completed! Best validation flow loss (Rank 0): {self.best_val_flow_loss:.4f}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def validate_step(self, batch):
        self.model_engine.eval()
        with torch.no_grad():
            data = self.prepare_batch(batch)
            full_mel = data['full_mel_specs']
            cond_embed_dict_for_loss = data['cond_embed_dict']

            B, _, _ = full_mel.shape
            t = torch.rand(B, device=self.device)

            # model_engine (forward) now returns fm_loss + repa_loss
            flow_loss_per_sample = self.model_engine(x=full_mel, t=t, cond_embed_dict=cond_embed_dict_for_loss)
            flow_loss = flow_loss_per_sample.mean()
            flow_loss_item = flow_loss.item()

            viz_data = {}
            if self.cmd_args.local_rank <= 0:
                viz_data = {
                    'ground_truth_full_mel_cpu': data['full_mel_specs'].cpu(),
                    'cond_embed_dict_cpu': {k: (v.cpu() if torch.is_tensor(v) else v)
                                            for k, v in data['cond_embed_dict'].items()},
                    'ref_mel_parts_for_viz_cpu': [item.cpu() for item in data['ref_mel_parts_for_viz']],
                    'actual_ref_lengths_cpu': data['actual_ref_lengths'],
                    'full_mel_lengths_for_viz_cpu': data['full_mel_lengths_for_viz'].cpu()
                }
            return {'val_flow_loss': flow_loss_item, 'batch_size': B, **viz_data}

    def validate(self, epoch):
        """Validation process, with visualization of samples"""
        is_main = self.cmd_args.local_rank <= 0
        self.model_engine.eval()

        loss_sum = torch.tensor(0.0, device=self.device)
        sample_count = torch.tensor(0, device=self.device)

        speech_sample_for_viz, audio_sample_for_viz = None, None

        with torch.no_grad():
            pbar = tqdm.tqdm(self.val_loader, desc=f"Validating Epoch {epoch + 1}", disable=not is_main)
            for batch in pbar:
                metrics = self.validate_step(batch)
                if metrics['val_flow_loss'] != float('inf'):
                    loss_sum += metrics['val_flow_loss'] * metrics['batch_size']
                    sample_count += metrics['batch_size']

                if is_main and (speech_sample_for_viz is None or audio_sample_for_viz is None):
                    for i, dtype in enumerate(batch['dataset_type']):
                        sample = {k: v[i:i + 1] if torch.is_tensor(v) else [v[i]] for k, v in batch.items()}
                        if dtype == 'librispeech' and speech_sample_for_viz is None:
                            speech_sample_for_viz = sample
                        elif dtype == 'audioset' and audio_sample_for_viz is None:
                            audio_sample_for_viz = sample

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sample_count, op=torch.distributed.ReduceOp.SUM)

        avg_loss = (loss_sum / sample_count).item() if sample_count > 0 else float('inf')

        if is_main and wandb.run:
            if speech_sample_for_viz:
                self._log_validation_sample(speech_sample_for_viz, epoch, "LibriSpeech")

            if audio_sample_for_viz:
                self._log_validation_sample(audio_sample_for_viz, epoch, "AudioSet")

        return {'val_flow_loss': avg_loss}

    def _log_validation_sample(self, batch_for_viz, epoch, data_type_str):
        """A wrapped helper function for visualization and logging, with the torchaudio.save logic restored"""
        if not (self.cmd_args.local_rank <= 0 and wandb.run): return
        logger.info(f"Rank 0: Generating visualization for a {data_type_str} sample...")
        try:
            with torch.no_grad():
                data = self.prepare_batch(batch_for_viz)
                output_dir = Path(self.config['paths'].get('output_dir', 'outputs_cfm')) / f"val_epoch{epoch + 1}"
                output_dir.mkdir(parents=True, exist_ok=True)

                # The teacher vectors 'teacher_...' are only used for the training/validation loss, not for sampling
                cond_dict_for_sample = {
                    'fused_embed': data['cond_embed_dict']['fused_embed'],
                    'ref_mel_for_cond': data['cond_embed_dict']['ref_mel_for_cond']
                }

                cond_dict_dev = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in
                                 cond_dict_for_sample.items()}
                target_len = data['full_mel_lengths_for_viz'][0].item()
                sway_coef = self.config['hyperparameters']['flow']['sway_coef']

                gen_mel = self.model_engine.module.sample(cond_embed_dict=cond_dict_dev,
                                                          target_duration_frames=target_len,
                                                          sway_sampling_coef=sway_coef)
                gen_mel_cpu = gen_mel.detach().cpu().squeeze(0)

                # --- Move all parts to the CPU before concatenation ---
                # Move the reference mel-spectrogram part from the GPU to the CPU
                ref_mel_cpu = data['ref_mel_parts_for_viz'][0].cpu()
                ref_len = data['actual_ref_lengths'][0]

                # Concatenate the CPU tensors
                recon_mel = torch.cat([ref_mel_cpu, gen_mel_cpu[:, ref_len:]], dim=1) if ref_len > 0 else gen_mel_cpu

                # Move the ground-truth mel-spectrogram from the GPU to the CPU
                gt_mel = data['full_mel_specs'][0, :, :recon_mel.shape[1]].cpu()

                mel_mean = self.config['hyperparameters']['flow']['mel_mean']
                mel_std = self.config['hyperparameters']['flow']['mel_std']
                raw_gt, raw_recon = gt_mel * mel_std + mel_mean, recon_mel * mel_std + mel_mean

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
                im1 = ax1.imshow(raw_gt.numpy(), aspect='auto', origin='lower')
                fig.colorbar(im1, ax=ax1)
                ax1.set_title(f"Ground Truth Mel ({data_type_str} - Ep{epoch + 1})")
                im2 = ax2.imshow(raw_recon.numpy(), aspect='auto', origin='lower')
                fig.colorbar(im2, ax=ax2)
                ax2.set_title(f"Reconstructed Mel ({data_type_str} - RefLen:{ref_len})")
                plt.tight_layout()
                wandb.log({f'Val Media/Mel Comparison {data_type_str}': wandb.Image(fig)}, step=self.global_step)
                plt.close(fig)

                # Vocos is on the GPU, so the mel-spectrogram needs to be moved back to the device for decoding
                audio_gt_dev = self.vocos.decode(raw_gt.unsqueeze(0).to(self.device))
                audio_reconstructed_dev = self.vocos.decode(raw_recon.unsqueeze(0).to(self.device))
                audio_gt_cpu, audio_reconstructed_cpu = peak_norm(audio_gt_dev.cpu()), peak_norm(
                    audio_reconstructed_dev.cpu())

                sample_rate = self.config['vocos']['sample_rate']
                gt_path = output_dir / f"ep{epoch + 1}_GT_{data_type_str}.wav"
                recon_path = output_dir / f"ep{epoch + 1}_Recon_{data_type_str}.wav"
                torchaudio.save(str(gt_path), audio_gt_cpu, sample_rate)
                torchaudio.save(str(recon_path), audio_reconstructed_cpu, sample_rate)

                wandb.log({
                    f'Val Media/Audio Real {data_type_str}': wandb.Audio(str(gt_path), sample_rate=sample_rate),
                    f'Val Media/Audio Reconstructed {data_type_str}': wandb.Audio(str(recon_path),
                                                                                  sample_rate=sample_rate)
                }, step=self.global_step)
        except Exception as e_viz:
            logger.error(f"Rank 0: Error during visualization for {data_type_str}: {e_viz}\n{traceback.format_exc()}")

    def save_checkpoint(self, tag: str, client_state: dict):
        # Ensure all processes synchronize before rank 0 starts saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.cmd_args.local_rank <= 0:
            checkpoint_file_path = self.checkpoint_dir / f"{tag}.pt"
            logger.info(f"Rank 0: Preparing to save checkpoint to {checkpoint_file_path}")
            model_state_dict = self.model_engine.module.state_dict()
            content_to_save = {
                'model_state_dict': model_state_dict,
                'pytorch_optimizer_state_dict': self.pytorch_optimizer.state_dict() if self.pytorch_optimizer else None,
                'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            }
            if client_state: content_to_save.update(client_state)
            try:
                torch.save(content_to_save, checkpoint_file_path)
                logger.info(f"Rank 0: Checkpoint successfully saved to {checkpoint_file_path}")
            except Exception as e:
                logger.error(f"Rank 0: Failed to save checkpoint {checkpoint_file_path}: {e}\n{traceback.format_exc()}")
                raise

        # Ensure all processes continue only after rank 0 finishes saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def main():
    global logger
    os.environ['MPLBACKEND'] = 'Agg'
    matplotlib.use('Agg', force=True)
    if sys.platform != 'win32':
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        if current_start_method not in ['spawn', 'forkserver']:
            try:
                multiprocessing.set_start_method('spawn', force=True)
                if int(os.getenv('LOCAL_RANK', -1)) <= 0: print("INFO: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e_mp_set:
                if int(os.getenv('LOCAL_RANK', -1)) <= 0: print(
                    f"WARNING: Failed to set 'spawn' start method: {e_mp_set}")
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    faulthandler.enable()

    parser = argparse.ArgumentParser(description="Distributed Flow Matching Training Script (CFM)")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the main configuration file")
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher.')
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    if cmd_args.local_rank == -1:
        env_local_rank = os.getenv('LOCAL_RANK')
        if env_local_rank is not None:
            try:
                cmd_args.local_rank = int(env_local_rank)
            except ValueError:
                pass
    try:
        config_obj = load_config(cmd_args.config)
    except Exception as e_cfg_load:
        logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.error(f"FATAL: Failed to load main config '{cmd_args.config}': {e_cfg_load}")
        sys.exit(1)

    log_level_from_config = config_obj.get('logging', {}).get('console', {}).get('level', 'INFO')
    logger = setup_logger_ds(cmd_args.local_rank, config_obj, level_override=log_level_from_config)
    deepspeed.runtime.utils.set_random_seed(config_obj.get('meta', {}).get('seed', 666))
    logger.info(f"Rank {cmd_args.local_rank}: Global random seed set via DeepSpeed.")

    if cmd_args.local_rank != -1:
        torch.cuda.set_device(cmd_args.local_rank)
        logger.info(f"Rank {cmd_args.local_rank}: Initializing DeepSpeed distributed environment...")
        deepspeed.init_distributed()
        logger.info(
            f"Rank {cmd_args.local_rank}: Distributed environment initialized. World size: {torch.distributed.get_world_size()}")
    else:
        logger.info(f"Rank {cmd_args.local_rank}: Not a distributed run (or local_rank is -1).")

    trainer_instance = None
    try:
        trainer_instance = FlowMatchingTrainer(config_obj, cmd_args)
        trainer_instance.train()
    except KeyboardInterrupt:
        if logger: logger.warning(f"Rank {cmd_args.local_rank}: Training interrupted by user (KeyboardInterrupt).")
    except Exception as e_main_loop:
        if logger:
            logger.error(
                f"Rank {cmd_args.local_rank}: Unhandled exception in main training loop: {e_main_loop}\n{traceback.format_exc()}")
        else:
            print(f"Rank {cmd_args.local_rank} MAIN_LOOP_ERROR: {e_main_loop}\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        if hasattr(cmd_args, 'local_rank') and cmd_args.local_rank != -1 and torch.distributed.is_initialized():
            logger.info(f"Rank {cmd_args.local_rank}: Reached end of main, waiting at final barrier.")
            torch.distributed.barrier()
            logger.info(f"Rank {cmd_args.local_rank}: Passed final barrier.")
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        logger.info(f"Rank {cmd_args.local_rank}: Main function finished.")


if __name__ == "__main__":
    main()
