import argparse
import faulthandler
import logging
import math
import multiprocessing
import os
import sys
import traceback
from pathlib import Path

import deepspeed
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

import wandb
from beats.BEATs import BEATs, BEATsConfig
from data_loader import AudioSetDataset
from models import AudioSetTokenizer
from utils import load_config

# Global variable declaration
global trainer_global_object

# --- Logging setup ---
global logger  # Will be initialized based on rank in the main function


def setup_logger(rank=-1):
    """Set up or update the global logger, adjusted according to the process rank"""
    # Get the root logger or a logger with a specific name
    current_logger = logging.getLogger(__name__)

    # Clear existing handlers to avoid duplicate additions causing duplicated log output
    if current_logger.hasHandlers():
        current_logger.handlers.clear()

    # Create a StreamHandler that writes logs to standard output
    handler = logging.StreamHandler(sys.stdout)
    # Define the log format, including the rank information
    log_format = f'%(asctime)s - RANK {rank} - %(name)s - %(levelname)s - %(message)s' if rank != -1 else '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    # Add the handler to the logger
    current_logger.addHandler(handler)
    # Set the logging level: processes with rank <= 0 output INFO and above, other processes output WARNING and above
    current_logger.setLevel(logging.INFO if rank <= 0 else logging.WARNING)
    # Prevent log messages from propagating to the parent logger to avoid duplicate handling
    current_logger.propagate = False

    logger = current_logger  # Update the global logger variable
    return logger


# --- Training state manager ---
class TrainingStateManager:
    """Manages the state during training, such as loss computation, metric computation, and optimization steps."""

    def __init__(self, model_engine, config):
        self.model_engine = model_engine
        self.device = self.model_engine.device  # Get the device from the DeepSpeed engine
        self.config = config
        self.loss_weights = self.initialize_loss_weights()  # Initialize the loss weights
        self.training_steps = 0  # Track the number of training steps

    def initialize_loss_weights(self):
        """Load the weights of the various loss terms from the configuration."""
        losses_specific_config = self.config['hyperparameters']['ast']['losses']
        return {
            'reconstruction_loss': losses_specific_config['reconstruction_loss_weight'],
            'commit_loss': losses_specific_config['commit_loss_weight'],
        }

    def compute_loss(self, model_outputs, targets):
        """
        Compute the total loss and each component loss, and store the weighted loss values under the main keys for logging.
        Args:
            model_outputs (dict): The model's output, which should contain various loss terms and possibly prediction results.
            targets (dict): The training targets, such as the style embedding target.
        Returns:
            tuple: (total_loss, losses_log, metrics)
        """
        losses_log = {}  # Used to record all loss values for logging
        total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)  # Initialize the total loss

        if not isinstance(model_outputs, dict):
            if logger: logger.error("The model output must be a dictionary. Returning zero loss.")
            return total_loss, losses_log, {}

        # Iterate over the model output, looking for tensors whose key ends with '_loss'
        for key, value in model_outputs.items():
            if key.endswith('_loss') and isinstance(value, torch.Tensor):
                # Get the weight of the corresponding loss, default to 1.0
                weight = self.loss_weights.get(key, 1.0)

                # Check whether the loss value is NaN or Inf
                if not torch.isfinite(value):
                    if logger: logger.warning(f"Loss component {key} is NaN or Inf ({value.item()}). Skipping this component.")
                    # Record the original non-finite value
                    losses_log[key] = value.item()
                    losses_log[f"{key}_raw"] = value.item()
                    continue

                # Compute the weighted loss component
                weighted_component = value * weight
                # Accumulate into the total loss
                total_loss = total_loss + weighted_component

                # Store the weighted loss value under the main key, for use by tqdm and wandb
                losses_log[key] = weighted_component.item()
                # Keep the original (unweighted) loss value under a new key, just in case
                losses_log[f"{key}_raw"] = value.item()

        # Check whether the total loss is NaN or Inf
        if not torch.isfinite(total_loss):
            if logger: logger.warning(f"The total loss is NaN or Inf ({total_loss.item()}). Resetting to 0.0.")
            total_loss = torch.tensor(0.0, requires_grad=True, device=self.device)  # Reset to 0 while keeping the gradient

        metrics = {}
        # If the model output contains prediction results, compute the relevant metrics
        if 'predictions' in model_outputs:
            metrics = self._compute_metrics(model_outputs['predictions'], targets)

        # In the returned losses_log, keys such as 'reconstruction_loss' and 'commit_loss' now correspond to the weighted values
        return total_loss, losses_log, metrics

    def _compute_metrics(self, predictions, targets):
        """Compute evaluation metrics during training, such as the cosine similarity of the style embeddings."""
        metrics = {}
        return metrics

    def optimize_step(self, loss):
        """Perform the backward pass and the optimizer step."""
        # Check whether the loss is valid
        if not (isinstance(loss, torch.Tensor) and torch.isfinite(loss)):
            if logger: logger.warning(f"The loss for the optimization step is invalid: {loss}. Skipping the backward pass and the optimization step.")
            return False  # Return False if step was skipped
        try:
            # Perform the backward pass
            self.model_engine.backward(loss)
            # Perform the optimizer step (parameter update)
            self.model_engine.step()
            # Increment the training step counter
            self.training_steps += 1
            return True  # Return True if step was successful
        except Exception as e:
            if logger: logger.error(f"Error in the optimization step: {str(e)}\n{traceback.format_exc()}")
            return False  # Return False on error


# --- Trainer class ---
class Trainer:
    """Responsible for managing the entire training pipeline, including the model, data loading, the training loop, validation, and checkpoint saving."""

    def __init__(self, config_path: str, cmd_args):
        self.cmd_args = cmd_args
        self.config = load_config(config_path)  # Load the configuration file
        global logger  # Use the global logger already initialized in main
        if logger is None:  # Check again, just in case
            logger = setup_logger(self.cmd_args.local_rank if hasattr(self.cmd_args, 'local_rank') else -1)

        logger.info(f"Rank {self.cmd_args.local_rank}: Starting trainer initialization...")

        # Initialize the various components to None
        self.beats_feature_extractor = None  # Replaces Whisper with BEATs
        self.train_loader = None
        self.val_loader = None
        self.model_engine = None
        self.pytorch_optimizer = None
        self.optimizer = None
        self.lr_scheduler = None
        self.global_step = 0  # Global training step counter
        self.current_epoch = 0  # Current training epoch
        # Initialize the best metric, used to save the best model
        self.best_metrics = {'val_loss': float('inf')}
        self.device = None  # Will be set by the DeepSpeed engine

        self.setup_environment()  # Environment setup, such as loading the BEATs model
        # Only rank 0 initializes WandB
        if self.cmd_args.local_rank <= 0: self.setup_wandb()
        self.setup_model_and_deepspeed()  # DeepSpeed model and StyleEncoder initialization

        # Set the checkpoint save directory
        self.checkpoint_dir = Path(self.config['paths']['checkpoint_dir'])
        self.checkpoint_ast_dir = self.checkpoint_dir / 'ast'
        # Only rank 0 creates the checkpoint directory
        if self.cmd_args.local_rank <= 0:
            self.checkpoint_ast_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Rank {self.cmd_args.local_rank}: Checkpoint directory created/confirmed: {self.checkpoint_ast_dir}")

    def setup_model_and_deepspeed(self):
        """Initialize the model, the PyTorch optimizer, and the DeepSpeed engine."""
        try:
            # Instantiate the AudioSetTokenizer model
            model = AudioSetTokenizer(
                input_dim=self.config['hyperparameters']['ast']['input_dim'],
                hidden_dim=self.config['hyperparameters']['ast']['hidden_dim'],
                vocab_size=self.config['hyperparameters']['ast']['vocab_size'],
            )
            try:
                self.pytorch_optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(self.config['hyperparameters']['ast']['optimizer']['scheduler']['max_lr']),
                    weight_decay=float(self.config['hyperparameters']['ast']['optimizer']['weight_decay']),
                    betas=eval(str(self.config['hyperparameters']['ast']['optimizer']['betas']))
                )
                logger.info(f"Rank {self.cmd_args.local_rank}: PyTorch AdamW optimizer created.")
            except KeyError as e:
                logger.error(f"Rank {self.cmd_args.local_rank}: Failed to create PyTorch optimizer."
                             f"Missing key in config: {e}. Please ensure config.yaml has"
                             f"['hyperparameters']['ast']['optimizer'] with 'scheduler.max_lr',"
                             f"'weight_decay', and 'betas'.")
                raise
            logger.info(f"Rank {self.cmd_args.local_rank}: Initializing the DeepSpeed engine...")
            # Check whether the DeepSpeed configuration file exists
            if not os.path.exists(self.cmd_args.deepspeed_config):
                raise FileNotFoundError(f"DeepSpeed configuration file not found: {self.cmd_args.deepspeed_config}")

            # Pass the PyTorch optimizer to DeepSpeed
            self.model_engine, self.optimizer, _, _ = deepspeed.initialize(
                args=self.cmd_args, model=model, optimizer=self.pytorch_optimizer
            )
            # self.device is obtained from the DeepSpeed engine; this is the device the model actually resides on
            self.device = self.model_engine.device
            logger.info(f"Rank {self.cmd_args.local_rank}: DeepSpeed engine initialization complete. Model on device: {self.device}")
            if self.optimizer is not None:
                logger.info(f"Rank {self.cmd_args.local_rank}: Type of optimizer returned by DeepSpeed: {type(self.optimizer)}")
            else:
                logger.warning(
                    f"Rank {self.cmd_args.local_rank}: DeepSpeed did not return an optimizer. Check DeepSpeed config.")

            # Move the BEATs model to the correct device
            if self.beats_feature_extractor:
                self.beats_feature_extractor.to(self.device)
                logger.info(f"Rank {self.cmd_args.local_rank}: BEATs feature extractor moved to device {self.device}")

        except Exception as e:
            logger.error(f"Rank {self.cmd_args.local_rank}: Model or DeepSpeed initialization failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def setup_environment(self):
        """Set up the training environment, mainly loading the BEATs model for feature extraction."""
        try:
            # Determine the device to use when loading the BEATs model
            device_rank = self.cmd_args.local_rank if torch.cuda.is_available() else 'cpu'
            beats_device_str = f'cuda:{device_rank}' if torch.cuda.is_available() and isinstance(
                device_rank, int) else 'cpu'

            logger.info(f"Rank {self.cmd_args.local_rank}: Loading the BEATs Feature Extractor...")
            beats_checkpoint_path = self.config['paths']['beats_feature_extractor_checkpoint']
            if not os.path.exists(beats_checkpoint_path):
                raise FileNotFoundError(f"BEATs feature extractor checkpoint not found at: {beats_checkpoint_path}")

            # Load onto a temporary device (CPU or GPU); it will later be moved to the final device specified by deepspeed
            checkpoint = torch.load(beats_checkpoint_path, map_location='cpu')
            cfg = BEATsConfig(checkpoint['cfg'])
            self.beats_feature_extractor = BEATs(cfg)
            self.beats_feature_extractor.load_state_dict(checkpoint['model'])
            self.beats_feature_extractor.eval()
            self.beats_feature_extractor.to(beats_device_str)  # Move to the temporary device
            logger.info(
                f"Rank {self.cmd_args.local_rank}: BEATs Feature Extractor loaded successfully from {beats_checkpoint_path} and moved to the temporary device {beats_device_str}.")

        except Exception as e:
            logger.error(f"Rank {self.cmd_args.local_rank}: Environment setup (BEATs) failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def setup_wandb(self):
        """Set up WandB logging (executed only on rank 0)."""
        # Only rank 0 performs WandB initialization
        if hasattr(self.cmd_args, 'local_rank') and self.cmd_args.local_rank != 0: return
        # Check whether WandB is enabled in the configuration file
        if self.config['logging']['wandb']['enabled']:
            try:
                project = self.config['logging']['wandb']['project']
                name = self.config['logging']['wandb']['name'] + "_AST"  # Add a suffix to distinguish the run
                # Initialize the WandB run
                wandb.init(project=project, name=name, config={**self.config, **vars(self.cmd_args)})
                logger.info(f"Rank 0: WandB initialization complete - project: {project}, run: {name}")
            except ImportError:
                logger.error("The wandb package was not found or failed to import. Please install wandb or disable it in config.yaml.")
                self.config['logging']['wandb']['enabled'] = False  # Disable WandB
            except Exception as e:
                logger.error(f"Rank 0: WandB setup failed: {str(e)}")
                self.config['logging']['wandb']['enabled'] = False  # Disable WandB
        else:
            logger.info(f"Rank 0: WandB logging has been disabled in config.yaml")

    def cleanup(self):
        """Clean up resources, such as closing the WandB run."""
        try:
            current_rank = self.cmd_args.local_rank if hasattr(self.cmd_args, 'local_rank') else -1
            # Usually only rank 0 handles wandb.finish
            if current_rank <= 0:
                if self.config['logging']['wandb']['enabled']:
                    try:
                        if wandb.run is not None: wandb.finish()  # End the WandB run
                        logger.info(f"Rank 0: WandB finished.")
                    except ImportError:
                        logger.info("Rank 0: wandb was not imported, no finish needed.")
                    except Exception as e_wandb:
                        logger.error(f"Rank 0: Error while closing WandB: {e_wandb}")
            logger.info(f"Rank {current_rank}: Resource cleanup attempt complete.")
        except Exception as e:
            current_rank_for_log = self.cmd_args.local_rank if hasattr(self.cmd_args, 'local_rank') else 'N/A'
            logger.error(f"Rank {current_rank_for_log}: Error during cleanup: {str(e)}")

    def setup_dataloaders(self):
        """Set up the training and validation data loaders."""
        try:
            # Check whether we are in a distributed environment
            is_distributed = torch.distributed.is_initialized()
            current_rank = torch.distributed.get_rank() if is_distributed else 0
            world_size = torch.distributed.get_world_size() if is_distributed else 1

            logger.info(f"Rank {current_rank}/{world_size}: Setting up the data loaders...")
            # Get the data directory paths
            train_dir = Path(self.config['data']['audioset']['train_root'])
            val_dir = Path(self.config['data']['audioset']['val_root'])
            # Check whether the data directories exist
            if not train_dir.exists(): raise ValueError(f"Training data directory does not exist: {train_dir}")
            if not val_dir.exists(): raise ValueError(f"Validation data directory does not exist: {val_dir}")

            # The target device of the dataset instances is now self.device (i.e., the DeepSpeed engine's device)
            target_device_for_dataset = self.device
            logger.info(f"Rank {current_rank}: Instantiating the dataset using target device {target_device_for_dataset}")

            # Instantiate the training dataset
            train_dataset = AudioSetDataset(root_dir=train_dir, config=self.config,
                                            target_device=target_device_for_dataset)
            # If distributed training, use a DistributedSampler
            train_sampler, shuffle_dl_train = (
                DistributedSampler(train_dataset, num_replicas=world_size, rank=current_rank, shuffle=True),
                False) if world_size > 1 else (None, True)
            # Create the training data loader
            self.train_loader = DataLoader(
                train_dataset, batch_size=self.model_engine.train_micro_batch_size_per_gpu(),
                # Use DeepSpeed's micro batch size
                sampler=train_sampler, shuffle=shuffle_dl_train,
                num_workers=self.config.get('data', {}).get('num_workers', 4),
                pin_memory=self.config.get('data', {}).get('pin_memory', True),
                drop_last=True if world_size > 1 else False  # Distributed training usually needs drop_last
            )
            logger.info(f"Rank {current_rank}: Training data loader created. Number of batches: {len(self.train_loader)}")

            # Instantiate the validation dataset
            val_dataset = AudioSetDataset(root_dir=val_dir, config=self.config,
                                          target_device=target_device_for_dataset)
            # If distributed training, use a DistributedSampler
            val_sampler, shuffle_dl_val = (
                DistributedSampler(val_dataset, num_replicas=world_size, rank=current_rank, shuffle=False),
                False) if world_size > 1 else (None, False)
            # Create the validation data loader
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.model_engine.train_micro_batch_size_per_gpu(), sampler=val_sampler,
                shuffle=shuffle_dl_val, num_workers=self.config.get('data', {}).get('num_workers', 4),
                pin_memory=self.config.get('data', {}).get('pin_memory', True),
            )
            logger.info(f"Rank {current_rank}: Validation data loader created. Number of batches: {len(self.val_loader)}")
        except Exception as e:
            rank_for_log = self.cmd_args.local_rank if hasattr(self.cmd_args, 'local_rank') else (
                torch.distributed.get_rank() if torch.distributed.is_initialized() else -1)
            logger.error(f"Rank {rank_for_log}: Failed to set up the data loaders: {str(e)}")
            logger.error(traceback.format_exc())
            if hasattr(self, 'cleanup'): self.cleanup()  # Attempt cleanup on failure
            raise

    def setup_lr_scheduler(self):
        """Sets up the OneCycleLR scheduler based on config, similar to train_cfm.py."""
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Optimizer or Train Loader not initialized. Cannot create LR scheduler.")
            return
        logger.info(f"Rank {self.cmd_args.local_rank}: Setting up LR scheduler...")
        try:
            scheduler_config = self.config['hyperparameters']['ast']['optimizer']['scheduler']
            num_epochs = self.config['hyperparameters']['ast']['num_epochs']
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
        except KeyError as e:
            logger.error(f"Rank {self.cmd_args.local_rank}: Failed to create LR scheduler."
                         f"Missing key in config: {e}. Please ensure config.yaml has"
                         f"['hyperparameters']['ast']['optimizer']['scheduler'] with necessary keys.")
            raise
        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Failed to setup LR Scheduler: {e}\n{traceback.format_exc()}")
            raise

    def train(self):
        """Run the main training and validation loop."""
        self.setup_dataloaders()
        self.setup_lr_scheduler()
        training_state = TrainingStateManager(self.model_engine, self.config)  # Instantiate the training state manager
        num_epochs = self.config['hyperparameters']['ast']['num_epochs']  # Get the total number of training epochs

        # Get distributed information
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        current_rank = torch.distributed.get_rank() if is_distributed else 0
        world_size = torch.distributed.get_world_size() if is_distributed else 1

        logger.info(f"Rank {current_rank}: Starting training for {num_epochs} epochs. Distributed: {is_distributed}, World Size: {world_size}")

        # Training loop
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.model_engine.train()  # Set the model to training mode

            # If distributed training, set the sampler's epoch to ensure a different data order each epoch
            if world_size > 1 and hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_iterator = self.train_loader
            # Only rank 0 shows the tqdm progress bar
            if current_rank == 0:
                epoch_iterator = tqdm(self.train_loader,
                                      desc=f"Epoch {epoch + 1}/{num_epochs} [Training...]",
                                      ncols=150, leave=True)

            # ---------- Training ----------
            for batch_idx, batch in enumerate(epoch_iterator):
                if batch is None:
                    logger.warning(f"Rank {current_rank}: Training skipped an invalid batch {batch_idx}")
                    continue
                try:
                    # Move the waveform data to the device
                    waveforms = batch['waveform'].to(self.device, non_blocking=True)

                    # Extract features using BEATs
                    with torch.no_grad():
                        # The features returned by BEATs have shape (B, T, D)
                        input_features, _ = self.beats_feature_extractor.extract_features(waveforms, padding_mask=None)
                        # Reshape to (B, D, T) as expected by the AudioSetTokenizer
                        input_features = input_features.permute(0, 2, 1)

                    # Move the input features to the device
                    input_features = input_features.to(self.device, non_blocking=True)

                    # Forward pass to obtain the model output
                    outputs = self.model_engine(input_features)

                    # Compute the loss and metrics
                    total_loss, losses, metrics = training_state.compute_loss(outputs, None)

                    # Perform the optimization step (backward pass and parameter update)
                    step_succeeded = training_state.optimize_step(total_loss)

                    # Step the LR scheduler (if the optimization step succeeded)
                    if step_succeeded and self.lr_scheduler:
                        self.lr_scheduler.step()

                    # Only rank 0 logs and updates the tqdm progress bar
                    if current_rank == 0:
                        current_lr = self.pytorch_optimizer.param_groups[0]['lr'] if self.pytorch_optimizer else float(
                            'nan')
                        lr_to_log = f"{current_lr:.2e}"

                        # Update the tqdm postfix display
                        postfix_dict = {
                            'loss': f"{total_loss.item():.4f}",
                            'recon_loss': f"{losses.get('reconstruction_loss', torch.tensor(0.0)):.4f}",
                            'commit_loss': f"{losses.get('commit_loss', torch.tensor(0.0)):.4f}",
                            'lr': lr_to_log  # Added LR to tqdm
                        }
                        if isinstance(epoch_iterator, tqdm):
                            epoch_iterator.set_postfix(postfix_dict)

                        # If WandB is enabled, log the training metrics
                        if self.config['logging']['wandb']['enabled']:
                            try:
                                if wandb.run is not None:
                                    # Helper function that ensures values can be converted to floats for WandB logging
                                    def to_float_for_wandb(value, default_if_none=0.0):
                                        if value is None: return default_if_none
                                        if hasattr(value, 'item'): return value.item()
                                        try:
                                            return float(value)
                                        except (TypeError, ValueError):
                                            logger.warning(
                                                f"Wandb: Could not convert value '{value}' (type: {type(value)}) to a float. Using the default value {default_if_none}.")
                                            return default_if_none

                                    # Prepare the WandB log data
                                    wandb_log_data = {
                                        "train/total_loss": to_float_for_wandb(total_loss),
                                        "train/recon_loss": to_float_for_wandb(losses.get('reconstruction_loss')),
                                        "train/commit_loss": to_float_for_wandb(losses.get('commit_loss')),
                                        "train/learning_rate": current_lr  # Added LR to WandB
                                    }
                                    # Log to WandB
                                    wandb.log(wandb_log_data, step=self.global_step)
                            except ImportError:
                                pass  # If wandb is not installed, ignore
                            except Exception as e_wandb_log:
                                logger.warning(f"Wandb logging of training data failed: {e_wandb_log}")
                    self.global_step += 1  # Increment the global step counter
                except Exception as e:
                    logger.error(f"Rank {current_rank}: Training step failed at batch {batch_idx}: {str(e)}")
                    if current_rank == 0: logger.error(traceback.format_exc())
                    raise  # Raise the exception to interrupt training

            # ---------- Validation ----------
            self.model_engine.eval()  # Set the model to evaluation mode
            if current_rank == 0:
                logger.info(
                    f"Rank {current_rank}: Entering validation mode. Number of validation batches: {len(self.val_loader) if self.val_loader else 'N/A'}")
                if self.val_loader and len(self.val_loader) == 0:
                    logger.warning("Rank 0: The validation data loader is empty; validation will be skipped.")

            # Initialize the validation metric accumulators
            total_val_loss_sum = torch.tensor(0.0, device=self.device)
            total_val_samples = torch.tensor(0, device=self.device, dtype=torch.long)

            # If distributed training, set the sampler's epoch
            if world_size > 1 and self.val_loader and hasattr(self.val_loader.sampler, 'set_epoch'):
                self.val_loader.sampler.set_epoch(epoch)

            val_iterator = self.val_loader
            # Only rank 0 shows the tqdm progress bar
            if current_rank == 0 and self.val_loader and len(self.val_loader) > 0:
                val_iterator = tqdm(self.val_loader,
                                    desc=f"Epoch {epoch + 1}/{num_epochs} [Validating...]",
                                    ncols=100)

            # Validation loop
            if self.val_loader and len(self.val_loader) > 0:
                with torch.no_grad():  # The validation phase does not need gradient computation
                    for batch_idx, val_batch in enumerate(val_iterator):
                        if val_batch is None:
                            if current_rank == 0: logger.warning(f"Rank {current_rank}: Validation skipped an invalid batch {batch_idx}")
                            continue
                        try:
                            # Move the waveform data to the device
                            waveforms = val_batch['waveform'].to(self.device, non_blocking=True)
                            bs = waveforms.size(0)  # Get the batch size

                            # Extract features using BEATs
                            with torch.no_grad():
                                input_features, _ = self.beats_feature_extractor.extract_features(waveforms,
                                                                                                  padding_mask=None)
                                input_features = input_features.permute(0, 2, 1)

                            # Move the features to the device
                            input_features = input_features.to(self.device, non_blocking=True)

                            # Forward pass to obtain the model output
                            outputs = self.model_engine(input_features)
                            # Compute the loss and metrics
                            val_loss_raw, _, val_metrics = training_state.compute_loss(outputs, None)

                            # Ensure the loss and metrics are tensors and moved to the device
                            if not isinstance(val_loss_raw, torch.Tensor):
                                current_val_loss = torch.tensor(val_loss_raw, device=self.device)
                            else:
                                current_val_loss = val_loss_raw.to(self.device)
                            current_val_loss = current_val_loss.squeeze()
                            current_val_loss = current_val_loss.float()

                            # Accumulate the loss, the similarity, and the sample count
                            total_val_loss_sum += current_val_loss * bs
                            total_val_samples += bs

                            # Only rank 0 updates the tqdm postfix display
                            if current_rank == 0 and isinstance(val_iterator, tqdm):
                                val_iterator.set_postfix({'loss': f"{current_val_loss.item():.4f}"})
                        except Exception as e:
                            logger.error(
                                f"Rank {current_rank}: Validation step failed at batch {batch_idx} (Epoch {epoch + 1}): {str(e)}")
                            if current_rank == 0: logger.error(traceback.format_exc())

                # If distributed training, all_reduce-sum the accumulated results across all ranks
                if world_size > 1:
                    torch.distributed.all_reduce(total_val_loss_sum, op=torch.distributed.ReduceOp.SUM)
                    torch.distributed.all_reduce(total_val_samples, op=torch.distributed.ReduceOp.SUM)

                # Compute the average validation metric
                avg_val_loss = float('inf')
                if total_val_samples.item() > 0:  # Ensure total_val_samples is not zero
                    avg_val_loss = total_val_loss_sum.item() / total_val_samples.item()

                # Only rank 0 logs the validation results
                if current_rank == 0:
                    logger.info(f"\nEpoch {epoch + 1} validation results: average loss {avg_val_loss:.4f}")

                # Check whether this is a new best model (based on the validation loss)
                best_improved_local = False
                if current_rank == 0:
                    if avg_val_loss <= self.best_metrics['val_loss']:
                        self.best_metrics['val_loss'] = avg_val_loss
                        logger.info(f"Rank 0: Found a new best model, validation loss {avg_val_loss:.4f}")
                        best_improved_local = True

                # In a distributed environment, broadcast rank 0's best-model information to the other processes
                best_improved_tensor = torch.tensor(int(best_improved_local), device=self.device, dtype=torch.int)
                if world_size > 1:
                    torch.distributed.broadcast(best_improved_tensor, src=0)
                globally_best_improved = bool(best_improved_tensor.item())

                # If it is the global best model, save a checkpoint
                if globally_best_improved:
                    tag = f"best_epoch_{epoch + 1}_loss_{avg_val_loss:.4f}"

                    current_client_state = {}  # Defaults to an empty dictionary
                    # Only rank 0 prepares the client_state
                    if current_rank == 0:
                        current_client_state = {
                            'epoch': epoch + 1,
                            'metrics': {'avg_val_loss': avg_val_loss},
                            'best_metrics': self.best_metrics,
                            'global_step': self.global_step,
                            'config_yaml': self.config
                        }

                    # Call the checkpoint-saving method
                    self.save_checkpoint(tag=tag, client_state=current_client_state)

                # Only rank 0 logs the validation metrics to WandB
                if current_rank == 0 and self.config['logging']['wandb']['enabled']:
                    try:
                        if wandb.run is not None:
                            wandb.log({
                                "val/epoch": epoch + 1,
                                "val/avg_loss": avg_val_loss,
                                "val/best_val_loss": self.best_metrics['val_loss']

                            }, step=self.global_step)  # Use the global step as the WandB step
                    except Exception as e:
                        logger.warning(f"Wandb logging of validation data failed: {e}")
            elif current_rank == 0:
                logger.info(f"Rank 0: Skipping validation, val_loader is empty or has no data.")

            logger.info(f"Rank {current_rank}: Epoch {epoch + 1} complete.")
            # Clear the CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def save_checkpoint(self, tag: str, client_state: dict):
        """
        Save the model checkpoint as a single .pth file.
        - Only rank 0 performs the actual file-saving operation.
        - The filename format is tag + '.pth'.
        - The saved content includes model_state_dict, optimizer state, scheduler state, and client_state.
        """
        is_distributed = torch.distributed.is_available() and torch.distributed.is_initialized()
        current_rank = torch.distributed.get_rank() if is_distributed else 0

        if is_distributed: torch.distributed.barrier()

        if current_rank == 0:
            checkpoint_file_path = None
            try:
                self.checkpoint_ast_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_file_path = self.checkpoint_ast_dir / f"{tag}.pth"
                logger.info(f"Rank 0: Preparing to save the checkpoint to {checkpoint_file_path}")

                model_state_dict = self.model_engine.module.state_dict()

                content_to_save = {
                    'model_state_dict': model_state_dict,
                    'pytorch_optimizer_state_dict': self.pytorch_optimizer.state_dict() if self.pytorch_optimizer else None,
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict() if self.lr_scheduler else None,
                }
                if client_state:
                    content_to_save.update(client_state)

                torch.save(content_to_save, checkpoint_file_path)
                logger.info(f"Rank 0: Checkpoint saved successfully to {checkpoint_file_path}")

            except Exception as e:
                log_path = checkpoint_file_path if checkpoint_file_path else self.checkpoint_ast_dir / f"{tag}.pth"
                logger.error(f"Rank 0: Failed to save the checkpoint (tag: {tag}) to {log_path}: {e}")
                logger.error(traceback.format_exc())
                raise

        if is_distributed: torch.distributed.barrier()


# --- Main function ---
def main():
    global trainer_global_object
    trainer_global_object = None

    # Create the command-line argument parser
    parser = argparse.ArgumentParser(description='AST training script with DeepSpeed')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration YAML file')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank passed by the distributed launcher (DeepSpeed handles this automatically)')
    # Add the DeepSpeed-related command-line arguments
    parser = deepspeed.add_config_arguments(parser)
    # Parse the command-line arguments
    cmd_args = parser.parse_args()

    # Set up the global logger, adjusting the logging level based on local_rank
    global logger
    logger = setup_logger(cmd_args.local_rank if hasattr(cmd_args, 'local_rank') else -1)

    # Check and log local_rank
    if not hasattr(cmd_args, 'local_rank') or cmd_args.local_rank == -1:
        # Try to get it from the environment variable, which may be useful in some single-GPU scenarios not started by a deepspeed launcher
        env_local_rank = os.environ.get('LOCAL_RANK')
        if env_local_rank is not None:
            try:
                effective_local_rank = int(env_local_rank)
                logger.info(f"Got local_rank from the environment variable LOCAL_RANK: {effective_local_rank}")
            except ValueError:
                logger.warning(f"The environment variable LOCAL_RANK ('{env_local_rank}') is not a valid integer; the default value 0 will be used.")
                effective_local_rank = 0
        else:  # If there is no environment variable either, assume single-machine non-distributed
            effective_local_rank = 0  # Or -1, depending on how the subsequent logic handles non-distributed
            logger.info(
                f"local_rank was not provided by the DeepSpeed launcher and is not in the environment either; assuming non-distributed or single-GPU run (local_rank={effective_local_rank}).")
        cmd_args.local_rank = effective_local_rank  # Update the value in cmd_args
    else:
        logger.info(f"Distributed mode: Rank {cmd_args.local_rank}")

    try:
        # DeepSpeed device initialization
        if cmd_args.local_rank != -1:  # Only initialize the process group in distributed mode
            deepspeed.init_distributed()  # Initialize the distributed environment
            # After init_distributed, torch.cuda.set_device(cmd_args.local_rank) will be called
            logger.info(
                f"Rank {cmd_args.local_rank}: DeepSpeed distributed environment initialized. Current device: {torch.cuda.current_device()}")
        else:  # Single GPU or CPU
            if torch.cuda.is_available():
                # In the single-GPU case, the device can be set manually, but DeepSpeed usually handles it
                # If deepspeed.initialize is used, it will handle the device
                logger.info(f"Rank {cmd_args.local_rank}: Non-distributed mode; will try to use an available GPU (if the configuration allows).")
            else:
                logger.info(f"Rank {cmd_args.local_rank}: Non-distributed mode; no GPU available, will use the CPU.")

        # Instantiate the Trainer and start training
        trainer_global_object = Trainer(config_path=cmd_args.config, cmd_args=cmd_args)
        trainer_global_object.train()
    except KeyboardInterrupt:
        # Catch the user interrupt signal (Ctrl+C)
        rank_for_log = cmd_args.local_rank if hasattr(cmd_args, 'local_rank') else 'N/A'
        if logger: logger.warning(f"\nRank {rank_for_log}: Training was interrupted by the user.")
        # Attempt to clean up resources
        if trainer_global_object is not None: trainer_global_object.cleanup()
    except Exception as e:
        # Catch other exceptions
        rank_for_log = cmd_args.local_rank if hasattr(cmd_args, 'local_rank') else 'N/A'
        if logger:
            logger.error(f"\nRank {rank_for_log}: An error occurred during training: {str(e)}")
            logger.error(traceback.format_exc())
        else:  # If the logger was not initialized successfully either
            print(f"\nRank {rank_for_log}: An error occurred during training (logger not initialized): {str(e)}")
            print(traceback.format_exc())
        # Attempt to clean up resources
        if trainer_global_object is not None: trainer_global_object.cleanup()
        sys.exit(1)  # Exit the program with a non-zero status code to indicate an error
    finally:
        # Ensure that in distributed mode all processes reach this point, then exit
        if hasattr(cmd_args, 'local_rank') and cmd_args.local_rank != -1 and torch.distributed.is_initialized():
            torch.distributed.barrier()  # Ensure that all processes reach this point
            if logger: logger.info(f"Rank {cmd_args.local_rank} passed the barrier and is exiting normally.")
        # For non-distributed or the main process, ensure the trainer is cleaned up
        elif trainer_global_object is not None and logger:  # Cleanup in the non-distributed case
            logger.info("Training finished or an error occurred; performing cleanup (non-distributed).")
        # Ensure the trainer object is cleaned up
        if trainer_global_object is not None and (
                not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
            if hasattr(trainer_global_object, 'cleanup') and callable(trainer_global_object.cleanup):
                trainer_global_object.cleanup()


if __name__ == "__main__":
    # Set the multiprocessing start method to 'spawn', which is important for CUDA and multi-process DataLoaders
    if sys.platform != 'win32':  # 'spawn' is the default on Windows, 'fork' is the default on Linux
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        # Only force it if the current method is neither 'spawn' nor started by forkserver
        # Some environments or libraries may have already set 'forkserver'
        if current_start_method != 'spawn' and current_start_method != 'forkserver':
            try:
                multiprocessing.set_start_method('spawn', force=True)
                print(f"INFO: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e:
                # If the context is already set, force=True may also fail; print a warning and continue
                print(
                    f"WARNING: Could not set multiprocessing start method to 'spawn' (current: {current_start_method}): {e}")
        elif current_start_method:
            print(f"INFO: Multiprocessing start method already '{current_start_method}'.")

    # Ignore a specific user warning
    os.environ['PYTHONWARNINGS'] = 'ignore:semaphore_tracker:UserWarning'
    # Enable faulthandler to print a Python traceback when a crash occurs
    faulthandler.enable()
    # Enable cuDNN benchmark mode, which is usually beneficial for training with fixed input sizes
    torch.backends.cudnn.benchmark = True

    # Set the random seed to ensure reproducibility
    seed = 666
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Call the main function to start the training pipeline
    main()
