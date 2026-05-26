import logging
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LibriSpeechDataset(Dataset):
    """
    LibriSpeech dataset implementation:
    This version integrates the stable, correct audio-processing logic from data_loader.py,
    including normalization, the resampling pipeline, and [fixed-position] cropping/padding,
    while preserving compatibility with AudioSetDataset and ConcatDataset.
    """

    def __init__(self, root_dir, config):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.config = config

        if not self.root_dir.exists():
            raise ValueError(f"Dataset path does not exist: {self.root_dir}")

        self.eps = 1e-8
        self.max_audio_value = 1.0
        self.min_audio_value = -1.0

        self.default_features = {
            'dataset_type': 'librispeech',
            'waveform': torch.zeros(self.config['audio']['max_length']),
            'speaker_id': -1,
            'file_path': "",
            'mel_lengths': torch.tensor(0, dtype=torch.long),
            's3_token': torch.zeros(500, dtype=torch.long)
        }
        self._init_dataset()

    def _init_dataset(self):
        self.sample_rate = self.config['audio']['sample_rate']
        self.target_vocos_sr = self.config['vocos']['sample_rate']
        self.mel_hop_length = self.config['vocos']['mel']['hop_length']
        self.mel_n_fft = self.config['vocos']['mel']['n_fft']

        self.file_paths = sorted([p for p in self.root_dir.rglob("*.flac")])
        if not self.file_paths:
            raise FileNotFoundError(f"No .flac files found in {self.root_dir}")

        self.speaker_map = {spk: i for i, spk in
                            enumerate(sorted(list(set(p.parent.parent.name for p in self.file_paths))))}

    def _safe_load_audio(self, audio_path):
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.numel() == 0:
                return self.default_features['waveform'].clone().unsqueeze(0), self.sample_rate

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            max_val = torch.abs(waveform).max()
            if max_val > self.eps:
                waveform = torch.clamp(waveform / max_val, self.min_audio_value, self.max_audio_value)

            return waveform, sr
        except Exception as e:
            logger.error(f"Failed to load audio file {str(audio_path)}: {e}")
            return self.default_features['waveform'].clone().unsqueeze(0), self.sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            waveform_orig_cpu, sr_orig = self._safe_load_audio(file_path)

            # --- 1. Compute mel_lengths ---
            waveform_for_mel_calc = waveform_orig_cpu
            if sr_orig != self.target_vocos_sr:
                waveform_for_mel_calc = torchaudio.functional.resample(
                    waveform_orig_cpu, sr_orig, self.target_vocos_sr
                )
            num_samples_for_mel = waveform_for_mel_calc.shape[-1]
            if self.config['vocos']['mel'].get('center', True):
                mel_length = num_samples_for_mel // self.mel_hop_length + 1
            else:
                mel_length = (
                                     num_samples_for_mel - self.mel_n_fft) // self.mel_hop_length + 1 if num_samples_for_mel >= self.mel_n_fft else 1
            mel_length = max(1, mel_length)

            # --- 2. Waveform processing ---
            waveform_main_sr = waveform_orig_cpu
            if sr_orig != self.sample_rate:
                waveform_main_sr = torchaudio.functional.resample(
                    waveform_orig_cpu, sr_orig, self.sample_rate
                )

            # --- 3. Fixed cropping/padding logic ---
            target_audio_samples = self.config['audio']['max_length']
            current_len_main_sr = waveform_main_sr.size(1)

            if current_len_main_sr > target_audio_samples:
                # Crop from the beginning
                waveform_processed = waveform_main_sr[:, :target_audio_samples]
            else:
                # Pad at the end
                pad_length = target_audio_samples - current_len_main_sr
                waveform_processed = F.pad(waveform_main_sr, (0, pad_length))

            speaker_id_str = file_path.parent.parent.name
            speaker_id = self.speaker_map.get(speaker_id_str, -1)

            # --- 4. Load the S3 token ---
            s3_token_tensor = torch.zeros(0, dtype=torch.long)
            s3_token_dir_path = self.config.get('data', {}).get('librispeech', {}).get('s3_token_dir')

            if s3_token_dir_path:
                SPEECH_TOKEN_BASE_DIR = Path(s3_token_dir_path)
                file_id = file_path.stem
                speech_token_file = SPEECH_TOKEN_BASE_DIR / f"{file_id}.pt"
                if speech_token_file.exists():
                    try:
                        loaded_tokens = torch.load(speech_token_file, map_location='cpu')
                        if isinstance(loaded_tokens, torch.Tensor):
                            s3_token_tensor = loaded_tokens.long()
                        elif isinstance(loaded_tokens, list):
                            s3_token_tensor = torch.tensor(loaded_tokens, dtype=torch.long)
                    except Exception as e_token:
                        logger.error(f"Failed to load S3 token file: {speech_token_file}, error: {e_token}")

            return {
                'dataset_type': 'librispeech',
                'waveform': waveform_processed.squeeze(0),
                'file_path': str(file_path),
                'mel_lengths': torch.tensor(mel_length, dtype=torch.long),
                'speaker_id': speaker_id,
                's3_token': s3_token_tensor
            }
        except Exception as e:
            logger.error(f"Skipping corrupted or failed LibriSpeech file (index: {idx}, path: {file_path}). Error: {e}")
            default_item = {k: v.clone() if torch.is_tensor(v) else v for k, v in self.default_features.items()}
            default_item['file_path'] = str(file_path)
            return default_item


class AudioSetDataset(Dataset):
    """
    AudioSet dataset implementation:
    This version integrates the stable, correct audio-processing logic from data_loader.py,
    while preserving compatibility with LibriSpeechDataset and ConcatDataset.
    """

    def __init__(self, root_dir, config, target_device=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.config = config
        self.processor_device = torch.device(target_device if target_device else 'cpu')

        if not self.root_dir.exists():
            raise ValueError(f"Dataset path does not exist: {self.root_dir}")

        self.eps = 1e-8
        self.max_audio_value = 1.0
        self.min_audio_value = -1.0

        self.default_features = {
            'dataset_type': 'audioset',
            'waveform': torch.zeros(self.config['audio']['max_length']),
            'file_path': "",
            'mel_lengths': torch.tensor(0, dtype=torch.long),
            'speaker_id': -1,  # Compatibility placeholder
            's3_token': torch.zeros(500, dtype=torch.long)  # Compatibility placeholder
        }
        self._init_dataset()

    def _init_dataset(self):
        self.sample_rate = self.config['audio']['sample_rate']
        self.target_vocos_sr = self.config['vocos']['sample_rate']
        self.mel_hop_length = self.config['vocos']['mel']['hop_length']
        self.mel_n_fft = self.config['vocos']['mel']['n_fft']
        self.file_paths = sorted(list(self.root_dir.glob("*.flac")))
        if not self.file_paths:
            raise ValueError(f"No audio files found in directory {self.root_dir}.")

    def _safe_load_audio(self, audio_path):
        try:
            waveform, sr = torchaudio.load(str(audio_path))
            if waveform.numel() == 0:
                # Return the default silent waveform
                return self.default_features['waveform'].clone().unsqueeze(0), self.sample_rate

            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            if waveform.size(0) > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Normalize and clamp
            max_val = torch.abs(waveform).max()
            safe_divisor = max_val if max_val > self.eps else torch.tensor(1.0, device=waveform.device,
                                                                           dtype=waveform.dtype)
            waveform = waveform / safe_divisor
            waveform = torch.clamp(waveform, self.min_audio_value, self.max_audio_value)

            return waveform, sr
        except Exception as e:
            # logger.error(f"Failed to load audio file {audio_path}: {e}")
            return self.default_features['waveform'].clone().unsqueeze(0), self.sample_rate

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            waveform_orig_cpu, sr_orig = self._safe_load_audio(file_path)

            # --- 1. Compute mel_lengths ---
            waveform_for_mel_calc = waveform_orig_cpu
            if sr_orig != self.target_vocos_sr:
                waveform_for_mel_calc = torchaudio.functional.resample(waveform_orig_cpu, sr_orig, self.target_vocos_sr)

            num_samples_for_mel = waveform_for_mel_calc.shape[-1]
            if self.config['vocos']['mel'].get('center', True):
                mel_length_actual = num_samples_for_mel // self.mel_hop_length + 1
            else:
                mel_length_actual = (
                                            num_samples_for_mel - self.mel_n_fft) // self.mel_hop_length + 1 if num_samples_for_mel >= self.mel_n_fft else 1
            mel_length_actual = max(1, mel_length_actual)

            # --- 2. Main waveform processing ---
            waveform_main_sr = waveform_orig_cpu
            if sr_orig != self.sample_rate:
                waveform_main_sr = torchaudio.functional.resample(waveform_orig_cpu, sr_orig, self.sample_rate)

            # --- 3. Produce the final output waveform (fixed cropping/padding) ---
            target_audio_samples = self.config['audio']['max_length']
            current_len_main_sr = waveform_main_sr.size(1)

            if current_len_main_sr > target_audio_samples:
                waveform_processed = waveform_main_sr[:, :target_audio_samples]
            else:
                pad_length = target_audio_samples - current_len_main_sr
                waveform_processed = F.pad(waveform_main_sr, (0, pad_length))

            # --- 4. Return a dictionary structure identical to LibriSpeechDataset ---
            return {
                'dataset_type': 'audioset',
                'waveform': waveform_processed.squeeze(0),
                'file_path': str(file_path),
                'mel_lengths': torch.tensor(mel_length_actual, dtype=torch.long),
                'speaker_id': self.default_features['speaker_id'],  # Use the default placeholder
                's3_token': self.default_features['s3_token'].clone()  # Use the default placeholder
            }
        except Exception as e:
            logger.error(f"Skipping corrupted or failed AudioSet file (index: {idx}, path: {file_path}). Error: {e}")
            # Return a deep copy of the default values
            default_item = {k: v.clone() if torch.is_tensor(v) else v for k, v in self.default_features.items()}
            default_item['file_path'] = str(file_path)
            return default_item
