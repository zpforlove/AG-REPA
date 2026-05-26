"""
S3 Speech Token extraction tool

Supports extracting S3 tokens from the following datasets:
1. LibriSpeech (.flac format)
2. Emilia-Large (.mp3 format, including Emilia and Emilia-YODAS)

Usage:
# Extract LibriSpeech
python extract_s3_tokens.py --dataset /path/to/librispeech --save_dir speech_tokens

# Extract Emilia-Large
python extract_s3_tokens.py --dataset /mnt/data/Emilia-large --save_dir emilia_speech_tokens --dataset_type emilia
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import onnxruntime
import torch
import torchaudio
import whisper
import yaml
from tqdm import tqdm


class SpeechTokenExtractor:
    def __init__(self, config_path: str, model_path: str, num_threads: int = 8, gpu_id: int = 0):
        """Initialize the Speech Token extractor"""
        # Set up the specified GPU device
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        # Load the configuration file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')

        # Configure the ONNX runtime
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = 1

        # Correctly configure the ONNX runtime to use the specified GPU
        provider_options = {'device_id': gpu_id}
        self.session = onnxruntime.InferenceSession(
            model_path,
            sess_options=options,
            providers=[("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]
        )

        # Get the audio parameters from the configuration
        self.sample_rate = self.config['audio']['sample_rate']
        self.max_length = self.config['audio']['max_length']
        self.num_threads = num_threads

        # Pre-create the resampler and move it to the GPU
        self.resampler = None  # Will be created when needed

    def process_audio(self, audio_path: str) -> Tuple[Optional[str], List[int]]:
        """Process a single audio file"""
        try:
            # Load the audio (must be on the CPU)
            audio, sample_rate = torchaudio.load(audio_path, backend='soundfile')

            # Move to the GPU
            audio = audio.to(self.device)

            # Resample to the target sample rate
            if sample_rate != self.sample_rate:
                if self.resampler is None or self.resampler.orig_freq != sample_rate:
                    self.resampler = torchaudio.transforms.Resample(
                        orig_freq=sample_rate,
                        new_freq=self.sample_rate
                    ).to(self.device)
                audio = self.resampler(audio)

            # Convert to mono
            if audio.shape[0] > 1:
                audio = audio.mean(dim=0)
            else:
                audio = audio.squeeze(0)  # Ensure it is 1D

            # Handle the audio length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                # Pad to a fixed length
                pad_length = self.max_length - len(audio)
                audio = torch.nn.functional.pad(audio, (0, pad_length))

            # Extract features using whisper (running on the GPU)
            # Note: whisper.log_mel_spectrogram internally uses the current device
            with torch.cuda.device(self.device):
                feat = whisper.log_mel_spectrogram(audio, n_mels=128)  # [n_mels, T]
                feat = feat.unsqueeze(0)  # [1, n_mels, T]

            # ONNX requires numpy input, so move it back to the CPU
            feat_np = feat.detach().cpu().numpy()

            # Extract tokens
            speech_token = self.session.run(
                None,
                {
                    self.session.get_inputs()[0].name: feat_np,
                    self.session.get_inputs()[1].name: np.array([feat_np.shape[2]], dtype=np.int32)
                }
            )[0].flatten().tolist()

            return Path(audio_path).stem, speech_token

        except Exception as e:
            logging.error(f'Failed to process audio ({audio_path}): {str(e)}')
            if 'audio' in locals():
                logging.error(f'Audio shape: {audio.shape}, device: {audio.device}')
            if 'feat' in locals():
                logging.error(f'Feature shape: {feat.shape}, device: {feat.device}')
            if 'feat_np' in locals():
                logging.error(f'Numpy feature shape: {feat_np.shape}')
            return None, []

    def extract_from_librispeech(self, dataset_path: str, save_path: str):
        """Extract speech tokens from the LibriSpeech dataset"""
        audio_files = list(Path(dataset_path).rglob('*.flac'))
        logging.info(f'Found {len(audio_files)} LibriSpeech audio files')

        os.makedirs(save_path, exist_ok=True)

        def process_and_save(audio_file):
            audio_id, tokens = self.process_audio(str(audio_file))
            if audio_id:
                save_file = Path(save_path) / f"{audio_id}.pt"
                torch.save(tokens, save_file)
                return audio_id
            return None

        processed_count = 0
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(process_and_save, f) for f in audio_files]

            for future in tqdm(futures, desc="Processing LibriSpeech audio"):
                result = future.result()
                if result:
                    processed_count += 1

        logging.info(f'Successfully processed and saved {processed_count} LibriSpeech audio files to: {save_path}')

    def _collect_json_files(self, root_path: Path, languages: List[str]) -> List[Path]:
        """Collect all qualifying JSON files under the specified directory, with progress display"""
        json_files = []

        for lang in languages:
            lang_dir = root_path / lang
            if not lang_dir.exists():
                logging.info(f"  Language directory does not exist, skipping: {lang_dir}")
                continue

            logging.info(f"  Scanning language directory: {lang} ...")

            # First, get all subdirectories
            sub_dirs = [d for d in lang_dir.iterdir() if d.is_dir()]
            logging.info(f"    Found {len(sub_dirs)} subdirectories")

            # Iterate over the subdirectories, with a progress bar
            lang_json_count = 0
            for sub_dir in tqdm(sub_dirs, desc=f"    Scanning {lang} subdirectories", leave=False):
                # Recursively find all JSON files
                for json_file in sub_dir.rglob("*.json"):
                    json_files.append(json_file)
                    lang_json_count += 1

            logging.info(f"    Found {lang_json_count} JSON files in the {lang} directory")

        return json_files

    def extract_from_emilia(
            self,
            dataset_path: str,
            save_path: str,
            languages: List[str] = None,
            min_duration: float = 1.0,
            max_duration: float = 30.0,
            min_dnsmos: float = 3.0
    ):
        """
        Extract speech tokens from the Emilia-Large dataset

        Args:
            dataset_path: Emilia-Large root directory, which should contain the Emilia and Emilia-YODAS subdirectories
            save_path: save directory
            languages: list of languages to process
            min_duration: minimum audio duration
            max_duration: maximum audio duration
            min_dnsmos: minimum DNSMOS score
        """
        if languages is None:
            languages = ["DE", "EN", "FR", "JA", "KO", "ZH"]

        dataset_path = Path(dataset_path)

        # ========== Stage 1: Collect all JSON files ==========
        logging.info("=" * 60)
        logging.info("Stage 1: Scanning the directory structure and collecting JSON files...")
        logging.info("=" * 60)

        all_json_files = []
        subdirs = ["Emilia", "Emilia-YODAS"]

        for subdir in subdirs:
            root_path = dataset_path / subdir
            if not root_path.exists():
                logging.warning(f"Subdirectory does not exist, skipping: {root_path}")
                continue

            logging.info(f"Scanning: {root_path}")
            json_files = self._collect_json_files(root_path, languages)
            all_json_files.extend([(root_path, jf) for jf in json_files])
            logging.info(f"  Found {len(json_files)} JSON files in {subdir}")

        logging.info(f"Found {len(all_json_files)} JSON files in total")

        # ========== Stage 2: Parse JSON metadata and filter ==========
        logging.info("=" * 60)
        logging.info("Stage 2: Parsing JSON metadata and filtering...")
        logging.info("=" * 60)

        audio_files = []
        skipped_stats = {
            'missing_fields': 0,
            'duration_filter': 0,
            'dnsmos_filter': 0,
            'file_not_found': 0,
            'parse_error': 0
        }

        for root_path, json_file in tqdm(all_json_files, desc="Parsing JSON files"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    meta = json.load(f)

                # Validate the required fields
                if not all(k in meta for k in ['id', 'wav', 'duration']):
                    skipped_stats['missing_fields'] += 1
                    continue

                duration = float(meta['duration'])
                dnsmos = float(meta.get('dnsmos', 4.0))

                # Filtering conditions
                if duration < min_duration or duration > max_duration:
                    skipped_stats['duration_filter'] += 1
                    continue
                if dnsmos < min_dnsmos:
                    skipped_stats['dnsmos_filter'] += 1
                    continue

                # Build the audio file path
                # Extract the language directory from the json_file path
                rel_parts = json_file.relative_to(root_path).parts
                lang = rel_parts[0] if rel_parts else None

                if lang:
                    lang_dir = root_path / lang
                    wav_path = lang_dir / meta['wav']
                else:
                    wav_path = root_path / meta['wav']

                if not wav_path.exists():
                    # Try to find an mp3 with the same name in the same directory as the json
                    wav_path = json_file.with_suffix('.mp3')
                    if not wav_path.exists():
                        skipped_stats['file_not_found'] += 1
                        continue

                audio_files.append({
                    'id': meta['id'],
                    'audio_path': str(wav_path),
                    'duration': duration
                })

            except Exception as e:
                skipped_stats['parse_error'] += 1
                logging.debug(f"Failed to parse JSON {json_file}: {e}")
                continue

        # Print the filtering statistics
        logging.info("Filtering statistics:")
        logging.info(f"  - Missing required fields: {skipped_stats['missing_fields']}")
        logging.info(f"  - Duration out of range: {skipped_stats['duration_filter']}")
        logging.info(f"  - DNSMOS out of range: {skipped_stats['dnsmos_filter']}")
        logging.info(f"  - Audio file not found: {skipped_stats['file_not_found']}")
        logging.info(f"  - JSON parse error: {skipped_stats['parse_error']}")
        logging.info(f"Qualifying audio files: {len(audio_files)}")

        if not audio_files:
            logging.warning("No qualifying audio files were found!")
            return

        # ========== Stage 3: Extract Speech Tokens ==========
        logging.info("=" * 60)
        logging.info("Stage 3: Extracting Speech Tokens...")
        logging.info("=" * 60)

        os.makedirs(save_path, exist_ok=True)

        def process_and_save(item):
            audio_id = item['id']
            audio_path = item['audio_path']

            # Check whether it already exists
            save_file = Path(save_path) / f"{audio_id}.pt"
            if save_file.exists():
                return ('skipped', audio_id)

            try:
                _, tokens = self.process_audio(audio_path)
                if tokens:
                    torch.save(tokens, save_file)
                    return ('processed', audio_id)
            except Exception as e:
                logging.error(f"Failed to process {audio_id}: {e}")
            return ('failed', None)

        processed_count = 0
        skipped_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(process_and_save, item) for item in audio_files]

            for future in tqdm(futures, desc="Extracting Speech Tokens"):
                status, result = future.result()
                if status == 'processed':
                    processed_count += 1
                elif status == 'skipped':
                    skipped_count += 1
                else:
                    failed_count += 1

        logging.info("=" * 60)
        logging.info("Processing complete!")
        logging.info(f"  - Newly processed: {processed_count}")
        logging.info(f"  - Skipped (already exists): {skipped_count}")
        logging.info(f"  - Failed: {failed_count}")
        logging.info(f"Save directory: {save_path}")
        logging.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Speech Token extraction tool')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--dataset', type=str, default='/mnt/data/Emilia-large',
                        help='Dataset path')
    parser.add_argument('--model', type=str,
                        default='pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx',
                        help='Path to the ONNX model')
    parser.add_argument('--save_dir', type=str, default='emilia_speech_tokens',
                        help='Save directory')
    parser.add_argument('--threads', type=int, default=16,
                        help='Number of threads')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of the GPU to use')
    parser.add_argument('--dataset_type', type=str, default='emilia',
                        choices=['librispeech', 'emilia'],
                        help='Dataset type: librispeech or emilia')
    parser.add_argument('--languages', type=str, nargs='+',
                        default=['DE', 'EN', 'FR', 'JA', 'KO', 'ZH'],
                        help='List of languages to process for the Emilia dataset')
    parser.add_argument('--min_duration', type=float, default=1.0,
                        help='Minimum audio duration (seconds)')
    parser.add_argument('--max_duration', type=float, default=30.0,
                        help='Maximum audio duration (seconds)')
    parser.add_argument('--min_dnsmos', type=float, default=3.0,
                        help='Minimum DNSMOS score')
    args = parser.parse_args()

    # Configure the logging format
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Check whether the configuration file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file does not exist: {args.config}")

    # Set up the GPU
    torch.cuda.set_device(args.gpu_id)

    extractor = SpeechTokenExtractor(args.config, args.model, args.threads, args.gpu_id)

    if args.dataset_type == 'librispeech':
        extractor.extract_from_librispeech(args.dataset, args.save_dir)
    elif args.dataset_type == 'emilia':
        extractor.extract_from_emilia(
            args.dataset,
            args.save_dir,
            languages=args.languages,
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            min_dnsmos=args.min_dnsmos
        )


if __name__ == '__main__':
    main()
