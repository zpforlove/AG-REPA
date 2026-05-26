import argparse
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
        # Set the specified GPU device
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

        # Get the audio parameters from the config
        self.sample_rate = self.config['audio']['sample_rate']
        self.max_length = self.config['audio']['max_length']
        self.num_threads = num_threads

        # Pre-create the resampler and move it to the GPU
        self.resampler = None  # Will be created when needed

    def process_audio(self, audio_path: str) -> tuple:
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

            # Extract features using whisper (runs on the GPU)
            # Note: whisper.log_mel_spectrogram internally uses the current device
            with torch.cuda.device(self.device):
                feat = whisper.log_mel_spectrogram(audio, n_mels=128)  # [n_mels, T]
                feat = feat.unsqueeze(0)  # [1, n_mels, T]

            # ONNX requires numpy input, so convert back to the CPU
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

    def extract_from_dataset(self, dataset_path: str, save_path: str):
        """Extract speech tokens from a dataset"""
        audio_files = list(Path(dataset_path).rglob('*.flac'))
        logging.info(f'Found {len(audio_files)} audio files')

        os.makedirs(save_path, exist_ok=True)

        def process_and_save(audio_file):
            audio_id, tokens = self.process_audio(str(audio_file))
            if audio_id:
                # Save a separate .pt file for each audio
                save_file = Path(save_path) / f"{audio_id}.pt"
                torch.save(tokens, save_file)
                return audio_id
            return None

        processed_count = 0
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = [executor.submit(process_and_save, f) for f in audio_files]

            for future in tqdm(futures, desc="Processing audio"):
                result = future.result()
                if result:
                    processed_count += 1

        logging.info(f'Successfully processed and saved {processed_count} audio files to: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Speech Token extraction tool')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')
    parser.add_argument('--dataset', type=str,
                        default='/mnt/data/Librispeech/dev/dev-all',
                        help='Dataset path')
    parser.add_argument('--model', type=str,
                        default='pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx',
                        help='ONNX model path')
    parser.add_argument('--save_dir', type=str, default='speech_tokens',
                        help='Save directory')
    parser.add_argument('--threads', type=int, default=16,
                        help='Number of threads')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='ID of the GPU to use')
    args = parser.parse_args()

    # Check whether the configuration file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file does not exist: {args.config}")

    # Set the GPU
    torch.cuda.set_device(args.gpu_id)

    extractor = SpeechTokenExtractor(args.config, args.model, args.threads, args.gpu_id)
    extractor.extract_from_dataset(args.dataset, args.save_dir)


if __name__ == '__main__':
    main()
