import argparse
import logging
import math
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LogitsProcessor,
    LogitsProcessorList,
)
from vocos import Vocos

# Import project-internal modules
from beats.BEATs import BEATs, BEATsConfig
from models import FlowMatchingModel, MelSpectrogramExtractor, AudioSetTokenizer
from utils import peak_norm

# --- Global logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [TTA_INFERENCE] %(message)s',
    stream=sys.stdout
)


def find_best_checkpoint(root_dir: str, sub_dir: str, prefix: str) -> str:
    """Find the checkpoint with the smallest loss"""
    dir_path = Path(root_dir) / sub_dir
    if not dir_path.exists():
        logging.warning(f"Directory does not exist: {dir_path}")
        return None

    best_loss = math.inf
    best_path = None

    # Match pattern: prefix + ... .pth
    for pth in dir_path.glob(f"{prefix}*.pth"):
        try:
            stem = pth.stem
            if "_loss_" in stem:
                parts = stem.split("_loss_")
                if len(parts) < 2:
                    continue
                loss_part = parts[1].split("_")[0]
                loss_val = float(loss_part)
                if loss_val < best_loss:
                    best_loss = loss_val
                    best_path = pth
        except Exception:
            continue

    if best_path:
        logging.info(f"Found best {sub_dir} model: {best_path.name} (Loss: {best_loss:.4f})")
        return str(best_path)

    logging.warning(f"No model file matching '{prefix}*loss*.pth' found in {dir_path}")
    return None


class AllowedVocabProcessor(LogitsProcessor):
    """Restrict generation to only the [AS_*] and [END] set"""

    def __init__(self, allowed_ids):
        self.allowed_ids = allowed_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_ids] = 0
        return scores + mask


class StyleAwareLLM:
    """
    Wrapper class: manages the LLM, the Tokenizer, and the additional Style Projection layer.
    It does not directly subclass nn.Module, but holds a model instance, making it convenient to call generate.
    """

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.llm_cfg = config['hyperparameters']['llm']

        # Read parameters
        self.model_name = self.llm_cfg['model_name']
        self.style_num_tokens = int(self.llm_cfg.get('style_num_tokens', 16))
        self.style_strength = float(self.llm_cfg.get('style_strength', 0.5))
        self.as_vocab_size = int(self.llm_cfg['as_vocab_size'])
        self.coarse_num_clusters = int(self.llm_cfg.get('coarse_num_clusters', 128))

        self._load_tokenizer()
        self._load_model()

    def _load_tokenizer(self):
        logging.info(f"Loading Tokenizer: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)

        # Extend the vocabulary (must match train_llm.py exactly)
        special_tokens = ['[TAG]', '[DES]', '[AT]', '[END]', '[PAD]']
        as_tokens = [f'[AS_{j}]' for j in range(self.as_vocab_size)]
        audio_style_tokens = [f'[AUDIO_{j}]' for j in range(self.style_num_tokens)]

        tokens_to_add = special_tokens + as_tokens + audio_style_tokens
        self.tokenizer.add_tokens(tokens_to_add, special_tokens=True)

        # Fix PAD/EOS
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        if getattr(self.tokenizer, "eos_token_id", None) is None:
            self.tokenizer.eos_token = '[END]'

        # Cache the Token IDs
        self.audio_token_ids = [
            self.tokenizer.convert_tokens_to_ids(f'[AUDIO_{i}]')
            for i in range(self.style_num_tokens)
        ]
        self.pad_token_id = self.tokenizer.pad_token_id
        self.eos_token_id = self.tokenizer.convert_tokens_to_ids('[END]')

        logging.info(f"Vocabulary extension complete, current size: {len(self.tokenizer)}")

    def _load_model(self):
        logging.info(f"Loading model: {self.model_name}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.resize_token_embeddings(len(self.tokenizer))

        # Configure EOS/PAD
        self.model.config.pad_token_id = self.pad_token_id
        self.model.config.eos_token_id = self.eos_token_id

        hidden_size = self.model.config.hidden_size
        beats_dim = 768

        # Get the dtype of the main model (usually torch.float16)
        target_dtype = self.model.dtype

        # 1. Style Projection Layer (cast to target_dtype immediately after creation)
        self.model.style_proj = nn.Linear(beats_dim, hidden_size).to(self.device).to(target_dtype)

        # 2. Coarse Head (cast to target_dtype immediately after creation)
        self.model.coarse_head = nn.Linear(hidden_size, self.coarse_num_clusters).to(self.device).to(target_dtype)

        self.model.to(self.device)
        self.model.eval()

    def load_weights(self, checkpoint_path):
        logging.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Handle the module. prefix saved by DeepSpeed
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
        logging.info(f"Weight loading result - Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        # Ensure style_proj is loaded
        if any("style_proj" in k for k in missing):
            logging.warning("style_proj weights not found! Style injection may have no effect.")

    def inject_style_and_generate(self, prompt_text, style_feats, max_new_tokens=1024, **gen_kwargs):
        """
        Perform generation with style injection.
        Principle: obtain the embeddings of the prompt, replace the values at the [AUDIO_i] positions, then pass them to generate.
        """
        # 1. Tokenize Prompt
        inputs = self.tokenizer(prompt_text, return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids']  # [1, T]

        # 2. Obtain the original embeddings
        with torch.no_grad():
            inputs_embeds = self.model.get_input_embeddings()(input_ids)  # [1, T, H]

        # 3. Style injection (reusing the training logic)
        # style_feats: [1, K, D_beats]
        if style_feats is not None:
            style_feats = style_feats.to(self.device).to(inputs_embeds.dtype)
            style_proj = self.model.style_proj
            style_H = style_proj(style_feats)  # [1, K, H]

            K = self.style_num_tokens
            alpha = self.style_strength

            # Find the [AUDIO_i] positions and replace them
            for i in range(K):
                tok_id = self.audio_token_ids[i]
                # Find the positions in input_ids equal to tok_id
                pos_mask = (input_ids == tok_id)  # [1, T]
                if pos_mask.any():
                    # Blend: (1-alpha)*Orig + alpha*Style
                    # Take out the style of the corresponding layer
                    style_vec = style_H[:, i, :].unsqueeze(1)  # [1, 1, H]

                    # Perform broadcast replacement
                    original_emb = inputs_embeds[pos_mask]  # [N, H]
                    # Expand style_vec to match N
                    style_expand = style_vec.expand(original_emb.shape[0], 1, -1).squeeze(1)

                    mixed_emb = (1.0 - alpha) * original_emb + alpha * style_expand
                    inputs_embeds[pos_mask] = mixed_emb

        # 4. Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.model.config.eos_token_id,
                max_new_tokens=max_new_tokens,
                **gen_kwargs
            )

        return outputs


def load_beats_and_ast(config, device, checkpoint_dir):
    """Load the BEATs and AST models for extracting features and GT"""
    logging.info("Loading BEATs and AST...")

    # BEATs
    beats_ckpt = config['paths']['beats_feature_extractor_checkpoint']
    ckpt = torch.load(beats_ckpt, map_location='cpu')
    beats = BEATs(BEATsConfig(ckpt['cfg']))
    beats.load_state_dict(ckpt['model'])
    beats.eval().to(device)

    # AST
    ast_ckpt = find_best_checkpoint(checkpoint_dir, "ast", "best_epoch")
    if not ast_ckpt:
        raise FileNotFoundError("AST checkpoint not found")

    ast_cfg = config['hyperparameters']['ast']
    ast = AudioSetTokenizer(ast_cfg['input_dim'], ast_cfg['hidden_dim'], ast_cfg['vocab_size'])
    ast.load_state_dict(torch.load(ast_ckpt, map_location='cpu')['model_state_dict'])
    ast.eval().to(device)

    return beats, ast


def process_audio(audio_path, config, beats_model, ast_model, style_num_tokens, device):
    """
    Process the audio:
    1. Extract BEATs features -> split -> average -> Style Feats
    2. Extract BEATs features -> AST -> GT Tokens
    """
    # 1. Load the audio
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample and pad/crop (consistent with data_loader)
    target_sr = config['audio']['sample_rate']
    max_len = config['audio']['max_length']

    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)

    cur_len = waveform.shape[1]
    if cur_len > max_len:
        waveform = waveform[:, :max_len]
    else:
        waveform = F.pad(waveform, (0, max_len - cur_len))

    waveform = waveform.to(device)

    with torch.no_grad():
        # 2. Extract BEATs
        beats_feats, _ = beats_model.extract_features(waveform)  # [1, T, D]

        # 3. Build the Style Feats (mirroring the logic in train_llm._extract_as_and_style)
        T, D = beats_feats.shape[1], beats_feats.shape[2]
        K = style_num_tokens
        idx_edges = torch.linspace(0, T, steps=K + 1, device=device).long()
        style_chunks = []
        for i in range(K):
            a, b = idx_edges[i].item(), idx_edges[i + 1].item()
            if b <= a:
                b = min(a + 1, T)
            seg = beats_feats[0, a:b, :]
            style_chunks.append(seg.mean(dim=0, keepdim=True))

        style_feats = torch.cat(style_chunks, dim=0).unsqueeze(0)  # [1, K, D]

        # 4. Build the GT Tokens
        input_feats_tok = beats_feats.permute(0, 2, 1)  # [1, D, T]
        gt_indices = ast_model.tokenize(input_feats_tok).view(-1)
        gt_tokens = [f"[AS_{i.item()}]" for i in gt_indices]

    return style_feats, gt_tokens


# =============================================================================
# ========================== TTA Inferencer main class ========================
# =============================================================================

class TTA_Inferencer:
    """
    Text-to-Audio inferencer: combines the LLM (token generation) and Flow Matching (audio generation).
    """

    def __init__(self, config_path, device_str='cuda:0'):
        logging.info("TTA_Inferencer.__init__ - START")
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")
        logging.info(f"Using device: {self.device}")

        self.config = self.load_config(config_path)

        # Initialize the model containers
        self.llm_wrapper = None
        self.beats_model = None
        self.ast_model = None
        self.flow_model = None
        self.mel_extractor = None
        self.vocos = None

        self.initialize_models()
        logging.info("TTA_Inferencer.__init__ - END")

    def load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config
        except Exception as e:
            logging.error(f"Failed to load configuration file: {e}")
            raise

    def initialize_models(self):
        """Initialize the LLM, BEATs, AST, CFM, and Vocos"""
        logging.info("Initializing model components...")

        # 1. Load the LLM-related components (for Stage 1: Text -> Tokens)
        logging.info("Loading StyleAwareLLM...")
        self.llm_wrapper = StyleAwareLLM(self.config, self.device)

        logging.info("Loading BEATs and AST (for feature extraction)...")
        self.beats_model, self.ast_model = load_beats_and_ast(
            self.config, self.device, self.config['paths']['checkpoint_dir']
        )

        # 2. Load the CFM-related components (for Stage 2: Tokens -> Mel)
        logging.info("Loading FlowMatchingModel...")
        teacher_dim_audio = self.config['hyperparameters']['ast']['input_dim']  # 768
        teacher_dim_speech = 1280  # Whisper large-v3 dim, fixed default

        self.flow_model = FlowMatchingModel(
            self.config,
            teacher_dim_speech=teacher_dim_speech,
            teacher_dim_audio=teacher_dim_audio
        ).to(self.device)

        logging.info("Loading MelSpectrogramExtractor...")
        self.mel_extractor = MelSpectrogramExtractor(self.config, target_device=self.device)

        logging.info("Loading Vocos...")
        self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)

        self.flow_model.eval()

    def load_checkpoints(self, llm_ckpt_dir=None, flow_ckpt_dir=None):
        """Load the weights of the LLM and CFM"""
        # 1. LLM Checkpoint
        if llm_ckpt_dir:
            best_llm = find_best_checkpoint(llm_ckpt_dir, "llm", "best_model")
            if best_llm:
                logging.info(f"Loading LLM Checkpoint: {best_llm}")
                self.llm_wrapper.load_weights(best_llm)
            else:
                logging.warning("No LLM Checkpoint found; random weights will be used (for debugging only)!")

        # 2. CFM Checkpoint
        if flow_ckpt_dir:
            chkpt_dir = Path(flow_ckpt_dir)
            best_loss = float('inf')
            best_chkpt_path = None
            loss_pattern = re.compile(r"val_loss_(\d+\.\d+)")

            # Find the best Flow model
            for f in list(chkpt_dir.glob("*.pt")) + list(chkpt_dir.glob("*.pth")):
                match = loss_pattern.search(f.name)
                if match:
                    loss = float(match.group(1))
                    if loss < best_loss:
                        best_loss, best_chkpt_path = loss, f

            if best_chkpt_path:
                logging.info(f"Loading Flow Checkpoint: {best_chkpt_path} (loss={best_loss:.4f})")
                checkpoint = torch.load(best_chkpt_path, map_location="cpu")

                # Handle a possible module prefix or key mismatch
                state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
                try:
                    self.flow_model.load_state_dict(state_dict, strict=True)
                except RuntimeError:
                    logging.warning("Strict loading failed, trying strict=False...")
                    self.flow_model.load_state_dict(state_dict, strict=False)
            else:
                logging.warning(f"No valid Flow model weights found in {flow_ckpt_dir}!")

    def generate_tokens_with_llm(self, prompt_text, prompt_audio_path, max_new_tokens=512):
        """Stage 1: Use the LLM to generate the AudioSet token sequence"""
        logging.info("--- Stage 1: LLM token generation ---")

        # 1. Process the reference audio and extract the style features (Style Feats)
        style_feats, _ = process_audio(
            prompt_audio_path, self.config, self.beats_model, self.ast_model,
            self.llm_wrapper.style_num_tokens, self.device
        )

        # 2. Build the prompt
        # Format: [TAG] {event} [DES] {desc} [AUDIO_0]...[AUDIO_K-1] [AT]
        audio_placeholders = "".join([f"[AUDIO_{i}]" for i in range(self.llm_wrapper.style_num_tokens)])
        full_prompt = f"{prompt_text} {audio_placeholders} [AT] "
        logging.info(f"Prompt: {full_prompt}")

        # 3. Restrict the generation vocabulary (only generate [AS_*] and [END])
        as_ids = [self.llm_wrapper.tokenizer.convert_tokens_to_ids(f'[AS_{i}]') for i in
                  range(self.llm_wrapper.as_vocab_size)]
        end_id = self.llm_wrapper.tokenizer.convert_tokens_to_ids('[END]')
        valid_ids = as_ids + [end_id]
        allowed_proc = LogitsProcessorList([AllowedVocabProcessor(torch.tensor(valid_ids, device=self.device))])

        # 4. Generate
        output_ids = self.llm_wrapper.inject_style_and_generate(
            full_prompt,
            style_feats,
            max_new_tokens=max_new_tokens,
            do_sample=True,  # Enable sampling here for diversity
            top_p=0.9,
            temperature=0.8,
            logits_processor=allowed_proc
        )

        # 5. Parse the Output IDs into Token Indices (int list)
        raw_tokens = self.llm_wrapper.tokenizer.convert_ids_to_tokens(output_ids[0], skip_special_tokens=False)

        pred_as_indices = []
        # Look for the content after [AT]
        try:
            at_index = raw_tokens.index('[AT]')
            scan_tokens = raw_tokens[at_index + 1:]
        except ValueError:
            scan_tokens = raw_tokens

        for t in scan_tokens:
            if t == '[END]':
                break
            if t.startswith('[AS_'):
                try:
                    # Parse [AS_123] -> 123
                    idx = int(t[4:-1])
                    pred_as_indices.append(idx)
                except ValueError:
                    continue

        logging.info(f"Number of generated AudioSet tokens: {len(pred_as_indices)}")
        return pred_as_indices

    def synthesize_audio_with_cfm(self, as_token_indices, prompt_audio_path, speed=1.0):
        """Stage 2: Use Flow Matching to generate audio"""
        logging.info("--- Stage 2: CFM audio synthesis ---")

        if not as_token_indices:
            raise ValueError("No valid token sequence was generated; cannot synthesize audio.")

        # 1. Prepare the reference audio (for ref_mel and teacher_audio_global)
        # Load the waveform
        wav, sr = torchaudio.load(prompt_audio_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        target_sr = self.config['audio']['sample_rate']
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        wav = wav.to(self.device)

        # Compute the Mel (for ref_mel_for_cond)
        full_mel = self.mel_extractor(wav, normalize=True)  # [1, n_mels, T]

        # Take a portion as the Reference Mel (e.g., the first 3 seconds or 30%)
        T_mel = full_mel.shape[-1]
        ref_len = int(T_mel * 0.3)
        # Number of frames corresponding to 3 seconds (Hop=256, SR=24000 -> ~93.75 fps)
        fps = self.config['vocos']['sample_rate'] / self.config['vocos']['mel']['hop_length']
        max_ref_frames = int(3.0 * fps)
        ref_len = min(ref_len, max_ref_frames)
        ref_len = max(ref_len, 1)  # At least 1 frame

        ref_mel_cond = full_mel[:, :, :ref_len]

        # 2. Compute the Teacher Audio Global (BEATs average feature)
        with torch.no_grad():
            beats_features, _ = self.beats_model.extract_features(wav)  # [1, T_beats, 768]
            teacher_audio_global = beats_features.mean(dim=1)  # [1, 768]

        # 3. Process the generated AudioSet Tokens
        # Convert the indices into a tensor
        token_tensor = torch.tensor(as_token_indices, dtype=torch.long, device=self.device).unsqueeze(0)  # [1, L]

        # Embed and Project (using the shared embedding layer of the Flow model)
        with torch.no_grad():
            fused_embed_raw = self.flow_model.embed_and_project_tokens('as', token_tensor)  # [1, D_tok, L]

        # 4. Determine the target audio length and interpolate
        # AudioSet tokens typically represent coarse-grained semantics. BEATs frame rate ~ 50Hz.
        # Vocos Mel frame rate ~ 93.75 Hz.
        # Simple duration estimate: Target Frames = Token Count * (Mel_Rate / BEATs_Rate) / Speed
        scale_factor = 1.875 / speed
        target_gen_len = int(len(as_token_indices) * scale_factor)
        target_gen_len = max(target_gen_len, 16)  # Minimum length safeguard

        # Total target length = Ref Len + Gen Len
        target_total_len = ref_len + target_gen_len

        # Interpolate fused_embed to match the target length
        fused_embed = F.interpolate(fused_embed_raw, size=target_total_len, mode='linear', align_corners=False)
        fused_embed = F.normalize(fused_embed, p=2, dim=1)

        # 5. Build the cond_embed_dict
        # Note: ref_mel_for_cond needs to be padded to the same length as fused_embed
        ref_mel_padded = F.pad(ref_mel_cond, (0, target_total_len - ref_len))

        cond_embed_dict = {
            'fused_embed': fused_embed,
            'ref_mel_for_cond': ref_mel_padded,
            'domain_ids': torch.tensor([1], device=self.device),  # 1 for AudioSet domain
            'teacher_audio_global': teacher_audio_global,
            'teacher_speech_global': None  # The TTA task usually does not need the Speech Teacher (Whisper)
        }

        # 6. Flow Matching sampling
        logging.info(f"Starting to generate the Mel-spectrogram, target frame count: {target_total_len}...")
        with torch.no_grad():
            gen_mel_norm = self.flow_model.sample(
                cond_embed_dict=cond_embed_dict,
                target_duration_frames=target_total_len,
                steps=self.config['hyperparameters']['flow']['n_timesteps'],
                cfg_scale=self.config['hyperparameters']['flow']['cfg_scale'],
                sway_sampling_coef=self.config['hyperparameters']['flow'].get('sway_coef', -1.0)
            )

        # 7. Denormalize the Mel
        mel_mean = self.config['hyperparameters']['flow']['mel_mean']
        mel_std = self.config['hyperparameters']['flow']['mel_std']

        # Take out the generated portion (removing the Ref prefix)
        gen_part_norm = gen_mel_norm[:, :, ref_len:]
        if gen_part_norm.shape[-1] == 0:
            logging.warning("Generated length is 0; returning the ref portion")
            gen_part_norm = gen_mel_norm

        gen_mel_denorm = gen_part_norm * mel_std + mel_mean

        return gen_mel_denorm

    def run_inference(self, event, description, prompt_audio, output_path, speed=1.0):
        """Run the complete TTA inference pipeline"""
        # 1. Build the complete prompt text
        # [TAG] {event} [DES] {description}
        prompt_text = f"[TAG] {event} [DES] {description}"

        # 2. LLM generation
        as_tokens = self.generate_tokens_with_llm(prompt_text, prompt_audio)

        # 3. CFM synthesis
        mel_denorm = self.synthesize_audio_with_cfm(as_tokens, prompt_audio, speed=speed)

        # 4. Vocos decoding
        logging.info("Vocos decoding...")
        with torch.no_grad():
            audio_wav = self.vocos.decode(mel_denorm)

        # 5. Save
        audio_cpu = peak_norm(audio_wav.cpu())
        sr = self.config['vocos']['sample_rate']

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, audio_cpu, sr)
        logging.info(f"Generation complete! Audio saved to: {output_path}")


# =============================================================================
# ================================= Main ======================================
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Inference Script for Text-to-Audio (TTA) using LLM + Flow Matching")

    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the configuration file')
    parser.add_argument('--llm_ckpt_dir', type=str, default='./checkpoints', help='Root directory containing the LLM checkpoints')
    parser.add_argument('--flow_ckpt_dir', type=str, default='./checkpoints/flow',
                        help='Directory containing the Flow Matching checkpoints')

    parser.add_argument('--event', type=str, default="Dog barking", help="Sound event category")
    parser.add_argument('--desc', type=str,
                        default="A large dog barking loudly and aggressively in an outdoor environment.",
                        help="Detailed sound description")
    parser.add_argument('--prompt_wav', type=str, default='./wav/dog.wav', help='Reference audio path (used to extract the style)')
    parser.add_argument('--output', type=str, default='./output/generated_tta.wav', help='Output audio path')
    parser.add_argument('--speed', type=float, default=1.0, help='Duration control factor (larger values make the audio longer/slower, smaller values make it shorter/faster)')
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    try:
        inferencer = TTA_Inferencer(args.config, device_str=f'cuda:{args.gpu_id}')

        # Load the weights
        inferencer.load_checkpoints(llm_ckpt_dir=args.llm_ckpt_dir, flow_ckpt_dir=args.flow_ckpt_dir)

        # Run inference
        inferencer.run_inference(
            event=args.event,
            description=args.desc,
            prompt_audio=args.prompt_wav,
            output_path=args.output,
            speed=args.speed
        )

    except Exception as e:
        logging.error(f"An error occurred during inference: {e}", exc_info=True)


if __name__ == "__main__":
    main()
