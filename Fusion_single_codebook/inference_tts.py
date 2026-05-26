import argparse
import logging
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import onnxruntime
import pysbd
import torch
import torch.nn.functional as F
import torchaudio
import whisper
import yaml
from faster_whisper import WhisperModel
from tqdm import tqdm
from vocos import Vocos

from cosyvoice.cli.cosyvoice import CosyVoice
from cosyvoice.utils.common import set_all_random_seed
from cosyvoice.utils.file_utils import load_wav
from models import FlowMatchingModel, MelSpectrogramExtractor
from utils import peak_norm

# --- Global configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [INFERENCE] %(message)s',
    stream=sys.stdout
)


class CFM_Inferencer:
    """
    A class that encapsulates all the logic required for Conditional Flow Matching model inference.
    """

    def __init__(self, config_path, device_str='cuda:0'):
        logging.info("CFM_Inferencer.__init__ - START")
        self.device = torch.device(device_str)
        logging.info(f"The inferencer will use device: {self.device}")
        self.config = self.load_config(config_path)

        # Initialize all required model components
        self.flow_model = None
        self.mel_extractor = None
        self.vocos = None
        self.s3_onnx_session = None
        self.cosyvoice_model = None
        self.whisper_model = None

        self.initialize_models()
        logging.info("CFM_Inferencer.__init__ - END")

    def load_config(self, config_path):
        logging.info("load_config - START")
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logging.info(f"Successfully loaded configuration file: {config_path}")
            logging.info("load_config - END")
            return config
        except Exception as e:
            logging.error(f"Failed to load configuration file: {e}")
            raise

    def initialize_models(self):
        """Initialize all models according to the configuration"""
        logging.info("initialize_models - START")
        try:
            logging.info("Initializing FlowMatchingModel...")
            teacher_dim_audio = self.config['hyperparameters']['ast']['input_dim']  # 768
            # Obtained from the Whisper configuration (fixed to the large-v3 d_model)
            teacher_dim_speech = 1280

            logging.info(
                f"Initializing FlowMatchingModel with teacher_dim_speech={teacher_dim_speech}, teacher_dim_audio={teacher_dim_audio}")

            self.flow_model = FlowMatchingModel(
                self.config,
                teacher_dim_speech=teacher_dim_speech,
                teacher_dim_audio=teacher_dim_audio
            ).to(self.device)

            logging.info("FlowMatchingModel initialized and moved to device.")

            logging.info("Initializing MelSpectrogramExtractor...")
            self.mel_extractor = MelSpectrogramExtractor(self.config, target_device=self.device)
            logging.info("MelSpectrogramExtractor initialized.")

            logging.info("Initializing Vocos...")
            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(self.device)
            logging.info("Vocos initialized and moved to device.")

            self.flow_model.eval()
            logging.info("Model initialization complete and set to evaluation mode.")
        except Exception as e:
            logging.error(f"An error occurred during model initialization: {e}\n{traceback.format_exc()}")
            raise
        logging.info("initialize_models - END")

    def initialize_asr_model(self):
        """Initialize the ASR model (Faster Whisper) if it is not already initialized"""
        if self.whisper_model is None:
            logging.info("initialize_asr_model - START")
            try:
                logging.info("Initializing Faster WhisperModel: large-v3, device=cpu, compute_type=int8")
                self.whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")
                logging.info("ASR model initialization on the CPU complete.")
            except Exception as e:
                logging.error(f"Failed to initialize the Faster Whisper model: {e}", exc_info=True)
                raise
            logging.info("initialize_asr_model - END")

    def transcribe_audio(self, audio_path: str):
        """Perform speech recognition with Faster Whisper and return the text and language code"""
        logging.info("transcribe_audio - START")
        self.initialize_asr_model()
        logging.info(f"Performing ASR recognition on '{Path(audio_path).name}'...")
        try:
            segments, info = self.whisper_model.transcribe(audio_path, beam_size=5, vad_filter=True)
            transcript_text = " ".join(seg.text for seg in segments).strip()
            logging.info(f"ASR recognition complete. Language: {info.language} (confidence: {info.language_probability:.2f})")
            logging.info("transcribe_audio - END")
            return transcript_text, info.language
        except Exception as e:
            logging.error(f"ASR recognition failed ({audio_path}): {e}", exc_info=True)
            return "", "en"

    def initialize_cosyvoice(self, model_dir):
        """Initialize the CosyVoice model separately"""
        if self.cosyvoice_model is None:
            logging.info(f"initialize_cosyvoice - START: Loading from '{model_dir}'")
            try:
                self.cosyvoice_model = CosyVoice(model_dir)
            except Exception as e:
                logging.error(f"Failed to load the CosyVoice model: {e}\n{traceback.format_exc()}")
                raise
            logging.info("initialize_cosyvoice - END")

    def load_checkpoint(self, checkpoint_dir):
        logging.info("load_checkpoint - START")
        checkpoint_path = self.find_best_checkpoint(checkpoint_dir)
        if not checkpoint_path:
            raise FileNotFoundError(f"No valid model checkpoint could be found in directory '{checkpoint_dir}'.")
        size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
        logging.info(f"Loading the model from the best checkpoint: {checkpoint_path.name} (size={size_mb:.1f} MB)")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if 'model_state_dict' not in checkpoint:
            raise KeyError("'model_state_dict' not found in the checkpoint file.")
        try:
            missing, unexpected = self.flow_model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            logging.info(
                f"Successfully loaded the model state dict (strict=True). Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
        except RuntimeError as e_strict:
            logging.warning(f"Loading with strict=True failed: {e_strict}. Trying strict=False ...")
            missing, unexpected = self.flow_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            logging.info(
                f"Successfully loaded the model state dict (strict=False). Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")

        logging.info("load_checkpoint - END")

    def find_best_checkpoint(self, checkpoint_dir):
        """Find the .pt or .pth file with the lowest val_loss in the specified directory"""
        chkpt_dir = Path(checkpoint_dir)
        if not chkpt_dir.is_dir(): return None
        best_loss = float('inf')
        best_chkpt_path = None
        loss_pattern = re.compile(r"val_loss_(\d+\.\d+)")

        # Support both .pt and .pth
        for f in list(chkpt_dir.glob("*.pt")) + list(chkpt_dir.glob("*.pth")):
            match = loss_pattern.search(f.name)
            if match:
                loss = float(match.group(1))
                if loss < best_loss:
                    best_loss, best_chkpt_path = loss, f
        if best_chkpt_path:
            logging.info(f"Found best model: {best_chkpt_path.name} (val_loss={best_loss:.4f})")
        return best_chkpt_path

    def load_and_process_audio(self, audio_path, target_sr):
        try:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
                waveform = resampler(waveform)
            return waveform.to(self.device)
        except Exception as e:
            logging.error(f"Failed to load or process audio '{audio_path}': {e}")
            raise

    def initialize_s3_extractor(self, onnx_model_path):
        if self.s3_onnx_session is None:
            logging.info("initialize_s3_extractor - START")
            try:
                options = onnxruntime.SessionOptions()
                providers = ["CPUExecutionProvider"]
                if self.device.type == 'cuda':
                    logging.info("CUDA device detected. Attempting to add CUDAExecutionProvider.")
                    providers.insert(0, ("CUDAExecutionProvider", {'device_id': self.device.index or 0}))

                logging.info(f"Initializing ONNX InferenceSession with providers: {providers}")
                self.s3_onnx_session = onnxruntime.InferenceSession(onnx_model_path, sess_options=options,
                                                                    providers=providers)
                logging.info(
                    f"ONNX Session initialized successfully with providers: {self.s3_onnx_session.get_providers()}")

            except Exception as e:
                logging.error(f"A catchable exception occurred while initializing the ONNX session: {e}", exc_info=True)
                raise RuntimeError(f"Failed to initialize the ONNX session: {e}") from e
            logging.info("initialize_s3_extractor - END")

    def get_s3_tokens_from_waveform(self, waveform, onnx_model_path):
        logging.info("get_s3_tokens_from_waveform - START")
        self.initialize_s3_extractor(onnx_model_path)
        logging.info("ONNX Session available. Processing waveform.")
        mel_feat = whisper.log_mel_spectrogram(waveform.squeeze(0).cpu(), n_mels=128).unsqueeze(0)
        inputs = {
            self.s3_onnx_session.get_inputs()[0].name: mel_feat.numpy(),
            self.s3_onnx_session.get_inputs()[1].name: np.array([mel_feat.shape[2]], dtype=np.int32)
        }
        tokens = self.s3_onnx_session.run(None, inputs)[0].flatten().tolist()
        logging.info("get_s3_tokens_from_waveform - END")
        return torch.tensor(tokens, dtype=torch.long, device=self.device)

    def get_s3_tokens_from_text_zeroshot(self, text, prompt_text, prompt_speech_16k, seed=666):
        set_all_random_seed(seed)
        normalized_text = self.cosyvoice_model.frontend.text_normalize(text, split=False)
        if not normalized_text:
            return torch.tensor([], dtype=torch.long, device=self.device)

        model_input = self.cosyvoice_model.frontend.frontend_zero_shot(
            normalized_text, prompt_text, prompt_speech_16k, self.cosyvoice_model.sample_rate, ''
        )
        with torch.no_grad():
            inference_generator = self.cosyvoice_model.model.llm.inference(
                text=model_input['text'].to(self.device),
                text_len=model_input['text_len'].to(self.device),
                prompt_text=model_input['prompt_text'].to(self.device),
                prompt_text_len=model_input['prompt_text_len'].to(self.device),
                prompt_speech_token=model_input['llm_prompt_speech_token'].to(self.device),
                prompt_speech_token_len=model_input['llm_prompt_speech_token_len'].to(self.device),
                embedding=model_input['llm_embedding'].to(self.device)
            )
            return torch.tensor([token for token in inference_generator], dtype=torch.long, device=self.device)

    def _split_text_semantically(
            self,
            sentence_chunks: list[str],
            target_words_per_chunk: int = 8,
            cjk_char_per_chunk: int = 10,
    ) -> tuple[list[str], list[bool]]:
        """
        Semantically aware fine-grained segmentation:
          - Within a sentence, divide into "soft boundaries" by weak-pause punctuation/conjunctions, then greedily pack into <=8 words (English) / <=10 characters (CJK);
          - Overly short chunks are automatically merged with adjacent chunks to avoid fragmentation;
          - Returns: (chunks, is_sentence_end_flags), where the "last chunk" of each sentence is True.
        """

        def is_cjk(s: str) -> bool:
            return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', s))

        def normalize_spaces(s: str) -> str:
            return " ".join(s.split())

        en_coord = r"(?:and|but|or|so|yet|nor|for)"
        en_subord = r"(?:which|that|who|whom|whose|where|when|because|if|although|though|while|since|unless|until|before|after)"
        # NOTE: the Chinese characters below are functional data, not a UI message:
        # this is the list of Chinese conjunctions used to split Chinese sentences at
        # clause boundaries during multilingual TTS. They must stay in Chinese to match Chinese input.
        zh_coord = r"(?:而且|但是|但|不过|或者|以及|因此|所以|然后|同时|然而|而|并且|而是|且)"
        weak_punct = r"[,\uFF0C\u3001;；:：—–\-]"

        def clause_split_by_punct(s: str) -> list[str]:
            parts = re.split(f"({weak_punct})", s)
            out = [];
            if not parts: return out
            cur = parts[0]
            for i in range(1, len(parts), 2):
                delim = parts[i]
                nxt = parts[i + 1] if i + 1 < len(parts) else ""
                out.append((cur + delim).strip())
                cur = nxt
            if cur.strip(): out.append(cur.strip())
            return out

        def further_split_on_conj(frags: list[str], cjk: bool) -> list[str]:
            out = []
            for frag in frags:
                if not frag.strip(): continue
                if cjk:
                    tmp = re.split(f"(?={zh_coord})", frag)
                    out.extend([t.strip() for t in tmp if t.strip()])
                else:
                    tmp = re.split(rf"(?=\b{en_coord}\b)|(?=\b{en_subord}\b)", frag, flags=re.I)
                    out.extend([normalize_spaces(t) for t in tmp if t.strip()])
            return out

        def pack_en(frag: str) -> list[str]:
            words = frag.split()
            res, i = [], 0
            while i < len(words):
                upper = min(i + target_words_per_chunk, len(words))
                cut = upper
                for k in range(upper, max(i + 2, i + 1), -1):
                    if re.search(rf"{weak_punct}$", words[k - 1]):
                        cut = k
                        break
                res.append(" ".join(words[i:cut]))
                i = cut
            return res

        def pack_cjk(frag: str) -> list[str]:
            s = re.sub(r"\s+", "", frag)
            res, i, n = [], 0, len(s)
            while i < n:
                upper = min(i + cjk_char_per_chunk, n)
                cut = upper;
                window = s[i:upper]
                m = re.search(rf"{weak_punct}", window)
                if m and (i + m.end()) >= (upper - 2):
                    cut = i + m.end()
                else:
                    m2 = re.search(zh_coord, window)
                    if m2 and (i + m2.start()) >= (upper - 3):
                        cut = i + m2.start()
                if cut <= i: cut = upper
                res.append(s[i:cut])
                i = cut
            return res

        def merge_tiny_one_sentence(chunks: list[str], cjk: bool) -> list[str]:
            def unit_len(x: str) -> int:
                return len(re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", x)) if not cjk else len(x)

            max_u = target_words_per_chunk if not cjk else cjk_char_per_chunk
            min_u = 2 if not cjk else 4
            merged, i = [], 0
            while i < len(chunks):
                cur = chunks[i]
                if i < len(chunks) - 1 and unit_len(cur) < min_u:
                    nxt = chunks[i + 1]
                    if unit_len(cur) + unit_len(nxt) <= max_u:
                        merged.append((cur + ("" if cjk else " ") + nxt).strip())
                        i += 2
                        continue
                merged.append(cur)
                i += 1
            return merged

        all_chunks, all_flags = [], []
        for sent in sentence_chunks:
            s = sent.strip()
            if not s: continue
            cjk = is_cjk(s)
            # Intra-sentence soft boundaries -> finer segmentation
            frags = further_split_on_conj(clause_split_by_punct(s), cjk=cjk)
            tmp = []
            for frag in frags:
                frag = frag.strip()
                if not frag: continue
                if cjk:
                    tmp.extend([c for c in pack_cjk(frag) if c])
                else:
                    wc = len(frag.split())
                    if wc <= target_words_per_chunk:
                        tmp.append(frag)
                    else:
                        tmp.extend(pack_en(frag))
            # Merge overly short chunks within the sentence
            tmp = merge_tiny_one_sentence(tmp, cjk=cjk)
            # Mark the end of the sentence (only the last chunk of this sentence is True)
            if tmp:
                flags = [False] * len(tmp)
                flags[-1] = True
                all_chunks.extend(tmp)
                all_flags.extend(flags)

        logging.info(f"Fine-grained segmentation complete. Number of segments: {len(all_chunks)}")

        # Space normalization for non-CJK
        def is_cjk_any(x: str) -> bool:
            return bool(re.search(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]', x))

        all_chunks = [c if is_cjk_any(c) else normalize_spaces(c) for c in all_chunks]
        return all_chunks, all_flags

    def _trim_silence(
            self,
            wav_1d: torch.Tensor,
            sr: int,
            thr_db: float = -50.0,
            frame_ms: int = 20,
            hop_ms: int = 10,
            pad_ms: int = 5,
    ) -> torch.Tensor:
        """
        Trim leading/trailing silence (mean-square energy threshold, -50 dB), keeping a very short guard margin (pad_ms).
        Uses nonzero() to locate the first and last valid frames, compatible with very short segments.
        Input/output: a 1D float32 CPU Tensor.
        """
        x = wav_1d.detach().cpu().float().contiguous()
        if x.ndim != 1: x = x.view(-1)
        if x.numel() == 0: return x

        frame = max(1, int(sr * frame_ms / 1000))
        hop = max(1, int(sr * hop_ms / 1000))
        pad = int(sr * pad_ms / 1000)

        total = x.numel()
        n_frames = 1 + max(0, (total - frame) // hop)
        if n_frames <= 1: return x

        frames = torch.stack([x[i * hop:i * hop + frame] for i in range(n_frames)], dim=0)
        energy = frames.pow(2).mean(dim=1).clamp_min_(1e-12)
        db = 10.0 * torch.log10(energy)
        mask = db > thr_db  # Bool
        idxs = torch.nonzero(mask, as_tuple=False).squeeze(-1)
        if idxs.numel() == 0: return x

        first_idx = int(idxs[0].item())
        last_idx = int(idxs[-1].item())
        start = max(0, first_idx * hop - pad)
        end = min(total, last_idx * hop + frame + pad)
        if end <= start: return x
        return x[start:end]

    def _smart_crossfade_concat(
            self,
            chunk_wavs_1d_cpu: list[torch.Tensor],
            chunk_texts: list[str],
            chunk_is_sentence_end: list[bool],
            sr: int,
            base_fade_ms: int = 60,
            sentence_pause_ms: int = 320,
            microfade_ms: int = 5,
    ) -> torch.Tensor:
        """
        Concatenation strategy:
          - Within a sentence: equal-power crossfade (length adjusted by the weak punctuation, ~48-85ms), trimming -50dB silence from each segment first;
          - Between sentences (the previous chunk is a sentence end): no crossfade, insert a fixed silence (320ms by default),
            but apply a very short "micro fade-in/fade-out" (5ms) on each side of the silence to prevent clicks.
        """
        if len(chunk_wavs_1d_cpu) == 0:
            return torch.zeros(0, dtype=torch.float32)
        if len(chunk_wavs_1d_cpu) == 1:
            return self._trim_silence(chunk_wavs_1d_cpu[0], sr)

        # Classify the "intra-sentence boundary strength" of the previous chunk
        def classify_end_char(text: str) -> str:
            t = (text or "").strip()
            if not t: return "none"
            ch = t[-1]
            if ch in [',', '，', '、']:            return "soft"
            if ch in [';', '；', ':', '：']:       return "medium"
            if ch in ['—', '-', '–']:             return "link"
            return "none"  # A regular small chunk without punctuation

        def get_fade_ms(kind: str) -> int:
            if kind == "soft":   return max(50, int(base_fade_ms * 1.15))  # ~70ms
            if kind == "medium": return max(60, int(base_fade_ms * 1.35))  # ~80ms
            if kind == "link":   return max(40, int(base_fade_ms * 0.90))  # ~55ms
            return max(35, int(base_fade_ms * 0.80))  # ~48ms

        out = None
        microfade = max(1, int(sr * microfade_ms / 1000))
        sentgap = max(0, int(sr * sentence_pause_ms / 1000))

        for i, wav in enumerate(chunk_wavs_1d_cpu):
            wav = self._trim_silence(wav, sr, thr_db=-50.0, frame_ms=20, hop_ms=10, pad_ms=5)
            if out is None:
                out = wav.clone()
                continue

            prev_is_sentence_end = bool(chunk_is_sentence_end[i - 1]) if i - 1 < len(chunk_is_sentence_end) else False

            if prev_is_sentence_end:
                # Between sentences: no crossfade, insert silence, and apply a very short equal-power micro fade-in/fade-out on each side
                if microfade >= 2 and out.numel() >= microfade and wav.numel() >= microfade:
                    a = out[-microfade:].clone()
                    b = wav[:microfade].clone()
                    t = torch.linspace(0.0, 1.0, steps=microfade, dtype=torch.float32)
                    fade_out = torch.cos(t * 0.5 * torch.pi)
                    fade_in = torch.sin(t * 0.5 * torch.pi)
                    a_fade = a * fade_out
                    b_fade = b * fade_in
                    gap = torch.zeros(sentgap, dtype=torch.float32)
                    out = torch.cat([out[:-microfade], a_fade, gap, b_fade, wav[microfade:]], dim=0)
                else:
                    gap = torch.zeros(sentgap, dtype=torch.float32)
                    out = torch.cat([out, gap, wav], dim=0)
                continue

            # Within a sentence: apply an equal-power crossfade whose length depends on the soft boundary
            kind = classify_end_char(chunk_texts[i - 1] if i - 1 < len(chunk_texts) else "")
            fade_ms = get_fade_ms(kind)
            fade_len = max(1, int(sr * fade_ms / 1000))
            # Safeguard for overly short segments (at most 1/3 on each side)
            fade_len = min(
                fade_len,
                out.numel() // 3 if out.numel() > 0 else 1,
                wav.numel() // 3 if wav.numel() > 0 else 1
            )
            if fade_len < 8 or out.numel() == 0 or wav.numel() == 0:
                out = torch.cat([out, wav], dim=0)
                continue

            a = out[-fade_len:].clone()
            b = wav[:fade_len].clone()
            t = torch.linspace(0.0, 1.0, steps=fade_len, dtype=torch.float32)
            fade_out = torch.cos(t * 0.5 * torch.pi)  # 1 -> 0
            fade_in = torch.sin(t * 0.5 * torch.pi)  # 0 -> 1
            mix = a * fade_out + b * fade_in
            out = torch.cat([out[:-fade_len], mix, wav[fade_len:]], dim=0)

        return out if out is not None else torch.zeros(0, dtype=torch.float32)

    def synthesize(self, text_to_speak, prompt_wav_path, s3_onnx_model_path, cosyvoice_model_dir, output_path,
                   speed=1.0):
        logging.info("synthesize - START")
        self.initialize_cosyvoice(cosyvoice_model_dir)

        # --- Step 1: Reference audio ---
        logging.info("--- synthesize: Step 1: Preprocessing the reference audio (Prompt) ---")
        prompt_text, lang_code = self.transcribe_audio(prompt_wav_path)
        if not prompt_text:
            raise ValueError(f"Could not recognize any text from '{prompt_wav_path}'; inference aborted.")

        prompt_speech_16k = load_wav(prompt_wav_path, 16000)
        prompt_waveform_s3 = self.load_and_process_audio(prompt_wav_path, self.config['audio']['sample_rate'])
        prompt_s3_tokens = self.get_s3_tokens_from_waveform(prompt_waveform_s3, s3_onnx_model_path)

        prompt_mel_norm = self.mel_extractor(prompt_waveform_s3, normalize=True)
        prompt_mel_len = prompt_mel_norm.shape[2]
        logging.info("--- synthesize: Step 1 complete ---")

        # --- Step 2: Sentence-level segmentation ---
        logging.info(f"--- Step 2: Using 'pysbd' to segment the long text (sentence level) ---")
        try:
            seg = pysbd.Segmenter(language=lang_code, clean=False)
            sentence_chunks = [chunk for chunk in seg.segment(text_to_speak) if chunk.strip()]
        except Exception as e:
            logging.warning(f"pysbd segmentation failed (language: {lang_code}): {e}. Will fall back to basic punctuation-based segmentation.")
            sentence_chunks = [s.strip() for s in re.split(r'[.。?!]', text_to_speak) if s.strip()]
        logging.info(f"The text was initially segmented into {len(sentence_chunks)} sentences.")

        # --- Step 2b: Intra-sentence fine-grained segmentation (returns flags marking sentence ends) ---
        text_chunks, is_sent_end_flags = self._split_text_semantically(
            sentence_chunks=sentence_chunks,
            target_words_per_chunk=8,
            cjk_char_per_chunk=10,
        )
        if not text_chunks:
            raise ValueError("After text segmentation there are no valid segments to synthesize.")

        # --- Step 3a: Precompute S3 Tokens & target lengths (aligned with flags) ---
        logging.info("--- Step 3a: Batch-precomputing S3 Tokens and target lengths ---")
        source_s3_tokens_list, target_mel_len_list = [], []
        valid_chunks, valid_flags = [], []
        original_rate = prompt_mel_len / len(prompt_s3_tokens) if len(prompt_s3_tokens) > 0 else 10.0
        rate = original_rate / speed
        logging.info(f"Original rate (frames/Token): {original_rate:.2f}, adjusted rate: {rate:.2f} (factor: {speed})")

        for chunk, flag in tqdm(list(zip(text_chunks, is_sent_end_flags)), desc="Precomputing S3 Tokens"):
            source_s3_tokens = self.get_s3_tokens_from_text_zeroshot(chunk, prompt_text, prompt_speech_16k)
            if source_s3_tokens.nelement() > 0:
                source_s3_tokens_list.append(source_s3_tokens)
                source_mel_len = int(len(source_s3_tokens) * rate)
                target_mel_len_list.append(prompt_mel_len + source_mel_len)
                valid_chunks.append(chunk)
                valid_flags.append(bool(flag))
            else:
                logging.warning(f"Segment '{chunk}' failed to generate S3 tokens and was skipped.")

        if not valid_chunks:
            raise RuntimeError("All text segments failed to generate S3 tokens; cannot continue synthesis.")

        batch_size = len(valid_chunks)
        logging.info(f"A total of {batch_size} segments will be synthesized in parallel.")

        # --- Step 3b: Build the batched tensors ---
        max_target_mel_len = max(target_mel_len_list)
        max_s3_len = len(prompt_s3_tokens) + max(len(s) for s in source_s3_tokens_list)

        batch_full_s3_tokens = torch.zeros((batch_size, max_s3_len), dtype=torch.long, device=self.device)
        batch_ref_mel_for_cond = torch.zeros(
            (batch_size, self.config['vocos']['mel']['n_mels'], max_target_mel_len),
            device=self.device
        )

        actual_full_s3_lengths = []
        for i in range(batch_size):
            full_s3 = torch.cat([prompt_s3_tokens, source_s3_tokens_list[i]], dim=0)
            batch_full_s3_tokens[i, :len(full_s3)] = full_s3
            batch_ref_mel_for_cond[i, :, :prompt_mel_len] = prompt_mel_norm
            actual_full_s3_lengths.append(len(full_s3))

        s3_embed_dim = self.config['hyperparameters']['flow']['token_embedding_dim']
        with torch.no_grad():
            batch_s3_embed_unpermuted_raw = self.flow_model.token_embedding_s3(batch_full_s3_tokens)
            batch_s3_embed_unpermuted = self.flow_model.P_s3(batch_s3_embed_unpermuted_raw)  # [B, T_s3_max, D_tok]

        # Align the lengths
        batch_s3_embed_resampled = torch.zeros((batch_size, s3_embed_dim, max_target_mel_len), device=self.device)
        for i in range(batch_size):
            target_len = target_mel_len_list[i]
            full_s3_len = actual_full_s3_lengths[i]
            if full_s3_len > 0 and target_len > 0:
                current = batch_s3_embed_unpermuted[i, :full_s3_len, :].permute(1, 0).unsqueeze(0)
                interpolated = F.interpolate(current, size=target_len, mode='linear', align_corners=False)
                batch_s3_embed_resampled[i, :, :target_len] = interpolated.squeeze(0)

        batch_s3_embed_resampled = F.normalize(batch_s3_embed_resampled, p=2, dim=1)
        fused_embed = batch_s3_embed_resampled
        cond_embed_dict = {'fused_embed': fused_embed, 'ref_mel_for_cond': batch_ref_mel_for_cond}

        # --- Step 3c: Generate the mel ---
        with torch.no_grad():
            batch_generated_full_mel_norm = self.flow_model.sample(
                cond_embed_dict=cond_embed_dict,
                target_duration_frames=max_target_mel_len,
                steps=self.config['hyperparameters']['flow']['n_timesteps'],
                cfg_scale=self.config['hyperparameters']['flow']['cfg_scale'],
                sway_sampling_coef=self.config['hyperparameters']['flow'].get('sway_coef', -1.0)
            )

        # --- Step 4: Decode + concatenate (intra-sentence crossfade; inter-sentence silence) ---
        logging.info("--- Step 4: Decode in parallel and perform smart concatenation ---")
        sr = self.config['vocos']['sample_rate']
        all_audio_1d_cpu = []

        max_gen_mel_len = max(t - prompt_mel_len for t in target_mel_len_list) if any(
            t > prompt_mel_len for t in target_mel_len_list) else 0

        if max_gen_mel_len > 0:
            batch_gen_mel_denorm = torch.zeros(
                (batch_size, self.config['vocos']['mel']['n_mels'], max_gen_mel_len),
                device=self.device
            )
            mel_std = self.config['hyperparameters']['flow']['mel_std']
            mel_mean = self.config['hyperparameters']['flow']['mel_mean']

            actual_mel_lengths = []
            for i in range(batch_size):
                target_len = target_mel_len_list[i]
                gen_len = target_len - prompt_mel_len
                actual_mel_lengths.append(gen_len)
                if gen_len > 0:
                    gen_mel_part_norm = batch_generated_full_mel_norm[i, :, prompt_mel_len:target_len]
                    gen_mel_part_denorm = gen_mel_part_norm * mel_std + mel_mean
                    batch_gen_mel_denorm[i, :, :gen_len] = gen_mel_part_denorm

            with torch.no_grad():
                batch_audio_chunks_dev = self.vocos.decode(batch_gen_mel_denorm)

            hop_length = self.mel_extractor.hop_length
            for i in range(batch_size):
                actual_audio_len = actual_mel_lengths[i] * hop_length
                audio_chunk_1d = batch_audio_chunks_dev[i, :actual_audio_len].detach().cpu().float()
                all_audio_1d_cpu.append(audio_chunk_1d)

        if len(all_audio_1d_cpu) == 0:
            full_audio_cpu_1d = torch.zeros(0, dtype=torch.float32)
        else:
            full_audio_cpu_1d = self._smart_crossfade_concat(
                chunk_wavs_1d_cpu=all_audio_1d_cpu,
                chunk_texts=valid_chunks,
                chunk_is_sentence_end=valid_flags,
                sr=sr,
                base_fade_ms=60,
                sentence_pause_ms=320,
                microfade_ms=5
            )

        audio_out_cpu = peak_norm(full_audio_cpu_1d.unsqueeze(0))
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(output_path, audio_out_cpu, sr)
        logging.info(f"Synthesis successful! The complete audio has been saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="CFM TTS inference script (supports multiple datasets, voice-embedding-free model) ")

    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the main configuration file (should be the yaml corresponding to config.py)')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/flow', help='Directory containing the CFM model checkpoints')
    parser.add_argument('--text', type=str,
                        default="I've seen things you people wouldn't believe. Attack ships on fire off the shoulder of Orion. I watched C Beams glitter in the dark near the Tannhauser Gate. All those moments will be lost in time, like tears in rain. Time to die. This is a classic line from Blade Runner.",
                        help='The text content to synthesize (can be multilingual long text)')
    parser.add_argument('--prompt_wav', type=str, default='./wav/english_male.flac',
                        help='WAV file providing the reference mel-spectrogram (will be automatically recognized by ASR)')
    parser.add_argument('--cosyvoice_model_dir', type=str, default='./pretrained_models/CosyVoice-300M',
                        help='Path to the CosyVoice base model')
    parser.add_argument('--s3_model', type=str, default='./pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx',
                        help='Path to the ONNX model of the S3 Token extractor')
    parser.add_argument('--output', type=str, default='./output/generated_tts.wav',
                        help='Save path for the output audio file')
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of the GPU to use')
    parser.add_argument('--speed', type=float, default=0.9, help='Speech-rate adjustment factor. >1.0 speeds up speech, <1.0 slows it down.')
    args = parser.parse_args()

    try:
        logging.info("main - START: Initializing Inferencer.")
        inferencer = CFM_Inferencer(config_path=args.config, device_str=f'cuda:{args.gpu_id}')

        logging.info("main - Loading checkpoint.")
        inferencer.load_checkpoint(args.checkpoint_dir)

        logging.info("main - Starting synthesis.")
        inferencer.synthesize(
            text_to_speak=args.text,
            prompt_wav_path=args.prompt_wav,
            s3_onnx_model_path=args.s3_model,
            cosyvoice_model_dir=args.cosyvoice_model_dir,
            output_path=args.output,
            speed=args.speed
        )
    except Exception as e:
        logging.error(f"A catchable error occurred during inference: {e}\n{traceback.format_exc()}")
    finally:
        logging.info("main - END: Script finished.")


if __name__ == '__main__':
    main()
