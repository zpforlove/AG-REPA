# AG-REPA: Attribution-Guided Representation Alignment for Audio Flow Matching

> Official code for the paper
> **"AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching"**
> *Pengfei Zhang, Tianxin Xie, Minghao Yang, Li Liu.* — ICML 2026.

> 📄 **Paper & poster:** [ICML 2026 Virtual](https://icml.cc/virtual/2026/poster/65899)
> &nbsp;|&nbsp; 🌐 **Language:** English | [简体中文](README.zh-CN.md)

A unified audio-generation framework that does both **Text-to-Speech (TTS)** and
**Text-to-Audio (TTA)** with a single Flow-Matching backbone. This repo holds the
**training, diagnostic, and inference code**, plus the interpretability toolkit
(**BiT-C / LASP / FoG-A**) and the **AG-REPA** training strategy from the paper.

**The idea in one line:** standard REPA aligns the layers that *store* the most
information; AG-REPA aligns the layers that *actually drive* the output — found
automatically by a causal probe — cutting Fréchet Audio Distance by
**18 % (speech) / 16 % (audio)** over the best fixed-layer REPA baseline.

> ℹ️ **This repo is code only** — no checkpoints or datasets. Pre-trained weights and
> diagnostic artifacts are on Hugging Face: **[🤗 AustinZhang/AG-REPA](https://huggingface.co/AustinZhang/AG-REPA)**.
> The frozen base models (BEATs, CosyVoice) are downloaded from their original sources — see the model card.

---

## 🔊 Audio samples — AG-REPA vs. baseline (single codebook, 1 epoch)

To make the effect *audible*, the clips below come from two models trained for **just one
epoch** on the **single main codebook** (Config A) — one **without** AG-REPA (baseline) and
one **with** it, everything else identical. After a single epoch the AG-REPA model is
already clearly cleaner and more stable, while the baseline is still noisy and
under-converged — a direct, audible sign that AG-REPA **speeds up training and stabilises
quality**.

**🗣️ TTS — zero-shot speech**

❌ *Baseline — no AG-REPA (1 epoch):*

https://github.com/user-attachments/assets/9d084624-dc6d-4f13-98ae-95d5367aac3e

✅ *AG-REPA (1 epoch):*

https://github.com/user-attachments/assets/4844662d-aafb-489d-9b04-dcf62c8d03bb

**🔊 TTA — general audio**

❌ *Baseline — no AG-REPA (1 epoch):*

https://github.com/user-attachments/assets/10fa3c94-8f70-4f89-a20b-13c734e6b1d3

✅ *AG-REPA (1 epoch):*

https://github.com/user-attachments/assets/04c1daf8-a976-4732-b3d0-0698b77f0e1b

> ▶ The clips play **inline** — just press play, no download. They are **one-epoch,
> single-codebook** samples showing *early-training* convergence (not final quality): even
> after one epoch AG-REPA is already cleaner and more stable than the baseline. Lossless
> WAVs: TTS [baseline](assets/audio/tts_no_agrepa.wav) / [AG-REPA](assets/audio/tts_agrepa.wav) ·
> TTA [baseline](assets/audio/tta_no_agrepa.wav) / [AG-REPA](assets/audio/tta_agrepa.wav).
> Fully-trained quality: [paper](https://icml.cc/virtual/2026/poster/65899) ·
> [🤗 model card](https://huggingface.co/AustinZhang/AG-REPA).

---

## ⚡ Quick start (inference in 3 steps)

Generate audio with the released AG-REPA weights. To train from scratch instead, jump to
[§5 Installation](#5-installation) → [§7 Training](#7-training-pipeline).

```bash
# 1) Environment
conda create -n agrepa python=3.10 -y && conda activate agrepa
pip install -r requirements.txt

# 2) Download the weights and wire them into a variant directory (full map in §9)
hf download AustinZhang/AG-REPA --local-dir AG-REPA-Model
cd Fusion_single_codebook
ln -s /path/to/pretrained_base_models pretrained_models          # BEATs + CosyVoice, see model card
mkdir -p checkpoints
ln -s /path/to/AG-REPA-Model/flow_matching/agrepa_single_codebook checkpoints/flow

# 3) Synthesize (zero-shot TTS, voice cloned from a reference clip)
python inference_tts.py \
    --text "This is a classic line from Blade Runner." \
    --prompt_wav ./wav/english_male.flac \
    --checkpoint_dir ./checkpoints/flow \
    --cosyvoice_model_dir ./pretrained_models/CosyVoice-300M \
    --output ./output/generated_tts.wav --speed 0.9 --gpu_id 0
```

**New here?** Read [§1](#1-what-problem-does-ag-repa-solve) for *what AG-REPA does*,
[§4](#4-repository-layout--the-four-variants) to *pick the right variant*, and
[§8](#8-inference)–[§9](#9-wiring-the-model-weights-to-the-code) for the full inference and
weight-wiring details.

---

## 1. What problem does AG-REPA solve?

REPresentation Alignment (REPA) speeds up training of Flow-Matching (FM) generative models
by aligning a network's intermediate hidden states with frozen pretrained teacher features.
But it only works if you align the **right layers** — and prior work picks them
*heuristically* (e.g. "always align the mid-block / layer 8").

The paper shows this heuristic is mis-targeted for **token-conditioned audio FM**, because
of what we call **Store–Contribute Dissociation (SCD)**:

* **Storage — what the network *knows*.** Deep layers (L20–L24) hold the richest
  semantic/acoustic information (high similarity to the teacher).
* **Contribution — what the network *uses*.** Shallow layers (L1–L3) and a mid-phase band
  (L6–L12, around diffusion time *t≈0.5*) are what actually drive the predicted velocity
  field.

These two sets **don't overlap**. So aligning the information-rich deep layers (standard
REPA) ends up supervising layers that are "rich but functionally passive." **AG-REPA**
instead aligns the *causally dominant* layers, picked automatically by a forward-only
causal-attribution probe.

**Result:** AG-REPA cuts Fréchet Audio Distance (FAD) by **18 % (speech)** and
**16 % (audio)** vs. the best fixed-layer REPA baseline, and transfers across Voicebox,
CosyVoice, and F5-TTS architectures.

---

## 2. The interpretability toolkit & AG-REPA in one picture

| Probe | Question it answers | What it measures |
|-------|---------------------|------------------|
| **BiT-C** (Bi-Stream Teacher Cosine) | *Is the conditioning interface aligned to both modalities?* | Cosine alignment of layer-0 representations to **Whisper** (speech) and **BEATs** (audio) teachers. |
| **LASP** (Layer-wise Analysis via Shared Projection) | *What does each layer **know**?* | Cosine similarity of every layer's pooled representation to the teacher, through a single frozen shared projection head (Store / "Cos-SEM", "Cos-EVT"). |
| **FoG-A** (Forward-only Gate Ablation) | *What does each layer **use**?* | Normalised change in the predicted velocity field when a layer's residual contribution is gated off (Contribute). |

<p align="center">
  <img src="assets/methodology_bitc_lasp.png" width="92%" alt="BiT-C dual-teacher supervision and LASP layer-wise shared-projection analysis">
</p>
<p align="center"><sub><b>Diagnosing representation storage.</b> (a) <b>BiT-C</b> anchors the conditioning interface to frozen <b>Whisper</b> (semantic) and <b>BEATs</b> (acoustic) teachers; (b) <b>LASP</b> probes "what each layer knows" by projecting every layer into a shared teacher space and measuring cosine similarity.</sub></p>

**AG-REPA** then (i) ranks layers by FoG-A causal attribution and keeps the **Top-K**, and
(ii) attaches a lightweight per-layer MLP projection head with an
**attribution-proportional weight** `λ_k ∝ FoG-A_k` — so the alignment loss is applied only
where it causally matters. Here `K = 3`; the probe-selected layers are:

* **Speech (Whisper teacher):** layers **L1, L9, L5** → `λ ≈ {0.334, 0.139, 0.118}`
* **Audio (BEATs teacher):** layers **L1, L21, L9** → `λ ≈ {0.278, 0.120, 0.112}`

(These exactly match Table 1 / Equation 11 of the paper and are hard-coded in
`REPA_*/models.py`.)

<p align="center">
  <img src="assets/methodology_foga_agrepa.png" width="80%" alt="FoG-A causal attribution and the AG-REPA targeted alignment objective">
</p>
<p align="center"><sub><b>From causal attribution to optimization.</b> (a) <b>FoG-A</b> gates off each layer's residual contribution and measures the induced change in the velocity field, producing a causal-importance map (red = high contribution). (b) <b>AG-REPA</b> applies alignment supervision <em>only</em> to the Top-K causally critical layers, each through a projection head weighted by its attribution score <code>λ_k</code>.</sub></p>

---

## 3. System architecture

A two-stage cascade splits high-level semantic planning from low-level acoustic rendering
(Appendix A of the paper):

<p align="center">
  <img src="assets/framework.png" width="100%" alt="The unified audio generation framework: tokenization, Stage-1 autoregressive LLM, and Stage-2 Flow Matching">
</p>
<p align="center"><sub><b>The unified audio generation framework.</b> (a) Domain-specific tokenization produces a unified discrete sequence (S³ tokens for speech, AudioSet tokens for audio, optionally interleaved with BEATs tokens); (b) a Stage-1 autoregressive LLM predicts the target acoustic tokens with reference-style injection; (c) a Stage-2 DiT Flow-Matching model synthesizes the mel-spectrogram, decoded to a waveform by the Vocos vocoder.</sub></p>

The same pipeline as a text schematic:

```
                 ┌──────────────────── Stage 1: Autoregressive LLM ───────────────────┐
 text + ref ───► │  Qwen3-0.6B-Base, fine-tuned to predict discrete acoustic tokens.   │
 audio           │  Reference style injected via BEATs-derived embeddings; auxiliary   │
                 │  coarse (cluster) prediction head.                                  │
                 └─────────────────────────────┬──────────────────────────────────────┘
                                               │  discrete tokens (S3 for speech, AudioSet for audio)
                                               ▼
                 ┌──────────────────── Stage 2: Flow Matching (DiT) ──────────────────┐
                 │  24-layer DiT backbone with adaLN-Zero predicts the velocity field  │
                 │  v_θ(x_t, t, c) transporting noise → target mel-spectrogram.        │
                 │  *** This is where BiT-C / LASP / FoG-A / AG-REPA operate. ***      │
                 └─────────────────────────────┬──────────────────────────────────────┘
                                               ▼
                                   Vocos vocoder ──► waveform (24 kHz)
```

### Tokenization pathways
* **Speech (S³ tokens).** CosyVoice's `speech_tokenizer_v1.onnx` produces semantic
  S³ tokens (vocab 4096).
* **Audio (AudioSet tokens).** A dedicated **AudioSet Tokenizer (AST)** — a RepCodec
  VQ-VAE trained on frozen **BEATs** features — produces discrete acoustic tokens
  (vocab 4096), aligned with the S³ vocabulary for unified downstream processing.

---

## 4. Repository layout — the four variants

The release ships **four self-contained variants**, differing along two axes:

|                | **Baseline + diagnostics** (`Fusion_*`) | **AG-REPA training** (`REPA_*`) |
|----------------|------------------------------------------|----------------------------------|
| **Single codebook** (Config A: S³ + AudioSet tokens) | `Fusion_single_codebook/` | `REPA_single_codebook/` |
| **Dual codebook** (Config B: Config A **+ interleaved BEATs** tokens) | `Fusion_dual_codebook/` | `REPA_dual_codebook/` |

* **`Fusion_*` — the diagnostic / Phase-I model.** Trains the FM model with the standard
  objective while running the **interpretability probes** (`LayerProbeLogger`,
  `fog_attribution`, `probe_layers` in `models.py`). It produces the layer-attribution CSVs
  and heat-maps that reproduce **Figure 1 / Table 1** of the paper, and doubles as the
  *no-alignment baseline*.
  > "Fusion" = the model carrying the **fused interpretability toolkit** (BiT-C + LASP +
  > FoG-A). The terminal capture `COS_FOG.png` in the model release is its Top-3 output.

* **`REPA_*` — the Phase-II model.** Starting from the same post-warm-up state, it applies
  **attribution-guided REPA** intra-layer bypass alignment at the FoG-A-selected layers
  (see §2). AG-REPA touches **only the Flow-Matching stage** — the Stage-1 LLM is shared
  with the corresponding `Fusion_*` codebook config.

* **`single` vs `dual` codebook.** The dual-codebook variants interleave a dense BEATs token
  after every primary token (`s = [t1, b1, t2, b2, …]`, Equation 15), giving a proxy
  manifold closer to the target acoustic manifold.

This follows the paper's strict **probe-then-intervene** protocol (Appendix A.5):
*Phase I (`Fusion_*`)* computes and freezes the Top-K causal layer set; *Phase II
(`REPA_*`)* trains with alignment applied only to those layers.

### Files inside each variant

| File | Role |
|------|------|
| `config.yaml` | Single source of truth: data paths, audio/mel settings, and all Stage-1/2 hyper-parameters. |
| `models.py` | `AudioSetTokenizer` (AST), `MelSpectrogramExtractor`, `DiTBlock`, `FlowMatchingModel`. In `Fusion_*` it also exposes `fog_attribution()` and `probe_layers()`; in `REPA_*` it implements the per-layer REPA heads + λ weighting. |
| `data_loader.py` | `LibriSpeechDataset` (speech) and `AudioSetDataset` (general audio). *(identical across variants)* |
| `extract_s3_tokens.py` | Pre-extracts CosyVoice S³ tokens for LibriSpeech into `speech_tokens/`. |
| `train_ast.py` | Stage-0: trains the AudioSet Tokenizer (RepCodec VQ-VAE on BEATs features). *(identical across variants)* |
| `train_llm.py` | Stage-1: fine-tunes the Qwen3-0.6B-Base LLM (token prediction + style injection + coarse head). |
| `train_cfm.py` | Stage-2: trains the DiT Flow-Matching model. `Fusion_*` embeds the probe logger; `REPA_*` runs AG-REPA training. |
| `utils.py` | Config loading + audio peak-normalisation helpers. *(identical across variants)* |
| `ds_config_{ast,cfm,llm}.json` | DeepSpeed ZeRO-1 configs for each training stage. |
| `cosyvoice/`, `beats/`, `repcodec/` | Vendored third-party encoders/tokenizers (see §12). |
| `wav/` | A handful of small reference-audio clips for inference demos. |
| `inference_tts.py`, `inference_tta.py` | **(`Fusion_single_codebook/` only)** End-to-end TTS / TTA inference. |
| `generate_descriptions.py` | **(`REPA_single_codebook/` only)** Auto-captions AudioSet clips with MiDashengLM-7B → `audioset_description.jsonl` (TTA text conditioning). |

> The four variants share a lot of code: `train_ast.py`, `data_loader.py`, `utils.py` and
> the vendored libraries are byte-identical across all four; `models.py`, `train_cfm.py`,
> `train_llm.py` and `config.yaml` differ per variant.

---

## 5. Installation

```bash
# Python 3.10 is recommended (matches the vendored .pyc / wheels).
conda create -n agrepa python=3.10 -y
conda activate agrepa

pip install -r requirements.txt

# CosyVoice text frontend (optional, only for TTS inference with text normalisation):
# install the ttsfrd wheels shipped with the CosyVoice-ttsfrd model package
# (https://www.modelscope.cn/models/iic/CosyVoice-ttsfrd).
```

Then get the pre-trained weights from **[🤗 AustinZhang/AG-REPA](https://huggingface.co/AustinZhang/AG-REPA)**
and wire them in per §9.

---

## 6. Data preparation

The models train on **LibriSpeech** (speech, 960 h) + **AudioSet** (general audio). Set the
dataset roots in `config.yaml` (`data.librispeech.*`, `data.audioset.*`), then:

```bash
cd REPA_single_codebook            # or any variant

# (a) Pre-extract S3 speech tokens for LibriSpeech  ->  speech_tokens/
python extract_s3_tokens.py --config config.yaml \
    --dataset librispeech \
    --model ./pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx \
    --save_dir speech_tokens --threads 16 --gpu_id 0

# (b) Generate natural-language captions for AudioSet  ->  audioset_description.jsonl
#     (only present in REPA_single_codebook/)
python generate_descriptions.py
```

---

## 7. Training pipeline

Every stage is launched with **DeepSpeed** (ZeRO-1) and reads its hyper-parameters from
`config.yaml` and the matching `ds_config_*.json`.

```bash
cd REPA_single_codebook            # pick the variant you want to train

# Stage 0 — AudioSet Tokenizer (RepCodec VQ-VAE on BEATs features)
deepspeed train_ast.py --config config.yaml

# Stage 1 — Autoregressive LLM (Qwen3-0.6B-Base; token + style + coarse losses)
deepspeed train_llm.py --config config.yaml

# Stage 2 — Flow-Matching DiT backbone
#   * In Fusion_* this also runs the BiT-C / LASP / FoG-A probes (Phase I).
#   * In REPA_*  this trains with attribution-guided alignment (Phase II).
deepspeed train_cfm.py --config config.yaml
```

Checkpoints land under `checkpoints/{ast,llm,flow}/`, and validation/visualisation samples
under `outputs_cfm/`.

### Reproducing the diagnostics (Figure 1 & Table 1)

Run **Stage 2 in a `Fusion_*` variant**. During the warm-up epoch the `LayerProbeLogger`
writes, per epoch, to `checkpoints/flow/`:

* `csv/probe_stats_epoch_XXXX.csv` — LASP (Cos-SEM / Cos-EVT) storage scores per layer.
* `csv/probe_stats_epoch_XXXX_foga.csv` — FoG-A causal-attribution scores per layer.
* `image/..._foga_heatmap.png` — the spatiotemporal SCD heat-map (Figure 1).
* `image/..._importance.png`, `image/..._trend_top3.png` — per-layer importance & Top-3
  stability across training (Table 7).

The Top-K layers and `λ_k` read off these CSVs are exactly what is hard-coded into the
`REPA_*` models for Phase II.

---

## 8. Inference

The end-to-end inference scripts live in **`Fusion_single_codebook/`**.

```bash
cd Fusion_single_codebook

# Text-to-Speech (zero-shot, voice cloned from a reference clip)
python inference_tts.py \
    --text "This is a classic line from Blade Runner." \
    --prompt_wav ./wav/english_male.flac \
    --checkpoint_dir ./checkpoints/flow \
    --cosyvoice_model_dir ./pretrained_models/CosyVoice-300M \
    --output ./output/generated_tts.wav --speed 0.9 --gpu_id 0

# Text-to-Audio (sound-event synthesis conditioned on an event tag + description)
python inference_tta.py \
    --event "Dog barking" \
    --desc "A medium-sized dog barking repeatedly in a quiet room" \
    --prompt_wav ./wav/dog.wav \
    --llm_ckpt_dir ./checkpoints --flow_ckpt_dir ./checkpoints/flow \
    --output ./output/generated_tta.wav --gpu_id 0
```

> To synthesise with the **AG-REPA** model, point `--checkpoint_dir` / `--flow_ckpt_dir`
> at a Flow-Matching checkpoint from `AG-REPA-Model/flow_matching/agrepa_*`.

---

## 9. Wiring the model weights to the code

Download the weights from Hugging Face (`hf download AustinZhang/AG-REPA --local-dir
AG-REPA-Model`) and the base models from their upstream sources. Each variant directory
expects these sub-folders — symlink or copy them in:

```
<variant>/
├── pretrained_models/        ←  BEATs + CosyVoice (download from upstream — see model card)
│   ├── BEATs_iter3_plus_AS2M.pt
│   ├── CosyVoice-300M/
│   └── CosyVoice-ttsfrd/
└── checkpoints/
    ├── ast/                  ←  AG-REPA-Model/audioset_tokenizer/
    ├── llm/                  ←  AG-REPA-Model/llm/<single|dual>_codebook/
    └── flow/                 ←  AG-REPA-Model/flow_matching/agrepa_<single|dual>_codebook/
```

> The Hugging Face release ships only the **final AG-REPA** checkpoints (`agrepa_*`, the
> `REPA_*` variants). The no-alignment baselines and FoG-A/LASP diagnostics are not bundled
> — produce them by training the `Fusion_*` variants from this code.

Example:

```bash
cd REPA_single_codebook
ln -s /path/to/pretrained_base_models                            pretrained_models
mkdir -p checkpoints
ln -s /path/to/AG-REPA-Model/audioset_tokenizer                  checkpoints/ast
ln -s /path/to/AG-REPA-Model/llm/single_codebook                 checkpoints/llm
ln -s /path/to/AG-REPA-Model/flow_matching/agrepa_single_codebook checkpoints/flow
```

See the [🤗 model card](https://huggingface.co/AustinZhang/AG-REPA) for the full mapping
table and base-model download links.

---

## 10. Key configuration knobs (`config.yaml`)

| Section | Highlights |
|---------|-----------|
| `audio` | 16 kHz input, 100-bin mel; `vocos` re-synthesises at 24 kHz. |
| `hyperparameters.ast` | `vocab_size: 4096`, reconstruction + VQ-commitment losses. |
| `hyperparameters.llm` | `Qwen/Qwen3-0.6B-Base`; `style_num_tokens (K)=16`, `style_strength (α)=0.5`, `style_dropout_p=0.7` (CFG); `coarse_num_clusters=128`, `coarse_loss_weight=0.5`. |
| `hyperparameters.flow` | 24-layer DiT, `hidden_dim 1024`, `n_heads 16`; teacher distillation weights `teacher_speech 0.5`, `teacher_audio 1.0`; `enable_whitebox_probe: true` turns on the diagnostics; `n_timesteps 32`, `cfg_scale 3.0` at inference. |

---

## 11. Citation

```bibtex
@inproceedings{zhang2026agrepa,
  title     = {AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching},
  author    = {Zhang, Pengfei and Xie, Tianxin and Yang, Minghao and Liu, Li},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## 12. Acknowledgements & third-party components

This work builds on several open-source projects, vendored under each variant directory or
referenced as teachers/base models:

* **CosyVoice** (Du et al., 2024) — S³ speech tokenizer & Stage-1 LLM design (`cosyvoice/`) — <https://github.com/FunAudioLLM/CosyVoice>
* **BEATs** (Chen et al., 2022) — acoustic teacher & audio-token features (`beats/`) — <https://github.com/microsoft/unilm/tree/master/beats>
* **RepCodec** (Huang et al., 2024) — VQ-VAE backbone of the AudioSet Tokenizer (`repcodec/`) — <https://github.com/mct10/RepCodec>
* **Vocos** (Siuzdak, 2023) — mel-spectrogram vocoder — <https://github.com/gemelo-ai/vocos>
* **Whisper** (Radford et al., 2022) — semantic teacher for speech — <https://github.com/openai/whisper>
* **Qwen3-0.6B-Base** (Yang et al., 2025) — Stage-1 LLM initialisation — <https://github.com/QwenLM/Qwen3>
* **MiDashengLM-7B** (Xiaomi) — AudioSet caption generation (data prep) — <https://github.com/xiaomi-research/dasheng-lm>

---

## 13. License

The AG-REPA-specific code in this repository is released under the **MIT License** — see
[`LICENSE`](LICENSE).

The vendored third-party components (`cosyvoice/`, `beats/`, `repcodec/`) and the referenced
base models keep their **own original licenses**; please check the upstream repositories
linked above before redistribution or commercial use.

As noted in the paper's Impact Statement, high-fidelity audio generation and voice cloning
carry risks (deepfakes, impersonation, voice spoofing). Responsible deployment should
include audio watermarking, spoofing detection, and restricted access to voice-cloning
capabilities.
