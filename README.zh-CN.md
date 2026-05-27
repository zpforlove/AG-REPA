# AG-REPA:面向音频流匹配的归因引导表征对齐

> 论文官方代码发布
> **《AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching》**
> (《AG-REPA:面向音频流匹配中表征对齐的因果层选择》)
> *Pengfei Zhang, Tianxin Xie, Minghao Yang, Li Liu.* —— ICML 2026。

> 📄 **论文与海报:** [ICML 2026 Virtual](https://icml.cc/virtual/2026/poster/65899)
> &nbsp;|&nbsp; 🌐 **语言:** [English](README.md) | 简体中文

一个统一的音频生成框架,用**单个流匹配(Flow Matching)主干**同时实现**文本转语音(TTS)**
与**文本转音频(TTA)**。本仓库包含其**训练、诊断与推理代码**,以及论文提出的可解释性工具集
(**BiT-C / LASP / FoG-A**)和 **AG-REPA** 训练策略。

> **一句话理解:** 标准 REPA 对齐的是"信息**存得最多**"的层;AG-REPA 转而对齐"**真正驱动
> 输出**"的层——这些层由一个因果探针自动找出——从而把 Fréchet 音频距离(FAD)相比最优固定层
> REPA 基线降低 **18%(语音)/ 16%(音频)**。

> ℹ️ **本仓库仅含代码**——不含任何检查点或数据集。预训练权重与诊断产物在 Hugging Face:
> **[🤗 AustinZhang/AG-REPA](https://huggingface.co/AustinZhang/AG-REPA)**。冻结的第三方基础
> 模型(BEATs、CosyVoice)需从各自原始来源下载,详见模型卡。

---

## 🔊 音频示例 —— AG-REPA 使用前后对比(单主码本,仅 1 个 Epoch)

为了让效果"听得见",下面的音频均由**仅训练 1 个 Epoch**、且**只用单个主码本**(配置 A)的
模型合成——一个**不用** AG-REPA(基线),一个**使用** AG-REPA,其余设置完全相同。仅一个
Epoch 后,AG-REPA 模型就已能合成明显更清晰、更稳定的音频,而基线仍嘈杂、欠收敛——直观体现
AG-REPA **加速训练、稳定音质**的作用。

**🗣️ TTS —— 零样本语音**

❌ *基线 —— 无 AG-REPA(1 Epoch):*

https://github.com/user-attachments/assets/9d084624-dc6d-4f13-98ae-95d5367aac3e

✅ *AG-REPA(1 Epoch):*

https://github.com/user-attachments/assets/4844662d-aafb-489d-9b04-dcf62c8d03bb

**🔊 TTA —— 通用音频**

❌ *基线 —— 无 AG-REPA(1 Epoch):*

https://github.com/user-attachments/assets/10fa3c94-8f70-4f89-a20b-13c734e6b1d3

✅ *AG-REPA(1 Epoch):*

https://github.com/user-attachments/assets/04c1daf8-a976-4732-b3d0-0698b77f0e1b

> ▶ 上方音频在本页面**直接播放**——点播放即可,无需下载。它们是**单码本、仅 1 个 Epoch**的
> 样本,展示*训练早期*的收敛差异(并非最终音质):仅一个 Epoch,AG-REPA 就已明显比基线更清晰、
> 更稳定。无损 WAV 下载:TTS [基线](assets/audio/tts_no_agrepa.wav) /
> [AG-REPA](assets/audio/tts_agrepa.wav) · TTA [基线](assets/audio/tta_no_agrepa.wav) /
> [AG-REPA](assets/audio/tta_agrepa.wav)。完整训练效果见
> [论文](https://icml.cc/virtual/2026/poster/65899)与
> [🤗 模型卡](https://huggingface.co/AustinZhang/AG-REPA)。

---

## ⚡ 快速上手(三步跑通推理)

用已发布的 AG-REPA 权重生成音频。若想**从头训练**,请直接看
[§5 安装](#5-安装) → [§7 训练流程](#7-训练流程)。

```bash
# 1) 环境
conda create -n agrepa python=3.10 -y && conda activate agrepa
pip install -r requirements.txt

# 2) 下载权重,并接入某个变体目录(完整映射见 §9)
hf download AustinZhang/AG-REPA --local-dir AG-REPA-Model
cd Fusion_single_codebook
ln -s /path/to/pretrained_base_models pretrained_models          # BEATs + CosyVoice,详见模型卡
mkdir -p checkpoints
ln -s /path/to/AG-REPA-Model/flow_matching/agrepa_single_codebook checkpoints/flow

# 3) 合成(零样本 TTS,从参考片段克隆音色)
python inference_tts.py \
    --text "This is a classic line from Blade Runner." \
    --prompt_wav ./wav/english_male.flac \
    --checkpoint_dir ./checkpoints/flow \
    --cosyvoice_model_dir ./pretrained_models/CosyVoice-300M \
    --output ./output/generated_tts.wav --speed 0.9 --gpu_id 0
```

**第一次接触?** 先读 [§1](#1-ag-repa-要解决什么问题) 了解 *AG-REPA 在做什么*,看
[§4](#4-仓库结构四个变体) *选对变体*,再看 [§8](#8-推理)–[§9](#9-将模型权重与代码对接)
了解完整的推理与权重接入细节。

---

## 1. AG-REPA 要解决什么问题?

表征对齐(REPresentation Alignment,REPA)通过将流匹配(FM)生成模型的中间隐藏状态与冻结的
预训练教师特征对齐来加速训练。但它只有在对齐**正确的层**时才有效——而以往工作往往凭**启发式**
做出选择(例如"总是对齐中间层 / 第 8 层")。

论文表明,对于**以 token 为条件的音频 FM**,这种启发式选择是"打偏了"的,原因是一个被我们称为
**存储—贡献分离(Store–Contribute Dissociation,SCD)** 的现象:

* **存储——网络"知道"什么。** 深层(L20–L24)携带最丰富的语义/声学信息(与教师空间的相似度高)。
* **贡献——网络实际"用"了什么。** 浅层(L1–L3),以及一个中段过渡带(在扩散时间 *t≈0.5* 附近
  的 L6–L12),才是真正驱动预测速度场的层。

这两组层**并不重合**。因此,对齐信息丰富的深层(标准 REPA)实际监督的是"表征丰富却在功能上
被动"的层。**AG-REPA** 转而把对齐施加到**因果上占主导**的层上——这些层由一个仅前向
(forward-only)的因果归因探针自动识别。

**结果:** 相比最优的固定层 REPA 基线,AG-REPA 将 Fréchet 音频距离(FAD)降低了
**18%(语音)** 与 **16%(音频)**,且可迁移到 Voicebox、CosyVoice、F5-TTS 等架构。

---

## 2. 一图看懂可解释性工具集与 AG-REPA

| 探针 | 回答的问题 | 度量的内容 |
|-------|---------------------|------------------|
| **BiT-C**(双流教师余弦,Bi-Stream Teacher Cosine) | *条件接口是否同时对齐了两种模态?* | 第 0 层表征与 **Whisper**(语音)、**BEATs**(音频)两个教师的余弦对齐度。 |
| **LASP**(共享投影下的逐层分析) | *每一层"知道"什么?* | 通过单个冻结的共享投影头,度量每层池化表征与教师的余弦相似度(存储 / "Cos-SEM"、"Cos-EVT")。 |
| **FoG-A**(仅前向门控消融,Forward-only Gate Ablation) | *每一层"用"了什么?* | 关闭某层的残差贡献后,预测速度场的归一化变化量(贡献)。 |

<p align="center">
  <img src="assets/methodology_bitc_lasp.png" width="92%" alt="BiT-C 双教师监督与 LASP 共享投影逐层分析">
</p>
<p align="center"><sub><b>诊断表征存储。</b>(a)<b>BiT-C</b> 将条件接口锚定到冻结的 <b>Whisper</b>(语义)与 <b>BEATs</b>(声学)教师;(b)<b>LASP</b> 通过把每一层投影到共享教师空间并度量余弦相似度,探查"每一层知道什么"。</sub></p>

随后,**AG-REPA**(i)按 FoG-A 因果归因对各层排序,保留 **Top-K** 层;(ii)为每个被选中的层
挂接一个轻量级逐层 MLP 投影头,并赋予**与归因成正比的权重** `λ_k ∝ FoG-A_k`——只在因果上
真正重要的位置施加对齐损失。本发布中 `K = 3`,探针选出的层为:

* **语音(Whisper 教师):** 层 **L1、L9、L5** → `λ ≈ {0.334, 0.139, 0.118}`
* **音频(BEATs 教师):** 层 **L1、L21、L9** → `λ ≈ {0.278, 0.120, 0.112}`

(这与论文表 1 / 公式 11 完全一致,并被硬编码在 `REPA_*/models.py` 中。)

<p align="center">
  <img src="assets/methodology_foga_agrepa.png" width="80%" alt="FoG-A 因果归因与 AG-REPA 的定向对齐目标">
</p>
<p align="center"><sub><b>从因果归因到优化。</b>(a)<b>FoG-A</b> 关闭每一层的残差贡献,度量速度场由此产生的变化,生成因果重要性图(红色 = 高贡献)。(b)<b>AG-REPA</b> <em>仅</em>对 Top-K 个因果关键层施加对齐监督,每层经由一个按其归因分数 <code>λ_k</code> 加权的投影头。</sub></p>

---

## 3. 系统架构

采用两阶段级联结构,将高层语义规划与低层声学渲染解耦(对应论文附录 A):

<p align="center">
  <img src="assets/framework.png" width="100%" alt="统一音频生成框架:Token 化、阶段 1 自回归 LLM、阶段 2 流匹配">
</p>
<p align="center"><sub><b>统一音频生成框架。</b>(a)按领域划分的 Token 化产生统一的离散序列(语音用 S³ token,音频用 AudioSet token,可选地与 BEATs token 交织);(b)阶段 1 自回归 LLM 在参考风格注入下预测目标声学 token;(c)阶段 2 DiT 流匹配模型合成梅尔谱,再由 Vocos 声码器解码为波形。</sub></p>

同一流程的文字示意图:

```
                 ┌──────────────────── 阶段 1:自回归 LLM ───────────────────┐
 文本 + 参考 ───► │  Qwen3-0.6B-Base,微调后用于预测离散声学 token。              │
 音频            │  参考风格通过 BEATs 派生的嵌入注入;附带粗粒度(聚类)预测头。  │
                 └─────────────────────────────┬──────────────────────────────┘
                                               │  离散 token(语音用 S3,音频用 AudioSet)
                                               ▼
                 ┌──────────────────── 阶段 2:流匹配(DiT) ──────────────────┐
                 │  24 层 DiT 主干(adaLN-Zero)预测速度场 v_θ(x_t, t, c),       │
                 │  将噪声搬运到目标梅尔谱。                                      │
                 │  *** BiT-C / LASP / FoG-A / AG-REPA 都作用在这一阶段。 ***   │
                 └─────────────────────────────┬──────────────────────────────┘
                                               ▼
                                   Vocos 声码器 ──► 波形(24 kHz)
```

### Token 化路径
* **语音(S³ token)。** 由 CosyVoice 的 `speech_tokenizer_v1.onnx` 产生语义 S³ token
  (词表 4096)。
* **音频(AudioSet token)。** 由专门的 **AudioSet 分词器(AST)** 产生离散声学 token
  (词表 4096)——AST 是在冻结的 **BEATs** 特征上训练的 RepCodec VQ-VAE,其词表与 S³
  对齐,以便统一下游处理。

---

## 4. 仓库结构——四个变体

本发布提供**四个自包含的变体**,沿两个维度区分:

|                | **基线 + 诊断**(`Fusion_*`) | **AG-REPA 训练**(`REPA_*`) |
|----------------|------------------------------------------|----------------------------------|
| **单码本**(配置 A:S³ + AudioSet token) | `Fusion_single_codebook/` | `REPA_single_codebook/` |
| **双码本**(配置 B:配置 A **+ 交织的 BEATs** token) | `Fusion_dual_codebook/` | `REPA_dual_codebook/` |

* **`Fusion_*` —— 诊断 / 第一阶段模型。** 用标准目标训练 FM 模型,同时运行**可解释性探针**
  (`models.py` 中的 `LayerProbeLogger`、`fog_attribution`、`probe_layers`)。它产出可复现
  论文 **图 1 / 表 1** 的逐层归因 CSV 与热力图,同时充当*无对齐基线*。
  > "Fusion" = 携带**融合可解释性工具集**(BiT-C + LASP + FoG-A)的模型。模型发布包中的
  > 终端截图 `COS_FOG.png` 即为它的 Top-3 输出。

* **`REPA_*` —— 第二阶段模型。** 从相同的预热后状态出发,在 FoG-A 选出的层上施加**归因引导的
  REPA** 层内旁路对齐(见 §2)。AG-REPA **只修改流匹配阶段**——阶段 1 的 LLM 与对应码本配置的
  `Fusion_*` 共享。

* **`single` 与 `dual` 码本。** 双码本变体在每个主 token 后交织一个稠密的 BEATs token
  (`s = [t1, b1, t2, b2, …]`,公式 15),构造出更接近目标声学流形的代理流形。

这与论文严格的**先探测后干预(probe-then-intervene)**协议(附录 A.5)一致:
*第一阶段(`Fusion_*`)* 计算并冻结 Top-K 因果层集合;*第二阶段(`REPA_*`)* 仅对这些层
施加对齐进行训练。

### 每个变体内部的文件

| 文件 | 作用 |
|------|------|
| `config.yaml` | 唯一配置来源:数据路径、音频/梅尔设置,以及全部阶段 1/2 超参数。 |
| `models.py` | `AudioSetTokenizer`(AST)、`MelSpectrogramExtractor`、`DiTBlock`、`FlowMatchingModel`。`Fusion_*` 中还提供 `fog_attribution()` 与 `probe_layers()`;`REPA_*` 中实现逐层 REPA 头 + λ 加权。 |
| `data_loader.py` | `LibriSpeechDataset`(语音)与 `AudioSetDataset`(通用音频)。*(各变体一致)* |
| `extract_s3_tokens.py` | 为 LibriSpeech 预提取 CosyVoice S³ token 至 `speech_tokens/`。 |
| `train_ast.py` | 阶段 0:训练 AudioSet 分词器(在 BEATs 特征上的 RepCodec VQ-VAE)。*(各变体一致)* |
| `train_llm.py` | 阶段 1:微调 Qwen3-0.6B-Base LLM(token 预测 + 风格注入 + 粗粒度头)。 |
| `train_cfm.py` | 阶段 2:训练 DiT 流匹配模型。`Fusion_*` 内嵌探针记录器;`REPA_*` 运行 AG-REPA 训练。 |
| `utils.py` | 配置加载 + 音频峰值归一化工具。*(各变体一致)* |
| `ds_config_{ast,cfm,llm}.json` | 各训练阶段的 DeepSpeed ZeRO-1 配置。 |
| `cosyvoice/`、`beats/`、`repcodec/` | 内置(vendored)的第三方编码器/分词器(见 §12)。 |
| `wav/` | 少量小体积参考音频,用于推理演示。 |
| `inference_tts.py`、`inference_tta.py` | **(仅 `Fusion_single_codebook/`)** 端到端 TTS / TTA 推理。 |
| `generate_descriptions.py` | **(仅 `REPA_single_codebook/`)** 用 MiDashengLM-7B 为 AudioSet 片段自动生成字幕 → `audioset_description.jsonl`(TTA 文本条件)。 |

> 四个变体共享大量代码:`train_ast.py`、`data_loader.py`、`utils.py` 以及内置库在四个变体间
> **逐字节一致**;`models.py`、`train_cfm.py`、`train_llm.py` 与 `config.yaml` 在各变体间不同。

---

## 5. 安装

```bash
# 推荐 Python 3.10(与内置的 .pyc / wheel 匹配)。
conda create -n agrepa python=3.10 -y
conda activate agrepa

pip install -r requirements.txt

# CosyVoice 文本前端(可选,仅在做带文本归一化的 TTS 推理时需要):
# 安装 CosyVoice-ttsfrd 模型包附带的 ttsfrd wheel
# (https://www.modelscope.cn/models/iic/CosyVoice-ttsfrd)。
```

随后从 Hugging Face 模型仓库 **[🤗 AustinZhang/AG-REPA](https://huggingface.co/AustinZhang/AG-REPA)**
下载预训练权重,并按下文 §9 接入。

---

## 6. 数据准备

模型在 **LibriSpeech**(语音,960 小时)+ **AudioSet**(通用音频)上训练。先在 `config.yaml`
中设置数据集根目录(`data.librispeech.*`、`data.audioset.*`),然后:

```bash
cd REPA_single_codebook            # 或任意变体

# (a) 为 LibriSpeech 预提取 S3 语音 token  ->  speech_tokens/
python extract_s3_tokens.py --config config.yaml \
    --dataset librispeech \
    --model ./pretrained_models/CosyVoice-300M/speech_tokenizer_v1.onnx \
    --save_dir speech_tokens --threads 16 --gpu_id 0

# (b) 为 AudioSet 生成自然语言字幕  ->  audioset_description.jsonl
#     (仅存在于 REPA_single_codebook/)
python generate_descriptions.py
```

---

## 7. 训练流程

所有训练阶段均通过 **DeepSpeed**(ZeRO-1)启动,并从 `config.yaml` 和对应的
`ds_config_*.json` 读取超参数。

```bash
cd REPA_single_codebook            # 选择要训练的变体

# 阶段 0 —— AudioSet 分词器(BEATs 特征上的 RepCodec VQ-VAE)
deepspeed train_ast.py --config config.yaml

# 阶段 1 —— 自回归 LLM(Qwen3-0.6B-Base;token + 风格 + 粗粒度损失)
deepspeed train_llm.py --config config.yaml

# 阶段 2 —— 流匹配 DiT 主干
#   * 在 Fusion_* 中会同时运行 BiT-C / LASP / FoG-A 探针(第一阶段)。
#   * 在 REPA_*  中以归因引导对齐进行训练(第二阶段)。
deepspeed train_cfm.py --config config.yaml
```

检查点写入 `checkpoints/{ast,llm,flow}/`,验证/可视化样本写入 `outputs_cfm/`。

### 复现诊断结果(图 1 与表 1)

在某个 **`Fusion_*` 变体中运行阶段 2**。在预热轮(warm-up epoch)期间,`LayerProbeLogger`
会逐 epoch 写入 `checkpoints/flow/`:

* `csv/probe_stats_epoch_XXXX.csv` —— 各层 LASP(Cos-SEM / Cos-EVT)存储分数。
* `csv/probe_stats_epoch_XXXX_foga.csv` —— 各层 FoG-A 因果归因分数。
* `image/..._foga_heatmap.png` —— SCD 时空热力图(图 1)。
* `image/..._importance.png`、`image/..._trend_top3.png` —— 逐层重要性 & Top-3 跨训练
  稳定性(表 7)。

从这些 CSV 中读出的 Top-K 层与 `λ_k`,正是被硬编码进 `REPA_*` 模型用于第二阶段的值。

---

## 8. 推理

端到端推理脚本位于 **`Fusion_single_codebook/`**。

```bash
cd Fusion_single_codebook

# 文本转语音(零样本,从参考片段克隆音色)
python inference_tts.py \
    --text "This is a classic line from Blade Runner." \
    --prompt_wav ./wav/english_male.flac \
    --checkpoint_dir ./checkpoints/flow \
    --cosyvoice_model_dir ./pretrained_models/CosyVoice-300M \
    --output ./output/generated_tts.wav --speed 0.9 --gpu_id 0

# 文本转音频(以事件标签 + 描述为条件的声音事件合成)
python inference_tta.py \
    --event "Dog barking" \
    --desc "A medium-sized dog barking repeatedly in a quiet room" \
    --prompt_wav ./wav/dog.wav \
    --llm_ckpt_dir ./checkpoints --flow_ckpt_dir ./checkpoints/flow \
    --output ./output/generated_tta.wav --gpu_id 0
```

> 若要使用 **AG-REPA** 模型合成,请将 `--checkpoint_dir` / `--flow_ckpt_dir` 指向
> `AG-REPA-Model/flow_matching/agrepa_*` 中的流匹配检查点。

---

## 9. 将模型权重与代码对接

从 Hugging Face 下载权重(`hf download AustinZhang/AG-REPA --local-dir AG-REPA-Model`),
并从上游来源下载基础模型。代码在**每个变体目录下**期望存在以下子目录——用软链接或复制方式接入:

```
<变体>/
├── pretrained_models/        ←  BEATs + CosyVoice(从上游下载 —— 详见模型卡)
│   ├── BEATs_iter3_plus_AS2M.pt
│   ├── CosyVoice-300M/
│   └── CosyVoice-ttsfrd/
└── checkpoints/
    ├── ast/                  ←  AG-REPA-Model/audioset_tokenizer/
    ├── llm/                  ←  AG-REPA-Model/llm/<single|dual>_codebook/
    └── flow/                 ←  AG-REPA-Model/flow_matching/agrepa_<single|dual>_codebook/
```

> Hugging Face 发布包仅含**最终 AG-REPA** 检查点(`agrepa_*`,对应 `REPA_*` 变体)。无对齐
> 基线与 FoG-A/LASP 诊断产物未打包——可通过训练 `Fusion_*` 变体从本代码生成。

示例:

```bash
cd REPA_single_codebook
ln -s /path/to/pretrained_base_models                            pretrained_models
mkdir -p checkpoints
ln -s /path/to/AG-REPA-Model/audioset_tokenizer                  checkpoints/ast
ln -s /path/to/AG-REPA-Model/llm/single_codebook                 checkpoints/llm
ln -s /path/to/AG-REPA-Model/flow_matching/agrepa_single_codebook checkpoints/flow
```

完整映射表与基础模型下载链接见 [🤗 模型卡](https://huggingface.co/AustinZhang/AG-REPA)。

---

## 10. 关键配置项(`config.yaml`)

| 区块 | 要点 |
|---------|-----------|
| `audio` | 16 kHz 输入,100 维梅尔;`vocos` 在 24 kHz 重新合成。 |
| `hyperparameters.ast` | `vocab_size: 4096`,重建 + VQ 提交(commitment)损失。 |
| `hyperparameters.llm` | `Qwen/Qwen3-0.6B-Base`;`style_num_tokens (K)=16`、`style_strength (α)=0.5`、`style_dropout_p=0.7`(CFG);`coarse_num_clusters=128`、`coarse_loss_weight=0.5`。 |
| `hyperparameters.flow` | 24 层 DiT,`hidden_dim 1024`、`n_heads 16`;教师蒸馏权重 `teacher_speech 0.5`、`teacher_audio 1.0`;`enable_whitebox_probe: true` 开启诊断;推理时 `n_timesteps 32`、`cfg_scale 3.0`。 |

---

## 11. 引用

```bibtex
@inproceedings{zhang2026agrepa,
  title     = {AG-REPA: Causal Layer Selection for Representation Alignment in Audio Flow Matching},
  author    = {Zhang, Pengfei and Xie, Tianxin and Yang, Minghao and Liu, Li},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

---

## 12. 致谢与第三方组件

本工作建立在多个开源项目之上,它们或被内置于各变体目录中,或作为教师/基础模型被引用:

* **CosyVoice**(Du et al., 2024)—— S³ 语音分词器 & 阶段 1 LLM 设计(`cosyvoice/`)—— <https://github.com/FunAudioLLM/CosyVoice>
* **BEATs**(Chen et al., 2022)—— 声学教师 & 音频 token 特征(`beats/`)—— <https://github.com/microsoft/unilm/tree/master/beats>
* **RepCodec**(Huang et al., 2024)—— AudioSet 分词器的 VQ-VAE 主干(`repcodec/`)—— <https://github.com/mct10/RepCodec>
* **Vocos**(Siuzdak, 2023)—— 梅尔谱声码器 —— <https://github.com/gemelo-ai/vocos>
* **Whisper**(Radford et al., 2022)—— 语音的语义教师 —— <https://github.com/openai/whisper>
* **Qwen3-0.6B-Base**(Yang et al., 2025)—— 阶段 1 LLM 初始化 —— <https://github.com/QwenLM/Qwen3>
* **MiDashengLM-7B**(小米)—— AudioSet 字幕生成(数据准备)—— <https://github.com/xiaomi-research/dasheng-lm>

---

## 13. 许可证

本仓库中 AG-REPA 专有代码以 **MIT 许可证**发布——见 [`LICENSE`](LICENSE)。

内置的第三方组件(`cosyvoice/`、`beats/`、`repcodec/`)及所引用的基础模型保留其**各自原始
许可证**;在再分发或商用前请查阅上方链接的上游仓库。

正如论文 Impact Statement(影响声明)所述,高保真音频生成与语音克隆存在风险(深度伪造、
冒充、语音欺骗)。负责任的部署应纳入音频水印、欺骗检测,并限制对语音克隆能力的访问。
