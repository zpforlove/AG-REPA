import logging
import math
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchdiffeq import odeint

from repcodec.modules.decoder import Decoder
from repcodec.modules.encoder import Encoder
from repcodec.modules.projector import Projector
from repcodec.modules.quantizer import Quantizer
from utils import timestep_embedding

# Set up the logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AudioSetTokenizer(nn.Module):
    """
    AudioSet Tokenizer (general-purpose audio tokenizer).
    This class implements an encoder-quantizer-decoder structure.
    """

    def __init__(self, input_dim, hidden_dim, vocab_size):
        """
        Initialize the AudioSetTokenizer.
        Args:
            input_dim (int): Dimension of the input features.
            hidden_dim (int): Dimension of the internal hidden layer, also the dimension of the codebook vectors.
            vocab_size (int): Size of the quantization codebook.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.encoder = Encoder(
            input_channels=input_dim,
            encode_channels=hidden_dim,
            channel_ratios=(1, 1),
            strides=(1, 1),
            kernel_size=3,
            bias=True,
            block_dilations=(1, 1)
        )
        self.projector = Projector(
            input_channels=hidden_dim,
            code_dim=hidden_dim,
            kernel_size=3,
            stride=1,
            bias=False
        )
        self.quantizer = Quantizer(
            code_dim=hidden_dim,
            codebook_num=1,
            codebook_size=vocab_size
        )
        self.decoder = Decoder(
            code_dim=hidden_dim,
            output_channels=input_dim,
            decode_channels=hidden_dim,
            channel_ratios=(1, 1),
            strides=(1, 1),
            kernel_size=3,
            bias=True,
            block_dilations=(1, 1),
            unit_kernel_size=3
        )

    def forward(self, x):
        """
        Forward pass.
        This function now computes and returns a dict containing the reconstruction loss and the VQ commitment loss.
        Args:
            x (torch.Tensor): Input feature tensor, expected shape (B, C, T),
                              where B is the batch size, C is the feature dimension (input_dim), T is the sequence length.
        Returns:
            dict: A dict containing the following keys:
                  'reconstruction_loss' (torch.Tensor): Mean squared error loss between the input and the reconstructed output.
                  'commit_loss' (torch.Tensor): VQ commitment loss.
        """
        try:
            # Encode -> project -> quantize
            encoded = self.encoder(x)
            proj_feat = self.projector(encoded)
            z_q, commit_loss, _ = self.quantizer(proj_feat)

            # Decode and reconstruct
            recon = self.decoder(z_q)

            # Compute reconstruction loss
            reconstruction_loss = F.mse_loss(recon, x, reduction='mean')

            # Check for NaN or Inf to prevent training from breaking
            if torch.isnan(reconstruction_loss) or torch.isinf(reconstruction_loss):
                logger.warning("Reconstruction loss is NaN or Inf. Setting it to 0.")
                reconstruction_loss = torch.tensor(0.0, device=x.device, requires_grad=True)
            if torch.isnan(commit_loss) or torch.isinf(commit_loss):
                was_requiring_grad = commit_loss.requires_grad
                logger.warning("Commitment loss is NaN or Inf. Setting it to 0.")
                commit_loss = torch.tensor(0.0, device=x.device, requires_grad=was_requiring_grad)

            # Return the loss dict in the required format
            return {
                'reconstruction_loss': reconstruction_loss,
                'commit_loss': commit_loss,
            }
        except Exception as e:
            logger.error(f"AudioSetTokenizer forward pass failed: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    def tokenize(self, x):
        """
        A convenience method dedicated to obtaining discrete tokens (indices).
        Args:
            x (torch.Tensor): Input feature tensor, shape (B, C, T).
        Returns:
            AudioSet Token (torch.Tensor): Discrete codebook indices, shape (B, T).
        """
        with torch.no_grad():
            encoded = self.encoder(x)
            proj_feat = self.projector(encoded)
            _, indices = self.quantizer.inference(proj_feat)
            # Handle the dimensionality of the AST token indices, ensuring they are 2D (batch_size, seq_len)
            tokens = indices.squeeze(0) if indices.dim() == 3 and indices.shape[0] == 1 else indices
        return tokens


class MelSpectrogramExtractor:
    """Mel-spectrogram extractor - configured to meet the requirements of the Vocos vocoder"""

    def __init__(self, config, target_device=None):
        """
        Args:
            config (dict): A dict containing the audio and vocoder configuration.
            target_device (str, optional): The desired compute device (e.g., "cuda:0", "cpu").
        """
        self.config = config
        self.vocos_config = config['vocos']  # Vocos vocoder-specific configuration

        # Mel-spectrogram parameters (obtained from the vocos config to ensure compatibility)
        self.mel_config = self.vocos_config['mel']
        self.n_fft = self.mel_config['n_fft']  # FFT window size
        self.hop_length = self.mel_config['hop_length']  # Hop length
        self.win_length = self.mel_config['win_length']  # Window length
        self.n_mels = self.mel_config['n_mels']  # Number of mel filterbanks
        self.f_min = self.mel_config['f_min']  # Minimum frequency
        self.f_max = self.mel_config['f_max']  # Maximum frequency
        self.center = self.mel_config['center']  # Whether to center the signal
        self.power = self.mel_config['power']  # Exponent of the magnitude spectrum (e.g., 2.0 for energy spectrum)

        # Resampling parameters
        self.input_sr = self.config['audio']['sample_rate']  # Sample rate of the input audio
        self.target_sr = self.vocos_config['sample_rate']  # Target sample rate expected by Vocos

        # Global statistics (used for mel-spectrogram normalization)
        self.mel_mean = self.config['hyperparameters']['flow']['mel_mean']  # Precomputed mel-spectrogram mean
        self.mel_std = self.config['hyperparameters']['flow']['mel_std']  # Precomputed mel-spectrogram standard deviation

        # Set up the device
        if target_device:
            self.device = torch.device(target_device)
        else:
            logger.warning("MelSpectrogramExtractor: target_device was not provided. Falling back to CPU.")
            self.device = torch.device('cpu')
        logger.info(f"MelSpectrogramExtractor instance will use device: {self.device}")

        # Initialize torchaudio's mel-spectrogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.target_sr,  # Target sample rate
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=self.center,
            power=self.power,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max
        ).to(self.device)  # Move the transform to the target device

    def __call__(self, waveform, normalize=True):
        """
        Replace the for loop with vectorized operations to improve efficiency and robustness.
        """
        # Ensure the input is on the correct device
        waveform = waveform.to(self.device)

        # If the input sample rate differs from the target sample rate, resample the entire batch
        if self.input_sr != self.target_sr:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=self.input_sr,
                new_freq=self.target_sr,
                lowpass_filter_width=self.vocos_config['resample']['lowpass_filter_width'],
                rolloff=self.vocos_config['resample']['rolloff'],
                resampling_method=self.vocos_config['resample']['method']
            )

        # --- Vectorized operations ---
        # 1. Apply the mel-spectrogram transform directly to the entire batch
        mel_specs = self.mel_transform(waveform)
        # 2. Take the log of the batch result and clamp it
        mel_specs = torch.log(torch.clamp(mel_specs, min=1e-5))

        # If requested, normalize using the global statistics
        if normalize:
            mel_specs = (mel_specs - self.mel_mean) / self.mel_std

        return mel_specs.to(self.device)  # Ensure the final output is on the target device


class DiTBlock(nn.Module):
    """
    DiT block (Diffusion Transformer Block), adapted from the DiT paper, using adaLN-Zero style modulation.
    A DiT block typically contains a self-attention layer and a feed-forward network layer, both with residual connections, and modulation is applied before the inputs are fed into these layers.
    The modulation parameters are derived from the time embedding.
    """

    def __init__(self, dim, heads=16, ff_mult=4, dropout=0.2):
        """
        Args:
            dim (int): Feature dimension of the input and output.
            heads (int, optional): Number of heads in the multi-head attention. Defaults to 16.
            ff_mult (int, optional): Multiplier of the feed-forward network hidden dimension relative to the input dimension. Defaults to 4.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
        super().__init__()
        assert dim % heads == 0, f"dim ({dim}) must be divisible by heads ({heads})"
        # The LayerNorm eps is usually a small value to improve numerical stability, e.g. 1e-5 or 1e-6
        # elementwise_affine=False means no learnable affine parameters (gamma, beta) are used, since these will be provided by the time modulation
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # Normalization before the attention layer
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)  # Normalization before the feed-forward network
        # Multi-head self-attention layer
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)

        ff_hidden_dim = int(dim * ff_mult)  # Hidden dimension of the feed-forward network
        # Feed-forward network (usually a two-layer linear network with an activation function in between)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim),  # First linear layer
            nn.GELU(),  # GELU is a commonly used activation function in Transformers
            nn.Dropout(dropout),  # Dropout
            nn.Linear(ff_hidden_dim, dim),  # Second linear layer (output dimension back to dim)
            nn.Dropout(dropout)  # Dropout
        )
        # Time modulation network (Time MLP), used to generate modulation parameters (scale, shift, gate, etc.) from the time embedding
        # adaLN-Zero requires 6 modulation parameters (gamma1_attn, beta1_attn, gamma2_ffn, beta2_ffn, alpha_scale_attn, alpha_scale_ffn)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),  # SiLU (Swish) activation function, applied to the input time embedding
            nn.Linear(dim, 6 * dim)  # Linear layer mapping the time embedding to 6x the dimension
        )
        self._init_weights()  # Call the weight initialization method

    def _init_weights(self):
        """Initialize the weights of specific layers, which helps stabilize training. The DiT paper usually initializes the last linear layer on the residual connection path to 0."""

        def _init_module_weights(m, gain=1.0, zero_out_last_linear=False):
            """Helper function for initializing the weights of a single module."""
            if isinstance(m, nn.Linear):
                if zero_out_last_linear:  # For the last linear layer of a residual block, usually initialized to 0
                    nn.init.zeros_(m.weight)
                else:
                    # Xavier uniform initialization with the gain parameter, which helps preserve signal variance
                    torch.nn.init.xavier_uniform_(m.weight, gain=gain)
                if m.bias is not None:  # The bias is usually initialized to 0
                    torch.nn.init.zeros_(m.bias)

        # Initialize the attention output projection layer (usually the output projection weights are initialized to 0, since it is part of the residual path)
        if hasattr(self.attn, 'out_proj'):
            _init_module_weights(self.attn.out_proj, zero_out_last_linear=True)

        # Initialize the linear layers of the feed-forward network
        _init_module_weights(self.ff[0])  # The first linear layer in the feed-forward network (dimension expansion)
        if hasattr(self.ff[3], 'weight'):  # The second linear layer in the feed-forward network (compressed back to dim, part of the residual path)
            _init_module_weights(self.ff[3], zero_out_last_linear=True)

        # Initialize the linear layer of the time modulation network
        if hasattr(self.time_mlp[1], 'weight'):  # The second element in time_mlp is the linear layer
            nn.init.normal_(self.time_mlp[1].weight, std=0.02)
            if self.time_mlp[1].bias is not None:
                nn.init.zeros_(self.time_mlp[1].bias)

    def forward(self, x, t_emb):  # x: [B, T_seq, HiddenDim], t_emb: [B, HiddenDim]
        """
        Forward pass of the DiT block, using adaLN-Zero style modulation.
        """
        # 1. Generate modulation parameters from the time embedding
        time_params = self.time_mlp(t_emb)
        gamma1_attn, beta1_attn, gamma2_ffn, beta2_ffn, alpha_scale_attn, alpha_scale_ffn = \
            [p.unsqueeze(1) for p in torch.chunk(time_params, 6, dim=1)]

        # 2. First sub-block: self-attention + adaLN-Zero
        normed_x_for_attention = self.norm1(x)
        modulated_x_for_attention = normed_x_for_attention * (1 + gamma1_attn) + beta1_attn
        attention_output_raw, _ = self.attn(query=modulated_x_for_attention,
                                            key=modulated_x_for_attention,
                                            value=modulated_x_for_attention)
        scaled_attention_output = alpha_scale_attn * attention_output_raw
        x_after_attention_block = x + scaled_attention_output

        # 3. Second sub-block: feed-forward network + adaLN-Zero
        normed_x_for_feedforward = self.norm2(x_after_attention_block)
        modulated_x_for_feedforward = normed_x_for_feedforward * (1 + gamma2_ffn) + beta2_ffn
        feedforward_output_raw = self.ff(modulated_x_for_feedforward)
        scaled_feedforward_output = alpha_scale_ffn * feedforward_output_raw
        x_after_feedforward_block = x_after_attention_block + scaled_feedforward_output
        return x_after_feedforward_block


class FlowMatchingModel(nn.Module):
    """
    Unified Flow Matching model (with REPA-style in-layer bypass alignment, dual teachers: Whisper/BEATs)
      - Backbone: DiT + FM velocity field loss
      - Input-side distillation: retains the main Flow Matching loss and input-side distillation
      - Bypass distillation: capture pooled vectors at the specified layers, project them into the teacher space via an MLP, and minimize the negative cosine
    """

    def __init__(self, config, teacher_dim_speech: int = 0, teacher_dim_audio: int = 0):
        super().__init__()
        self.config = config
        h = config['hyperparameters']['flow']

        # ===== Basic shape parameters =====
        self.n_mels = config['vocos']['mel']['n_mels']
        self.ref_mel_dim = self.n_mels
        hidden_dim = h['hidden_dim']
        self.sigma = h['sigma']
        self.cond_drop_prob = h['cond_drop_prob']
        self.drop_ref_prob = h.get('drop_ref_prob', 0.0)
        self.n_timesteps = h['n_timesteps']
        self.cfg_scale = h['cfg_scale']

        D_tok = h['token_embedding_dim']
        vs3 = h['s3_vocab_size']
        vas = h.get('as_vocab_size', vs3)

        # ===== Per-vocabulary embeddings + linear projection to a shared space =====
        self.token_embedding_s3 = nn.Embedding(vs3, D_tok)
        self.token_embedding_as = nn.Embedding(vas, D_tok)

        self.P_s3 = nn.Linear(D_tok, D_tok, bias=False)
        self.P_as = nn.Linear(D_tok, D_tok, bias=False)
        with torch.no_grad():
            self.P_s3.weight.copy_(torch.eye(D_tok))
            self.P_as.weight.copy_(torch.eye(D_tok))

        # Condition-side fusion: concatenate [token_shared | ref_mel], then feed into the backbone together with x
        self.max_fused_embed_dim = D_tok
        self.input_proj = nn.Linear(self.n_mels + self.max_fused_embed_dim + self.ref_mel_dim, hidden_dim)

        # ===== Time embedding + DiT backbone + output layer =====
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Linear(4 * hidden_dim, hidden_dim),
        )
        self.blocks = nn.ModuleList([
            DiTBlock(dim=hidden_dim,
                     heads=h['n_heads'],
                     ff_mult=h['ff_mult'],
                     dropout=h['dropout'])
            for _ in range(h['n_layers'])
        ])
        self.output_layer = nn.Sequential(
            nn.LayerNorm(hidden_dim, eps=1e-6, elementwise_affine=True),
            nn.Linear(hidden_dim, self.n_mels)
        )

        max_len = h.get('max_pos_emb_len', 8192)
        pe = self._get_sinusoidal_embedding(max_len, hidden_dim)
        self.register_buffer('pos_emb', pe)

        # ===== Input-side distillation projections =====
        self.teacher_proj_speech = (
            nn.Linear(D_tok, teacher_dim_speech, bias=False) if teacher_dim_speech > 0 else None
        )
        self.teacher_proj_audio = (
            nn.Linear(D_tok, teacher_dim_audio, bias=False) if teacher_dim_audio > 0 else None
        )
        with torch.no_grad():
            if self.teacher_proj_speech is not None:
                nn.init.orthogonal_(self.teacher_proj_speech.weight, gain=1.0)
            if self.teacher_proj_audio is not None:
                nn.init.orthogonal_(self.teacher_proj_audio.weight, gain=1.0)

        lw = h.get('loss_weights', {})
        self.w_teacher_speech = float(lw.get('teacher_speech', 0.10))
        self.w_teacher_audio = float(lw.get('teacher_audio', 0.10))

        # ====== REPA layer-wise bypass: layer set & λ (FoG-A×2) ======
        # 1-based -> 0-based
        self.repa_layers_speech = [0, 8, 4]  # L1=0, L9=8, L5=4
        self.repa_layers_audio = [0, 20, 8]  # L1=0, L21=20, L9=8
        # λ per layer
        self.repa_lambda_speech = {0: 0.334, 8: 0.139, 4: 0.1176}
        self.repa_lambda_audio = {0: 0.278, 20: 0.12, 8: 0.1122}

        # A separate set of MLP heads for each layer: hidden_dim -> teacher_dim
        def _make_mlp_head(out_dim: int):
            # A concise two-layer MLP; REPA suggests using an MLP for h_φ(·); no activation at the end so it can be cosine-aligned with the teacher
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )

        self.repa_heads_speech = nn.ModuleDict()
        self.repa_heads_audio = nn.ModuleDict()
        if teacher_dim_speech > 0:
            for li in self.repa_layers_speech:
                self.repa_heads_speech[str(li)] = _make_mlp_head(teacher_dim_speech)
        if teacher_dim_audio > 0:
            for li in self.repa_layers_audio:
                self.repa_heads_audio[str(li)] = _make_mlp_head(teacher_dim_audio)

        # Initialization: apply reasonable initialization to the linear layers
        for md in (self.repa_heads_speech, self.repa_heads_audio):
            for _k, mlp in md.items():
                for m in mlp:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

    # === Positional encoding ===
    def _get_sinusoidal_embedding(self, max_len, dim):
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        emb = torch.zeros(max_len, dim)
        emb[:, 0::2] = torch.sin(pos * div)
        emb[:, 1::2] = torch.cos(pos * div)
        return emb

    # === Velocity field prediction ===
    def _predict_velocity(
            self,
            xt: torch.Tensor,
            t: torch.Tensor,
            cond_with_ref_mel: torch.Tensor,
            capture_layers: "set[int] | None" = None,
            return_layer_pooled: bool = False,
            skip_layers: "set[int] | None" = None
    ):
        """
        Velocity field prediction (optionally returns each layer's "temporally average-pooled vector" for use in bypass alignment).
        Returns:
          return_layer_pooled=False : v ∈ [B, n_mels, T]
          return_layer_pooled=True  : (v, pooled_dict{layer_idx: [B, hidden_dim]})
        """
        B, _, T = xt.shape
        x_f = xt.permute(0, 2, 1)  # [B, T, Cx]
        c_f = cond_with_ref_mel.permute(0, 2, 1)  # [B, T, Cc]
        h = torch.cat([x_f, c_f], dim=-1)  # [B, T, Cx+Cc]
        h = self.input_proj(h)
        if T <= self.pos_emb.size(0):
            h = h + self.pos_emb[:T].unsqueeze(0)
        else:
            h = h + self.pos_emb.unsqueeze(0)

        time_emb_dim = h.shape[-1]
        te = timestep_embedding(t, time_emb_dim)
        te = self.time_embed(te)

        pooled_dict = {}
        skip_layers = skip_layers or set()
        for li, block in enumerate(self.blocks):
            if li in skip_layers:
                if return_layer_pooled and (capture_layers is None or li in capture_layers):
                    pooled_dict[li] = h.mean(dim=1)
                continue
            h = block(h, te)
            if return_layer_pooled and (capture_layers is None or li in capture_layers):
                pooled_dict[li] = h.mean(dim=1)  # [B, hidden_dim]

        v = self.output_layer(h)  # [B, T, n_mels]
        v = v.permute(0, 2, 1)  # [B, n_mels, T]
        if return_layer_pooled:
            return v, pooled_dict
        return v

    # ===== Unified interface for per-vocabulary embedding + projection =====
    @torch.no_grad()
    def embed_and_project_tokens(self, domain: str, token_indices: torch.LongTensor):
        """
        domain: 's3' or 'as'
        token_indices: [N, L] (Long)
        return: shared token embedding after projection, [N, D_tok, L]
        """
        assert domain in ('s3', 'as')
        if token_indices.numel() == 0:
            N = token_indices.shape[0]
            D = self.token_embedding_s3.embedding_dim
            return torch.zeros(N, D, 0, device=token_indices.device)

        if domain == 's3':
            emb = self.token_embedding_s3(token_indices)
            emb = self.P_s3(emb)
        else:
            emb = self.token_embedding_as(token_indices)
            emb = self.P_as(emb)
        return emb.permute(0, 2, 1).contiguous()

    # ===== Forward: FM main loss + input-side distillation + REPA layer bypass distillation =====
    def forward(self, x, t, cond_embed_dict):
        """
        x: [B, n_mels, T_full]
        t: [B]
        cond_embed_dict:
          'fused_embed'           : [B, D_tok, T]
          'ref_mel_for_cond'      : [B, n_mels, T]
          'domain_ids'            : [B] (0=LibriSpeech, 1=AudioSet)
          'teacher_speech_global' : [B, C_w] or None
          'teacher_audio_global'  : [B, C_a] or None
        return: per-sample total loss [B]
        """
        B, _, _ = x.shape
        device = x.device

        fused_embed = cond_embed_dict['fused_embed']
        ref_mel_for_cond = cond_embed_dict['ref_mel_for_cond']

        # Condition dropout
        mask_fused = (torch.rand(B, device=device) > self.cond_drop_prob).float().view(B, 1, 1)
        masked_fused_embed = fused_embed * mask_fused
        if self.training and self.drop_ref_prob > 0.0:
            mask_ref = (torch.rand(B, device=device) > self.drop_ref_prob).float().view(B, 1, 1)
            ref_mel_for_cond = ref_mel_for_cond * mask_ref

        final_cond = torch.cat([masked_fused_embed, ref_mel_for_cond], dim=1)

        # FM construction path
        noise = torch.randn_like(x) * self.sigma
        tb = t.view(B, 1, 1)
        xt = (1 - tb) * noise + tb * x

        # Only capture the layers needed for REPA to reduce overhead
        repa_capture = set(self.repa_layers_speech) | set(self.repa_layers_audio)
        pred_v, pooled = self._predict_velocity(
            xt, t, final_cond, capture_layers=repa_capture, return_layer_pooled=True
        )

        # ===== FM main loss =====
        target_v = x - noise
        fm_loss = F.mse_loss(pred_v, target_v, reduction='none').mean(dim=[1, 2])  # [B]
        total_loss = fm_loss.clone()

        # ===== Input-side distillation =====
        dom_ids = cond_embed_dict.get('domain_ids', None)
        if dom_ids is not None:
            pooled_input = masked_fused_embed.mean(dim=2)
            pooled_input = F.layer_norm(pooled_input, normalized_shape=(pooled_input.shape[-1],))
            cond_kept_mask = mask_fused.squeeze().float()
            used_speech_input = False
            used_audio_input = False

            # Speech → Whisper
            ts = cond_embed_dict.get('teacher_speech_global', None)
            if self.teacher_proj_speech is not None and ts is not None and self.w_teacher_speech > 0.0:
                proj_s = self.teacher_proj_speech(pooled_input)
                cos_s = F.cosine_similarity(F.normalize(proj_s, dim=-1),
                                            F.normalize(ts.to(device), dim=-1), dim=-1)
                ls_mask = (dom_ids == 0).float() * cond_kept_mask
                if ls_mask.any():
                    teacher_loss_s = (1.0 - cos_s) * ls_mask
                    total_loss = total_loss + self.w_teacher_speech * teacher_loss_s
                    used_speech_input = True

            # Audio → BEATs
            ta = cond_embed_dict.get('teacher_audio_global', None)
            if self.teacher_proj_audio is not None and ta is not None and self.w_teacher_audio > 0.0:
                proj_a_input = self.teacher_proj_audio(pooled_input)
                cos_a_input = F.cosine_similarity(F.normalize(proj_a_input, dim=-1),
                                                  F.normalize(ta.to(device), dim=-1), dim=-1)
                as_mask = (dom_ids == 1).float() * cond_kept_mask
                if as_mask.any():
                    teacher_loss_a_input = (1.0 - cos_a_input) * as_mask
                    total_loss = total_loss + self.w_teacher_audio * teacher_loss_a_input
                    used_audio_input = True

            # keepalive (avoids a head having no gradient graph when there are no samples from that domain)
            if (self.teacher_proj_speech is not None) and (not used_speech_input):
                total_loss = total_loss + (self.teacher_proj_speech.weight.sum() * 0.0)
            if (self.teacher_proj_audio is not None) and (not used_audio_input):
                total_loss = total_loss + (self.teacher_proj_audio.weight.sum() * 0.0)

        # ===== REPA layer-wise bypass distillation =====
        # REPA objective: maximize the similarity between the teacher y* and h_φ(h_t); here we use (1 - cosine) as the loss and add it to the main loss weighted by λ
        # Only compute the layer-wise alignment with the corresponding teacher for the samples of each domain

        # +++ KEEPALIVE trackers +++
        used_repa_speech_layers = set()
        used_repa_audio_layers = set()

        if dom_ids is not None and (len(self.repa_heads_speech) > 0 or len(self.repa_heads_audio) > 0):
            ts = cond_embed_dict.get('teacher_speech_global', None)
            ta = cond_embed_dict.get('teacher_audio_global', None)

            for li, vec in pooled.items():  # vec: [B, hidden_dim]
                # Normalize (REPA's sim(·) suggests normalizing the representations before computing similarity)
                vec_n = F.normalize(vec, dim=-1)

                # —— Speech bypass (only for dom==0) ——
                if str(li) in self.repa_heads_speech and ts is not None:
                    mask_sem = (dom_ids == 0)
                    if mask_sem.any():
                        z_sem = self.repa_heads_speech[str(li)](vec_n[mask_sem])  # [B_s, C_w]
                        cos_sem = F.cosine_similarity(F.normalize(z_sem, dim=-1),
                                                      F.normalize(ts[mask_sem].to(device), dim=-1), dim=-1)
                        repa_loss_sem = (1.0 - cos_sem) * self.repa_lambda_speech.get(int(li), 0.0)
                        # Scatter back into total_loss per sample
                        idx = torch.nonzero(mask_sem, as_tuple=True)[0]
                        total_loss[idx] = total_loss[idx] + repa_loss_sem

                        used_repa_speech_layers.add(str(li))  # +++ mark as used +++

                # —— Audio bypass (only for dom==1) ——
                if str(li) in self.repa_heads_audio and ta is not None:
                    mask_evt = (dom_ids == 1)
                    if mask_evt.any():
                        z_evt = self.repa_heads_audio[str(li)](vec_n[mask_evt])  # [B_a, C_a]
                        cos_evt = F.cosine_similarity(F.normalize(z_evt, dim=-1),
                                                      F.normalize(ta[mask_evt].to(device), dim=-1), dim=-1)
                        repa_loss_evt = (1.0 - cos_evt) * self.repa_lambda_audio.get(int(li), 0.0)
                        idx = torch.nonzero(mask_evt, as_tuple=True)[0]
                        total_loss[idx] = total_loss[idx] + repa_loss_evt

                        used_repa_audio_layers.add(str(li))  # +++ mark as used +++

            # +++ KEEPALIVE check (fixes the multi-GPU hang issue) +++
            # Check the Speech REPA heads
            for li_key in self.repa_heads_speech:  # li_key is a str
                if li_key not in used_repa_speech_layers:
                    # This head was not used in this batch, so add a dummy loss
                    dummy_loss = 0.0 * sum(p.sum() for p in self.repa_heads_speech[li_key].parameters())
                    total_loss = total_loss + dummy_loss

            # Check the Audio REPA heads
            for li_key in self.repa_heads_audio:  # li_key is a str
                if li_key not in used_repa_audio_layers:
                    # This head was not used in this batch, so add a dummy loss
                    dummy_loss = 0.0 * sum(p.sum() for p in self.repa_heads_audio[li_key].parameters())
                    total_loss = total_loss + dummy_loss

        return total_loss

    # ===== ODE inference =====
    def _ode_func(self, t_scalar, x_current, cond_embed_dict_full, cfg_scale):
        B = x_current.shape[0]
        current_T_duration = x_current.shape[-1]
        device = x_current.device

        t_batch = torch.full((B,), t_scalar, device=device, dtype=torch.float32)

        fused_embed_full = cond_embed_dict_full['fused_embed']
        ref_mel_for_cond_full = cond_embed_dict_full['ref_mel_for_cond']

        # Truncate / zero-pad to match the current length
        if fused_embed_full.shape[-1] < current_T_duration:
            fused_effective = F.pad(fused_embed_full, (0, current_T_duration - fused_embed_full.shape[-1]))
        else:
            fused_effective = fused_embed_full[:, :, :current_T_duration]
        if ref_mel_for_cond_full.shape[-1] < current_T_duration:
            ref_mel_effective = F.pad(ref_mel_for_cond_full, (0, current_T_duration - ref_mel_for_cond_full.shape[-1]))
        else:
            ref_mel_effective = ref_mel_for_cond_full[:, :, :current_T_duration]

        final_cond_effective = torch.cat([fused_effective, ref_mel_effective], dim=1)

        v_cond = self._predict_velocity(x_current, t_batch, final_cond_effective)
        if cfg_scale > 1e-8:
            null_fused_cond = torch.zeros_like(fused_effective)
            null_cond_with_ref = torch.cat([null_fused_cond, ref_mel_effective], dim=1)
            v_uncond = self._predict_velocity(x_current, t_batch, null_cond_with_ref)
            effective_velocity = v_uncond + cfg_scale * (v_cond - v_uncond)
        else:
            effective_velocity = v_cond
        return effective_velocity

    @torch.no_grad()
    def sample(self, cond_embed_dict, *, target_duration_frames=None, steps=None, cfg_scale=None, method="euler",
               sway_sampling_coef=None):
        """
        Inference interface
        cond_embed_dict: 'fused_embed': [B, D, T_cond], 'ref_mel_for_cond': [B, n_mels, T_cond]
        """
        self.eval()
        device = next(self.parameters()).device

        fused_embed = cond_embed_dict['fused_embed']
        B, _, T_cond_full = fused_embed.shape

        steps_to_take = steps if steps is not None else self.n_timesteps
        current_cfg_scale = cfg_scale if cfg_scale is not None else self.cfg_scale

        max_supported_len = self.pos_emb.size(0)
        if target_duration_frames is None:
            actual_duration = T_cond_full
        else:
            actual_duration = target_duration_frames
        if actual_duration > max_supported_len:
            actual_duration = max_supported_len

        x0 = torch.randn(B, self.n_mels, actual_duration, device=device) * self.sigma

        t_span = torch.linspace(0., 1., steps_to_take, device=device)
        if sway_sampling_coef is not None:
            t_span = t_span + sway_sampling_coef * (torch.cos(math.pi / 2 * t_span) - 1 + t_span)

        solution_trajectory = odeint(
            lambda t_scalar, x_current: self._ode_func(t_scalar, x_current, cond_embed_dict, current_cfg_scale),
            x0,
            t_span,
            method=method
        )
        final_mel = solution_trajectory[-1]
        return final_mel
