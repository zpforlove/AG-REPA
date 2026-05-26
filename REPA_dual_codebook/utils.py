import logging
import math
import sys

import torch
import yaml

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - UTILS - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def load_config(config_path):
    """Load and validate the configuration file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required configuration keys
    required_keys = ['data', 'audio', 'hyperparameters', 'vocos', 'logging', 'paths']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    return config


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Convert timesteps into sinusoidal positional embeddings, encoding the global
    time information of the "denoising progress".
    Args:
        timesteps: timestep tensor [B,]
        dim: output embedding dimension
        max_period: maximum period length
    Returns:
        embedding vector [B, dim]
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def peak_norm(wav: torch.Tensor, peak: float = 0.99, eps: float = 1e-9) -> torch.Tensor:
    """
    Apply peak normalization to an audio waveform.

    Parameters
    ----------
    wav : torch.Tensor
        Input waveform tensor, whose shape can be (T), (1, T), or (C, T);
        values typically lie within [-1, 1] or a relatively small amplitude range.
    peak : float
        The desired "maximum absolute amplitude" after normalization.
        Using 0.99 leaves a small margin below full scale 1.0 to avoid
        clipping when writing PCM-16/PCM-24.
    eps : float
        A tiny constant that prevents a zero denominator on silent segments
        (max==0) and also improves numerical stability.

    Returns
    -------
    torch.Tensor
        The normalized waveform, with a tensor shape identical to the input.
    """

    # 1) Compute the peak (maximum absolute amplitude) of the current waveform.
    #    .abs() takes the absolute value first; .max() returns a scalar tensor.
    current_peak = wav.abs().max()

    # 2) Compute the scaling factor.
    #    If current_peak is very small (close to 0), this yields a
    #    large scale; eps guards against division by zero.
    scale = peak / (current_peak + eps)

    # 3) Apply the same scaling factor to the whole waveform, scaling it up or down linearly.
    wav_norm = wav * scale

    # 4) Return the processed waveform.
    return wav_norm
