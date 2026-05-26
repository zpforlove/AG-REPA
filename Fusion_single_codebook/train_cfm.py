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
import matplotlib
import torch
import torch.nn.functional as F
import torchaudio
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import ConcatDataset, DataLoader, DistributedSampler
from torch.utils.data.dataloader import default_collate
from vocos import Vocos

import wandb
from beats.BEATs import BEATs, BEATsConfig
from data_loader import AudioSetDataset, LibriSpeechDataset
from models import AudioSetTokenizer, FlowMatchingModel, MelSpectrogramExtractor
from utils import load_config, peak_norm

# Global variable declaration
global logger


def setup_logger_ds(rank=-1, config=None, level_override=None):
    current_logger = logging.getLogger("train_cfm_script")
    if current_logger.hasHandlers():
        current_logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    log_format = (
        f"%(asctime)s - RANK {rank} - %(module)s - %(levelname)s - %(message)s"
        if rank != -1
        else "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
    )
    formatter = logging.Formatter(log_format)
    handler.setFormatter(formatter)
    current_logger.addHandler(handler)
    log_level_str = "INFO"
    if (
            config
            and "logging" in config
            and isinstance(config["logging"], dict)
            and "console" in config["logging"]
            and isinstance(config["logging"]["console"], dict)
            and "level" in config["logging"]["console"]
    ):
        log_level_str = config["logging"]["console"]["level"].upper()
    if level_override:
        log_level_str = level_override.upper()
    log_level_resolved = getattr(logging, log_level_str, logging.INFO)
    current_logger.setLevel(log_level_resolved if rank <= 0 else logging.WARNING)
    current_logger.propagate = False
    return current_logger


def custom_collate_fn(batch):
    """
    Custom collate function:
    1. Fixed-length fields (waveform, mel_lengths, etc.) -> stacked into a Tensor using default_collate
    2. Variable-length fields (s3_token) -> kept as a List[Tensor] to avoid resize errors
    """
    if not batch:
        return {}

    output = {}
    # Get the keys of the first sample
    keys = batch[0].keys()

    for key in keys:
        if key == 's3_token':
            # Variable-length field: do not stack, return the list directly; padding is handled later in prepare_batch
            output[key] = [sample[key] for sample in batch]
        else:
            # Fixed-length field: collect the list and stack using default_collate
            items = [sample[key] for sample in batch]
            output[key] = default_collate(items)

    return output


class LayerProbeLogger:
    """
    White-box probe logger (enabled only on Rank 0):
      - Alternate between probing "odd/even layers" (based on the parity of global_step)
      - CSV: one file per epoch, saved in <flow>/csv/
      - Plotting: only plot the figure for an epoch "at the end of each epoch"; plot the "overall comparison figure" at the end of training
      - Printing: print a Top-3 summary every 200 steps (three significant digits)
    """

    def __init__(self, num_layers: int, csv_path: "Path | str", enabled: bool = True):
        self.enabled = bool(enabled)
        self.num_layers = int(num_layers)

        # Base path and derived directories
        base_path = Path(csv_path)  # e.g., <...>/flow/probe_stats.csv
        flow_dir = base_path.parent  # e.g., <...>/flow
        self.csv_dir = flow_dir / "csv"
        self.img_dir = flow_dir / "image"
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.img_dir.mkdir(parents=True, exist_ok=True)

        # Base filename and suffix (used for naming the epoch files)
        self.base_name = base_path.stem  # "probe_stats"
        self.suffix = base_path.suffix or ".csv"

        # Cumulative statistics over the whole training run (used for printing Top-3 and the final overall comparison figure)
        self.sem_sum = [0.0] * self.num_layers
        self.evt_sum = [0.0] * self.num_layers
        self.counts = [0] * self.num_layers
        # Global time series (aggregated across epochs; used for the overall-trend Top-3 plot)
        self.series_sem = {
            i: [] for i in range(self.num_layers)
        }  # [(global_step, value), ...]
        self.series_evt = {i: [] for i in range(self.num_layers)}

        # Record the "last epoch seen" (used to trigger plotting the "previous epoch" on the first step of a new epoch)
        self._last_epoch_seen: int | None = None
        # Set of epochs that have already been plotted (to avoid duplicate plotting)
        self._finalized_epochs: set[int] = set()

    # ----------------------------- Sampling-layer selection -----------------------------
    def layers_to_probe(self, global_step: int) -> list[int]:
        if global_step % 2 == 0:
            return list(range(0, self.num_layers, 2))
        else:
            return list(range(1, self.num_layers, 2))

    # ----------------------------- Per-step update & conditional printing -----------------------------
    def update(
            self,
            step: int,
            epoch: int,
            results: dict[int, dict],
            log_to_console: bool = True,
    ):
        logger = logging.getLogger(__name__)

        # 0) Write to this epoch's CSV (if writing to this epoch for the first time, write the header first)
        epoch_csv = self.csv_dir / f"{self.base_name}_epoch_{epoch:04d}{self.suffix}"
        if not epoch_csv.exists():
            with open(epoch_csv, "w", encoding="utf-8") as f:
                f.write(
                    "step,epoch,layer,sem_sim,evt_sim\n"
                )

        # ------ Grab the two fused-based batch-level metrics from this batch's results and cache them on the instance ------
        sem_fused_current, evt_fused_current = None, None
        for _li, _entry in results.items():
            if isinstance(_entry, dict):
                if ("sem_fused" in _entry) and (_entry["sem_fused"] is not None):
                    try:
                        sem_fused_current = float(_entry["sem_fused"])
                    except Exception:
                        sem_fused_current = None
                if ("evt_fused" in _entry) and (_entry["evt_fused"] is not None):
                    try:
                        evt_fused_current = float(_entry["evt_fused"])
                    except Exception:
                        evt_fused_current = None
            if (sem_fused_current is not None) or (evt_fused_current is not None):
                break
        self._last_sem_fused = sem_fused_current
        self._last_evt_fused = evt_fused_current
        # ---------------------------------------------------------------------

        lines = []
        # 1) Write to the CSV, and simultaneously update the "whole-training-run" cumulative statistics and the global time series
        for li, entry in results.items():
            sem = entry.get("sem", None)
            evt = entry.get("evt", None)

            if sem is not None:
                v = float(sem)
                self.sem_sum[li] += v
                self.counts[li] += 1
                self.series_sem[li].append((step, v))
            if evt is not None:
                v = float(evt)
                self.evt_sum[li] += v
                self.counts[li] += 1
                self.series_evt[li].append((step, v))

            # Record to the CSV step by step
            lines.append(
                f"{step},{epoch},{li},{'' if sem is None else sem},{'' if evt is None else evt}\n"
            )

        if lines:
            with open(epoch_csv, "a", encoding="utf-8") as f:
                f.writelines(lines)

        # 2) On the first step of a new epoch, plot the "previous epoch"
        self._maybe_finalize_previous_epoch(current_epoch=epoch)

    def update_foga(
            self,
            step: int,
            epoch: int,
            foga_results: "dict[int, dict]",
            log_to_console: bool = True,
    ):
        if not self.enabled or not foga_results:
            return

        # 1) Append step by step to "this epoch's FoG-A CSV"
        epoch_csv = self.csv_dir / f"{self.base_name}_epoch_{epoch:04d}_foga.csv"
        if not epoch_csv.exists():
            with open(epoch_csv, "w", encoding="utf-8") as f:
                f.write("step,epoch,layer,speech_foga,audio_foga\n")
        lines = []
        for li, vals in sorted(foga_results.items(), key=lambda kv: kv[0]):
            sp = vals.get("speech", None)
            au = vals.get("audio", None)
            lines.append(
                f"{step},{epoch},{li},{'' if sp is None else sp},{'' if au is None else au}\n"
            )
        if lines:
            with open(epoch_csv, "a", encoding="utf-8") as f:
                f.writelines(lines)

        # 2) Maintain running means internally for the Top-3
        if not hasattr(self, "_foga_sp_sum"):
            self._foga_sp_sum = [0.0] * self.num_layers
            self._foga_sp_cnt = [0] * self.num_layers
            self._foga_au_sum = [0.0] * self.num_layers
            self._foga_au_cnt = [0] * self.num_layers

        for li, vals in foga_results.items():
            if vals.get("speech", None) is not None:
                self._foga_sp_sum[li] += float(vals["speech"])
                self._foga_sp_cnt[li] += 1
            if vals.get("audio", None) is not None:
                self._foga_au_sum[li] += float(vals["audio"])
                self._foga_au_cnt[li] += 1

        # 3) Print a joint table every 200 steps
        if log_to_console and (step % 200 == 0):
            logger = logging.getLogger(__name__)

            def _means_valid(sum_arr, cnt_arr):
                return [
                    (i, sum_arr[i] / cnt_arr[i])
                    for i in range(self.num_layers)
                    if cnt_arr[i] > 0
                ]

            def _top3(pairs):
                return sorted(pairs, key=lambda x: x[1], reverse=True)[:3]

            def _fmt3(x):
                try:
                    return f"{float(x):.3g}"
                except:
                    return str(x)

            # --- Reuse the original _log_top3 fused-metric printing ---
            fused_parts = []
            if hasattr(self, "_last_sem_fused") and (self._last_sem_fused is not None):
                fused_parts.append(f"SEM_fused: {_fmt3(self._last_sem_fused)}")
            if hasattr(self, "_last_evt_fused") and (self._last_evt_fused is not None):
                fused_parts.append(f"EVT_fused: {_fmt3(self._last_evt_fused)}")
            fused_suffix = "" if not fused_parts else " | " + " | ".join(fused_parts)

            # Cosine Top-3
            cos_sem = _means_valid(self.sem_sum, self.counts)
            cos_evt = _means_valid(self.evt_sum, self.counts)
            cos_sem_top3 = _top3(cos_sem)
            cos_evt_top3 = _top3(cos_evt)

            # FoG-A Top-3
            foga_sp = _means_valid(self._foga_sp_sum, self._foga_sp_cnt)
            foga_au = _means_valid(self._foga_au_sum, self._foga_au_cnt)
            foga_sp_top3 = _top3(foga_sp)
            foga_au_top3 = _top3(foga_au)

            # Joint-table printing
            rows = []
            for k in range(3):
                cs = (
                    f"L{cos_sem_top3[k][0] + 1}={_fmt3(cos_sem_top3[k][1])}"
                    if k < len(cos_sem_top3)
                    else "-"
                )
                ce = (
                    f"L{cos_evt_top3[k][0] + 1}={_fmt3(cos_evt_top3[k][1])}"
                    if k < len(cos_evt_top3)
                    else "-"
                )
                fs = (
                    f"L{foga_sp_top3[k][0] + 1}={_fmt3(foga_sp_top3[k][1])}"
                    if k < len(foga_sp_top3)
                    else "-"
                )
                fa = (
                    f"L{foga_au_top3[k][0] + 1}={_fmt3(foga_au_top3[k][1])}"
                    if k < len(foga_au_top3)
                    else "-"
                )
                rows.append(f"{k + 1:>2} | {cs:<18} | {ce:<18} | {fs:<20} | {fa:<20}")

            header = " rk | Cos-SEM(top3)      | Cos-EVT(top3)      | FoG-A Speech(top3)     | FoG-A Audio(top3)     "
            logger.info(
                f"[Probe+FoG-A@step {step}]{fused_suffix}\n{header}\n" + "\n".join(rows)
            )

    # ----------------------------- On entering a new epoch, plot the "previous epoch" figure -----------------------------
    def _maybe_finalize_previous_epoch(self, current_epoch: int):
        logger = logging.getLogger(__name__)
        if self._last_epoch_seen is None:
            self._last_epoch_seen = current_epoch
            return

        if (
                current_epoch != self._last_epoch_seen
                and self._last_epoch_seen not in self._finalized_epochs
        ):
            try:
                self._finalize_single_epoch(self._last_epoch_seen)
                self._finalized_epochs.add(self._last_epoch_seen)
            except Exception as e_epoch:
                logger.warning(
                    f"[Probe] finalize epoch {self._last_epoch_seen} failed: {e_epoch}"
                )
            finally:
                self._last_epoch_seen = current_epoch

    # ----------------------------- Plotting implementation for a single epoch -----------------------------
    def _finalize_single_epoch(self, epoch_id: int):
        import numpy as np
        from matplotlib import pyplot as plt

        logger = logging.getLogger(__name__)

        # ---------- Read this epoch's "cosine probe" CSV ----------
        csv_path = self.csv_dir / f"{self.base_name}_epoch_{epoch_id:04d}{self.suffix}"
        if not csv_path.exists():
            logger.warning(
                f"[Probe] Cosine probe CSV for epoch {epoch_id:04d} not found: {csv_path.name}, skipping plots."
            )
            return

        per_layer_steps_sem = {i: [] for i in range(self.num_layers)}
        per_layer_steps_evt = {i: [] for i in range(self.num_layers)}
        sem_sum = [0.0] * self.num_layers
        evt_sum = [0.0] * self.num_layers
        cnts = [0] * self.num_layers

        with open(csv_path, "r", encoding="utf-8") as f:
            _ = f.readline()  # Skip the header
            for line in f:
                try:
                    step_s, epoch_s, layer_s, sem_s, evt_s = line.strip().split(",")
                    li = int(layer_s)
                    if sem_s != "":
                        v = float(sem_s)
                        sem_sum[li] += v
                        cnts[li] += 1
                        per_layer_steps_sem[li].append((int(step_s), v))
                    if evt_s != "":
                        v = float(evt_s)
                        evt_sum[li] += v
                        cnts[li] += 1
                        per_layer_steps_evt[li].append((int(step_s), v))
                except Exception:
                    continue

        means_sem = [
            (sem_sum[i] / cnts[i]) if cnts[i] > 0 else 0.0
            for i in range(self.num_layers)
        ]
        means_evt = [
            (evt_sum[i] / cnts[i]) if cnts[i] > 0 else 0.0
            for i in range(self.num_layers)
        ]
        x_ticks = list(range(1, self.num_layers + 1))

        # Figure 1: Layer importance comparison
        plt.figure(figsize=(12, 6))
        width = 0.35
        plt.bar(
            [i - width / 2 for i in x_ticks],
            means_sem,
            width=width,
            label="Semantic(Whisper)",
        )
        plt.bar(
            [i + width / 2 for i in x_ticks],
            means_evt,
            width=width,
            label="Event(BEATs)",
        )
        plt.xlabel("Layer Index (1-based)")
        plt.ylabel("Mean Cosine Similarity")
        plt.title(f"Layer Importance: Semantic vs Event (Epoch {epoch_id})")
        plt.legend()
        out1 = self.img_dir / f"{self.base_name}_epoch_{epoch_id:04d}_importance.png"
        plt.tight_layout()
        plt.savefig(out1, dpi=150)
        plt.close()

        # Figure 2: Trend (Top-3)
        def _topk(means, k=3):
            return sorted(range(len(means)), key=lambda i: means[i], reverse=True)[:k]

        top_sem = _topk(means_sem, 3)
        top_evt = _topk(means_evt, 3)

        plt.figure(figsize=(12, 6))
        for li in top_sem:
            s = sorted(per_layer_steps_sem[li], key=lambda x: x[0])
            if s:
                steps, vals = zip(*s)
                plt.plot(steps, vals, label=f"Sem-L{li + 1}")
        for li in top_evt:
            s = sorted(per_layer_steps_evt[li], key=lambda x: x[0])
            if s:
                steps, vals = zip(*s)
                plt.plot(steps, vals, label=f"Evt-L{li + 1}")
        plt.xlabel("Global Step (within epoch)")
        plt.ylabel("Cosine Similarity")
        plt.title(f"Probe Trend (Top-3 Semantic/Event) - Epoch {epoch_id}")
        plt.legend()
        out2 = self.img_dir / f"{self.base_name}_epoch_{epoch_id:04d}_trend_top3.png"
        plt.tight_layout()
        plt.savefig(out2, dpi=150)
        plt.close()

        # ---------- Figure 3: FoG-A heatmap (newly added) ----------
        foga_csv = self.csv_dir / f"{self.base_name}_epoch_{epoch_id:04d}_foga.csv"
        if foga_csv.exists():
            sp_sum = np.zeros(self.num_layers, dtype=float)
            sp_cnt = np.zeros(self.num_layers, dtype=int)
            au_sum = np.zeros(self.num_layers, dtype=float)
            au_cnt = np.zeros(self.num_layers, dtype=int)
            with open(foga_csv, "r", encoding="utf-8") as f:
                _ = f.readline()
                for line in f:
                    try:
                        step_s, epoch_s, layer_s, sp_s, au_s = line.strip().split(",")
                        li = int(layer_s)
                        if sp_s != "":
                            sp_sum[li] += float(sp_s)
                            sp_cnt[li] += 1
                        if au_s != "":
                            au_sum[li] += float(au_s)
                            au_cnt[li] += 1
                    except Exception:
                        continue
            sp_avg = np.divide(
                sp_sum, np.maximum(sp_cnt, 1), where=(np.maximum(sp_cnt, 1) > 0)
            )
            au_avg = np.divide(
                au_sum, np.maximum(au_cnt, 1), where=(np.maximum(au_cnt, 1) > 0)
            )
            heat = np.vstack([sp_avg, au_avg])  # shape (2, L)

            fig3 = plt.figure(figsize=(max(8, self.num_layers / 1.5), 3.2))
            ax3 = fig3.add_subplot(111)
            im = ax3.imshow(heat, aspect="auto", cmap="viridis")
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(["Speech", "General"])
            ax3.set_xticks(range(self.num_layers))
            ax3.set_xticklabels(
                [f"L{i + 1}" for i in range(self.num_layers)], rotation=45
            )
            ax3.set_title(f"FoG-A Layer Contribution Heatmap (Epoch {epoch_id})")
            cbar = fig3.colorbar(im)
            cbar.set_label("Δv (relative L2)")
            fig3.tight_layout()
            out3 = (
                    self.img_dir / f"{self.base_name}_epoch_{epoch_id:04d}_foga_heatmap.png"
            )
            fig3.savefig(out3, dpi=150)
            plt.close(fig3)
            logger.info(
                f"[Probe] Epoch {epoch_id} figures saved: {out1.name}, {out2.name}, {out3.name}"
            )
        else:
            logger.info(
                f"[Probe] Epoch {epoch_id} figures saved: {out1.name} , {out2.name}"
            )

    # ----------------------------- End of all training: plot the overall comparison figures -----------------------------
    def finalize(self, out_dir: "Path | str"):
        from matplotlib import pyplot as plt

        logger = logging.getLogger(__name__)

        # 1) Fallback: if the last epoch has not yet been plotted, plot it first
        if (
                self._last_epoch_seen is not None
                and self._last_epoch_seen not in self._finalized_epochs
        ):
            try:
                self._finalize_single_epoch(self._last_epoch_seen)
                self._finalized_epochs.add(self._last_epoch_seen)
            except Exception as e_last:
                logger.warning(
                    f"[Probe] finalize(last epoch={self._last_epoch_seen}) failed: {e_last}"
                )

        # 2) Aggregate all epoch CSVs
        csv_files = sorted(self.csv_dir.glob(f"{self.base_name}_epoch_*.csv"))
        if not csv_files:
            logger.warning(
                "[Probe] No CSV files found under flow/csv; skip overall plotting."
            )
            return

        # Aggregation containers
        all_sem_sum = [0.0] * self.num_layers
        all_evt_sum = [0.0] * self.num_layers
        all_cnts = [0] * self.num_layers
        series_sem_all = {i: [] for i in range(self.num_layers)}
        series_evt_all = {i: [] for i in range(self.num_layers)}

        for csv_path in csv_files:
            with open(csv_path, "r", encoding="utf-8") as f:
                _ = f.readline()  # Skip the header
                for line in f:
                    try:
                        step_s, epoch_s, layer_s, sem_s, evt_s = line.strip().split(",")
                        li = int(layer_s)
                        if sem_s != "":
                            v = float(sem_s)
                            all_sem_sum[li] += v
                            all_cnts[li] += 1
                            series_sem_all[li].append((int(step_s), v))
                        if evt_s != "":
                            v = float(evt_s)
                            all_evt_sum[li] += v
                            all_cnts[li] += 1
                            series_evt_all[li].append((int(step_s), v))
                    except Exception:
                        continue

        means_sem_all = [
            (all_sem_sum[i] / all_cnts[i]) if all_cnts[i] > 0 else 0.0
            for i in range(self.num_layers)
        ]
        means_evt_all = [
            (all_evt_sum[i] / all_cnts[i]) if all_cnts[i] > 0 else 0.0
            for i in range(self.num_layers)
        ]
        x_ticks = list(range(1, self.num_layers + 1))

        # Figure A: Overall layer importance comparison (semantic vs event)
        plt.figure(figsize=(12, 6))
        width = 0.35
        plt.bar(
            [i - width / 2 for i in x_ticks],
            means_sem_all,
            width=width,
            label="Semantic(Whisper)",
        )
        plt.bar(
            [i + width / 2 for i in x_ticks],
            means_evt_all,
            width=width,
            label="Event(BEATs)",
        )
        plt.xlabel("Layer Index (1-based)")
        plt.ylabel("Mean Cosine Similarity")
        plt.title("Overall Layer Importance: Semantic vs Event")
        plt.legend()
        overall_1 = self.img_dir / f"{self.base_name}_overall_importance.png"
        plt.tight_layout()
        plt.savefig(overall_1, dpi=150)
        plt.close()

        # Figure B: Overall trend
        def _topk(means, k=3):
            return sorted(range(len(means)), key=lambda i: means[i], reverse=True)[:k]

        top_sem = _topk(means_sem_all, 3)
        top_evt = _topk(means_evt_all, 3)

        plt.figure(figsize=(12, 6))
        for li in top_sem:
            s = sorted(series_sem_all[li], key=lambda x: x[0])
            if s:
                steps, vals = zip(*s)
                plt.plot(steps, vals, label=f"Sem-L{li + 1}")
        for li in top_evt:
            s = sorted(series_evt_all[li], key=lambda x: x[0])
            if s:
                steps, vals = zip(*s)
                plt.plot(steps, vals, label=f"Evt-L{li + 1}")
        plt.xlabel("Global Step (all epochs)")
        plt.ylabel("Cosine Similarity")
        plt.title("Overall Probe Trend (Top-3 Semantic/Event)")
        plt.legend()
        overall_2 = self.img_dir / f"{self.base_name}_overall_trend_top3.png"
        plt.tight_layout()
        plt.savefig(overall_2, dpi=150)
        plt.close()

        logger.info(
            f"[Probe] Overall figures saved: {overall_1.name} , {overall_2.name}"
        )


class BalancedDistributedSampler(DistributedSampler):
    """
    A distributed sampler designed to balance data from the different subsets within a ConcatDataset.
    By oversampling the smaller datasets, it ensures that in each epoch the number of samples provided by each subset
    is equal to the number of samples in the largest subset.
    """

    def __init__(
            self,
            dataset,
            num_replicas=None,
            rank=None,
            shuffle=True,
            seed=0,
            drop_last=False,
    ):
        if not isinstance(dataset, ConcatDataset):
            raise TypeError("The dataset must be of type ConcatDataset")

        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

        self.dataset_lengths = [len(d) for d in dataset.datasets]
        self.max_len = max(self.dataset_lengths)

        unpadded_total_size = self.max_len * len(dataset.datasets)

        if self.drop_last:
            self.num_samples = unpadded_total_size // self.num_replicas
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.num_samples = math.ceil(unpadded_total_size / self.num_replicas)
            self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        indices = []
        for i, dataset_len in enumerate(self.dataset_lengths):
            sub_indices = torch.randint(
                high=dataset_len, size=(self.max_len,), generator=g
            ).tolist()
            offset = self.dataset.cumulative_sizes[i - 1] if i > 0 else 0
            indices.extend([idx + offset for idx in sub_indices])

        if self.shuffle:
            shuffled_order = torch.randperm(len(indices), generator=g).tolist()
            indices = [indices[i] for i in shuffled_order]

        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            indices = indices[: self.total_size]

        subset_indices = indices[self.rank: self.total_size: self.num_replicas]
        return iter(subset_indices)

    def __len__(self):
        return self.num_samples


class FlowMatchingTrainer:
    def __init__(self, config_dict, cmd_args):
        self.cmd_args = cmd_args
        self.config = config_dict
        self._is_shutdown = False
        global logger
        log_level_from_config = (
            self.config.get("logging", {}).get("console", {}).get("level", "INFO")
        )
        logger = setup_logger_ds(
            self.cmd_args.local_rank, self.config, level_override=log_level_from_config
        )
        logger.info(
            f"Rank {self.cmd_args.local_rank}: Starting Unified Flow Matching Trainer initialization..."
        )

        self.mel_extractor = None
        self.flow_model = None
        self.pytorch_optimizer = None
        self.optimizer = None
        self.model_engine = None
        self.lr_scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.wandb_initialized = False
        self.device = None

        self.beats_feature_extractor = None

        self._setup_device()
        self._setup_external_models()
        self.setup_models()

        if self.cmd_args.local_rank <= 0:
            self.setup_wandb()
        self._init_data_loaders()

        self.checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"]) / "flow"
        if self.cmd_args.local_rank <= 0:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_val_flow_loss = float("inf")
        self.current_epoch = 0

    def _setup_device(self):
        if self.model_engine:
            self.device = self.model_engine.device
        elif torch.cuda.is_available() and self.cmd_args.local_rank != -1:
            self.device = torch.device(f"cuda:{self.cmd_args.local_rank}")
        elif torch.cuda.is_available():
            gpu_id_to_use = self.config.get("device", {}).get("device_ids", [0])[0]
            self.device = torch.device(f"cuda:{gpu_id_to_use}")
        else:
            self.device = torch.device("cpu")
        logger.info(
            f"Rank {self.cmd_args.local_rank}: Device determined: {self.device}"
        )

    def _setup_external_models(self):
        try:
            # ====== AudioSetTokenizer ======
            ast_checkpoint_dir = Path(self.config["paths"]["checkpoint_dir"]) / "ast"
            if not ast_checkpoint_dir.exists() or not any(ast_checkpoint_dir.iterdir()):
                raise FileNotFoundError(
                    f"AudioSetTokenizer checkpoint directory is empty or not found: {ast_checkpoint_dir}"
                )

            best_loss = float("inf")
            best_checkpoint_path = None
            for pth_file in ast_checkpoint_dir.glob("best_epoch_*_loss_*.pth"):
                try:
                    loss_str = pth_file.stem.split("_loss_")[-1]
                    loss_val = float(loss_str)
                    if loss_val < best_loss:
                        best_loss = loss_val
                        best_checkpoint_path = pth_file
                except (ValueError, IndexError):
                    logger.warning(
                        f"Rank {self.cmd_args.local_rank}: Could not parse loss from filename: {pth_file.name}"
                    )
                    continue
            if best_checkpoint_path is None:
                raise FileNotFoundError(
                    f"No valid 'best_epoch_*_loss_*.pth' checkpoints found in {ast_checkpoint_dir}"
                )
            logger.info(
                f"Rank {self.cmd_args.local_rank}: Found best AST checkpoint: {best_checkpoint_path} (loss {best_loss:.4f})"
            )

            ast_checkpoint = torch.load(best_checkpoint_path, map_location="cpu")
            ast_config = self.config["hyperparameters"]["ast"]
            self.audioset_tokenizer = AudioSetTokenizer(
                input_dim=ast_config["input_dim"],
                hidden_dim=ast_config["hidden_dim"],
                vocab_size=ast_config["vocab_size"],
            )
            self.audioset_tokenizer.load_state_dict(ast_checkpoint["model_state_dict"])
            self.audioset_tokenizer.to(self.device).eval()
            logger.info(
                f"Rank {self.cmd_args.local_rank}: AudioSet Tokenizer loaded from best checkpoint."
            )

            # ====== BEATs (Audio teacher) ======
            beats_extractor_path = self.config["paths"][
                "beats_feature_extractor_checkpoint"
            ]
            if not os.path.exists(beats_extractor_path):
                raise FileNotFoundError(
                    f"BEATs Feature Extractor checkpoint not found at: {beats_extractor_path}"
                )

            extractor_checkpoint = torch.load(beats_extractor_path, map_location="cpu")
            extractor_cfg = BEATsConfig(extractor_checkpoint["cfg"])
            self.beats_feature_extractor = BEATs(extractor_cfg)
            self.beats_feature_extractor.load_state_dict(extractor_checkpoint["model"])
            self.beats_feature_extractor.to(self.device).eval()
            logger.info(
                f"Rank {self.cmd_args.local_rank}: BEATs Feature Extractor loaded."
            )

            with torch.no_grad():
                dummy = torch.zeros(
                    1, int(self.config["audio"]["sample_rate"]), device=self.device
                )
                feats, _ = self.beats_feature_extractor.extract_features(dummy)
                self.teacher_dim_audio = int(feats.shape[-1])
            logger.info(
                f"Rank {self.cmd_args.local_rank}: teacher_dim_audio = {self.teacher_dim_audio}"
            )

            # ====== Whisper-large-v3 (Speech teacher) ======
            self.whisper_processor, self.whisper_model = None, None
            self.teacher_dim_speech = 0
            try:
                from transformers import WhisperModel, WhisperProcessor

                whisper_name_or_path = self.config["paths"].get(
                    "whisper_large_v3", "openai/whisper-large-v3"
                )
                self.whisper_processor = WhisperProcessor.from_pretrained(
                    whisper_name_or_path
                )
                self.whisper_model = WhisperModel.from_pretrained(whisper_name_or_path)
                self.whisper_model.to(self.device).eval()

                self.teacher_dim_speech = int(
                    getattr(self.whisper_model.config, "d_model", 0)
                )
                if self.teacher_dim_speech <= 0:
                    with torch.no_grad():
                        sr16 = 16000
                        zeros = torch.zeros(1, sr16)
                        inputs = self.whisper_processor(
                            zeros.numpy(), sampling_rate=sr16, return_tensors="pt"
                        )
                        enc_out = self.whisper_model.encoder(
                            input_features=inputs.input_features.to(self.device)
                        )
                        self.teacher_dim_speech = int(
                            enc_out.last_hidden_state.shape[-1]
                        )
                self.whisper_target_sr = 16000
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Whisper-large-v3 loaded. teacher_dim_speech = {self.teacher_dim_speech}"
                )
            except Exception as e_w:
                logger.error(
                    f"Rank {self.cmd_args.local_rank}: Failed to load Whisper-large-v3: {e_w}. Speech teacher disabled."
                )
                self.whisper_processor, self.whisper_model = None, None
                self.teacher_dim_speech = 0

        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: External models setup failed: {e}\n{traceback.format_exc()}"
            )
            raise

    def setup_wandb(self):
        if self.cmd_args.local_rank != 0:
            self.wandb_initialized = False
            return
        try:
            if self.config["logging"]["wandb"]["enabled"]:
                project = self.config["logging"]["wandb"]["project"]
                name_prefix = self.config["logging"]["wandb"]["name"]
                exp_name_suffix = self.config.get("meta", {}).get("exp_name", "CFM")
                name = f"{name_prefix}_{exp_name_suffix}"
                wandb.init(
                    project=project,
                    name=name,
                    config={**self.config, **vars(self.cmd_args)},
                )
                logger.info(
                    f"Rank 0: WandB initialized - Project: {project}, Run: {name}"
                )
                self.wandb_initialized = True
            else:
                logger.info("Rank 0: WandB logging disabled in config.")
                self.wandb_initialized = False
        except Exception as e:
            logger.error(f"Rank 0: WandB setup failed: {str(e)}")
            self.wandb_initialized = False

    def setup_models(self):
        logger.info(
            f"Rank {self.cmd_args.local_rank}: Initializing models and optimizer..."
        )
        try:
            self.mel_extractor = MelSpectrogramExtractor(
                self.config, target_device=self.device
            )
            logger.info(
                f"Rank {self.cmd_args.local_rank}: Mel Spectrogram Extractor Initialized."
            )

            t_s = int(getattr(self, "teacher_dim_speech", 0))
            t_a = int(getattr(self, "teacher_dim_audio", 0))
            self.flow_model = FlowMatchingModel(
                self.config, teacher_dim_speech=t_s, teacher_dim_audio=t_a
            )
            logger.info(
                f"Rank {self.cmd_args.local_rank}: Flow Matching Model instantiated "
                f"(teacher_dim_speech={t_s}, teacher_dim_audio={t_a})."
            )

            opt_cfg = self.config["hyperparameters"]["flow"]["optimizer"]
            self.pytorch_optimizer = torch.optim.AdamW(
                self.flow_model.parameters(),
                lr=float(opt_cfg["scheduler"]["max_lr"]),
                weight_decay=float(opt_cfg["weight_decay"]),
                betas=eval(str(opt_cfg["betas"])),
            )

            self.model_engine, _, _, _ = deepspeed.initialize(
                args=self.cmd_args,
                model=self.flow_model,
                optimizer=self.pytorch_optimizer,
            )
            logger.info(
                f"Rank {self.cmd_args.local_rank}: DeepSpeed engine initialized on device: {self.model_engine.device}"
            )
            self.device = self.model_engine.device

            self.vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(
                self.device
            )
            logger.info(
                f"Rank {self.cmd_args.local_rank}: Vocos loaded to {self.device}."
            )

        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Model/DeepSpeed init failed: {str(e)}\n{traceback.format_exc()}"
            )
            raise

    def setup_lr_scheduler(self):
        if self.pytorch_optimizer is None or self.train_loader is None:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Optimizer or Train Loader not initialized. Cannot create LR scheduler."
            )
            return
        logger.info(f"Rank {self.cmd_args.local_rank}: Setting up LR scheduler...")
        scheduler_config = self.config["hyperparameters"]["flow"]["optimizer"][
            "scheduler"
        ]
        num_epochs = self.config["hyperparameters"]["flow"]["num_epochs"]
        gradient_accumulation_steps = self.model_engine.gradient_accumulation_steps()
        num_optimizer_steps_per_epoch = math.ceil(
            len(self.train_loader) / gradient_accumulation_steps
        )
        total_steps = num_optimizer_steps_per_epoch * num_epochs
        if total_steps <= 0:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Calculated total_steps ({total_steps}) is invalid. LR scheduler not created."
            )
            return
        logger.info(
            f"Rank {self.cmd_args.local_rank}: Total steps for LR scheduler: {total_steps}"
        )
        self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.pytorch_optimizer,
            max_lr=float(scheduler_config["max_lr"]),
            total_steps=total_steps,
            pct_start=float(scheduler_config["pct_start"]),
            div_factor=float(scheduler_config["div_factor"]),
            final_div_factor=float(scheduler_config["final_div_factor"]),
        )
        logger.info(f"Rank {self.cmd_args.local_rank}: OneCycleLR scheduler created.")

    def _init_data_loaders(self):
        """
        Use ConcatDataset to mix LibriSpeech and AudioSet, with a balanced sampling strategy.
        Now, BalancedDistributedSampler is also used in single-GPU mode to align the data-distribution behavior,
        and custom_collate_fn is enforced to fix the crash caused by variable-length s3_token.
        """
        try:
            is_dist = torch.distributed.is_initialized()
            rank, world_size = (
                (torch.distributed.get_rank(), torch.distributed.get_world_size())
                if is_dist
                else (0, 1)
            )
            batch_size = self.model_engine.train_micro_batch_size_per_gpu()

            ls_train_ds = LibriSpeechDataset(
                self.config["data"]["librispeech"]["train_root"], self.config
            )

            as_train_ds = AudioSetDataset(
                self.config["data"]["audioset"]["train_root"],
                self.config,
                str(self.device),
            )
            train_dataset = ConcatDataset([ls_train_ds, as_train_ds])

            # --- Key change: use BalancedDistributedSampler regardless of single-GPU or multi-GPU ---
            # This ensures that the mixing ratio of LibriSpeech and AudioSet is exactly the same as in multi-GPU training
            train_sampler = BalancedDistributedSampler(
                train_dataset, num_replicas=world_size, rank=rank, shuffle=True
            )

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                shuffle=False,  # The sampler already handles shuffling
                drop_last=True,
                num_workers=self.config["data"]["num_workers"],
                pin_memory=self.config["data"]["pin_memory"],
                persistent_workers=True,
                collate_fn=custom_collate_fn,  # Use the custom collate_fn to fix the variable-length s3_token issue
            )

            ls_val_ds = LibriSpeechDataset(
                self.config["data"]["librispeech"]["val_root"], self.config
            )

            as_val_ds = AudioSetDataset(
                self.config["data"]["audioset"]["val_root"],
                self.config,
                str(self.device),
            )
            val_dataset = ConcatDataset([ls_val_ds, as_val_ds])

            val_sampler = (
                DistributedSampler(
                    val_dataset, num_replicas=world_size, rank=rank, shuffle=False
                )
                if is_dist
                else None
            )
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=self.config["data"]["num_workers"],
                pin_memory=self.config["data"]["pin_memory"],
                persistent_workers=True,
                collate_fn=custom_collate_fn,  # The validation set also needs collate_fn to prevent a crash when batch > 1
            )

            mode_str = "Distributed" if is_dist else "Single-Card"
            logger.info(
                f"Rank {rank}: DataLoaders initialized in {mode_str} mode with Balanced Sampling & Custom Collate."
            )
            self.setup_lr_scheduler()
        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Failed to set up data loaders: {e}\n{traceback.format_exc()}"
            )
            raise

    def prepare_batch(self, batch):
        """
        Prepare the batch data without mixing.
        - LibriSpeech and AudioSet samples are kept pure.
        - The target mel-spectrogram is computed directly from the raw waveform.
        - The conditioning `fused_embed` comes directly from the respective S3 or AS tokens
        """
        try:
            device = self.device

            # -------------------- 0) Raw batch --------------------
            waveforms = batch["waveform"].to(device, non_blocking=True)  # [B, T_audio]
            dataset_types = batch["dataset_type"]  # list[str]
            mel_lengths_batch = batch["mel_lengths"]  # [B]
            B, T_audio = waveforms.shape

            ls_idx = [i for i, dt in enumerate(dataset_types) if dt == "librispeech"]
            as_idx = [i for i, dt in enumerate(dataset_types) if dt == "audioset"]

            # -------------------- 1) Target mel (computed directly from the raw waveform) --------------------
            full_mel_specs = self.mel_extractor(waveforms)
            _, n_mels, T_full = full_mel_specs.shape

            # -------------------- 2) Reference mel (prefix <=30%) --------------------
            ref_mel_parts, actual_ref_lengths = [], []
            for i in range(B):
                current_mel_len = mel_lengths_batch[i].item()
                max_ref_len_abs = max(1, int(current_mel_len * 0.3))
                ref_len = (
                    torch.randint(1, max_ref_len_abs + 1, (1,)).item()
                    if max_ref_len_abs > 0
                    else 0
                )
                actual_ref_lengths.append(ref_len)
                if ref_len > 0:
                    ref_part = full_mel_specs[i, :, : min(ref_len, T_full)]
                else:
                    ref_part = torch.empty(
                        (n_mels, 0), device=self.device, dtype=full_mel_specs.dtype
                    )
                ref_mel_parts.append(ref_part)
            ref_mel_for_cond = torch.stack(
                [F.pad(p, (0, T_full - p.shape[1])) for p in ref_mel_parts]
            )

            # -------------------- 3) Conditioning fused_embed (shared token space) --------------------
            D_tok = self.model_engine.module.max_fused_embed_dim
            fused_embed = torch.zeros(
                B, D_tok, T_full, device=device, dtype=full_mel_specs.dtype
            )

            # Process the AudioSet samples
            if len(as_idx) > 0:
                as_wave = waveforms[
                    torch.as_tensor(as_idx, device=device, dtype=torch.long)
                ]
                with torch.no_grad():
                    beats_features, _ = self.beats_feature_extractor.extract_features(
                        as_wave
                    )
                    input_feats_tok = beats_features.permute(0, 2, 1)
                    as_tokens = self.audioset_tokenizer.tokenize(input_feats_tok)
                    as_shared = self.model_engine.module.embed_and_project_tokens(
                        "as", as_tokens.to(device)
                    )
                if as_shared.shape[-1] > 0:
                    as_shared = F.interpolate(
                        as_shared, size=T_full, mode="linear", align_corners=False
                    )
                    as_shared = F.normalize(as_shared, p=2, dim=1)
                fused_embed[as_idx, :, :] = as_shared

            # Process the LibriSpeech samples
            if len(ls_idx) > 0:
                s3_field = batch.get("s3_token", None)
                if s3_field is None:
                    raise RuntimeError(
                        "The 's3_token' field is missing from the batch; please make sure LibriSpeechDataset returns this key correctly."
                    )
                ls_tokens_list = []

                # After being processed by custom_collate_fn, s3_field is necessarily a list or tuple
                if isinstance(s3_field, (list, tuple)):
                    for i in ls_idx:
                        t = s3_field[i]
                        if not torch.is_tensor(t):
                            t = torch.as_tensor(t, dtype=torch.long)
                        ls_tokens_list.append(t)
                elif torch.is_tensor(s3_field):
                    # Compatibility code, just in case
                    for i in ls_idx:
                        t = s3_field[i]
                        if t.dim() > 1:
                            t = t.view(-1)
                        ls_tokens_list.append(t.to(torch.long))
                else:
                    raise RuntimeError(f"Unsupported 's3_token' type: {type(s3_field)}")

                max_token_len = (
                    max(t.numel() for t in ls_tokens_list) if ls_tokens_list else 0
                )
                if max_token_len == 0:
                    max_token_len = 1
                    ls_tokens_list = [
                        torch.zeros(1, dtype=torch.long, device=device)
                        for _ in ls_tokens_list
                    ]
                padded_tokens = torch.stack(
                    [
                        F.pad(t.to(device), (0, max_token_len - t.numel()), value=0)
                        for t in ls_tokens_list
                    ],
                    dim=0,
                )

                with torch.no_grad():
                    ls_shared = self.model_engine.module.embed_and_project_tokens(
                        "s3", padded_tokens
                    )
                if ls_shared.shape[-1] > 0:
                    ls_shared = F.interpolate(
                        ls_shared, size=T_full, mode="linear", align_corners=False
                    )
                    ls_shared = F.normalize(ls_shared, p=2, dim=1)
                fused_embed[ls_idx, :, :] = ls_shared

            # -------------------- 4) Teacher vectors --------------------
            teacher_audio_global = None
            if len(as_idx) > 0:
                as_wave = waveforms[
                    torch.as_tensor(as_idx, device=device, dtype=torch.long)
                ]
                with torch.no_grad():
                    beats_features, _ = self.beats_feature_extractor.extract_features(
                        as_wave
                    )
                    ta_local = beats_features.mean(dim=1)
                Cb = ta_local.shape[-1]
                ta_list = [
                    torch.zeros(Cb, device=device, dtype=ta_local.dtype)
                    for _ in range(B)
                ]
                for k, i in enumerate(as_idx):
                    ta_list[i] = ta_local[k]
                teacher_audio_global = torch.stack(ta_list, dim=0)

            teacher_speech_global = None
            if len(ls_idx) > 0 and (
                    self.whisper_model is not None and self.whisper_processor is not None
            ):
                ls_wave = waveforms[
                    torch.as_tensor(ls_idx, device=device, dtype=torch.long)
                ]
                with torch.no_grad():
                    sr_conf = int(self.config["audio"]["sample_rate"])
                    if sr_conf != self.whisper_target_sr:
                        ls_wave_16k = torchaudio.functional.resample(
                            ls_wave, orig_freq=sr_conf, new_freq=self.whisper_target_sr
                        )
                    else:
                        ls_wave_16k = ls_wave
                    ls_inputs = self.whisper_processor(
                        ls_wave_16k.detach().cpu().numpy(),
                        sampling_rate=self.whisper_target_sr,
                        return_tensors="pt",
                    )
                    input_features = ls_inputs.input_features.to(self.device)
                    enc_out = self.whisper_model.encoder(input_features=input_features)
                    ts_local = enc_out.last_hidden_state.mean(dim=1)
                Cw = ts_local.shape[-1]
                ts_list = [
                    torch.zeros(Cw, device=device, dtype=ts_local.dtype)
                    for _ in range(B)
                ]
                for k, i in enumerate(ls_idx):
                    ts_list[i] = ts_local[k]
                teacher_speech_global = torch.stack(ts_list, dim=0)

            # -------------------- 5) Domain labels and return --------------------
            domain_ids = torch.tensor(
                [0 if dt == "librispeech" else 1 for dt in dataset_types],
                device=device,
                dtype=torch.long,
            )
            ret = {
                "full_mel_specs": full_mel_specs,
                "cond_embed_dict": {
                    "fused_embed": fused_embed,
                    "ref_mel_for_cond": ref_mel_for_cond,
                    "domain_ids": domain_ids,
                    "teacher_speech_global": teacher_speech_global,
                    "teacher_audio_global": teacher_audio_global,
                },
                "ref_mel_parts_for_viz": ref_mel_parts,
                "actual_ref_lengths": actual_ref_lengths,
                "full_mel_lengths_for_viz": mel_lengths_batch,
            }
            return ret

        except Exception as e:
            logger.error(
                f"Rank {self.cmd_args.local_rank}: Batch preparation failed: {e}\n{traceback.format_exc()}"
            )
            return None

    def train_step(self, batch):
        self.model_engine.train()
        data = self.prepare_batch(batch)
        if data is None:
            return {"flow_loss": float("inf"), "is_valid_step": False}

        x = data["full_mel_specs"]  # [B, n_mels, T]
        cond_embed_dict = data["cond_embed_dict"]  # Conditioning dictionary
        B = x.size(0)
        t = torch.rand(B, device=x.device)

        # --- Main task: per-sample loss ---
        per_sample_flow_loss = self.model_engine(
            x=x, t=t, cond_embed_dict=cond_embed_dict
        )
        flow_loss = per_sample_flow_loss.mean()

        is_loss_valid = True
        if not torch.isfinite(flow_loss):
            logger.error(
                f"Rank {self.cmd_args.local_rank}: NaN/Inf detected in training loss: {flow_loss.item()}. Skipping step."
            )
            scaled_flow_loss_item = 0.0
            is_loss_valid = False
            if self.optimizer:
                self.optimizer.zero_grad()
        else:
            scaled_flow_loss_item = flow_loss.item()

        if is_loss_valid:
            self.model_engine.backward(flow_loss)
            self.model_engine.step()

        if (
                self.lr_scheduler
                and is_loss_valid
                and self.model_engine.is_gradient_accumulation_boundary()
        ):
            self.lr_scheduler.step()

        # --- White-box probe (Rank 0 only; no gradients) ---
        try:
            if self.cmd_args.local_rank <= 0:
                if not hasattr(self, "probe_logger") or self.probe_logger is None:
                    n_layers = int(self.config["hyperparameters"]["flow"]["n_layers"])
                    enable_probe = bool(
                        self.config["hyperparameters"]["flow"].get(
                            "enable_whitebox_probe", True
                        )
                    )
                    csv_path = self.checkpoint_dir / "probe_stats.csv"
                    self.probe_logger = LayerProbeLogger(
                        num_layers=n_layers, csv_path=csv_path, enabled=enable_probe
                    )

                if self.probe_logger.enabled and is_loss_valid:
                    layers = self.probe_logger.layers_to_probe(self.global_step)
                    with torch.no_grad():
                        results = self.model_engine.module.probe_layers(
                            x_full_mel=x,
                            cond_embed_dict=cond_embed_dict,
                            layer_indices=layers,
                        )
                    self.probe_logger.update(
                        self.global_step,
                        self.current_epoch,
                        results,
                        log_to_console=False,
                    )

                    if self.global_step % 200 == 0:
                        dom_ids = cond_embed_dict.get("domain_ids", None)
                        if dom_ids is not None:
                            idx_s = (
                                (dom_ids == 0)
                                .nonzero(as_tuple=False)
                                .flatten()
                                .tolist()
                            )
                            idx_a = (
                                (dom_ids == 1)
                                .nonzero(as_tuple=False)
                                .flatten()
                                .tolist()
                            )
                            pick = []
                            if len(idx_s) > 0:
                                pick.append(idx_s[0])
                            if len(idx_a) > 0:
                                pick.append(idx_a[0])
                            if not pick:
                                pick = [0]
                            pick = torch.tensor(pick, device=x.device, dtype=torch.long)
                            x_sub = x.index_select(dim=0, index=pick)
                            cond_sub = {}
                            for k, v in cond_embed_dict.items():
                                if torch.is_tensor(v):
                                    cond_sub[k] = v.index_select(dim=0, index=pick)
                                else:
                                    try:
                                        cond_sub[k] = [
                                            v[i] for i in pick.cpu().tolist()
                                        ]
                                    except Exception:
                                        cond_sub[k] = v
                        else:
                            x_sub = x[:1]
                            cond_sub = {
                                k: (v[:1] if torch.is_tensor(v) else v)
                                for k, v in cond_embed_dict.items()
                            }

                        with torch.no_grad():
                            foga = self.model_engine.module.fog_attribution(
                                x_full_mel=x_sub,
                                cond_embed_dict=cond_sub,
                                layer_indices=layers,
                            )
                        self.probe_logger.update_foga(
                            self.global_step,
                            self.current_epoch,
                            foga,
                            log_to_console=True,
                        )

        except Exception as e_probe:
            logger.warning(
                f"Rank 0: probe / FoG-A failed at step {self.global_step}: {e_probe}\n{traceback.format_exc()}"
            )

        return {"flow_loss": scaled_flow_loss_item, "is_valid_step": is_loss_valid}

    def train(self):
        logger.info(
            f"Rank {self.cmd_args.local_rank}: Starting Conditional Flow Matching training..."
        )
        num_epochs = self.config["hyperparameters"]["flow"]["num_epochs"]
        is_main_process = self.cmd_args.local_rank <= 0

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            self.model_engine.train()
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            epoch_flow_loss_sum = 0.0
            num_valid_batches_epoch = 0

            pbar_desc = f"Epoch {epoch + 1}/{num_epochs} [Training...]"
            pbar = tqdm.tqdm(
                enumerate(self.train_loader),
                total=len(self.train_loader),
                desc=pbar_desc,
                ncols=150,
                disable=not is_main_process,
            )

            for step, batch in pbar:
                loss_info = self.train_step(batch)

                if loss_info["is_valid_step"]:
                    epoch_flow_loss_sum += loss_info["flow_loss"]
                    num_valid_batches_epoch += 1
                self.global_step += 1

                if is_main_process:
                    current_lr = (
                        self.pytorch_optimizer.param_groups[0]["lr"]
                        if self.pytorch_optimizer
                        else float("nan")
                    )
                    avg_epoch_loss_so_far = (
                        epoch_flow_loss_sum / num_valid_batches_epoch
                        if num_valid_batches_epoch > 0
                        else 0.0
                    )
                    pbar.set_postfix(
                        {
                            "flow_loss": f"{loss_info['flow_loss']:.4f}",
                            "avg_flow": f"{avg_epoch_loss_so_far:.4f}",
                            "lr": f"{current_lr:.2e}",
                        }
                    )
                    if self.wandb_initialized and loss_info["is_valid_step"]:
                        try:
                            wandb.log(
                                {
                                    "train/step_flow_loss": loss_info["flow_loss"],
                                    "train/learning_rate": current_lr,
                                },
                                step=self.global_step,
                            )
                        except Exception as e:
                            logger.error(
                                f"Rank 0: WandB step logging error at global_step {self.global_step}: {e}\n{traceback.format_exc()}"
                            )

            # --- Post-epoch processing logic ---
            epoch_stats_tensor = torch.tensor(
                [epoch_flow_loss_sum, float(num_valid_batches_epoch)],
                dtype=torch.float32,
                device=self.device,
            )
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(
                    epoch_stats_tensor, op=torch.distributed.ReduceOp.SUM
                )

            global_epoch_flow_loss_sum = epoch_stats_tensor[0].item()
            global_num_valid_batches_epoch = epoch_stats_tensor[1].item()
            avg_epoch_flow_loss = (
                global_epoch_flow_loss_sum / global_num_valid_batches_epoch
                if global_num_valid_batches_epoch > 0
                else 0.0
            )

            if is_main_process:
                logger.info(
                    f"Rank 0: Epoch {epoch + 1} completed. Average Training Flow Loss (Global): {avg_epoch_flow_loss:.4f}"
                )

            logger.info(
                f"Rank {self.cmd_args.local_rank}: Epoch {epoch + 1} training completed. Starting validation..."
            )

            val_metrics = self.validate(epoch)
            val_flow_loss = val_metrics["val_flow_loss"]

            should_save_this_epoch = torch.tensor(
                [0], dtype=torch.int, device=self.device
            )
            if is_main_process:
                logger.info(
                    f"Rank 0: Epoch {epoch + 1} Validation Flow Loss (Global): {val_flow_loss:.4f}"
                )
                if val_flow_loss < self.best_val_flow_loss:
                    self.best_val_flow_loss = val_flow_loss
                    logger.info(
                        f"Rank 0: New best validation flow loss: {self.best_val_flow_loss:.4f}. Marking for checkpoint save."
                    )
                    should_save_this_epoch[0] = 1

                if self.wandb_initialized:
                    try:
                        wandb.log(
                            {
                                "epoch/epoch_num": epoch + 1,
                                "epoch/train_avg_flow_loss": avg_epoch_flow_loss,
                                "epoch/val_avg_flow_loss": val_flow_loss,
                                "epoch/best_val_flow_loss": self.best_val_flow_loss,
                            },
                            step=self.global_step,
                        )
                    except Exception as e:
                        logger.error(
                            f"Rank 0: WandB epoch logging error for epoch {epoch + 1}: {e}\n{traceback.format_exc()}"
                        )

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(should_save_this_epoch, src=0)

            if should_save_this_epoch.item() == 1:
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Received signal to save checkpoint for epoch {epoch + 1}."
                )
                tag_to_save = ""
                client_state_to_save = {}
                if is_main_process:
                    tag_to_save = f"best_ep{epoch + 1}_val_loss_{self.best_val_flow_loss:.4f}_step{self.global_step}"
                    client_state_to_save = {
                        "epoch": epoch + 1,
                        "global_step": self.global_step,
                        "best_val_metric": self.best_val_flow_loss,
                        "config_snapshot": self.config,
                    }
                    logger.info(f"Rank 0: Checkpoint tag: {tag_to_save}")
                self.save_checkpoint(tag=tag_to_save, client_state=client_state_to_save)

            if torch.distributed.is_initialized():
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Reached end of epoch {epoch + 1}. Synchronizing all processes."
                )
                torch.distributed.barrier()
                logger.info(
                    f"Rank {self.cmd_args.local_rank}: Passed end of epoch {epoch + 1} barrier."
                )

        if is_main_process:
            logger.info(
                f"Training completed! Best validation flow loss (Rank 0): {self.best_val_flow_loss:.4f}"
            )

        # ====== Training complete: output the white-box probe figures (Rank 0 only) ======
        if (
                self.cmd_args.local_rank <= 0
                and hasattr(self, "probe_logger")
                and self.probe_logger
        ):
            try:
                self.probe_logger.finalize(self.checkpoint_dir)
            except Exception as e_fin:
                logger.error(f"[Probe] finalize failed: {e_fin}")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def validate_step(self, batch):
        self.model_engine.eval()
        with torch.no_grad():
            data = self.prepare_batch(batch)
            full_mel = data["full_mel_specs"]
            cond_embed_dict_for_loss = data["cond_embed_dict"]

            B, _, _ = full_mel.shape
            t = torch.rand(B, device=self.device)

            # model_engine returns the per-sample loss
            flow_loss_per_sample = self.model_engine(
                x=full_mel, t=t, cond_embed_dict=cond_embed_dict_for_loss
            )
            flow_loss = flow_loss_per_sample.mean()
            flow_loss_item = flow_loss.item()

            viz_data = {}
            if self.cmd_args.local_rank <= 0:
                viz_data = {
                    "ground_truth_full_mel_cpu": data["full_mel_specs"].cpu(),
                    "cond_embed_dict_cpu": {
                        k: (v.cpu() if torch.is_tensor(v) else v)
                        for k, v in data["cond_embed_dict"].items()
                    },
                    "ref_mel_parts_for_viz_cpu": [
                        item.cpu() for item in data["ref_mel_parts_for_viz"]
                    ],
                    "actual_ref_lengths_cpu": data["actual_ref_lengths"],
                    "full_mel_lengths_for_viz_cpu": data[
                        "full_mel_lengths_for_viz"
                    ].cpu(),
                }
            return {"val_flow_loss": flow_loss_item, "batch_size": B, **viz_data}

    def validate(self, epoch):
        """The validation process, with visualization of the samples"""
        is_main = self.cmd_args.local_rank <= 0
        self.model_engine.eval()

        loss_sum = torch.tensor(0.0, device=self.device)
        sample_count = torch.tensor(0, device=self.device)

        speech_sample_for_viz, audio_sample_for_viz = None, None

        with torch.no_grad():
            pbar = tqdm.tqdm(
                self.val_loader,
                desc=f"Validating Epoch {epoch + 1}",
                disable=not is_main,
            )
            for batch in pbar:
                metrics = self.validate_step(batch)
                if metrics["val_flow_loss"] != float("inf"):
                    loss_sum += metrics["val_flow_loss"] * metrics["batch_size"]
                    sample_count += metrics["batch_size"]

                if is_main and (
                        speech_sample_for_viz is None or audio_sample_for_viz is None
                ):
                    for i, dtype in enumerate(batch["dataset_type"]):
                        sample = {
                            k: v[i: i + 1] if torch.is_tensor(v) else [v[i]]
                            for k, v in batch.items()
                        }
                        if dtype == "librispeech" and speech_sample_for_viz is None:
                            speech_sample_for_viz = sample
                        elif dtype == "audioset" and audio_sample_for_viz is None:
                            audio_sample_for_viz = sample

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(loss_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                sample_count, op=torch.distributed.ReduceOp.SUM
            )

        avg_loss = (
            (loss_sum / sample_count).item() if sample_count > 0 else float("inf")
        )

        if is_main and wandb.run:
            if speech_sample_for_viz:
                self._log_validation_sample(speech_sample_for_viz, epoch, "LibriSpeech")

            if audio_sample_for_viz:
                self._log_validation_sample(audio_sample_for_viz, epoch, "AudioSet")

        return {"val_flow_loss": avg_loss}

    def _log_validation_sample(self, batch_for_viz, epoch, data_type_str):
        """An encapsulated visualization and logging helper function that restores the torchaudio.save logic"""
        if not (self.cmd_args.local_rank <= 0 and wandb.run):
            return
        logger.info(f"Rank 0: Generating visualization for a {data_type_str} sample...")
        try:
            with torch.no_grad():
                data = self.prepare_batch(batch_for_viz)
                output_dir = (
                        Path(self.config["paths"].get("output_dir", "outputs_cfm"))
                        / f"val_epoch{epoch + 1}"
                )
                output_dir.mkdir(parents=True, exist_ok=True)

                cond_dict_dev = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in data["cond_embed_dict"].items()
                }
                target_len = data["full_mel_lengths_for_viz"][0].item()
                sway_coef = self.config["hyperparameters"]["flow"]["sway_coef"]

                gen_mel = self.model_engine.module.sample(
                    cond_embed_dict=cond_dict_dev,
                    target_duration_frames=target_len,
                    sway_sampling_coef=sway_coef,
                )
                gen_mel_cpu = gen_mel.detach().cpu().squeeze(0)

                # --- Move all parts to the CPU before concatenation ---
                # Move the reference mel-spectrogram part from the GPU to the CPU
                ref_mel_cpu = data["ref_mel_parts_for_viz"][0].cpu()
                ref_len = data["actual_ref_lengths"][0]

                # Concatenate the CPU tensors
                recon_mel = (
                    torch.cat([ref_mel_cpu, gen_mel_cpu[:, ref_len:]], dim=1)
                    if ref_len > 0
                    else gen_mel_cpu
                )

                # Move the ground-truth mel-spectrogram from the GPU to the CPU
                gt_mel = data["full_mel_specs"][0, :, : recon_mel.shape[1]].cpu()

                mel_mean = self.config["hyperparameters"]["flow"]["mel_mean"]
                mel_std = self.config["hyperparameters"]["flow"]["mel_std"]
                raw_gt, raw_recon = (
                    gt_mel * mel_std + mel_mean,
                    recon_mel * mel_std + mel_mean,
                )

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
                im1 = ax1.imshow(raw_gt.numpy(), aspect="auto", origin="lower")
                fig.colorbar(im1, ax=ax1)
                ax1.set_title(f"Ground Truth Mel ({data_type_str} - Ep{epoch + 1})")
                im2 = ax2.imshow(raw_recon.numpy(), aspect="auto", origin="lower")
                fig.colorbar(im2, ax=ax2)
                ax2.set_title(f"Reconstructed Mel ({data_type_str} - RefLen:{ref_len})")
                plt.tight_layout()
                wandb.log(
                    {f"Val Media/Mel Comparison {data_type_str}": wandb.Image(fig)},
                    step=self.global_step,
                )
                plt.close(fig)

                # Vocos is on the GPU, so the mel-spectrogram needs to be moved back to the device for decoding
                audio_gt_dev = self.vocos.decode(raw_gt.unsqueeze(0).to(self.device))
                audio_reconstructed_dev = self.vocos.decode(
                    raw_recon.unsqueeze(0).to(self.device)
                )
                audio_gt_cpu, audio_reconstructed_cpu = (
                    peak_norm(audio_gt_dev.cpu()),
                    peak_norm(audio_reconstructed_dev.cpu()),
                )

                sample_rate = self.config["vocos"]["sample_rate"]
                gt_path = output_dir / f"ep{epoch + 1}_GT_{data_type_str}.wav"
                recon_path = output_dir / f"ep{epoch + 1}_Recon_{data_type_str}.wav"
                torchaudio.save(str(gt_path), audio_gt_cpu, sample_rate)
                torchaudio.save(str(recon_path), audio_reconstructed_cpu, sample_rate)

                wandb.log(
                    {
                        f"Val Media/Audio Real {data_type_str}": wandb.Audio(
                            str(gt_path), sample_rate=sample_rate
                        ),
                        f"Val Media/Audio Reconstructed {data_type_str}": wandb.Audio(
                            str(recon_path), sample_rate=sample_rate
                        ),
                    },
                    step=self.global_step,
                )
        except Exception as e_viz:
            logger.error(
                f"Rank 0: Error during visualization for {data_type_str}: {e_viz}\n{traceback.format_exc()}"
            )

    def save_checkpoint(self, tag: str, client_state: dict):
        # Ensure all processes synchronize before rank 0 starts saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self.cmd_args.local_rank <= 0:
            checkpoint_file_path = self.checkpoint_dir / f"{tag}.pt"
            logger.info(
                f"Rank 0: Preparing to save checkpoint to {checkpoint_file_path}"
            )
            model_state_dict = self.model_engine.module.state_dict()
            content_to_save = {
                "model_state_dict": model_state_dict,
                "pytorch_optimizer_state_dict": self.pytorch_optimizer.state_dict()
                if self.pytorch_optimizer
                else None,
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
                if self.lr_scheduler
                else None,
            }
            if client_state:
                content_to_save.update(client_state)
            try:
                torch.save(content_to_save, checkpoint_file_path)
                logger.info(
                    f"Rank 0: Checkpoint successfully saved to {checkpoint_file_path}"
                )
            except Exception as e:
                logger.error(
                    f"Rank 0: Failed to save checkpoint {checkpoint_file_path}: {e}\n{traceback.format_exc()}"
                )
                raise

        # Ensure all processes continue only after rank 0 has finished saving
        if torch.distributed.is_initialized():
            torch.distributed.barrier()


def main():
    global logger
    os.environ["MPLBACKEND"] = "Agg"
    matplotlib.use("Agg", force=True)
    if sys.platform != "win32":
        current_start_method = multiprocessing.get_start_method(allow_none=True)
        if current_start_method not in ["spawn", "forkserver"]:
            try:
                multiprocessing.set_start_method("spawn", force=True)
                if int(os.getenv("LOCAL_RANK", -1)) <= 0:
                    print("INFO: Multiprocessing start method set to 'spawn'.")
            except RuntimeError as e_mp_set:
                if int(os.getenv("LOCAL_RANK", -1)) <= 0:
                    print(f"WARNING: Failed to set 'spawn' start method: {e_mp_set}")
    os.environ["PYTHONWARNINGS"] = "ignore:semaphore_tracker:UserWarning"
    faulthandler.enable()

    parser = argparse.ArgumentParser(
        description="Distributed Flow Matching Training Script (CFM)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the main configuration file",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank passed from distributed launcher.",
    )
    parser = deepspeed.add_config_arguments(parser)
    cmd_args = parser.parse_args()

    if cmd_args.local_rank == -1:
        env_local_rank = os.getenv("LOCAL_RANK")
        if env_local_rank is not None:
            try:
                cmd_args.local_rank = int(env_local_rank)
            except ValueError:
                pass
    try:
        config_obj = load_config(cmd_args.config)
    except Exception as e_cfg_load:
        logging.basicConfig(
            level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logging.error(
            f"FATAL: Failed to load main config '{cmd_args.config}': {e_cfg_load}"
        )
        sys.exit(1)

    log_level_from_config = (
        config_obj.get("logging", {}).get("console", {}).get("level", "INFO")
    )
    logger = setup_logger_ds(
        cmd_args.local_rank, config_obj, level_override=log_level_from_config
    )
    deepspeed.runtime.utils.set_random_seed(config_obj.get("meta", {}).get("seed", 666))
    logger.info(f"Rank {cmd_args.local_rank}: Global random seed set via DeepSpeed.")

    if cmd_args.local_rank != -1:
        torch.cuda.set_device(cmd_args.local_rank)
        logger.info(
            f"Rank {cmd_args.local_rank}: Initializing DeepSpeed distributed environment..."
        )
        deepspeed.init_distributed()
        logger.info(
            f"Rank {cmd_args.local_rank}: Distributed environment initialized. World size: {torch.distributed.get_world_size()}"
        )
    else:
        logger.info(
            f"Rank {cmd_args.local_rank}: Not a distributed run (or local_rank is -1)."
        )

    trainer_instance = None
    try:
        trainer_instance = FlowMatchingTrainer(config_obj, cmd_args)
        trainer_instance.train()
    except KeyboardInterrupt:
        if logger:
            logger.warning(
                f"Rank {cmd_args.local_rank}: Training interrupted by user (KeyboardInterrupt)."
            )
    except Exception as e_main_loop:
        if logger:
            logger.error(
                f"Rank {cmd_args.local_rank}: Unhandled exception in main training loop: {e_main_loop}\n{traceback.format_exc()}"
            )
        else:
            print(
                f"Rank {cmd_args.local_rank} MAIN_LOOP_ERROR: {e_main_loop}\n{traceback.format_exc()}"
            )
        sys.exit(1)
    finally:
        if (
                hasattr(cmd_args, "local_rank")
                and cmd_args.local_rank != -1
                and torch.distributed.is_initialized()
        ):
            logger.info(
                f"Rank {cmd_args.local_rank}: Reached end of main, waiting at final barrier."
            )
            torch.distributed.barrier()
            logger.info(f"Rank {cmd_args.local_rank}: Passed final barrier.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Rank {cmd_args.local_rank}: Main function finished.")


if __name__ == "__main__":
    main()
