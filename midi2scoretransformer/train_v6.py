#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import argparse
import random
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from dataset_v6 import ASAPDataset
from dataset_v6 import PianoCoReDataset
from dataset_v6 import UnpairedXMLDataset

from config import MyModelConfig, FEATURES
from models.roformer import Roformer

try:
    from transformers import get_cosine_schedule_with_warmup
except Exception:
    get_cosine_schedule_with_warmup = None

import warnings
# filter music21 warnings for unpaired xml
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"music21\..*",
)

def _apply_history_dropout(
    y: Dict[str, torch.Tensor],
    keep_prob: float,
    pad_key: str = "pad",
) -> Dict[str, torch.Tensor]:
    """
    Randomly drop (mask to 0) a portion of the *decoder input history* tokens
    (teacher forcing). This approximates "model only sees 25% of output history".

    y: bucketed one-hot streams, plus y["pad"] (B,T) long/bool.
    keep_prob: probability to keep a timestep token (for each timestep).
    """
    if keep_prob >= 1.0:
        return y

    # pad: (B,T) where 1 indicates real token, 0 padding
    pad = y[pad_key]
    if pad.dtype != torch.bool:
        pad_mask = pad > 0
    else:
        pad_mask = pad

    B, T = pad_mask.shape
    device = pad_mask.device

    # Sample keep mask per timestep; always keep t=0 to avoid empty history edge cases
    keep = (torch.rand((B, T), device=device) < keep_prob)
    keep[:, 0] = True

    # Do NOT keep anything outside real tokens
    keep = keep & pad_mask

    out = {}
    for k, v in y.items():
        if k == pad_key:
            out[k] = v
            continue
        # v: (B,T,C)
        v2 = v.clone()
        # where keep is False, zero out the one-hot (no teacher signal)
        v2[~keep] = 0
        out[k] = v2
    out[pad_key] = y[pad_key]
    return out

class MixedBatchIterator:  # [ADDED]
    """
    Minimal mixed loader:
    yield (x, y) batches from paired_loader / unpaired_loader with probability.
    Keep output format identical to your original code: batch == (x, y).
    """
    def __init__(self, paired_loader: DataLoader, unpaired_loader: DataLoader, unpaired_ratio: float, seed: int = 42):
        self.paired_loader = paired_loader
        self.unpaired_loader = unpaired_loader
        self.unpaired_ratio = float(unpaired_ratio)
        self.rng = random.Random(seed)

    def __iter__(self):
        paired_it = iter(self.paired_loader)
        unpaired_it = iter(self.unpaired_loader)

        while True:
            use_unpaired = (self.rng.random() < self.unpaired_ratio)
            if use_unpaired:
                try:
                    batch = next(unpaired_it)
                except StopIteration:
                    unpaired_it = iter(self.unpaired_loader)
                    batch = next(unpaired_it)
            else:
                try:
                    batch = next(paired_it)
                except StopIteration:
                    paired_it = iter(self.paired_loader)
                    batch = next(paired_it)

            # IMPORTANT: keep exactly (x, y) format
            yield batch[0], batch[1]


class MultiSourceBatchIterator:
    """
    Yield homogeneous batches from ASAP paired, PianoCoRe paired, and optional unpaired XML loaders.

    Sampling order:
      1) unpaired batch with probability unpaired_ratio;
      2) otherwise PianoCoRe paired batch with probability pianocore_ratio;
      3) otherwise ASAP paired batch.
    """

    def __init__(
        self,
        asap_loader: DataLoader,
        pianocore_loader: DataLoader | None = None,
        pianocore_ratio: float = 0.0,
        unpaired_loader: DataLoader | None = None,
        unpaired_ratio: float = 0.0,
        seed: int = 42,
    ):
        self.asap_loader = asap_loader
        self.pianocore_loader = pianocore_loader
        self.pianocore_ratio = float(pianocore_ratio)
        self.unpaired_loader = unpaired_loader
        self.unpaired_ratio = float(unpaired_ratio)
        self.rng = random.Random(seed)

    @staticmethod
    def _next(loader: DataLoader, state: dict, name: str):
        try:
            return next(state[name])
        except (KeyError, StopIteration):
            state[name] = iter(loader)
            return next(state[name])

    def __iter__(self):
        state = {}
        while True:
            if self.unpaired_loader is not None and self.rng.random() < self.unpaired_ratio:
                batch = self._next(self.unpaired_loader, state, "unpaired")
            elif self.pianocore_loader is not None and self.rng.random() < self.pianocore_ratio:
                batch = self._next(self.pianocore_loader, state, "pianocore")
            else:
                batch = self._next(self.asap_loader, state, "asap")
            yield batch[0], batch[1]

class TrainRoformer(Roformer):
    """
    Add training_step / configure_optimizers on top of the model (Roformer is a LightningModule already).
    """

    def __init__(
        self,
        enc_configuration: MyModelConfig,
        dec_configuration: MyModelConfig,
        hyperparameters: dict,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        pad_loss_weight: float = 0.1,
        teacher_keep_prob_paired: float = 0.25, 
        teacher_keep_prob_unpaired: float = 0.50,
        warmup_steps: int = 4000,
        max_steps: int = 40000,
    ):
        super().__init__(enc_configuration, dec_configuration, hyperparameters)
        self.lr = lr
        self.weight_decay = weight_decay
        self.pad_loss_weight = pad_loss_weight
        self.teacher_keep_prob_paired = teacher_keep_prob_paired
        self.teacher_keep_prob_unpaired = teacher_keep_prob_unpaired
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        # Pre-build loss weights/ignore_index map from FEATURES
        self._feat_loss_weight = {k: v["loss_weight"] for k, v in FEATURES.items()}
        self._feat_ignore_index = {k: v["ignore_index"] for k, v in FEATURES.items()}

        self._bce = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def _ce_onehot(
        logits: torch.Tensor,         # (B,T,C)
        target_onehot: torch.Tensor,  # (B,T,C)
        pad_mask: torch.Tensor,       # (B,T) bool, True for real tokens
        ignore_index: int,
    ) -> torch.Tensor:
        """
        Cross entropy for one-hot targets, masked by pad.
        """
        target_idx = target_onehot.argmax(dim=-1).long()  # (B,T)

        B, T, C = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, C),
            target_idx.reshape(B * T),
            ignore_index=ignore_index,
            reduction="none",
        ).reshape(B, T)

        loss = loss * pad_mask.float()
        denom = pad_mask.float().sum().clamp_min(1.0)
        return loss.sum() / denom

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch  # both are dicts of bucketed tensors; y has "pad" (B,T)
        # pad mask for loss
        pad_mask = (y["pad"] > 0)

        # Determine if this batch is unpaired via x["unconditional"] (B,T,1)
        # unconditional==1 -> unpaired, unconditional==0 -> paired
        if "unconditional" in x:
            # x["unconditional"] could be (B,T,1) float
            is_unpaired = (x["unconditional"][:, 0, 0] > 0.5)  # (B,) bool
            # assume batch is homogeneous; MixedBatchIterator yields whole batches from one loader
            use_unpaired = bool(is_unpaired.all().item())
        else:
            use_unpaired = False

        keep_prob = self.teacher_keep_prob_unpaired if use_unpaired else self.teacher_keep_prob_paired
        
        y_in = _apply_history_dropout(y, keep_prob=keep_prob)
        self.log("train/is_unpaired", float(use_unpaired), prog_bar=False, on_step=True, on_epoch=True)
        self.log("train/teacher_keep_prob", float(keep_prob), prog_bar=False, on_step=True, on_epoch=True)
        
        y_hat = self.forward(x, y_in)  # dict: streams + "pad" logits

        total = 0.0
        OUT_FEATURE_KEYS = [k for k in FEATURES.keys() if k != "onset"]
        
        # stream classification losses
        for k in OUT_FEATURE_KEYS:
            w = self._feat_loss_weight.get(k, 0.0)
            if w <= 0:
                continue
            loss_k = self._ce_onehot(
                logits=y_hat[k],
                target_onehot=y[k],
                pad_mask=pad_mask,
                ignore_index=self._feat_ignore_index[k],
            )
            self.log(f"train/loss_{k}", loss_k, prog_bar=False, on_step=True, on_epoch=True)
            total = total + w * loss_k

        # pad head BCE loss: y_hat["pad"] (B,T,1), y["pad"] (B,T)
        pad_logits = y_hat["pad"].squeeze(-1)
        pad_target = y["pad"].float()
        pad_loss = self._bce(pad_logits, pad_target)
        pad_loss = (pad_loss * pad_mask.float()).sum() / pad_mask.float().sum().clamp_min(1.0)
        total = total + self.pad_loss_weight * pad_loss
        self.log("train/loss_pad", pad_loss, prog_bar=False, on_step=True, on_epoch=True)

        self.log("train/loss_total", total, prog_bar=True, on_step=True, on_epoch=True)
        return total

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        pad_mask = (y["pad"] > 0)
        y_hat = self.forward(x, y)  # val 不做 history dropout

        total = 0.0
        OUT_FEATURE_KEYS = [k for k in FEATURES.keys() if k != "onset"]
        for k in OUT_FEATURE_KEYS:
            w = self._feat_loss_weight.get(k, 0.0)
            if w <= 0:
                continue
            loss_k = self._ce_onehot(
                logits=y_hat[k],
                target_onehot=y[k],
                pad_mask=pad_mask,
                ignore_index=self._feat_ignore_index[k],
            )
            total = total + w * loss_k

        pad_logits = y_hat["pad"].squeeze(-1)
        pad_target = y["pad"].float()
        pad_loss = self._bce(pad_logits, pad_target)
        pad_loss = (pad_loss * pad_mask.float()).sum() / pad_mask.float().sum().clamp_min(1.0)
        total = total + self.pad_loss_weight * pad_loss

        self.log("val/loss_total", total, prog_bar=True, on_step=False, on_epoch=True)
        return total

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if get_cosine_schedule_with_warmup is None:
            return opt

        sch = get_cosine_schedule_with_warmup(
            opt,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=self.max_steps,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "interval": "step"},
        }


def build_model(args) -> TrainRoformer:
    # Encoder config
    enc_conf = MyModelConfig(
        is_decoder=False,
        add_cross_attention=False,
        is_autoregressive=False,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        embedding_size=args.hidden_size,  # embedding.py uses config.embedding_size
        bias=True,
    )
    # Decoder config
    dec_conf = MyModelConfig(
        is_decoder=True,
        add_cross_attention=True,
        is_autoregressive=True,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_attention_heads=args.num_heads,
        intermediate_size=args.intermediate_size,
        hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout,
        embedding_size=args.hidden_size,
        bias=True,
    )

    hyper = {"components": ["encoder", "decoder"]}

    return TrainRoformer(
        enc_configuration=enc_conf,
        dec_configuration=dec_conf,
        hyperparameters=hyper,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pad_loss_weight=args.pad_loss_weight,
        teacher_keep_prob_paired=args.teacher_keep_prob_paired,
        teacher_keep_prob_unpaired=args.teacher_keep_prob_unpaired,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--run_name", type=str, default="pm2s_roformer")
    parser.add_argument("--seed", type=int, default=42)

    # dataloader
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument(
        "--unpaired_list",
        type=str,
        default="",
        help="txt file, each line is an unpaired MusicXML absolute path",
    )
    parser.add_argument("--unpaired_ratio", type=float, default=0.5, help="probability of sampling an unpaired batch per step")
    parser.add_argument(
        "--surrogate_mode",
        type=str,
        default="pitch_onset_duration",
        choices=["pitch_only", "pitch_onset", "pitch_onset_duration"],
        help="How much rendered-MIDI surrogate information to use for unpaired data",
    )
    parser.add_argument(
        "--pdmx_mxl_root",
        type=str,
        default="/mnt/ssd/hbli/datasets/PDMX/mxl",
        help="Root directory of PDMX MXL files, used to infer corresponding rendered MIDI path",
    )
    parser.add_argument(
        "--pdmx_render_root",
        type=str,
        default="/mnt/ssd/hbli/datasets/PDMX/render",
        help="Root directory of MusicRender-exported MIDI files",
    )
    parser.add_argument(
        "--pianocore_manifest",
        type=str,
        default="",
        help="PianoCoRe manifest CSV for high-quality paired samples. Enable with --pianocore_ratio > 0.",
    )
    parser.add_argument(
        "--pianocore_root",
        type=str,
        default="/mnt/ssd/hbli/datasets/pianocore/",
        help="Root directory that contains PianoCoRe/raw files.",
    )
    parser.add_argument(
        "--pianocore_ratio",
        type=float,
        default=0.0,
        help="Probability of sampling PianoCoRe among paired batches after unpaired sampling.",
    )
    parser.add_argument("--pianocore_split", type=str, default="train")
    parser.add_argument("--pianocore_cache_dir", type=str, default="")
    parser.add_argument("--pianocore_max_rows", type=int, default=0)
    parser.add_argument(
        "--pianocore_no_require_chunk_file",
        action="store_true",
        help="Do not prefilter PianoCoRe manifest rows by existing *_pianocore_chunks.json files.",
    )
    parser.add_argument(
        "--pianocore_ignore_alignment_segments",
        action="store_true",
        help="Allow PianoCoRe training samples to concatenate across alignment_segment_id boundaries.",
    )

    # training schedule
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--warmup_steps", type=int, default=4000)

    # optimizer
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # model arch
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--intermediate_size", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)

    # loss tricks
    parser.add_argument("--pad_loss_weight", type=float, default=0.1)
    parser.add_argument("--teacher_keep_prob_paired", type=float, default=0.25)
    parser.add_argument("--teacher_keep_prob_unpaired", type=float, default=0.50)
    
    # runtime/debug
    parser.add_argument("--gpu_id", type=int, default=0, help="physical GPU id (use CUDA_VISIBLE_DEVICES or set this)")
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--fast_dev_run", action="store_true", help="Lightning fast dev run (1 train/val batch).")
    parser.add_argument("--limit_train_batches", type=float, default=1.0)
    parser.add_argument("--limit_val_batches", type=float, default=1.0)
    # output
    parser.add_argument("--out_dir", type=str, default="./runs", help="Root output directory for logs and checkpoints",)
    
    args = parser.parse_args()

    pl.seed_everything(args.seed, workers=True)

    # datasets
    train_set = ASAPDataset(
        data_dir=args.data_dir,
        split="train",
        seq_length=args.seq_length,
        cache=True,
        padding="per-beat",
        augmentations={
            "transpose": 12,
            "tempo_jitter": (0.8, 1.2),
            "duration_jitter": (0.95, 1.05),
            "onset_jitter": 0.05,
            "random_crop": True,
        },
        return_continous=False,
    )
    val_set = ASAPDataset(
        data_dir=args.data_dir,
        split="validation",
        seq_length=args.seq_length,
        cache=True,
        padding="per-beat",
        augmentations={},
        return_continous=False,
    )

    paired_loader = DataLoader(
        # [CHANGED] train_loader -> paired_loader
        train_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    pianocore_loader = None
    if args.pianocore_manifest and args.pianocore_ratio > 0:
        pianocore_set = PianoCoReDataset(
            manifest_path=args.pianocore_manifest,
            pianocore_root=args.pianocore_root,
            split=args.pianocore_split,
            seq_length=args.seq_length,
            cache=True,
            cache_dir=args.pianocore_cache_dir,
            padding="per-beat",
            augmentations={
                "transpose": 12,
                "tempo_jitter": (0.8, 1.2),
                "duration_jitter": (0.95, 1.05),
                "onset_jitter": 0.05,
                "random_crop": True,
            },
            return_continous=False,
            require_chunk_file=(not args.pianocore_no_require_chunk_file),
            max_rows=args.pianocore_max_rows,
            skip_on_error=True,
            respect_alignment_segments=(not args.pianocore_ignore_alignment_segments),
        )
        print(f"PianoCoRe paired train samples: {len(pianocore_set)}")
        pianocore_loader = DataLoader(
            pianocore_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            persistent_workers=(args.num_workers > 0),
        )

    # ---- Unpaired loader (optional) [ADDED]----
    unpaired_loader = None
    if args.unpaired_list and os.path.exists(args.unpaired_list):
        unpaired_set = UnpairedXMLDataset(
            mxl_paths=args.unpaired_list,
            seq_length=args.seq_length,
            cache=True,
            augmentations={"transpose": 12},
            surrogate_mode=args.surrogate_mode,
            mxl_root=args.pdmx_mxl_root,
            rendered_midi_root=args.pdmx_render_root,
            skip_on_error=True,
        )
        unpaired_loader = DataLoader(
            unpaired_set,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
            drop_last=True,
            persistent_workers=(args.num_workers > 0)
        )

    if pianocore_loader is not None or unpaired_loader is not None:
        train_loader = MultiSourceBatchIterator(
            asap_loader=paired_loader,
            pianocore_loader=pianocore_loader,
            pianocore_ratio=args.pianocore_ratio,
            unpaired_loader=unpaired_loader,
            unpaired_ratio=args.unpaired_ratio if unpaired_loader is not None else 0.0,
            seed=args.seed,
        )
    else:
        train_loader = paired_loader

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        num_workers=max(0, args.num_workers // 2),
        pin_memory=True,
        shuffle=False,
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(args)

    logger = CSVLogger(save_dir=args.out_dir, name=args.run_name)
    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.out_dir, args.run_name, "checkpoints"),
        filename="{step}-{val/loss_total:.4f}",
        save_top_k=3,
        monitor="val/loss_total",
        mode="min",
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="step")

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = [args.gpu_id] if accelerator == "gpu" else 1

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision=args.precision if accelerator == "gpu" else "32-true",
        max_steps=args.max_steps,
        gradient_clip_val=0.5,
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=20,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
        val_check_interval=1000,
        check_val_every_n_epoch=None,
        enable_checkpointing=True,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    main()
