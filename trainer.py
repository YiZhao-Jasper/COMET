"""
COMET Trainer — Contrastive Warm-Start Flow Matching
=====================================================

Based on MOTFM/trainer.py (the original flow matching trainer), modified to
incorporate two key innovations:

1. **Warm-Start Coupling** (from Conditional Prior FM, TMLR):
   Replace x_0 ~ N(0, I) with x_0 = alpha * blur(mask) + sqrt(1-alpha^2) * eps.

2. **Contrastive Flow Loss** (from DeltaFM, ICCV 2025):
   loss = pos_error - temperature * neg_error, where negatives are
   created by shuffling target velocities within the batch.

Changes vs MOTFM/trainer.py are marked with `# [COMET]` comments.
"""
import argparse
import os
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.loggers import TensorBoardLogger

from flow_matching.path import AffineProbPath
from flow_matching.path.scheduler import CondOTScheduler

from utils.general_utils import create_dataloader, load_and_prepare_data, load_config
from utils.utils_fm import build_model, validate_and_save_samples

# [COMET] Import our contrastive + warm-start components
from losses.contrastive_flow import ContrastiveFlowLoss, create_warm_start_noise


class FlowMatchingDataModule(pl.LightningDataModule):
    """Lightning ``DataModule`` wrapping the existing data helpers.

    (Unchanged from MOTFM/trainer.py)
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.train_data: Optional[dict] = None
        self.val_data: Optional[dict] = None

    def setup(self, stage: Optional[str] = None) -> None:
        data_config = self.config["data_args"]
        model_config = self.config.get("model_args", {})

        spatial_dims = model_config.get("spatial_dims", None)
        if spatial_dims is not None:
            spatial_dims = int(spatial_dims)

        # Normalization knobs (optional in config; defaults preserve existing behavior).
        image_norm = data_config.get("image_norm", "minmax_0_1")
        mask_norm = data_config.get("mask_norm", "minmax_0_1")
        norm_scope = data_config.get("norm_scope", "global")
        clip_percentiles = data_config.get("clip_percentiles", None)
        if clip_percentiles is not None:
            clip_percentiles = (float(clip_percentiles[0]), float(clip_percentiles[1]))
        norm_eps = float(data_config.get("norm_eps", 1e-6))

        # Class mapping: prefer explicit ordering if provided.
        class_values = data_config.get("class_values", None)
        class_to_idx = {c: i for i, c in enumerate(class_values)} if class_values else None

        class_conditioning = bool(model_config.get("with_conditioning", False))
        expected_num_classes = None
        if class_conditioning:
            if model_config.get("cross_attention_dim", None) is None:
                raise ValueError(
                    "`model_args.with_conditioning` is True but `model_args.cross_attention_dim` is missing."
                )
            expected_num_classes = int(model_config["cross_attention_dim"])
            if class_values and expected_num_classes != len(class_values):
                raise ValueError(
                    f"`model_args.cross_attention_dim`={expected_num_classes} does not match "
                    f"`data_args.class_values` length ({len(class_values)})."
                )

        def _load(split: str) -> dict:
            return load_and_prepare_data(
                pickle_path=data_config["pickle_path"],
                split=split,
                convert_classes_to_onehot=True,
                spatial_dims=spatial_dims,
                image_norm=image_norm,
                mask_norm=mask_norm,
                norm_scope=norm_scope,
                clip_percentiles=clip_percentiles,
                norm_eps=norm_eps,
                class_to_idx=class_to_idx,
                num_classes=expected_num_classes,
                class_mapping_split=data_config.get("split_train", "train"),
            )

        mask_conditioning = bool(model_config.get("mask_conditioning", False))

        def _assert_required_keys(data: dict, *, split_name: str) -> None:
            if mask_conditioning and "masks" not in data:
                raise ValueError(
                    f"`model_args.mask_conditioning` is True but split '{split_name}' has no masks."
                )
            if class_conditioning and "classes" not in data:
                raise ValueError(
                    f"`model_args.with_conditioning` is True but split '{split_name}' has no classes."
                )

        if stage in (None, "fit"):
            self.train_data = _load(data_config["split_train"])
            self.val_data = _load(data_config["split_val"])
            _assert_required_keys(self.train_data, split_name=data_config["split_train"])
            _assert_required_keys(self.val_data, split_name=data_config["split_val"])
        elif stage == "validate":
            self.val_data = _load(data_config["split_val"])
            _assert_required_keys(self.val_data, split_name=data_config["split_val"])

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        tr_args = self.config["train_args"]
        sampler = None
        shuffle = True

        if bool(tr_args.get("class_balanced_sampling", False)):
            classes = None if self.train_data is None else self.train_data.get("classes")
            if classes is None:
                warnings.warn(
                    "`train_args.class_balanced_sampling` is enabled but no classes were found; "
                    "falling back to shuffling."
                )
            else:
                if classes.ndim == 2:
                    class_idxs = classes.argmax(dim=1).to(dtype=torch.long)
                    num_classes = int(classes.shape[1])
                elif classes.ndim == 1:
                    class_idxs = classes.to(dtype=torch.long)
                    num_classes = int(class_idxs.max().item() + 1)
                else:
                    raise ValueError(
                        f"Unexpected classes tensor shape {tuple(classes.shape)}; "
                        "expected [N] indices or [N, K] one-hot."
                    )

                counts = torch.bincount(class_idxs, minlength=num_classes).to(dtype=torch.float32)
                power = float(tr_args.get("class_balance_power", 1.0))
                class_weights = counts.clamp_min(1.0).pow(-power)
                sample_weights = class_weights[class_idxs].to(dtype=torch.double)
                sampler = torch.utils.data.WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                shuffle = False

        return create_dataloader(
            Images=self.train_data["images"],
            Masks=self.train_data.get("masks"),
            classes=self.train_data.get("classes"),
            batch_size=tr_args["batch_size"],
            shuffle=shuffle,
            sampler=sampler,
            num_workers=int(tr_args.get("num_workers", 0)),
            pin_memory=tr_args.get("pin_memory", None),
            persistent_workers=tr_args.get("persistent_workers", None),
            drop_last=bool(tr_args.get("drop_last", False)),
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        tr_args = self.config["train_args"]
        return create_dataloader(
            Images=self.val_data["images"],
            Masks=self.val_data.get("masks"),
            classes=self.val_data.get("classes"),
            batch_size=tr_args["batch_size"],
            shuffle=False,
            num_workers=int(tr_args.get("num_workers", 0)),
            pin_memory=tr_args.get("pin_memory", None),
            persistent_workers=tr_args.get("persistent_workers", None),
            drop_last=False,
        )


class FlowMatchingLightningModule(pl.LightningModule):
    """Lightning ``Module`` for COMET (Contrastive Warm-Start Flow Matching).

    Based on MOTFM/trainer.py::FlowMatchingLightningModule, with:
    - [COMET] Warm-start coupling replacing pure Gaussian noise
    - [COMET] Contrastive flow loss (from DeltaFM) added to training
    """

    def __init__(self, config: dict):
        super().__init__()
        self.save_hyperparameters(config)
        self.model = build_model(config["model_args"])
        self.mask_conditioning = config["model_args"]["mask_conditioning"]
        self.class_conditioning = config["model_args"]["with_conditioning"]
        self.path = AffineProbPath(scheduler=CondOTScheduler())

        # ---- [COMET] Contrastive loss (from DeltaFM/triplet_loss.py) ----
        comet_args = config.get("comet_args", config.get("cwsfm_args", {}))
        self.contrastive_weight = float(comet_args.get("contrastive_weight", 0.1))
        contrastive_temp = float(comet_args.get("contrastive_temperature", 1.0))
        self.contrastive_loss_fn = ContrastiveFlowLoss(temperature=contrastive_temp)
        self.contrastive_warmup_epochs = int(comet_args.get("contrastive_warmup_epochs", 5))

        # ---- [COMET] Warm-start parameters (from ConditionalPriorFM) ----
        self.warm_start_alpha = float(comet_args.get("warm_start_alpha", 0.3))
        self.warm_start_blur_kernel = int(comet_args.get("warm_start_blur_kernel", 15))
        self.warm_start_blur_sigma = float(comet_args.get("warm_start_blur_sigma", 5.0))
        self.use_warm_start = bool(comet_args.get("use_warm_start", True))
        self.use_contrastive = bool(comet_args.get("use_contrastive", True))

    def _compute_loss(self, batch: dict) -> torch.Tensor:
        im_batch = batch["images"]
        if self.mask_conditioning:
            if "masks" not in batch:
                raise KeyError(
                    "mask_conditioning is enabled but the dataloader batch has no 'masks' key."
                )
            mask_batch = batch["masks"]
        else:
            mask_batch = None

        if self.class_conditioning:
            if "classes" not in batch:
                raise KeyError(
                    "class_conditioning is enabled but the dataloader batch has no 'classes' key."
                )
            class_batch = batch["classes"]
        else:
            class_batch = None

        # ---- [COMET] Warm-start coupling ----
        # Original MOTFM: x_0 = torch.randn_like(im_batch)
        # Adapted from ConditionalPriorFM/train.py (lines 180-182):
        #   x0 = decoder(clip_texts, zi)
        #   x0 = x0 + (torch.randn_like(x0) * sigma)
        # Our version: x0 = alpha * blur(mask) + sqrt(1-alpha^2) * noise
        noise = torch.randn_like(im_batch)
        if self.use_warm_start and mask_batch is not None:
            x_0 = create_warm_start_noise(
                masks=mask_batch,
                noise=noise,
                alpha=self.warm_start_alpha,
                blur_kernel_size=self.warm_start_blur_kernel,
                blur_sigma=self.warm_start_blur_sigma,
            )
        else:
            x_0 = noise

        t = torch.rand(im_batch.shape[0], device=im_batch.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=im_batch)

        # Forward pass (same as MOTFM)
        v_pred = self.model(
            x=sample_info.x_t,
            t=sample_info.t,
            masks=mask_batch,
            cond=class_batch,
        )

        # ---- [COMET] Contrastive flow loss ----
        # Adapted from DeltaFM/triplet_loss.py::TripletSILoss.__call__
        # Original: denoising_loss = self.contrastive_loss(pred, images, d_alpha_t, d_sigma_t, noises)
        # In DeltaFM, they compute: model_target = d_alpha_t * images + d_sigma_t * noises
        # For CondOTScheduler, the target velocity dx_t = x_1 - x_0, which is sample_info.dx_t
        target_velocity = sample_info.dx_t

        # Check if contrastive should be active (with warm-up following DeltaFM practice)
        use_contrastive_now = (
            self.use_contrastive
            and self.contrastive_weight > 0
            and self.current_epoch >= self.contrastive_warmup_epochs
            and im_batch.shape[0] > 1  # need batch > 1 for negatives
        )

        # ---- Standard FM loss (always the primary objective, same as MOTFM) ----
        standard_mse = F.mse_loss(v_pred, target_velocity)

        if use_contrastive_now:
            # Contrastive regularizer: pos_error - temp * neg_error
            # (from DeltaFM/triplet_loss.py::compute_triplet_loss_efficiently)
            #
            # CRITICAL: In DeltaFM, the contrastive loss is stabilized by an
            # additional projection loss (proj_loss). Without it, using
            # contrastive_loss alone causes unbounded negative loss and divergence.
            #
            # Fix: Use standard MSE as PRIMARY loss + contrastive as REGULARIZER.
            # loss = standard_mse + weight * (pos_error - temp * neg_error)
            #      ≈ (1 + weight) * mse - weight * temp * neg_error
            # With weight=0.1, temp=0.01: loss ≈ 1.1*mse - 0.001*neg (always positive)
            contrastive_loss, diag = self.contrastive_loss_fn(v_pred, target_velocity)
            loss = standard_mse + self.contrastive_weight * contrastive_loss

            # Log diagnostics
            self.log("train/standard_mse", standard_mse,
                     prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/contrastive_loss", diag["contrastive/loss"],
                     prog_bar=False, on_step=True, on_epoch=True)
            self.log("train/cos_pos", diag["contrastive/cos_pos"],
                     prog_bar=False, on_step=False, on_epoch=True)
            self.log("train/cos_neg", diag["contrastive/cos_neg"],
                     prog_bar=False, on_step=False, on_epoch=True)
            self.log("train/contrastive_acc", diag["contrastive/accuracy"],
                     prog_bar=False, on_step=False, on_epoch=True)
        else:
            # Before warm-up: standard FM loss only (same as MOTFM)
            loss = standard_mse

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._compute_loss(batch)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        # Validation always uses standard MSE (no contrastive) for clean monitoring
        im_batch = batch["images"]
        mask_batch = batch.get("masks") if self.mask_conditioning else None
        class_batch = batch["classes"] if self.class_conditioning and "classes" in batch else None

        noise = torch.randn_like(im_batch)
        if self.use_warm_start and mask_batch is not None:
            x_0 = create_warm_start_noise(
                masks=mask_batch,
                noise=noise,
                alpha=self.warm_start_alpha,
                blur_kernel_size=self.warm_start_blur_kernel,
                blur_sigma=self.warm_start_blur_sigma,
            )
        else:
            x_0 = noise

        t = torch.rand(im_batch.shape[0], device=im_batch.device)
        sample_info = self.path.sample(t=t, x_0=x_0, x_1=im_batch)

        v_pred = self.model(
            x=sample_info.x_t,
            t=sample_info.t,
            masks=mask_batch,
            cond=class_batch,
        )
        loss = F.mse_loss(v_pred, sample_info.dx_t)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)

    def configure_optimizers(self) -> optim.Optimizer:
        lr = self.hparams["train_args"]["lr"]
        return optim.Adam(self.model.parameters(), lr=lr)

    def on_validation_epoch_end(self) -> None:
        """Run sampling/visualization at epoch end (same as MOTFM)."""
        # Avoid duplicate work under DDP.
        if hasattr(self.trainer, "is_global_zero") and not self.trainer.is_global_zero:
            return

        tr = self.hparams.get("train_args", {})
        solver_args = self.hparams.get("solver_args", {})

        # [COMET] Pass warm-start config to validation sampling
        comet_args = self.hparams.get("comet_args", self.hparams.get("cwsfm_args", {}))

        log_dir = None
        if getattr(self.trainer, "logger", None) is not None and hasattr(
            self.trainer.logger, "log_dir"
        ):
            log_dir = self.trainer.logger.log_dir
        if not log_dir:
            log_dir = self.trainer.default_root_dir

        val_loader = self.trainer.datamodule.val_dataloader()

        validate_and_save_samples(
            model=self.model,
            val_loader=val_loader,
            device=self.device,
            checkpoint_dir=log_dir,
            epoch=self.current_epoch,
            solver_config=solver_args,
            max_samples=tr.get("num_val_samples", 16),
            class_map=None,
            mask_conditioning=self.mask_conditioning,
            class_conditioning=self.class_conditioning,
            # [COMET] warm-start params for inference
            warm_start_config=comet_args if bool(comet_args.get("use_warm_start", True)) else None,
        )


def _resolve_resume_checkpoint(
    explicit_ckpt_path: Optional[str], root_ckpt_dir: str, run_name: str
) -> Optional[str]:
    """Return the checkpoint path to resume from, if any.

    (Unchanged from MOTFM/trainer.py)
    """
    if explicit_ckpt_path:
        return explicit_ckpt_path

    ckpt_dir = os.path.join(root_ckpt_dir, run_name)
    if not os.path.isdir(ckpt_dir):
        return None

    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    if os.path.isfile(last_ckpt):
        return last_ckpt

    _candidates = [
        os.path.join(ckpt_dir, fname)
        for fname in os.listdir(ckpt_dir)
        if fname.endswith(".ckpt") and os.path.isfile(os.path.join(ckpt_dir, fname))
    ]
    if not _candidates:
        return None

    return max(_candidates, key=os.path.getmtime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train COMET with Lightning.")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/comet_camus.yaml",
        help="Path to the configuration file.",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    run_name = os.path.splitext(os.path.basename(args.config_path))[0]
    tr = config["train_args"]
    root_ckpt_dir = tr["checkpoint_dir"]

    # Data and model modules
    datamodule = FlowMatchingDataModule(config)
    model = FlowMatchingLightningModule(config)

    # Logging and callbacks
    logger = TensorBoardLogger(save_dir=root_ckpt_dir, name=run_name)
    ckpt_dir = os.path.join(root_ckpt_dir, run_name)
    val_freq = max(1, int(tr.get("val_freq", 1)))
    save_every_n = max(1, int(tr.get("save_every_n_epochs", 10)))

    # Callback 1: Save BEST model (top-1 by val/loss)
    best_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-epoch{epoch:03d}-valloss{val/loss:.6f}",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
        every_n_epochs=val_freq,
    )

    # Callback 2: Save every N epochs (periodic snapshots)
    periodic_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="epoch{epoch:03d}-valloss{val/loss:.6f}",
        monitor=None,            # no metric filtering — save unconditionally
        save_top_k=-1,           # keep ALL periodic saves
        auto_insert_metric_name=False,
        every_n_epochs=save_every_n,
        save_last=True,          # always keep last.ckpt for resume
    )

    lr_cb = LearningRateMonitor(logging_interval="step")
    cbs = [best_cb, periodic_cb, lr_cb]

    # Precision setup with safe bf16/fp16 detection
    _bf16_supported = (
        torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)()
    )
    _fp16_supported = torch.cuda.is_available()
    default_precision = (
        "bf16-mixed" if _bf16_supported else ("16-mixed" if _fp16_supported else "32-true")
    )
    precision = tr.get("precision", default_precision)

    resume_ckpt = _resolve_resume_checkpoint(tr.get("ckpt_path"), root_ckpt_dir, run_name)
    if resume_ckpt:
        print(f"Resuming training from checkpoint: {resume_ckpt}")
    else:
        print("No checkpoint found. Starting training from scratch.")

    trainer = pl.Trainer(
        default_root_dir=root_ckpt_dir,
        max_epochs=tr["num_epochs"],
        precision=precision,
        accumulate_grad_batches=tr.get("gradient_accumulation_steps", 8),
        gradient_clip_val=tr.get("grad_clip_norm", 0.0) or None,
        check_val_every_n_epoch=val_freq,
        enable_progress_bar=True,
        logger=logger,
        callbacks=cbs,
        # Distributed/accelerator knobs
        accelerator=tr.get("accelerator", "auto"),
        devices=tr.get("devices", "auto"),
        strategy=DDPStrategy(find_unused_parameters=True),
        deterministic=tr.get("deterministic", False),
        log_every_n_steps=tr.get("log_every_n_steps", 50),
        num_sanity_val_steps=tr.get("num_sanity_val_steps", 0),
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    main()
