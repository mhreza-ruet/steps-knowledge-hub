"""Compatibility fixes for DocLayout-YOLO 0.0.4's DocStructBench model."""

from doclayout_yolo.models.yolov10.train import YOLOv10DetectionTrainer
from doclayout_yolo.nn.tasks import attempt_load_one_weight
from doclayout_yolo.utils import LOGGER
from doclayout_yolo.utils.torch_utils import strip_optimizer
from doclayout_yolo.utils import metrics as yolo_metrics
import numpy as np
import random
import warnings
from contextlib import contextmanager
import torch


class ColorAugmentations:
    """The upstream active image-only transforms, without unused bbox options."""

    def __init__(self, p=1.0):
        import albumentations as A
        self.p = p
        self.transform = A.Compose([
            A.Blur(p=0.01), A.MedianBlur(p=0.01),
            A.ToGray(p=0.01), A.CLAHE(p=0.01),
        ])

    def __call__(self, labels):
        if len(labels["cls"]) and random.random() < self.p:
            labels["img"] = self.transform(image=labels["img"])["image"]
        return labels


@contextmanager
def training_compatibility(device):
    """Scope upstream warning fixes to this training/validation invocation."""
    from doclayout_yolo.data import augment, build
    original_augment, original_pin = augment.Albumentations, build.PIN_MEMORY
    try:
        augment.Albumentations = ColorAugmentations
        build.PIN_MEMORY = original_pin if str(device) not in ("cpu", "mps", "mps:0") else False
        with warnings.catch_warnings():
            # The package still uses these CUDA-only APIs, disabled on MPS.
            # Keep unrelated FutureWarnings and all runtime errors visible.
            warnings.filterwarnings("ignore", message=r"`torch\.cuda\.amp\.(GradScaler|autocast).*",
                                    category=FutureWarning, module=r"doclayout_yolo\.engine\.trainer")
            yield
    finally:
        augment.Albumentations, build.PIN_MEMORY = original_augment, original_pin


def compute_ap_numpy2(recall, precision):
    """Upstream's 101-point AP calculation using NumPy's current API."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    ap = np.trapezoid(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def plot_training_progress(csv_path):
    """Plot measured values only, excluding disabled/uncomputed zero losses."""
    import pandas as pd
    import matplotlib.pyplot as plt
    frame = pd.read_csv(csv_path)
    frame.columns = frame.columns.str.strip()
    # Upstream appends to results.csv when a run name is reused.
    restarts = np.flatnonzero(frame["epoch"].diff().fillna(1).to_numpy() <= 0)
    if len(restarts):
        frame = frame.iloc[restarts[-1]:]
    columns = [
        ("train/box_om", "Box error (one-to-many)"),
        ("train/cls_om", "Class error (one-to-many)"),
        ("train/box_oo", "Box error (one-to-one)"),
        ("train/cls_oo", "Class error (one-to-one)"),
        ("metrics/precision(B)", "Validation precision"),
        ("metrics/recall(B)", "Validation recall"),
        ("metrics/mAP50(B)", "Validation mAP50"),
        ("metrics/mAP50-95(B)", "Validation mAP50–95"),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, columns):
        ax.plot(frame["epoch"], frame[key])
        ax.set(title=title, xlabel="Epoch")
        ax.grid(alpha=0.2)
        if key.startswith("metrics/"):
            ax.set_ylim(0, 1)
    fig.suptitle("Training errors: lower is better • Validation scores: higher is better")
    return fig


# Patch only DocLayout's metric function, without changing NumPy globally.
if not hasattr(np, "trapz"):
    yolo_metrics.compute_ap = compute_ap_numpy2


class DocLayoutTrainer(YOLOv10DetectionTrainer):
    def plot_metrics(self):
        import matplotlib.pyplot as plt
        fig = plot_training_progress(self.csv)
        destination = self.save_dir / "results.png"
        fig.savefig(destination, dpi=160)
        plt.close(fig)
        self.on_plot(destination)

    def progress_string(self):
        # Upstream's numeric column queries CUDA only; label it accurately.
        result = super().progress_string().replace("GPU_mem", "CUDA_mem")
        if self.device.type == "mps":
            used = torch.mps.current_allocated_memory() / 1024**3
            LOGGER.info(f"Apple GPU (MPS) tensor memory: {used:.2f} GiB. "
                        "CUDA_mem does not apply on this Mac.")
        return result

    def final_eval(self):
        """Keep G2L_CRM's shared convolutions unfused when validating best.pt.

        Upstream's validator disables fusion for a model object containing
        G2L_CRM, but misses that check when given a checkpoint filename.
        Load the checkpoint unfused before passing it to the validator.
        """
        for checkpoint in (self.last, self.best):
            if checkpoint.exists():
                strip_optimizer(checkpoint)
                if checkpoint == self.best:
                    LOGGER.info(f"\nValidating {checkpoint} (without layer fusion)...")
                    self.validator.args.plots = self.args.plots
                    model, _ = attempt_load_one_weight(checkpoint, device=self.device, fuse=False)
                    self.metrics = self.validator(model=model)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
