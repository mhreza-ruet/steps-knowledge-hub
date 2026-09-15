#!/usr/bin/env python3
"""
step04a_finetune_doclayout_yolo.py
-----------------------------------
Fine-tunes DocLayout-YOLO (or plain YOLOv10) on the dataset produced by
step04a_prepare_yolo_dataset.py.

IMPORTANT -- pretrained weights:
This sandbox's network is restricted to package registries (pypi, npm,
etc.) and can't reach huggingface.co or github.com release assets, so it
can't download the official DocStructBench pretrained checkpoint itself.
On your machine (normal internet access), download it once from:
    https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench
  (the .pt file, e.g. doclayout_yolo_docstructbench_imgsz1024.pt)
and pass its local path via --weights. Without --weights, this script
falls back to a random-initialized yolov10s architecture (bundled locally,
no download needed) -- useful only for a quick smoke test of the training
code path, NOT for real results: fine-tuning from a document-pretrained
checkpoint is what actually matters for a dataset this small.

Usage:
    python step04a_finetune_doclayout_yolo.py \\
        --data ./yolo_dataset/data.yaml \\
        --weights /path/to/doclayout_yolo_docstructbench_imgsz1024.pt \\
        --epochs 50 --imgsz 1024 --batch 8

Requirements:
    pip install doclayout-yolo torch
"""

import argparse
import sys
from contextlib import nullcontext
from functools import wraps

try:
    import torch
except ImportError:
    sys.exit("[Error] torch not installed. Run: pip install torch")

# doclayout_yolo (and older ultralytics) call torch.load() without weights_only=False
# when stripping the optimizer state from a finished run's checkpoint. torch >= 2.6
# changed that default to True, which makes the (harmless, first-party) checkpoint
# unpickling fail right at the end of training -- after the weights are already
# saved. Patch it back to the pre-2.6 behavior for this trusted, locally-produced
# checkpoint rather than letting a cosmetic crash follow a successful training run.
if (tuple(int(p) for p in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6)
        and not getattr(torch.load, "_doclayout_compatible", False)):
    _orig_torch_load = torch.load

    @wraps(_orig_torch_load)
    def _patched_torch_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return _orig_torch_load(*a, **kw)

    _patched_torch_load._doclayout_compatible = True
    torch.load = _patched_torch_load

try:
    from doclayout_yolo import YOLOv10
    _BACKEND = "doclayout_yolo"
except ImportError:
    try:
        from ultralytics import YOLO as YOLOv10
        _BACKEND = "ultralytics"
    except ImportError:
        sys.exit("[Error] Neither doclayout_yolo nor ultralytics installed. "
                  "Run: pip install doclayout-yolo")


def select_device() -> str:
    """Select CUDA, Apple Metal (MPS), or CPU for YOLO."""
    if torch.cuda.is_available():
        return "0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def finetune(data: str, weights: str = "yolov10s.yaml", epochs: int = 50, imgsz: int = 1024,
             batch: int = 8, device: str = None, project: str = "./runs",
             name: str = "doclayout_finetune", patience: int = None):
    """
    Fine-tunes (or, with a bare .yaml, trains from random init) a DocLayout-YOLO /
    YOLOv10 model. Returns (model, metrics) so a notebook can inspect results,
    plot metrics.results_dict, etc. right after the call.
    """
    device = device or select_device()
    print(f"Backend:  {_BACKEND}")
    print(f"Weights:  {weights}")
    print(f"Device:   {device}")
    if device == "cpu":
        print("[Note] Using CPU. Automatic GPU selection requires CUDA or MPS "
              "to be available in the installed PyTorch build.")

    model = YOLOv10(weights)
    training_options = {}
    compatibility = nullcontext()
    if _BACKEND == "doclayout_yolo":
        from doclayout_compat import DocLayoutTrainer, training_compatibility
        training_options["trainer"] = DocLayoutTrainer
        compatibility = training_compatibility(device)
    with compatibility:
        model.train(
            **training_options,
            data=data,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            exist_ok=True,
            patience=patience if patience is not None else max(20, epochs),
            plots=True,
        )

        print("\nRunning validation on the held-out val split...")
        metrics = model.val(data=data, device=device)
    print("\n" + "─" * 54)
    print(f"mAP50:    {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    names = model.names
    for class_id, ap in zip(metrics.box.ap_class_index, metrics.box.ap50):
        print(f"  {names[int(class_id)]:<18} AP50={ap:.4f}")

    print(f"\nBest weights saved under: {project}/{name}/weights/best.pt")
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to data.yaml from step04a_prepare_yolo_dataset.py.")
    parser.add_argument("--weights", default="yolov10s.yaml",
                         help="Pretrained .pt checkpoint (recommended: DocStructBench weights, "
                              "downloaded on a machine with internet access) OR a bundled .yaml "
                              "architecture config for a random-init smoke test (default: "
                              "yolov10s.yaml, no download needed, NOT useful for real accuracy).")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None,
                         help="'cpu', 'mps' (Apple GPU), '0' (first CUDA GPU), etc. "
                              "Default: auto-detect CUDA, then MPS, then CPU.")
    parser.add_argument("--project", default="./runs", help="Output directory for training runs.")
    parser.add_argument("--name", default="doclayout_finetune", help="Run name.")
    args = parser.parse_args()

    finetune(args.data, args.weights, args.epochs, args.imgsz, args.batch,
             args.device, args.project, args.name)


if __name__ == "__main__":
    main()
