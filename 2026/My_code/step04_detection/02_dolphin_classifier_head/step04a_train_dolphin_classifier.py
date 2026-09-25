#!/usr/bin/env python3
"""
step04a_train_dolphin_classifier.py
-------------------------------------
Trains the classifier HEAD that sits on top of Dolphin's (over-)detections:
given a crop of one Dolphin-detected box, predict which of our classes
(Table / Plot / Diagram / Figure_Mixed) it is, or Background if it isn't
one of the things we want.

Approach: a small pretrained CNN backbone (default: torchvision ResNet18,
ImageNet weights) with its final layer replaced by a fresh Linear "head"
for our classes -- literally the classifier head described in the plan.
The backbone is frozen by default (only the new head trains), which is
the right call for a dataset this small (a few hundred crops at most,
mostly from ~10 annotated papers): fine-tuning the whole backbone would
overfit fast. Set --no-freeze-backbone once there's enough data to fine-tune
the whole thing.

Expects the ImageFolder-style layout step04a_build_classifier_dataset.py
writes:
    classifier_dataset/
      train/<class>/*.png
      val/<class>/*.png

The loss is class-weighted by default (inverse frequency) since Dolphin's
Background crops (headings, equations, etc.) heavily outnumber the classes
we actually want -- without weighting, a classifier can get "good" overall
accuracy just by predicting Background every time.

Usage:
    python step04a_train_dolphin_classifier.py \\
        ./classifier_dataset \\
        --epochs 30 --backbone resnet18 --pretrained

Requirements:
    pip install torch torchvision pillow
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    import torchvision
    from torchvision import datasets, transforms
except ImportError:
    sys.exit("[Error] torch/torchvision not installed. Run: pip install torch torchvision")


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

BACKBONES = {
    "resnet18": (torchvision.models.resnet18, "fc"),
    "resnet34": (torchvision.models.resnet34, "fc"),
    "mobilenet_v3_small": (torchvision.models.mobilenet_v3_small, "classifier"),
}


def select_device(preferred: str = None) -> str:
    if preferred:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model(backbone: str, num_classes: int, pretrained: bool, freeze_backbone: bool):
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from: {list(BACKBONES)}")
    ctor, head_attr = BACKBONES[backbone]

    weights = "DEFAULT" if pretrained else None
    model = ctor(weights=weights)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    # Replace the final classification layer with a fresh, trainable head
    # for our classes -- this is the only part that trains when the
    # backbone is frozen.
    if head_attr == "fc":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:  # mobilenet_v3_small: classifier is a small Sequential, last layer is Linear
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def build_dataloaders(data_root: Path, batch_size: int, image_size: int, num_workers: int = 0):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def _image_folder(root, tf):
        # allow_empty=True: a rare class (e.g. Figure_Mixed, only a handful
        # of examples total) can easily land with zero crops in one split
        # just from the paper-level train/val split -- without this,
        # ImageFolder raises instead of treating it as a legitimately empty
        # class in that split. Older torchvision doesn't have the kwarg.
        try:
            return datasets.ImageFolder(str(root), transform=tf, allow_empty=True)
        except TypeError:
            return datasets.ImageFolder(str(root), transform=tf)

    train_ds = _image_folder(data_root / "train", train_tf)
    val_ds = _image_folder(data_root / "val", val_tf)

    if train_ds.classes != val_ds.classes:
        raise ValueError(
            f"train/val class folders don't match: {train_ds.classes} vs {val_ds.classes} "
            f"-- likely a class with zero examples in one split (too little data for that class)."
        )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, train_ds.classes


def class_weights_from_dataset(dataset, num_classes: int) -> "torch.Tensor":
    counts = [0] * num_classes
    for _, label in dataset.samples:
        counts[label] += 1
    total = sum(counts)
    weights = [total / (num_classes * max(c, 1)) for c in counts]
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model, loader, device, num_classes: int):
    model.eval()
    confusion = [[0] * num_classes for _ in range(num_classes)]
    correct, total, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss_sum += criterion(outputs, labels).item() * images.size(0)
        preds = outputs.argmax(dim=1)
        for t, p in zip(labels.tolist(), preds.tolist()):
            confusion[t][p] += 1
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return {"loss": loss_sum / max(total, 1), "accuracy": correct / max(total, 1),
            "confusion": confusion, "n": total}


def train_classifier_head(data, backbone: str = "resnet18", pretrained: bool = True,
                           freeze_backbone: bool = True, epochs: int = 30, batch_size: int = 16,
                           image_size: int = 224, lr: float = 1e-3, class_weighted: bool = True,
                           device: str = None, num_workers: int = 0, verbose: bool = True):
    """
    Full job (usable from a notebook): builds dataloaders from an
    ImageFolder-style dataset, trains the classifier head, and returns
    (model, history, classes) where history is a list of per-epoch dicts.
    """
    data_root = Path(data).expanduser().resolve()
    if not (data_root / "train").exists() or not (data_root / "val").exists():
        raise FileNotFoundError(
            f"Expected {data_root}/train and {data_root}/val -- run "
            f"step04a_build_classifier_dataset.py first."
        )

    device = select_device(device)
    train_loader, val_loader, classes = build_dataloaders(data_root, batch_size, image_size, num_workers)
    num_classes = len(classes)

    model = build_model(backbone, num_classes, pretrained, freeze_backbone).to(device)

    weight = class_weights_from_dataset(train_loader.dataset, num_classes).to(device) if class_weighted else None
    criterion = nn.CrossEntropyLoss(weight=weight)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr)

    if verbose:
        n_trainable = sum(p.numel() for p in trainable_params)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Device: {device}  |  Backbone: {backbone} (pretrained={pretrained}, "
              f"frozen={freeze_backbone})  |  Classes: {classes}")
        print(f"Trainable params: {n_trainable:,} / {n_total:,}")

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            running_correct += (outputs.argmax(dim=1) == labels).sum().item()
            seen += images.size(0)

        train_loss = running_loss / max(seen, 1)
        train_acc = running_correct / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device, num_classes)

        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"]})
        if verbose:
            print(f"Epoch {epoch:>3}/{epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.3f}  "
                  f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.3f}")

    final_val = evaluate(model, val_loader, device, num_classes)
    return model, {"history": history, "classes": classes, "final_val": final_val, "device": device}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data", help="Path to the classifier_dataset/ folder "
                                      "(from step04a_build_classifier_dataset.py).")
    parser.add_argument("--backbone", choices=list(BACKBONES), default="resnet18")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                         help="Random-init backbone instead of ImageNet weights "
                              "(only useful as a network-free smoke test).")
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false",
                         default=True, help="Fine-tune the whole backbone instead of just the head.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no-class-weight", dest="class_weighted", action="store_false", default=True)
    parser.add_argument("--device", default=None, help="cpu / cuda / mps (default: auto-detect).")
    args = parser.parse_args()

    try:
        train_classifier_head(
            args.data, args.backbone, args.pretrained, args.freeze_backbone, args.epochs,
            args.batch_size, args.image_size, args.lr, args.class_weighted, args.device,
        )
    except FileNotFoundError as e:
        sys.exit(f"[Error] {e}")


if __name__ == "__main__":
    main()
