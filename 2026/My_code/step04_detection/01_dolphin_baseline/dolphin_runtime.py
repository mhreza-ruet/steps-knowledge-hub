"""Portable model loading; keep Dolphin's upstream inference/parsing methods."""

import torch
import demo_layout


def select_device():
    """Choose an available PyTorch GPU backend, otherwise CPU.

    ROCm-enabled PyTorch also exposes AMD GPUs through the CUDA API.
    Availability depends on the installed PyTorch build and drivers.
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    xpu = getattr(torch, "xpu", None)
    if xpu is not None and xpu.is_available():
        return "xpu"
    return "cpu"


class PortableDolphin(demo_layout.DOLPHIN):
    """Use upstream chat with automatic device placement at model load time."""

    def __init__(self, model_id_or_path):
        self.device = select_device()
        # Preserve upstream BF16 on capable CUDA/ROCm GPUs. Use FP32 on
        # other backends for conservative precision compatibility.
        dtype = (
            torch.bfloat16
            if self.device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        print(f"Loading Dolphin on {self.device} ({dtype})...")
        if self.device == "cpu":
            print("No supported GPU backend is available in this PyTorch installation.")
        self.processor = demo_layout.AutoProcessor.from_pretrained(model_id_or_path)
        self.model = demo_layout.Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id_or_path, torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
