import torch


def detect_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
