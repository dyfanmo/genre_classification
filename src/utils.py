from pathlib import Path

import torch


def get_device() -> str:
    """
    Return 'cuda' if a CUDA-capable GPU is available, otherwise 'cpu'.
    Logs basic information for debugging.
    """
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"CUDA available: {gpu_name}")
        except Exception:
            print("CUDA available but device name not accessible.")
        return "cuda"

    print("CUDA not available — using CPU")
    return "cpu"


def ensure_directory(path: Path) -> None:
    """
    Ensure that the given directory exists. If it does not, create it.
    """
    path.mkdir(parents=True, exist_ok=True)
