import torch


def get_device(use_gpu: bool = True, use_deterministic_ops: bool = False) -> str:
    """
    References:
        https://pytorch.org/docs/stable/notes/mps.html
        https://pytorch.org/docs/stable/notes/randomness.html
    """
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    elif use_gpu and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if use_deterministic_ops:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False

    return device
