import torch


def custom_kernel(data: torch.Tensor) -> torch.Tensor:
    """
    RGB to Grayscale conversion.

    Args:
        data: RGB tensor of shape (H, W, 3) with values in [0, 1]
    Returns:
        Grayscale tensor of shape (H, W) with values in [0, 1]
    """
    weights = torch.tensor([0.2989, 0.5870, 0.1140], device=data.device, dtype=data.dtype)
    return torch.sum(data * weights, dim=-1)
