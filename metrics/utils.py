import torch
import torch.nn as nn


# Base class for all the metrics to extend
class BaseMetric:
    def __init__(self, name: str):
        self.name = name
        pass

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
    ) -> torch.Tensor:
        raise NotImplementedError


def mix_image_and_saliency(
    image: torch.Tensor, saliency_map: torch.Tensor
) -> torch.Tensor:
    """
    Mixes the image and the saliency map to create a new image.

    Parameters:

    image: torch.Tensor
        The input image. Shape: (B, C, H, W)

    saliency_map: torch.Tensor
        The saliency map. Shape: (B, C, H, W)
        Each element of the saliency map should be between 0 and 1.
    """
    # assert saliency_map.max() == 1 and saliency_map.min() == 0
    if saliency_map.max() != 1 or saliency_map.min() != 0:
        print("Saliency map should be normalized between 0 and 1")
        print(saliency_map.max(), saliency_map.min())
        raise ValueError("Saliency map should be normalized between 0 and 1")
    return image * saliency_map
