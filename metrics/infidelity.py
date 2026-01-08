from .utils import BaseMetric
import torch
import torch.nn as nn
import numpy as np
from captum.metrics import infidelity
from utils import AttributionMethod


# define a perturbation function for the input
def perturb_fn(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    noise = torch.tensor(
        np.random.normal(0, 0.003, inputs.shape), device=inputs.device
    ).float()
    return noise, inputs - noise


class Infidelity(BaseMetric):
    def __init__(self):
        super().__init__("infidelity")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        # attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        # apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        saliency_maps = saliency_maps.expand(-1, 3, -1, -1).to(device)
        test_images = test_images.to(device)
        class_idx = class_idx.to(device)
        res = infidelity(
            model, perturb_fn, test_images, saliency_maps, target=class_idx
        )

        if return_mean:
            res = res.mean()

        return res.detach().cpu().item()
