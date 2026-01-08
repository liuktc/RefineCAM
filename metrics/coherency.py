import torch
import torch.nn as nn
from .utils import mix_image_and_saliency, BaseMetric
from utils import AttributionMethod, Mixer
import matplotlib.pyplot as plt


def batch_pearson_coherency(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Computes Pearson correlation for a batch of matrices.

    Args:
        A (torch.Tensor): Tensor of shape (batch_size, m, n).
        B (torch.Tensor): Tensor of shape (batch_size, m, n).

    Returns:
        torch.Tensor: PCC for each matrix pair in the batch.
    """
    # Reshape to (batch_size, m*n)
    a = A.view(A.shape[0], -1)
    b = B.view(B.shape[0], -1)

    # Compute mean-centered matrices
    a_centered = a - a.mean(dim=1, keepdim=True)
    b_centered = b - b.mean(dim=1, keepdim=True)

    # Compute covariance and stds
    cov = (a_centered * b_centered).sum(dim=1) / (a.shape[1] - 1)
    std_a = torch.sqrt((a_centered**2).sum(dim=1) / (a.shape[1] - 1))
    std_b = torch.sqrt((b_centered**2).sum(dim=1) / (b.shape[1] - 1))

    # Avoid division by zero
    eps = 1e-8
    rho = cov / (std_a * std_b + eps)
    return rho


class Coherency(BaseMetric):
    def __init__(self):
        super().__init__("coherency")
        pass

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        previous_attributions: list[torch.Tensor],
        layer: nn.Module,
        upsample_method,
        mixer: Mixer,
        return_mean: bool = True,
        device: str = "cpu",
        reshape_transform=lambda x: x,
        **kwargs,
    ) -> torch.Tensor:
        # Coherency is defined as as the pearson correlation between the attribution on the image and the attribution on the image * saliency map
        mixed_images = mix_image_and_saliency(test_images, saliency_maps)

        mixed_attributions = attribution_method.attribute(
            input_tensor=mixed_images,
            model=model,
            layer=layer,
            target=class_idx,
            reshape_transform=reshape_transform,
        )

        mixed_attributions = upsample_method(
            attribution=mixed_attributions,
            image=mixed_images,
            device=device,
            model=model,
            layer=layer,
        )

        # Filter using the previous attributions, but firts make a copy of the previous attributions
        previous_attributions_copy = []

        for i in range(len(previous_attributions)):
            previous_attributions_copy.append(previous_attributions[i].detach().clone())

        previous_attributions_copy.append(mixed_attributions)
        mixed_attributions = mixer(previous_attributions_copy)

        if mixed_attributions.shape != saliency_maps.shape:
            raise ValueError(
                f"Mixed attributions shape {mixed_attributions.shape} does not match saliency maps shape {saliency_maps.shape}."
            )

        if len(mixed_attributions.shape) == 4:
            mixed_attributions = mixed_attributions.squeeze(1)
            saliency_maps = saliency_maps.squeeze(1)

        # Compute the correlation between mixed_attributions and saliency_maps
        pearson = (batch_pearson_coherency(mixed_attributions, saliency_maps) + 1) / 2

        if return_mean:
            return pearson.mean()

        return pearson.item()
