import torch
from sklearn.metrics import roc_auc_score

from .utils import BaseMetric


class Mass_IOU(BaseMetric):
    def __init__(self):
        super().__init__("mass_iou")

    def __call__(
        self,
        saliency_maps: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        mask = mask.detach().clone()
        attribution = saliency_maps.detach().clone()

        if mask.shape != attribution.shape:
            raise ValueError(
                f"mask and attribution shape mismatch, {mask.shape} != {attribution.shape}"
            )

        if len(mask.shape) != 4:
            raise ValueError(
                f"mask and attribution should have 4 dimensions, actual shape: {mask.shape}"
            )

        if mask.shape[0] != 1 or mask.shape[1] != 1:
            raise ValueError(
                f"mask and attribution should have dimensions (1, 1, H, W), actual shape: {mask.shape}"
            )

        mask = mask.flatten()
        attribution = attribution.flatten()

        # Check that the mask is binary
        if not torch.logical_or(mask == 0, mask == 1).all():
            raise ValueError("mask should be binary (0 or 1)")

        mass_inside = (attribution * mask).sum()
        mass_outside = (attribution * (1 - mask)).sum()

        return mass_inside / (mass_inside + mass_outside)
