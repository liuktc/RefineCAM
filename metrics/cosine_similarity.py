import torch

from .utils import BaseMetric
import torch.nn.functional as F


class CosineSimilarity(BaseMetric):
    def __init__(self):
        super().__init__("cosine_similarity")

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

        # Normalize mask and attribution to have L2 norm of 1
        mask_norm = mask / mask.norm(p=2)
        attribution_norm = attribution / attribution.norm(p=2)
        # Compute the cosine similarity
        cosine_sim = F.cosine_similarity(mask_norm, attribution_norm, dim=0)
        return cosine_sim.item()  # Return as a scalar value
