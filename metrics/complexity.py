import torch
import torch.nn as nn
from .utils import mix_image_and_saliency, BaseMetric
from utils import AttributionMethod


class Complexity(BaseMetric):
    def __init__(self):
        super().__init__("complexity")
        pass

    def __call__(
        self,
        saliency_maps: torch.Tensor,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        saliency_maps_clone = saliency_maps.detach().clone()
        if len(saliency_maps.shape) == 4:
            # If the saliency maps are 4D, we need to reduce them to 3D
            # by removing the channel dimension
            saliency_maps_clone = saliency_maps_clone.squeeze(1)

        # res = torch.linalg.norm(saliency_maps, ord=1, dim=(-2, -1))
        # Iterate over the batch dimension
        res = torch.zeros(
            saliency_maps_clone.shape[0], device=saliency_maps_clone.device
        )
        for i in range(saliency_maps_clone.shape[0]):
            res[i] = torch.linalg.vector_norm(saliency_maps_clone[i], ord=1)
        if return_mean:
            res = res.mean()
        return res.item()
