"""
Metric based on the diffusion of an attribution map.
"""
from .utils import BaseMetric
from typing import List
import torch

def linear_interpolate(map_start: torch.Tensor, map_end: torch.Tensor, n:int = 20):
    # Compute n maps linearly interpolated between map_start and map_end
    return torch.stack([map_start + (map_end - map_start) * i / (n - 1) for i in range(n)], axis=0)

class DiffusionCurves():
    def __init__(self):
        pass

    def compute_curves(self,
                       test_images: torch.Tensor,
                       saliency_maps: torch.Tensor,
                       metrics: List[BaseMetric],
                       device: str,
                       n: int = 20,
                       **kwargs):
        test_images = test_images.to(device)
        saliency_maps = saliency_maps.to(device)

        B = test_images.shape[0]
        # Sample B uniformly distributed maps
        uniform_maps = torch.rand_like(saliency_maps)

        diffused_maps = linear_interpolate(saliency_maps, uniform_maps, n=n)
        diffused_maps = diffused_maps.to(device)

        scores_per_metric = {m.name: torch.zeros((B, n), device=device) for m in metrics}

        for map in diffused_maps:
            for m in metrics:
                # print(test_images.device, map.device)
                scores_per_metric[m.name] += m(test_images=test_images, saliency_maps=map, device=device, **kwargs)

        return scores_per_metric