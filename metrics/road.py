from .utils import BaseMetric
from pytorch_grad_cam.metrics.road import (
    ROADCombined,
    ROADMostRelevantFirst,
    ROADLeastRelevantFirst,
)
import torch.nn as nn
import torch
from pytorch_grad_cam.utils.model_targets import ClassifierOutputSoftmaxTarget
from utils import AttributionMethod


class RoadCombined(BaseMetric):
    def __init__(self):
        super().__init__("road_combined")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        # attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        return_visualization: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        percentiles = [20, 40, 60, 80]
        road_combined = ROADCombined(percentiles=percentiles)
        targets = [ClassifierOutputSoftmaxTarget(i.item()) for i in class_idx]
        if len(saliency_maps.shape) == 4:
            saliency_maps = saliency_maps.squeeze(1)

        saliency_maps = saliency_maps.detach().cpu().numpy()
        scores_combined = road_combined(test_images.detach(), saliency_maps, targets, model)

        if return_mean:
            scores_combined = scores_combined.mean()

        if not return_visualization:
            return scores_combined

        # Calculate visualization
        visualization_results = []
        scores = []
        for imputer in [
            ROADMostRelevantFirst,
            ROADLeastRelevantFirst,
        ]:
            for perc in percentiles:
                score, visualizations = imputer(perc)(
                    test_images,
                    saliency_maps,
                    targets,
                    model,
                    return_visualization=True,
                )

                scores.append(score)

                # if imputer.__class__.__name__ not in visualization_results:
                #     visualization_results[imputer.__class__.__name__] = []

                visualization_results.append(visualizations[0].detach().cpu())

        return scores, scores_combined, visualization_results
