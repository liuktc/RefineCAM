from .utils import BaseMetric
import torch
import torch.nn as nn
import numpy as np
from captum.metrics import sensitivity_max
from utils import AttributionMethod


class Sensitivity(BaseMetric):
    def __init__(self):
        super().__init__("sensitivity")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        # saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        # apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        # **kwargs needs to contain baseline_dist and layer
        def attribution_wrapper(images: torch.Tensor, model, layer, targets, **kwargs):
            if type(images) is tuple and len(images) == 1:
                images = images[0]

            BATCH_SIZE = 1
            FINAL_SIZE = 10 // BATCH_SIZE
            ATTRIBUTION_SHAPE = None

            res = []
            for i in range(0, len(images), BATCH_SIZE):
                batch = images[i : i + BATCH_SIZE].requires_grad_().to(device)
                # Repeat targets
                batch_targets = torch.repeat_interleave(targets, BATCH_SIZE, dim=0)

                attribution_res = (
                    attribution_method.attribute(
                        input_tensor=batch,
                        model=model,
                        layer=layer,
                        target=batch_targets,
                    )
                    .detach()
                    .cpu()
                )
                ATTRIBUTION_SHAPE = attribution_res.shape

                # If any of the attributions is NaN, skip the batch
                if torch.isnan(attribution_res).any():
                    print("A saliency map is NaN, skipping batch")
                    del batch, batch_targets, attribution_res
                    torch.cuda.empty_cache()
                    continue
                res.append(attribution_res)

            if len(res) == 0:
                # Build a random very big tensor of the same shape of attributions
                res = [
                    torch.randn(ATTRIBUTION_SHAPE) * 9999999 for _ in range(FINAL_SIZE)
                ]

            if len(res) != FINAL_SIZE:
                remaining = FINAL_SIZE - len(res)
                res += [res[-1]] * remaining

            res = torch.cat(res, dim=0)

            return res

        # Set the **kwargs to contain model, layer, targets, baseline_dist
        kwargs["model"] = model
        kwargs["targets"] = class_idx

        sens = sensitivity_max(attribution_wrapper, test_images, **kwargs)

        if (sens > 100).any():
            return None

        if return_mean:
            sens = torch.mean(sens)

        return sens.detach().cpu().item()
