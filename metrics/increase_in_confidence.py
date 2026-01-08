import torch
import torch.nn as nn
from .utils import mix_image_and_saliency, BaseMetric


from utils import AttributionMethod


class IncreaseInConfidence(BaseMetric):
    def __init__(self):
        super().__init__("increase_in_confidence")
        pass

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: int | torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        The number of times in the entire dataset that the model's confidence increased when providing only
        the saliency map as input.

        Parameters:

        model: torch.nn.Module
            The model to be evaluated.

        test_images: torch.Tensor
            The test images to be evaluated. Shape: (N, C, H, W)

        saliency_maps: torch.Tensor
            The saliency maps to be evaluated. Shape: (N, C, H, W)

        class_idx: int | torch.Tensor
            If int: the index of the class to be evaluated, the same for all the input images.
            if torch.Tensor: the index of the class to be evaluated for each input image. Shape (N,)
        """

        test_images = test_images.to(device)
        saliency_maps = saliency_maps.to(device)
        saliency_images = mix_image_and_saliency(test_images, saliency_maps)

        test_preds = model(test_images)  # Shape: (N, num_classes)
        saliency_preds = model(saliency_images)  # Shape: (N, num_classes)

        if apply_softmax:
            test_preds = nn.functional.softmax(test_preds, dim=1)
            saliency_preds = nn.functional.softmax(saliency_preds, dim=1)

        # Select only the relevant class
        if isinstance(class_idx, int):
            test_preds = test_preds[:, class_idx]  # Shape: (N,)
            saliency_preds = saliency_preds[:, class_idx]  # Shape: (N,)
        elif isinstance(class_idx, torch.Tensor):
            test_preds = test_preds[torch.arange(test_preds.size(0)), class_idx]
            saliency_preds = saliency_preds[
                torch.arange(saliency_preds.size(0)), class_idx
            ]
        else:
            raise ValueError("class_idx should be either an int or a torch.Tensor")

        numerator = test_preds - saliency_preds
        numerator[numerator > 0] = 0
        numerator[numerator < 0] = 1

        denominator = len(test_preds)  # N

        res = torch.sum(numerator / denominator) * 100

        if return_mean:
            res = res.mean()

        return res.item()
