import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur
from torcheval.metrics.aggregation.auc import AUC
from .utils import BaseMetric
from utils import AttributionMethod


class InsertionCurveAUC(BaseMetric):
    def __init__(self):
        super().__init__("insertion_curve_AUC")

    def __call__(
        self,
        model: nn.Module,
        test_images: torch.Tensor,
        saliency_maps: torch.Tensor,
        class_idx: torch.Tensor,
        attribution_method: AttributionMethod,
        device: torch.device | str = "cpu",
        apply_softmax: bool = True,
        return_mean: bool = True,
        **kwargs,
    ):
        B, C, H, W = test_images.shape
        ins_range, insertion = insertion_curve(
            model, test_images, saliency_maps, class_idx, device, apply_softmax
        )
        res = torch.zeros(B)
        for i in range(B):
            insertion_auc = AUC()
            insertion_auc.update(ins_range[i], insertion[i])
            res[i] = insertion_auc.compute()

        if return_mean:
            res = res.mean()
        return res.item()


def insertion_curve(
    model: nn.Module,
    images: torch.Tensor,
    saliency_maps: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device | str = "cpu",
    apply_softmax: bool = True,
    num_points: int = 30,
):
    """Generate the insertion curve as defined in https://arxiv.org/abs/1806.07421

    Args:
        model (nn.Module): The model to be evaluated
        image (torch.Tensor): The input image. Shape: (C, H, W)
        saliency_map (torch.Tensor): The saliency map. Shape: (H, W)
    """
    assert saliency_maps.shape[1] == 1, "Saliency map should be single channel"
    saliency_maps = saliency_maps.squeeze(1)

    B, C, H, W = images.shape
    num_pixels = H * W

    insertion_ranges = torch.zeros(B, num_points)
    insertion_values = torch.zeros(B, num_points)

    for b in range(B):
        image = images[b].unsqueeze(0)  # Shape: (1, C, H, W)
        saliency_map = saliency_maps[b]  # Shape: (H, W)
        # Apply gaussian filter to the image
        # Values taken from https://github.com/eclique/RISE/blob/master/Evaluation.ipynb
        kernel_size = 11
        sigma = 5
        blurred_image = gaussian_blur(image, kernel_size, sigma)

        sm_flatten = saliency_map.flatten()
        best_indices = sm_flatten.argsort().flip(
            0
        )  # Indices of the saliency map sorted in descending order

        pixel_removed_perc = torch.linspace(0, 1, num_points)
        res = torch.zeros_like(pixel_removed_perc)

        # plt.figure(figsize=(20, 20))
        # for i, perc in tqdm(enumerate(pixel_removed_perc)):
        for i, perc in enumerate(pixel_removed_perc):
            # plt.subplot(6, 5, i + 1)
            num_pixels_to_remove = int(num_pixels * perc)

            pixels_to_be_removed = best_indices[:num_pixels_to_remove]

            new_image = blurred_image.clone()
            # new_image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W] = (
            #     0  # Remove the pixel by setting it to a constant value
            # )
            new_image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W] = (
                image[0, :, pixels_to_be_removed // W, pixels_to_be_removed % W]
            )

            new_image = new_image.to(device)

            # plt.imshow(new_image[0].permute(1, 2, 0).detach().cpu().numpy())
            # print(new_image[0].max(), new_image[0].min())

            # Compute the prediction confidence on the class_idx
            with torch.no_grad():
                preds = model(new_image)[0]
                if apply_softmax:
                    preds = nn.functional.softmax(preds, dim=0)[labels[b]]
                res[i] = preds

        insertion_ranges[b] = pixel_removed_perc
        insertion_values[b] = res

    return insertion_ranges, insertion_values
