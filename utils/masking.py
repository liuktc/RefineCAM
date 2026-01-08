import torch
import numpy as np


class ChannelMeanMask:
    def __init__(self, image):
        self.image = image  # Store the original image

    def __call__(self, mask):
        return channel_mean_masking(self.image, mask)


def channel_mean_masking(
    image: torch.Tensor,
    masks: torch.Tensor,
) -> torch.Tensor:
    """
    Replace masked-out pixels in the image with the average value of their channel.

    Args:
        image (torch.Tensor): The input image of shape (C, H, W).
        mask (torch.Tensor): The binary mask of shape (nsamples, H, W), where 1 means keep the pixel, 0 means mask it out.

    Returns:
        torch.Tensor: The masked images of shape (nsamples, C, H, W).
    """
    # if mask and image are torch.tensor
    if isinstance(masks, torch.Tensor) and isinstance(image, torch.Tensor):
        # Compute the channel-wise spatial average
        channel_means = image.mean(dim=(1, 2), keepdim=True)  # Shape: (C, 1, 1)

        # Expand mask to apply to all channels
        masks = masks.unsqueeze(1)  # Shape: (nsamples, 1, H, W)

        # Replace masked-out pixels with the channel average
        masked_image = image * masks + (1 - masks) * channel_means
        return masked_image
    elif isinstance(masks, np.ndarray) and isinstance(masks, np.ndarray):
        # Compute the channel-wise spatial average
        channel_means = image.mean(axis=(1, 2), keepdims=True)  # Shape: (C, 1, 1)

        # Expand mask to apply to all channels
        mask = np.expand_dims(masks, axis=0)

        # Replace masked-out pixels with the channel average
        masked_image = image * mask + (1 - mask) * channel_means
        return masked_image
    else:
        raise ValueError("mask and image must be either torch.Tensor or np.ndarray")
