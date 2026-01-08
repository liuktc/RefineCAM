import os
import hashlib
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from .util import draw_random_shapes, Binarize
from typing import Tuple, List
from torchvision import transforms
from torchvision.transforms import InterpolationMode

DEFAULT_BACKGROUND_TRANSFORM = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(
            brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15
        ),
    ]
)

def DEFAULT_MASK_PREPROCESS(size): 
        return transforms.Compose(
        [
            transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
            transforms.GaussianBlur(kernel_size=1),
            transforms.ToTensor(),
            Binarize(),
        ]
    )

def deterministic_walk(directory):
    for root, dirs, files in sorted(
        os.walk(directory), key=lambda x: x[0]
    ):  # Sort by root directory name
        dirs.sort()  # Sort directories in-place
        files.sort()  # Sort files in-place
        yield root, dirs, files


class SynteticFigures(Dataset):
    def __init__(
        self,
        background_path,
        num_shapes_per_image=1,
        size_ranges: Tuple[int, int] | List[Tuple[int, int]] = (80, 100),
        num_images=1000,
        split="train",
        image_transform=None,
        background_transform=None,
        mask_preprocess=None,
        include_crowns=True,
        return_shape_dim=False,
        image_size=(224, 224),
    ):
        super().__init__()
        # DEFAULT TRANSFORMATIONS
        if background_transform is None:
            background_transform = DEFAULT_BACKGROUND_TRANSFORM

        if mask_preprocess is None:
            mask_preprocess = DEFAULT_MASK_PREPROCESS(image_size)

        self.background_path = background_path
        self.image_transform = image_transform
        self.background_transform = background_transform
        self.mask_preprocess = mask_preprocess
        self.num_shapes_per_image = num_shapes_per_image
        self.return_shape_dim = return_shape_dim
        if not isinstance(size_ranges, (list, tuple)):
            raise ValueError(
                "size_ranges must be a tuple or a list of tuples"
            )
        self.size_ranges = size_ranges
        self.num_images = num_images
        self.include_crowns = include_crowns

        def hash_string(s: str) -> int:
            return int(hashlib.sha256(s.encode()).hexdigest(), 16) % 2**32

        self.initial_seed = hash_string(split)

        # Read all the images in the background path
        self.background_images = []
        for root, _, files in deterministic_walk(self.background_path):
            for file in files:
                if file.endswith(".jpg"):
                    self.background_images.append(os.path.join(root, file))

    def __len__(self):
        return self.num_images

    def __getitem__(self, index):
        seed = self.initial_seed + index
        # Seed the random generator with the index
        np.random.seed(seed)
        torch.manual_seed(seed)

        if index >= self.num_images:
            raise IndexError("Index out of bounds")

        background = cv2.imread(
            self.background_images[index % len(self.background_images)]
        )
        background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        background = torch.Tensor(background).type(torch.uint8).permute(2, 0, 1)
        ###############################################
        # background = torch.zeros_like(background)
        ###############################################

        if self.background_transform:
            background = self.background_transform(background)

        # Set the background back to numpy array
        background = background.permute(1, 2, 0).numpy().astype(np.int16)

        # If we include crowns, we have 6 classes (0-5), otherwise 3 classes (0-2)
        if self.include_crowns:
            label = np.random.randint(0, 6)
        else:
            label = np.random.randint(0, 3)

        # Select a random size range from the list or use the single tuple
        if isinstance(self.size_ranges, list):
            shape_dim = np.random.randint(0, len(self.size_ranges))
            size_range = self.size_ranges[shape_dim]
        else:
            shape_dim = 0
            size_range = self.size_ranges


        img, mask = draw_random_shapes(
            background,
            shape_type=label,
            num_shapes=self.num_shapes_per_image,
            size_range=size_range,
            seed=seed,
        )

        img = img.astype(np.uint8)
        mask = mask.astype(np.uint8)
        # img = np.transpose(img, (2, 0, 1))

        img = Image.fromarray(img)
        mask = Image.fromarray(mask)

        if self.image_transform:
            img = self.image_transform(img)

        if self.mask_preprocess:
            mask = self.mask_preprocess(mask)

        if not self.return_shape_dim:
            return img, mask, label
        else:
            return img, mask, label, shape_dim


class SyntheticFiguresSmall(SynteticFigures):
    def __init__(self, background_path, num_images, split="train",**kwargs):
        super().__init__(
            background_path=background_path,
            num_shapes_per_image=1,
            size_ranges=(15, 16),
            num_images=num_images,
            split=split,
            **kwargs
        )


class SyntheticFiguresAll(SynteticFigures):
    def __init__(self, background_path, num_images, split="train", **kwargs):
        super().__init__(
            background_path=background_path,
            num_shapes_per_image=1,
            size_ranges=[(80, 100), (40, 50), (20, 25)],
            num_images=num_images,
            split=split,
            **kwargs
        )