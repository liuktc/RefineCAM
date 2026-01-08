from torch.utils.data import Dataset, ConcatDataset, Subset
from .imagenette import Imagenette
from .imagewoof import Imagewoof
import numpy as np


# Custom wrapper to offset labels in ImageWoof
class OffsetDataset(Dataset):
    def __init__(self, dataset, label_offset):
        self.dataset = dataset
        self.label_offset = label_offset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label + self.label_offset  # Offset label


def imagenettewoof(
    root: str = "data",
    split: str = "train",
    size: str = "full",
    download: bool = False,
    transform=None,
):
    TEST_SIZE = 1000
    if split == "train":
        imagenette = Imagenette(root, split, size, download, transform)
        imagewoof = Imagewoof(root, split, size, download, transform)

        # Offset ImageWoof labels
        imagewoof = OffsetDataset(imagewoof, 10)

        merged_dataset = ConcatDataset([imagenette, imagewoof])

        return merged_dataset
    elif split == "val":
        imagenette = Imagenette(root, split, size, download, transform)
        imagewoof = Imagewoof(root, split, size, download, transform)

        imagenette_total_size = len(imagenette)
        imagewoof_total_size = len(imagewoof)

        imagenette_val_size = imagenette_total_size - TEST_SIZE // 2
        imagewoof_val_size = imagewoof_total_size - TEST_SIZE // 2

        # Produces a random permutation with a fixed seed
        # to ensure reproducibility
        np.random.seed(42)
        imagenette_val_indices = np.random.permutation(imagenette_total_size)[
            :imagenette_val_size
        ]
        imagewoof_val_indices = np.random.permutation(imagewoof_total_size)[
            :imagewoof_val_size
        ]

        # imagenette_val_indices = list(range(imagenette_val_size))
        # imagewoof_val_indices = list(range(imagewoof_val_size))

        imagenette_val = Subset(imagenette, imagenette_val_indices)
        imagewoof_val = Subset(imagewoof, imagewoof_val_indices)

        # Offset ImageWoof labels
        imagewoof_val = OffsetDataset(imagewoof_val, 10)

        merged_dataset = ConcatDataset([imagenette_val, imagewoof_val])

        return merged_dataset
    elif split == "test":
        imagenette = Imagenette(root, "val", size, download, transform)
        imagewoof = Imagewoof(root, "val", size, download, transform)

        imagenette_total_size = len(imagenette)
        imagewoof_total_size = len(imagewoof)

        imagenette_test_size = TEST_SIZE // 2
        imagewoof_test_size = TEST_SIZE // 2

        np.random.seed(42)
        imagenette_test_indices = np.random.permutation(imagenette_total_size)[
            -imagenette_test_size:
        ]

        imagewoof_test_indices = np.random.permutation(imagewoof_total_size)[
            -imagewoof_test_size:
        ]

        # imagenette_test_indices = list(
        #     range(imagenette_total_size - imagenette_test_size, imagenette_total_size)
        # )
        # imagewoof_test_indices = list(
        #     range(imagewoof_total_size - imagewoof_test_size, imagewoof_total_size)
        # )

        imagenette_test = Subset(imagenette, imagenette_test_indices)
        imagewoof_test = Subset(imagewoof, imagewoof_test_indices)

        # Offset ImageWoof labels
        imagewoof_test = OffsetDataset(imagewoof_test, 10)

        merged_dataset = ConcatDataset([imagenette_test, imagewoof_test])

        return merged_dataset
    else:
        raise ValueError("split must be 'train' or 'val'")
