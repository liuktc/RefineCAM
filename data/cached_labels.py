import os
import pickle
from collections import defaultdict
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class CachedLabelIndexDataset(Dataset):
    def __init__(
        self, dataset, cache_path="label_index_cache.pkl", force_rebuild=False
    ):
        """
        Args:
            dataset: Your original dataset that has labels
            cache_path: Path to save/load the label-to-indices mapping
            force_rebuild: If True, always rebuild the cache even if it exists
        """
        self.dataset = dataset
        self.cache_path = Path(cache_path)

        # Try to load cached mapping if available and not forced to rebuild
        if not force_rebuild and self.cache_path.exists():
            with open(self.cache_path, "rb") as f:
                self.label_to_indices = pickle.load(f)
            print(f"Loaded label-to-indices mapping from {cache_path}")
        else:
            # Build the mapping from scratch
            self.label_to_indices = defaultdict(list)
            for idx in tqdm(
                range(len(dataset)), desc="Building label index cache", leave=False
            ):
                label = self._get_label(dataset, idx)
                self.label_to_indices[label].append(idx)

            # Convert to regular dict and store sorted indices
            self.label_to_indices = {
                label: sorted(indices)
                for label, indices in self.label_to_indices.items()
            }

            # Save the mapping
            with open(self.cache_path, "wb") as f:
                pickle.dump(self.label_to_indices, f)
            print(f"Saved label-to-indices mapping to {cache_path}")

        self.labels = list(self.label_to_indices.keys())

    def _get_label(self, dataset, idx):
        """Helper to handle different dataset return formats"""
        item = dataset[idx]
        if isinstance(item, (tuple, list)) and len(item) == 2:
            item = item[1]  # (image, label) format
            if isinstance(item, torch.Tensor):
                return item.item()
            else:
                return item
        elif isinstance(item, (tuple, list)) and len(item) == 3:
            item = item[2]
            if isinstance(item, torch.Tensor):
                return item.item()
            else:
                return item
        elif hasattr(dataset, "targets"):  # For datasets like torchvision ImageFolder
            return dataset.targets[idx]
        else:
            raise RuntimeError("Couldn't determine how to get labels from dataset")

    def get_by_label_and_index(self, label, index):
        """Get the index-th example of the specified label"""
        indices = self.label_to_indices[label]
        if index >= len(indices):
            raise IndexError(f"Label {label} only has {len(indices)} examples")
        return self.dataset[indices[index]]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
