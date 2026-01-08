import os
from typing import Literal
import torch
from torch.utils.data import Dataset
from torchvision.datasets import VOCDetection
from tqdm.auto import tqdm

FROM_LABEL_TO_IDX = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}

FROM_IDX_TO_LABEL = {v: k for k, v in FROM_LABEL_TO_IDX.items()}


class PascalVOC2007(Dataset):
    def __init__(
        self,
        image_set: Literal["train", "val", "trainval", "test"],
        skip_difficult: bool = True,
        transform=None,
    ):
        super().__init__()
        self.dataset = VOCDetection(
            root="data", year="2007", image_set=image_set, download=True
        )
        self.skip_difficult = skip_difficult
        self.transform = transform

        self.indices = []
        self.cache_name = f"./data/pascal_voc_2007_{image_set}{'_no_diff' if skip_difficult else ''}.pt"
        self.create_indices()

    def create_indices(self):
        if os.path.exists(self.cache_name):
            self.indices = torch.load(self.cache_name)
            return

        for idx in tqdm(range(len(self.dataset)), desc="Creating indices"):
            img, details = self.dataset[idx]
            objects = details["annotation"]["object"]
            cont = 0
            for obj in objects:
                if self.skip_difficult and int(obj["difficult"]) == 1:
                    continue

                cont += 1

                label = FROM_LABEL_TO_IDX[obj["name"]]
                label = torch.Tensor([label]).long().reshape(-1)

                self.indices.append((idx, cont, label))
        # Save the indices in cache
        torch.save(self.indices, self.cache_name)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        idx, cont, _ = self.indices[index]
        img, details = self.dataset[idx]
        objects = details["annotation"]["object"]
        n = 0
        for obj in objects:
            if self.skip_difficult and int(obj["difficult"]) == 1:
                continue

            n += 1
            if n == cont:
                bounding_box = obj["bndbox"]

                x_min = int(bounding_box["xmin"])
                y_min = int(bounding_box["ymin"])
                x_max = int(bounding_box["xmax"])
                y_max = int(bounding_box["ymax"])

                img_obj = img.crop((x_min, y_min, x_max, y_max))
                if self.transform:
                    img_obj = self.transform(img_obj)

                label = FROM_LABEL_TO_IDX[obj["name"]]
                label = torch.Tensor([label]).long().reshape(-1)

                return img_obj, label
