import timm
from timm.data import create_transform
import torch

MODEL_NAME = "vit_base_patch16_224"

model = timm.create_model(MODEL_NAME, pretrained=True)

vit_preprocess = create_transform(
    input_size=model.default_cfg["input_size"],  # (3, 224, 224)
    mean=model.default_cfg["mean"],
    std=model.default_cfg["std"],
    interpolation=model.default_cfg["interpolation"],
    crop_pct=model.default_cfg["crop_pct"],
)


def vit_imagenet():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    return model


def vit_PascalVOC():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = torch.nn.Linear(768, 20)
    return model


def vit_Synthetic():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = torch.nn.Linear(768, 6)
