import timm
from timm.data import create_transform
import torch

MODEL_NAME = "swin_tiny_patch4_window7_224"

model = timm.create_model(MODEL_NAME, pretrained=True)

swin_preprocess = create_transform(
    input_size=model.default_cfg["input_size"],  # (3, 224, 224)
    mean=model.default_cfg["mean"],
    std=model.default_cfg["std"],
    interpolation=model.default_cfg["interpolation"],
    crop_pct=model.default_cfg["crop_pct"],
)


def swin_imagenet():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    return model


def swin_PascalVOC():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = torch.nn.Linear(768, 20)
    return model


def swin_Synthetic():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = torch.nn.Linear(768, 6)
    return model


def swin_Synthetic_Small():
    model = timm.create_model(MODEL_NAME, pretrained=True)
    model.head.fc = torch.nn.Linear(768, 3)
    return model


# def swin_Funnybirds():
#     model = timm.create_model(MODEL_NAME, pretrained=True)
#     model.head.fc = torch.nn.Linear(768, 50)
#     return model


def swin_Funnybirds():
    pretrained_cfg_overlay = {"input_size": (3, 256, 256)}
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=50,
        pretrained_cfg_overlay=pretrained_cfg_overlay,
    )
    for param in model.parameters():
        param.requires_grad = True
    return model
