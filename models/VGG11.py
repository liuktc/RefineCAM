import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg11
from torchvision.models.vgg import VGG11_Weights

vgg_preprocess = transforms.Compose(
    [
        # transforms.Pad(112),
        transforms.Resize((224, 224)),
        # transforms.Resize((112, 112)),

        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize(  # Normalize using ImageNet mean and std
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)


def vgg11_PascalVOC():
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(4096, 20)
    return model


def vgg11_Synthetic():
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(4096, 6)
    return model

def vgg11_Synthetic_Small():
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(4096, 3)
    return model


def vgg11_Imagenettewoof():
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(4096, 20)
    return model

def vgg11_Imagenet():
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    return model

def vgg11_Funnybirds():
    model = vgg11(weights=VGG11_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(4096, 50)
    return model