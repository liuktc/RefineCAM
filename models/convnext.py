import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_small
from torchvision.models.convnext import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights

convnext_preprocess = ConvNeXt_Tiny_Weights.DEFAULT.transforms()

def convnext_tiny_Imagenet():
    model = convnext_tiny(ConvNeXt_Tiny_Weights.DEFAULT)
    return model

def convnext_tiny_Synthetic():
    model = convnext_tiny(ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(768, 6)
    return model

def convnext_tiny_Funnybirds():
    model = convnext_tiny(ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(768, 50)
    return model

def convnext_small_Imagenet():
    model = convnext_small(ConvNeXt_Small_Weights.DEFAULT)
    return model

def convnext_small_Synthetic():
    model = convnext_small(ConvNeXt_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(768, 6)
    return model

def convnext_small_Funnybirds():
    model = convnext_small(ConvNeXt_Small_Weights.DEFAULT)
    model.classifier[-1] = nn.Linear(768, 50)
    return model