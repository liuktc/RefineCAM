from .VGG11 import (
    vgg11_PascalVOC,
    vgg_preprocess,
    vgg11_Synthetic,
    vgg11_Synthetic_Small,
    vgg11_Imagenet,
    vgg11_Funnybirds,
)
from .resnet import (
    resnet18_PascalVOC,
    resnet50_PascalVOC,
    resnet_preprocess,
    resnet18_Synthetic,
    resnet50_Synthetic,
    resnet18_Synthetic_Small,
    resnet50_Synthetic_Small,
    resnet18_Imagenet,
    resnet50_Imagenet,
    resnet18_Funnybirds,
    resnet50_Funnybirds,
)
from .Swin import (
    swin_imagenet,
    swin_PascalVOC,
    swin_Synthetic,
    swin_Synthetic_Small,
    swin_preprocess,
    swin_Funnybirds,
)
from .vit import vit_imagenet, vit_PascalVOC, vit_Synthetic, vit_preprocess
from .convnext import (
    convnext_tiny_Imagenet,
    convnext_tiny_Synthetic,
    convnext_small_Imagenet,
    convnext_small_Synthetic,
    convnext_preprocess,
    convnext_tiny_Funnybirds,
    convnext_small_Funnybirds,
)
