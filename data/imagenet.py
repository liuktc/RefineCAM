from torchvision.datasets import ImageNet

def imagenet(root:str = "data", split: str = "val", transform=None):
    if split != "val":
        raise ValueError("Only 'val' split is downloaded.")
    return ImageNet(root=root, split=split, transform=transform)