from io import BytesIO
from urllib.request import urlopen

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAMPlusPlus, RefineCAM
from pytorch_grad_cam.metrics.arcc import ARCC
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.transforms import CenterCrop, Compose, Resize

# Model and image setup
model = resnet18(weights=ResNet18_Weights.DEFAULT).eval()

url = "https://github.com/pytorch/hub/raw/master/images/dog.jpg"
image = Image.open(BytesIO(urlopen(url).read())).convert("RGB")

input_tensor = ResNet18_Weights.DEFAULT.transforms()(image).unsqueeze(0)
rgb_img = np.asarray(Compose([Resize(256), CenterCrop(224)])(image)) / 255.0

predicted_class = [
    ClassifierOutputTarget(model(input_tensor).argmax(dim=1).item())
]  # Class 258: Samoyed dog

# Define and compute GradCAM++ and RefineCAM
gradcam = GradCAMPlusPlus(
    model=model,
    target_layers=[model.layer4[-1]],  # GradCAM++ uses a single layer
)
gradcam_result = gradcam(input_tensor, targets=predicted_class)

refinecam = RefineCAM(
    model=model,
    target_layers=[
        model.layer1[-1],  # RefineCAM uses multiple layers
        model.layer2[-1],  # and combines their outputs to produce
        model.layer3[-1],  # a refined attribution map.
        model.layer4[-1],
    ],
)
refinecam_result = refinecam(input_tensor, targets=predicted_class)

# Compute ARCC score on GradCAM++ and RefineCAM
gradcam_arcc = ARCC(base_method=gradcam)(
    input_tensor, gradcam_result, targets=predicted_class, model=model
)
refinecam_arcc = ARCC(base_method=refinecam)(
    input_tensor, refinecam_result, targets=predicted_class, model=model
)

# Visualize the results
fig, ax = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
for a, cam, title, arcc in zip(
    ax,
    [gradcam_result, refinecam_result],
    ["GradCAM++", "RefineCAM"],
    [gradcam_arcc, refinecam_arcc],
):
    a.imshow(show_cam_on_image(rgb_img, cam.squeeze(), use_rgb=True))
    a.set_title(f"{title}\nARCC: {arcc.item():.4f}")
    a.axis("off")

plt.tight_layout()
plt.show()
