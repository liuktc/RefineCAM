import torch
import torch.nn as nn
from captum.attr import DeepLiftShap,DeepLift, IntegratedGradients, InputXGradient, Lime, KernelShap, LRP, GuidedGradCam
from .util import (
    cut_model_to_layer,
    cut_model_from_layer,
    min_max_normalize,
    calculate_erf,
    post_process_erf,
    calculate_erf_on_attribution,
)
from skimage.segmentation import slic
from pytorch_grad_cam import GradCAMPlusPlus, ScoreCAM, EigenCAM, LayerCAM, ShapleyCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from typing import List, Callable
import numpy as np

import psutil
import os

class FunnyBirdsExplainer():
    def explain(self, input):
        return self.explainer.explain(self.model, input)


    # image: the input image
    # part map: the corresponding segmentation map where one color denotes one part
    # target: the target class
    # colors_to_part: a list that maps colors to parts
    # thresholds: the different thresholds to use to estimate which parts are important
    # with_bg: include the background parts in the computation
    def get_important_parts(self, image, part_map, target, colors_to_part, thresholds,attribution=None, with_bg = False):
        """
        Outputs parts of the bird that are important according to the explanation.
        This must be reimplemented for different explanation types.
        Output is of the form: ['beak', 'wing', 'tail']
        """
        assert image.shape[0] == 1 # B = 1
        if attribution is None:
            attribution = self.explain(image, target=target)
        #m = nn.ReLU()
        #positive_attribution = m(attribution)

        part_importances = self.get_part_importance(image, part_map, target, colors_to_part, with_bg = with_bg, attribution=attribution)
        #total_attribution_in_parts = 0
        #for key in part_importances.keys():
        #    total_attribution_in_parts += abs(part_importances[key])

        important_parts_for_thresholds = []
        for threshold in thresholds:
            important_parts = []
            for key in part_importances.keys():
                if part_importances[key] > (attribution.sum() * threshold):
                    important_parts.append(key)
            important_parts_for_thresholds.append(important_parts)
        return important_parts_for_thresholds


    # image: the input image
    # part map: the corresponding segmentation map where one color denotes one part
    # target: the target class
    # colors_to_part: a list that maps colors to parts
    # with_bg: include the background parts in the computation
    def get_part_importance(self, image, part_map, target, colors_to_part,attribution=None, with_bg = False):
        """
        Outputs part importances for each part.
        """
        assert image.shape[0] == 1 # B = 1
        if attribution is None:
            attribution = self.explain(image, target=target)


        
        part_importances = {}

        dilation1 = nn.MaxPool2d(5, stride=1, padding=2)
        for part_color in colors_to_part.keys():
            torch_color = torch.zeros(1,3,1,1).to(image.device)
            torch_color[0,0,0,0] = part_color[0]
            torch_color[0,1,0,0] = part_color[1]
            torch_color[0,2,0,0] = part_color[2]
            color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
            
            color_available_dilated = dilation1(color_available)
            attribution_in_part = attribution * color_available_dilated
            attribution_in_part = attribution_in_part.sum()

            part_string = colors_to_part[part_color]
            part_string = ''.join((x for x in part_string if x.isalpha()))
            if part_string in part_importances.keys():
                part_importances[part_string] += attribution_in_part.item()
            else:
                part_importances[part_string] = attribution_in_part.item()

        if with_bg:
            for i in range(50):
                torch_color = torch.zeros(1,3,1,1).to(image.device)
                torch_color[0,0,0,0] = 204
                torch_color[0,1,0,0] = 204
                torch_color[0,2,0,0] = 204+i
                color_available = torch.all(part_map == torch_color, dim = 1, keepdim=True).float()
                color_available_dilated = dilation1(color_available)

                attribution_in_part = attribution * color_available_dilated
                attribution_in_part = attribution_in_part.sum()
                
                bg_string = 'bg_' + str(i).zfill(3)
                part_importances[bg_string] = attribution_in_part.item()

        return part_importances


    def get_p_thresholds(self):
        return np.linspace(0.01, 0.50, num=80)



def print_memory_usage():
    process = psutil.Process(os.getpid())
    print(
        f"Memory used: {process.memory_info().rss / 1024**2:.2f} MB"
    )  # Resident Set Size (RSS)


class AttributionMethod(FunnyBirdsExplainer):
    def __init__(self, name: str = None):
        self.name = name
        pass

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: nn.Module,
        target: torch.Tensor,
        baseline_dist: torch.Tensor = None,
    ):
        raise NotImplementedError()


class SimpleUpsampling(nn.Module):
    def __init__(self, size, mode="bilinear", name: str = "bilinearUpsampling"):
        super().__init__()
        self.size = size
        self.mode = mode
        self.up = nn.Upsample(size=self.size, mode=self.mode)
        self.name = name

    def forward(self, attribution, **kwargs):
        return self.up(attribution)


class ERFUpsampling(nn.Module):
    """
    Upsampling using effective receptive field (ERF) upsampling.
    """

    def __init__(
        self,
        # model,
        # layer: nn.Module,
        post_process_filter: Callable = post_process_erf,
    ):
        super().__init__()
        # self.model = model
        self.post_process_filter = post_process_filter

    def forward(
        self,
        attribution: torch.Tensor,
        image,
        device,
        model,
        layer,
        return_erf: bool = False,
        **kwargs,
    ):
        self.model = cut_model_to_layer(model, layer, included=True)

        erf = calculate_erf(self.model, image, device=device)
        self.layer_number = int(get_layer_name(model, layer).split(".")[1])

        # Rescale the attribution map using the ERF values
        result = np.zeros(erf.shape[2:], dtype=np.float32)
        assert attribution.shape[0] == 1 and attribution.shape[1] == 1
        attribution = attribution[0][0].detach().cpu().numpy()
        attribution = attribution.astype(np.float32)

        for i in range(erf.shape[0]):
            for j in range(erf.shape[1]):
                result += attribution[i, j] * erf[i, j]

        # Sum over channels
        result = result.sum(axis=0, keepdims=True)

        result = torch.Tensor(result).unsqueeze(0)

        result = min_max_normalize(result)

        if self.post_process_filter is not None:
            result = self.post_process_filter(result, self.layer_number)

        if return_erf:
            return torch.Tensor(result).to(device), erf
        else:
            return torch.Tensor(result).to(device)


def get_layer_name(model: nn.Module, layer: nn.Module):
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None


class ERFUpsamplingFast(nn.Module):
    def __init__(
        self,
        post_process_filter: Callable = post_process_erf,
        name: str = "ERFUpsamplingFast",
    ):
        super().__init__()
        self.post_process_filter = post_process_filter
        self.name = name

    def forward(self, attribution: torch.Tensor, image, device, model, layer, **kwargs):
        self.model = cut_model_to_layer(model, layer, included=True)
        self.layer_number = int(get_layer_name(model, layer).split(".")[1])
        # erf = calculate_erf(self.model, image, device=self.device)
        result = calculate_erf_on_attribution(self.model, image, attribution, device)

        result = result.to(device=device)
        result = result.sum(axis=0).unsqueeze(0).unsqueeze(0)

        result = min_max_normalize(result)

        if self.post_process_filter is not None:
            result = self.post_process_filter(result, self.layer_number)

        return torch.Tensor(result).to(device)


class CaptumWrapper(AttributionMethod):
    def __init__(self, attribution_method, name: str, use_baseline: bool, needs_layer: bool = False, use_feature_mask: bool = False):
        name = attribution_method.__name__
        super().__init__(name=name)
        self.attribution_method = attribution_method
        self.use_baseline = use_baseline
        self.needs_layer = needs_layer
        self.use_feature_mask = use_feature_mask


    def attribute(self, input_tensor, model, layer, target, normalize = True, feature_mask=None, **kwargs):
        if self.needs_layer:
            attr = self.attribution_method(model, layer=layer)
        else:
            attr = self.attribution_method(model)

        if self.use_baseline and self.use_feature_mask:
            baseline_dist = torch.randn_like(input_tensor) * 0.001
            attributions = attr.attribute(
                input_tensor, baseline_dist, target=target, feature_mask=feature_mask
            )
        elif self.use_baseline:
            baseline_dist = torch.randn_like(input_tensor) * 0.001
            attributions = attr.attribute(
                input_tensor, baseline_dist, target=target
            )
        elif self.use_feature_mask:
            attributions = attr.attribute(
                input_tensor, target=target, feature_mask=feature_mask
            )
        else:
            attributions = attr.attribute(
                input_tensor, target=target
            )
        attributions = attributions.sum(dim=1, keepdim=True)

        # This methods also returns negative attributions
        attributions = torch.abs(attributions)


        if normalize:
            attributions = min_max_normalize(attributions)

        return attributions


class CaptumDeepLift(CaptumWrapper):
    def __init__(self):
        super().__init__(DeepLift, name="DeepLiftShap", use_baseline=True)


class CaptumIntegratedGradients(CaptumWrapper):
    def __init__(self):
        super().__init__(IntegratedGradients, name="IntegratedGradients", use_baseline=False)


class CaptumInputXGradient(CaptumWrapper):
    def __init__(self):
        super().__init__(InputXGradient, name="InputXGradient", use_baseline=False)


class CaptumLime(CaptumWrapper):
    def __init__(self):
        super().__init__(Lime, name="Lime", use_baseline=False, use_feature_mask=True)

    def attribute(self, input_tensor, model, layer, target, normalize=True, **kwargs):
        # input_img: shape (1, 3, H, W)
        device = input_tensor.device
        input_img = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)

        # Create ~100 superpixels
        segments = slic(input_img, n_segments=100, compactness=10, sigma=1, start_label=0)

        # Convert segmentation mask into tensor
        feature_mask = torch.tensor(segments).unsqueeze(0).to(device)  # shape (1, H, W)

        # Restore input shape for Captum
        input_tensor = torch.tensor(input_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        return super().attribute(input_tensor, model, layer, target, normalize,feature_mask=feature_mask, **kwargs)


class CaptumKernelShap(CaptumWrapper):
    def __init__(self):
        super().__init__(KernelShap, name="KernelShap", use_baseline=False, use_feature_mask=True)

    def attribute(self, input_tensor, model, layer, target, normalize=True, **kwargs):
        # input_img: shape (1, 3, H, W)
        device = input_tensor.device
        input_img = input_tensor.squeeze().permute(1, 2, 0).detach().cpu().numpy()  # (H, W, 3)

        # Create ~100 superpixels
        segments = slic(input_img, n_segments=100, compactness=10, sigma=1, start_label=0)

        # Convert segmentation mask into tensor
        feature_mask = torch.tensor(segments).unsqueeze(0).to(device)  # shape (1, H, W)

        # Restore input shape for Captum
        input_tensor = torch.tensor(input_img).permute(2, 0, 1).unsqueeze(0).float().to(device)

        return super().attribute(input_tensor, model, layer, target, normalize,feature_mask=feature_mask, **kwargs)


class CaptumLRP(CaptumWrapper):
    def __init__(self):
        super().__init__(LRP, name="LRP", use_baseline=False)


class CaptumGuidedGradCam(CaptumWrapper):
    def __init__(self):
        super().__init__(GuidedGradCam, name="GuidedGradCam", use_baseline=False, needs_layer=True)


class _DeepLiftShap(AttributionMethod):
    def __init__(
        self, baseline_images: torch.Tensor = None, name: str = "DeepLiftShap"
    ):
        super().__init__(name=name)
        self.baseline_images = baseline_images

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        model_to = cut_model_to_layer(
            model, layer, included=True
        )  # Model from the start up to the layer
        model_from = cut_model_from_layer(
            model, layer, included=False
        )  # Model from the layer to the end

        input_to_model = model_to(input_tensor)
        dl = DeepLiftShap(model_from)
        if self.baseline_images is None:
            baseline_dist = torch.randn_like(input_to_model) * 0.001
        else:
            baseline_dist = model_to(self.baseline_images)

        attributions, delta = dl.attribute(
            input_to_model, baseline_dist, target=target, return_convergence_delta=True
        )
        attributions = attributions.sum(dim=1, keepdim=True)

        # This methods also returns negative attributions
        attributions = torch.abs(attributions)

        if normalize:
            attributions = min_max_normalize(attributions)

        return attributions


class _ScoreCAM(AttributionMethod):
    def __init__(
        self,
        name: str = "ScoreCAM",
    ):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = ScoreCAMNoResize(model=model, target_layers=[layer], **kwargs)

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute ScoreCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor


class ScoreCAMNoResize(ScoreCAM):
    """Just the ScoreCAM but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class _GradCAMPlusPlus(AttributionMethod):
    def __init__(
        self,
        name: str = "GradCAMPlusPlus",
    ):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = GradCAMPlusPlusNoResize(model=model, target_layers=[layer], **kwargs)

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute GradCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor


class GradCAMPlusPlusNoResize(GradCAMPlusPlus):
    """Just the GradCAMPlusPlus but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class EigenCAMNoResize(EigenCAM):
    """Just the EigenCAM but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class _EigenCAM(AttributionMethod):
    def __init__(
        self,
        name: str = "EigenCAM",
    ):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = EigenCAMNoResize(model=model, target_layers=[layer], **kwargs)

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute EigenCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor


class LayerCAMNoResize(LayerCAM):
    """Just the LayerCAM but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class _LayerCAM(AttributionMethod):
    def __init__(self, name: str = "LayerCAM"):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = LayerCAMNoResize(model=model, target_layers=[layer], **kwargs)

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute LayerCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor


class ShapleyCAMNoResize(ShapleyCAM):
    """Just the ShapleyCAM but without the rescaling of the CAM attribution map."""

    def __init__(self, *args, **kwargs):
        if "reshape_transform" in kwargs:
            self.trans = kwargs.pop("reshape_transform")
        super().__init__(*args, **kwargs)

    def compute_cam_per_layer(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module],
        eigen_smooth: bool,
    ) -> np.ndarray:
        activations_list = [
            a.cpu().data.numpy() for a in self.activations_and_grads.activations
        ]
        grads_list = [
            g.cpu().data.numpy() for g in self.activations_and_grads.gradients
        ]
        # target_size = self.get_target_width_height(input_tensor)
        device = input_tensor.device

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            if type(layer_activations) is np.ndarray:
                layer_activations = torch.from_numpy(layer_activations).requires_grad_().to(device)
                layer_activations = self.trans(layer_activations)
                # print(layer_activations.shape)
            if type(layer_grads) is np.ndarray:
                layer_grads = torch.from_numpy(layer_grads).requires_grad_().to(device)
                layer_grads = self.trans(layer_grads)
                # print(layer_grads.shape)

            cam = self.get_cam_image(
                input_tensor,
                target_layer,
                targets,
                layer_activations,
                layer_grads,
                eigen_smooth,
            )
            cam = np.maximum(cam, 0)
            # scaled = scale_cam_image(cam, target_size)
            # cam_per_target_layer.append(scaled[:, None, :])
            cam_per_target_layer.append(cam[:, None, :])

        return cam_per_target_layer


class _ShapleyCAM(AttributionMethod):
    def __init__(self, name: str = "ShapleyCAM"):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        model: nn.Module,
        layer: nn.Module,
        target: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        with torch.enable_grad():
            cam = ShapleyCAMNoResize(model=model, target_layers=[layer], **kwargs)

            targets = [ClassifierOutputTarget(t) for t in target]

            # Compute ShapleyCAM
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)

            # Convert to tensor efficiently and move to same device as input
            grayscale_cam_tensor = torch.from_numpy(grayscale_cam).to(
                input_tensor.device
            )
            grayscale_cam_tensor = grayscale_cam_tensor.unsqueeze(1)

            if normalize:
                grayscale_cam_tensor = min_max_normalize(grayscale_cam_tensor)

            return grayscale_cam_tensor


class _RandomCAM(AttributionMethod):
    def __init__(self, name: str = "RandomCAM"):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        normalize: bool = True,
        **kwargs,
    ):
        if input_tensor.dim() != 4:
            raise ValueError(
                f"Input tensor must be 4D (batch_size, channels, height, width), but got {input_tensor.dim()}D"
            )
        
        # Random CAM
        random_cam = torch.rand((input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3])).to(input_tensor.device)
        if normalize:
            random_cam = min_max_normalize(random_cam)

        return random_cam
    
class _All1sCAM(AttributionMethod):
    def __init__(self, name: str = "All1sCAM"):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        **kwargs,
    ):
        if input_tensor.dim() != 4:
            raise ValueError(
                f"Input tensor must be 4D (batch_size, channels, height, width), but got {input_tensor.dim()}D"
            )
        
        # All 1s CAM
        all_ones_cam = torch.ones((input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3])).to(input_tensor.device)
        
        # Set a single pixel to 0
        all_ones_cam[:, :, 0, 0] = 0

        return all_ones_cam

class _HalfCAM(AttributionMethod):
    def __init__(self, name: str = "HalfCAM"):
        super().__init__(name=name)

    def attribute(
        self,
        input_tensor: torch.Tensor,
        **kwargs,
    ):
        if input_tensor.dim() != 4:
            raise ValueError(
                f"Input tensor must be 4D (batch_size, channels, height, width), but got {input_tensor.dim()}D"
            )
        
        # Half CAM
        half_cam = torch.ones((input_tensor.shape[0], 1, input_tensor.shape[2], input_tensor.shape[3])).to(input_tensor.device)

        half_cam*= 0.5

        # Set a single pixel to 0 and a single pixel to 1
        half_cam[:, :, 0, 0] = 0
        half_cam[:, :, -1, -1] = 1

        return half_cam
        