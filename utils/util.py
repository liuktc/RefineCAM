import torch.nn as nn
from collections import OrderedDict
import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.ndimage import gaussian_filter


def cut_model_from_layer(
    model: nn.Module,
    layer_name: str,
    included: bool = False,
    add_flatten_before_linear: bool = True,
) -> nn.Module:
    """
    Creates a new model that starts from the given layer of the original model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_name (str): The name of the layer from which to start the new model.
        included (bool): Whether to include the specified layer in the new model.
        add_flatten_before_linear (bool): Whether to add a flatten layer before the first linear layer.

    Returns:
        torch.nn.Module: A new model starting from the specified layer.
    """
    if isinstance(layer_name, nn.Module):
        layer_name = get_layer_name(model, layer_name)

    # Get all layers as an ordered dictionary
    modules = dict(model.named_modules())

    # Find the target layer
    if layer_name not in modules:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    # Create a new sequential model
    layers = OrderedDict()
    names = []
    layer_found = False
    flatten_added = False
    for name, module in model.named_modules():
        if name == layer_name:
            layer_found = True
        if layer_found:
            if name == layer_name and not included:
                continue
            else:
                if "." not in name:
                    continue

                if (
                    add_flatten_before_linear
                    and isinstance(module, nn.Linear)
                    and not flatten_added
                ):
                    layers["flatten"] = nn.Flatten()
                    flatten_added = True
                name = name.replace(".", "_")
                layers[name] = module
                names.append(name)

    if not layers or len(layers) == 0:
        raise ValueError(f"Could not cut the model from layer {layer_name}.")

    # Build and return the new model
    return nn.Sequential(layers)


def cut_model_to_layer(
    model: nn.Module,
    layer_name: nn.Module,
    included: bool = False,
) -> nn.Module:
    """
    Creates a new model that ends at the given layer of the original model.

    Args:
        model (torch.nn.Module): The PyTorch model.
        layer_name (str): The name of the layer at which to end the new model.
        included (bool): Whether to include the specified layer in the new model.

    Returns:
        torch.nn.Module: A new model ending at the specified layer.
    """
    if isinstance(layer_name, nn.Module):
        layer_name = get_layer_name(model, layer_name)

    # Get all layers as an ordered dictionary
    modules = dict(model.named_modules())

    # Find the target layer
    if layer_name not in modules:
        raise ValueError(f"Layer {layer_name} not found in the model.")

    # Create a new sequential model
    layers = OrderedDict()
    names = []
    for name, module in model.named_modules():
        if name == layer_name:
            if included:
                name = name.replace(".", "_")
                layers[name] = module
            break
        else:
            if "." not in name:
                continue

            name = name.replace(".", "_")
            layers[name] = module
            names.append(name)

    if not layers or len(layers) == 0:
        raise ValueError(f"Could not cut the model to layer {layer_name}.")

    # Build and return the new model
    return nn.Sequential(layers)


def set_relu_inplace(model: nn.Module, inplace=False):
    """
    Sets the inplace parameter of all ReLU layers to False.
    This is needed for the DeepExplainer rom the shap library to work correctly.
    """
    for child in model.children():
        if isinstance(child, nn.ReLU):
            child.inplace = inplace
        else:
            set_relu_inplace(child)


def scale_single_tensor(y: torch.Tensor, perc: float = 0.5, tolerance: float = 1e-5):
    """
    Scale the input tensor so that the area under the curve is perc of the total area.
    y is assumed to be a 2D tensor whose values are in the range [0,1].

    Args:
        - y (torch.Tensor): 2D tensor whose values are in the range [0,1].
        - perc (float, optional): Percentage of the total area. Defaults to 0.5.
        - tolerance (float, optional): Tolerance on the result. Defaults to 1e-5.

    Returns:
        torch.Tensor: The scaled version of y.
    """
    # Scale y so that the area under the curve is perc of the total area
    assert 0 <= perc <= 1, "perc must be in the range [0,1]"
    assert 0 <= tolerance, "tolerance must be positive"
    assert y.min() == 0 and y.max() == 1, "y must be in the range [0,1]"

    y = y.clone().detach()

    H, W = y.shape
    TOT = H * W

    def loss(y: torch.Tensor, perc: float):
        return (y.sum() - TOT * perc) ** 2

    alpha = torch.tensor(1.0, requires_grad=True)
    optimizer = torch.optim.Adam([alpha], lr=0.1)

    for _ in range(10000):
        optimizer.zero_grad()
        loss_val = loss(y ** (alpha**2), perc)
        loss_val.backward()
        optimizer.step()
        if loss_val.item() < tolerance:
            break

    return y ** (alpha**2)


def scale_saliencies(
    saliencies: torch.Tensor, perc: float = 0.5, tolerance: float = 1e-5
):
    """
    Scale the input tensor so that the area under the curve is perc of the total area.
    y is assumed to be a 2D tensor whose values are in the range [0,1].

    Args:
        - y (torch.Tensor): 2D tensor whose values are in the range [0,1].
        - perc (float, optional): Percentage of the total area. Defaults to 0.5.
        - tolerance (float, optional): Tolerance on the result. Defaults to 1e-3.

    Returns:
        torch.Tensor: The scaled version of y.
    """
    # y.shape = (B, C, H, W)
    B, C, H, W = saliencies.shape

    assert C == 1, "Only single channel saliencies are supported"

    for b in range(B):
        saliencies[b, 0] = scale_single_tensor(saliencies[b, 0], perc, tolerance)

    return saliencies


def get_layer_name(model: nn.Module, layer: nn.Module):
    for name, module in model.named_modules():
        if module == layer:
            return name
    return None


# def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
#     """Normalize the input tensor to the range [0,1].
#         It works along the last 2 dimensions.

#     Args:
#         x (torch.Tensor): The input tensor of shape (B, C, H, W).

#     Returns:
#         (torch.Tensor): The normalized tensor.
#     """
#     # Normalize the input tensor to the range [0,1]
#     # It works along the last 2 dimensions
#     return (x - x.amin((2, 3), True)) / (x.amax((2, 3), True) - x.amin((2, 3), True))
def min_max_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize the input tensor to [0,1] along H and W dims,
    while safely handling NaNs, infs, and zero-range cases.
    """

    # Replace NaN with 0, +inf with max finite, -inf with min finite
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # Compute per-channel min and max
    x_min = x.amin((2, 3), keepdim=True)
    x_max = x.amax((2, 3), keepdim=True)

    # Avoid division by zero: if max == min, the map is constant
    range_val = x_max - x_min
    zero_range_mask = (range_val < 1e-12)

    # Standard normalization
    out = (x - x_min) / (range_val + 1e-12)

    # For constant maps (max=min), return zeros instead of NaNs
    out = torch.where(zero_range_mask, torch.zeros_like(out), out)

    # Ensure result is strictly within [0,1]
    return out.clamp(0.0, 1.0)

def calculate_erf(model, input, device):
    # input_tensor = torch.tensor(input, device=device, requires_grad=True)
    input_tensor = input.clone().detach().requires_grad_(True).to(device)
    model.eval()
    output = model(input_tensor)

    input_shape = input_tensor.shape
    result = np.zeros(
        (
            output.shape[2],
            output.shape[3],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        )
    )
    for x in range(output.shape[2]):
        for y in range(output.shape[3]):
            input_tensor = input.clone().detach().requires_grad_(True).to(device)
            # Set the model to evaluation mode
            model.eval()

            # Forward pass: compute the output tensor
            output = model(input_tensor)
            # Initialize the output gradient as zero everywhere and 1 at a specific location
            grad_output = torch.zeros_like(output)

            grad_output[0, :, x, y] = 1  # Target a specific output unit

            # Backward pass: compute gradient of the output with respect to the input image
            output.backward(grad_output, retain_graph=True)

            # Retrieve the gradient of the input
            grad_input = input_tensor.grad.data[0].cpu().numpy()
            grad_input = np.abs(grad_input)  # Get the absolute values of the gradients
            grad_input = grad_input / (
                np.max(grad_input) + 1e-6
            )  # Normalize the gradients

            # Save the gradient of the input
            result[x, y] = grad_input

    return result


def calculate_erf_on_attribution(model, input, attribution_map, device):
    input_tensor = input.clone().detach().requires_grad_(True).to(device)
    model.eval()
    output = model(input_tensor)

    # Initialize the output gradient as zero everywhere and 1 at a specific location
    grad_output = torch.zeros_like(output)
    COUNT = 0
    for x in range(output.shape[2]):
        for y in range(output.shape[3]):
            COUNT += 1
            grad_output[0, :, x, y] = attribution_map[
                0, 0, x, y
            ]  # Target a specific output unit

    output.backward(grad_output, retain_graph=True)
    grad_input = input_tensor.grad.data[0].cpu()
    grad_input = torch.abs(grad_input)
    grad_input = grad_input / (torch.max(grad_input) + 1e-6)

    return grad_input


def get_sigma(n: int):
    # Assume n in [0,20]
    a = -20 / (np.log(1 / 5))
    return 5 * np.exp((n - 20) / a)


def post_process_erf(attribution_map: torch.Tensor, layer_number: int):
    # Assume the layer_number is in [0, 20]
    attribution_map = attribution_map.detach().cpu().numpy()
    attribution_map = np.where(
        attribution_map > np.percentile(attribution_map, 80), 1, attribution_map
    )

    attribution_map = gaussian_filter(attribution_map, sigma=get_sigma(layer_number))

    return attribution_map
