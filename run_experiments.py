import yaml
import torch
import numpy as np
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader, Subset
import logging  # Logging added
import colorlog
# Import your modules
from utils import (
    _GradCAMPlusPlus,
    _ShapleyCAM,
    _ScoreCAM,
    _EigenCAM,
    _LayerCAM,
    _RandomCAM,
    _All1sCAM,
    _HalfCAM,
    SimpleUpsampling,
    ERFUpsamplingFast,
    min_max_normalize,
    MultiplierMix,
    IdentityMix,
    NthRootMultiplierMix,
    LogExpMix,
    ExpMeanMix,
    CaptumDeepLift,
    CaptumIntegratedGradients,
    CaptumInputXGradient,
    CaptumLime,
    CaptumKernelShap,
    CaptumLRP,
    CaptumGuidedGradCam
)
from data import imagenettewoof, SynteticFigures, Binarize, imagenet, SyntheticFiguresSmall, SyntheticFiguresAll, FunnyBirds, FunnyBirdsSubset

from results.results_metrics import ResultMetrics
from metrics import (
    ROC_AUC,
    DeletionCurveAUC,
    InsertionCurveAUC,
    Infidelity,
    AverageDrop,
    Coherency,
    Complexity,
    RoadCombined,
    Mass_IOU,
    BCE,
    L2MaskNorm,
    CosineSimilarity,
    Accuracy,
    ControlledSyntheticDataCheck,
    SingleDeletion,
    PreservationCheck,
    DeletionCheck,
    TargetSensitivity,
    Sensitivity,
    Distractibility,
    BackgroundIndependence,
    DiffusionCurves
)
from models import (
    vgg11_Imagenettewoof,
    vgg11_Synthetic,
    vgg11_Imagenet,
    vgg11_Synthetic_Small,
    vgg11_Funnybirds,
    vgg_preprocess,
    resnet18_Imagenettewoof,
    resnet50_Imagenettewoof,
    resnet18_Synthetic,
    resnet50_Synthetic,
    resnet18_Synthetic_Small,
    resnet50_Synthetic_Small,
    resnet18_Imagenet,
    resnet50_Imagenet,
    resnet18_Funnybirds,
    resnet50_Funnybirds,
    resnet_preprocess,
    vit_imagenet,
    vit_imagenettewoof,
    vit_PascalVOC,
    vit_Synthetic,
    vit_preprocess,
    swin_imagenettewoof,
    swin_imagenet,
    swin_PascalVOC,
    swin_Synthetic,
    swin_Synthetic_Small,
    swin_Funnybirds,
    swin_preprocess,
    convnext_small_Synthetic,
    convnext_small_Imagenet,
    convnext_tiny_Synthetic,
    convnext_tiny_Imagenet,
    convnext_tiny_Funnybirds,
    convnext_preprocess,
)

import argparse
from tqdm.auto import tqdm

logging.root.handlers = []  # Remove all root handlers
logging.basicConfig(handlers=[], force=True)  # Override any basicConfig

formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(reset)s %(message)s',
    log_colors={
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
    }
)

# Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s - %(levelname)s - %(message)s",
#     handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
# )
logging.getLogger().handlers = []
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False 

handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)


logger.info("Starting program and setting up environment.")



# Read the config filename from command line arguments
parser = argparse.ArgumentParser(description="Run experiments with ML4CV_XAI.")
parser.add_argument(
    "--config_file",
    type=str,
    required=True,
    # default="config.yaml",
    help="Path to the configuration file.",
)
parser.add_argument(
    "--device",
    type=str,
    default=None,
    help="Device to run the experiments on (e.g., 'cpu', 'cuda:0'). Overrides config file.",
)
args = parser.parse_args()
logger.info(f"Using config file: {args.config_file}")
device = args.device

# Load config
with open(args.config_file, "r") as f:
    config = yaml.safe_load(f)
    config = {k.lower(): v for k, v in config.items()}
    for k, v in config.items():
        if "path" in k:
            continue
        if isinstance(v, list):
            config[k] = [x.lower() for x in v]
        elif isinstance(v, str):
            config[k] = v.lower()
        else:
            config[k] = v

# Override device if provided via command line
if device is None:
    device = config.get("device", "cpu")

if "cuda" in device and not torch.cuda.is_available():
    raise ValueError("CUDA is not available. Please set device to 'cpu'.")

if "seed" not in config:
    config["seed"] = 123
    logger.warning("No seed provided. Using default seed 123.")

torch.manual_seed(123)
np.random.seed(123)

logger.info("Loaded and processed configuration from config.yaml")

##################################################
# CONSTANTS
##################################################

MODELS = ["vgg11", "resnet18", "resnet50", "vit", "swin", "convnext_tiny", "convnext_small"]
DATASETS = ["imagenettewoof", "synthetic", "synthetic_small","synthetic_all", "imagenet", "funnybirds"]
ATTRIBUTION_METHODS = [
    "GradCAMPlusPlus",
    "ShapleyCAM",
    "ScoreCAM",
    "EigenCAM",
    "LayerCAM",
    "RandomCAM",
    "All1sCAM",
    "HalfCAM",
    "DeepLift",
    "IntegratedGradients",
    "InputXGradient",
    "Lime",
    "KernelShap",
    "LRP",
    "GuidedGradCam"
]
UPSCALE_METHODS = ["SimpleUpsampling", "ERFUpsamplingFast"]
METRICS = [
    "ROAD_combined",
    "ROC_AUC",
    "DeletionCurveAUC",
    "InsertionCurveAUC",
    "Infidelity",
    "AverageDrop",
    "Coherency",
    "Complexity",
    "Mass_IOU",	
    "BCE",
    "L2MaskNorm",
    "CosineSimilarity",
    "FunnyBirds"
]
MIXING_METHODS = [
    "IdentityMix",
    "MultiplierMix",
    "NthRootMultiplierMix",
    "LogExpMix",
    "ExpMeanMix"
]
# LAYERS = [4,3,2,1]

# Convert all the constant lists to lowercase
MODELS = [m.lower() for m in MODELS]
DATASETS = [d.lower() for d in DATASETS]
ATTRIBUTION_METHODS = [m.lower() for m in ATTRIBUTION_METHODS]
UPSCALE_METHODS = [m.lower() for m in UPSCALE_METHODS]
METRICS = [m.lower() for m in METRICS]
MIXING_METHODS = [m.lower() for m in MIXING_METHODS]


###############################################
# CONFIGURATION VALIDATION
###############################################

if config["model"] not in MODELS:
    raise ValueError(f"Model {config['model']} not in {MODELS}.")
if config["dataset"] not in DATASETS:
    raise ValueError(f"Dataset {config['dataset']} not in {DATASETS}.")
if not all(m in ATTRIBUTION_METHODS for m in config["attribution_methods"]):
    raise ValueError(
        f"Attribution methods {config['attribution_methods']} not in {ATTRIBUTION_METHODS}."
    )
if not all(m in UPSCALE_METHODS for m in config["upscale_methods"]):
    raise ValueError(
        f"Upscale methods {config['upscale_methods']} not in {UPSCALE_METHODS}."
    )
if not all(m in METRICS for m in config["metrics"]):
    raise ValueError(f"Metrics {config['metrics']} not in {METRICS}.")
if not all(m in MIXING_METHODS for m in config.get("mixing_methods", [])):
    raise ValueError(
        f"Mixing methods {config.get('mixing_methods', [])} not in {MIXING_METHODS}."
    )
logger.info(
    f"Configuration validated: Model={config['model']}, Dataset={config['dataset']}, "
    f"Attribution Methods={config['attribution_methods']}, Upscale Methods={config['upscale_methods']}, "
    f"Metrics={config['metrics']}"
)


#################################################
# MODEL
#################################################

if config["dataset"] == "imagenettewoof":
    models_map = {
        "vgg11": vgg11_Imagenettewoof,
        "resnet18": resnet18_Imagenettewoof,
        "resnet50": resnet50_Imagenettewoof,
        "vit": vit_imagenettewoof,
        "swin": swin_imagenettewoof,
        "convnext_tiny": None,
        "convnext_small": None,
    }
elif config["dataset"] == "synthetic" or config["dataset"] == "synthetic_all":
    models_map = {
        "vgg11": vgg11_Synthetic,
        "resnet18": resnet18_Synthetic,
        "resnet50": resnet50_Synthetic,
        "vit": vit_Synthetic,
        "swin": swin_Synthetic,
        "convnext_tiny": convnext_tiny_Synthetic,
        "convnext_small": convnext_small_Synthetic,
    }
elif config["dataset"] == "imagenet":
    models_map = {
        "vgg11": vgg11_Imagenet,
        "resnet18": resnet18_Imagenet,
        "resnet50": resnet50_Imagenet,
        "vit": vit_imagenet,
        "swin": swin_imagenet,
        "convnext_tiny": convnext_tiny_Imagenet,
        "convnext_small": convnext_small_Imagenet,
    }
elif config["dataset"] == "synthetic_small":
    models_map = {
        "vgg11": vgg11_Synthetic_Small,
        "resnet18": resnet18_Synthetic_Small,
        "resnet50": resnet50_Synthetic_Small,
        "vit": None,  # Assuming vit is used for PascalVOC in small dataset
        "swin": swin_Synthetic_Small,
        "convnext_tiny": None,
        "convnext_small": None,
    }
elif config["dataset"] == "funnybirds":
    models_map = {
        "vgg11": vgg11_Funnybirds,
        "resnet18": resnet18_Funnybirds,
        "resnet50": resnet50_Funnybirds,
        "vit": None,
        "swin": swin_Funnybirds,
        "convnext_tiny": convnext_tiny_Funnybirds,
        "convnext_small": None,
    }
else:
    raise ValueError(f"Dataset {config['dataset']} not supported.")

model = models_map[config["model"]]()
model.to(device)
model.eval()


logger.info(f"Model '{config['model']}' initialized.")
if "model_path" in config and config["model_path"]:
    if config["model_path"].endswith(".tar"):
        model.load_state_dict(torch.load(config["model_path"], map_location=device)['state_dict'])
    else:
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
    logger.info(f"Model weights loaded from {config['model_path']}.")
else:
    logger.warning("No model path provided. Using default weights.")
    logger.info("Model weights not loaded.")

preprocess_map = {
    "vgg11": vgg_preprocess,
    "resnet18": resnet_preprocess,
    "resnet50": resnet_preprocess,
    "vit": vit_preprocess,
    "swin": swin_preprocess,
    "convnext_tiny": convnext_preprocess,
    "convnext_small": convnext_preprocess,
}
preprocess = preprocess_map[config["model"]]

###############################################
# DATASET
###############################################

if config["dataset"] == "imagenettewoof":
    test_data = imagenettewoof(
        split="test", size="320px", download=False, transform=preprocess
    )
elif config["dataset"] == "synthetic":
    TRAIN_SIZE = 8
    TEST_SIZE = 6 * 100
    background_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15
            ),
        ]
    )
    mask_preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            transforms.GaussianBlur(kernel_size=3),
            transforms.ToTensor(),
            Binarize(),
        ]
    )
    test_data = SynteticFigures(
        background_path="./data/WaldoNoise",
        num_images=TEST_SIZE,
        split="test",
        num_shapes_per_image=1,
        image_transform=preprocess,
        background_transform=background_transform,
        mask_preprocess=mask_preprocess,
        size_ranges=(80, 100),
    )
elif config["dataset"] == "imagenet":
    test_data = imagenet(root="/media/data/ldomeniconi/imagenet_root", split="val", transform=preprocess)
elif config["dataset"] == "synthetic_small":
    TRAIN_SIZE = 8
    TEST_SIZE = 6 * 100
    background_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(
                brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15
            ),
        ]
    )
    mask_preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.NEAREST),
            transforms.GaussianBlur(kernel_size=1),
            transforms.ToTensor(),
            Binarize(),
        ]
    )
    test_data = SyntheticFiguresSmall(
        background_path="./data/WaldoNoise_test",
        num_images=TEST_SIZE,
        split="test",
        image_transform=preprocess,
        background_transform=background_transform,
        mask_preprocess=mask_preprocess,
    )
elif config["dataset"] == "synthetic_all":
    TEST_SIZE = 6 * 100

    test_data = SyntheticFiguresAll(
        background_path="./data/WaldoNoise_test",
        num_images=TEST_SIZE,
        split="test",
        image_transform=preprocess,
        image_size=(224,224),
    )
elif config["dataset"] == "funnybirds":
    test_data = FunnyBirds("/media/data/ldomeniconi/funnybirds/FunnyBirds", "test", get_part_map=True, transform=None)
else:
    raise ValueError(f"Dataset {config['dataset']} not supported.")
logger.info(f"Dataset '{config['dataset']}' loaded with {len(test_data)} samples.")


#############################################
# ATTRIBUTION METHODS
#############################################

attr_methods_map = {
    "gradcamplusplus": _GradCAMPlusPlus,
    "shapleycam": _ShapleyCAM,
    "scorecam": _ScoreCAM,
    "eigencam": _EigenCAM,
    "layercam": _LayerCAM,
    "randomcam": _RandomCAM,
    "all1scam": _All1sCAM,
    "halfcam": _HalfCAM,
    "deeplift": CaptumDeepLift,
    "integratedgradients": CaptumIntegratedGradients,
    "inputxgradient": CaptumInputXGradient,
    "lime": CaptumLime,
    "kernelshap": CaptumKernelShap,
    "lrp": CaptumLRP,
    "guidedgradcam": CaptumGuidedGradCam
}
attr_methods = [attr_methods_map[m]() for m in config["attribution_methods"]]
logger.info(f"Attribution methods initialized: {config['attribution_methods']}")

# Upscale methods
upscale_map = {
    "simpleupsampling": lambda: SimpleUpsampling((256, 256) if config["dataset"] == "funnybirds" else (224, 224)),
    "erupsamplingfast": ERFUpsamplingFast,
}
upscale_methods = [upscale_map[m]() for m in config["upscale_methods"]]
logger.info(f"Upscale methods initialized: {config['upscale_methods']}")

# ConvNext and LRP are not compatible
if config["model"] in ["convnext_tiny", "convnext_small"] and "lrp" in config["attribution_methods"]:
    logger.error("LRP is not compatible with ConvNext models. Please remove LRP from attribution methods.")
    exit(1)


#############################################
# METRICS
#############################################

metric_map = {
    "road_combined": RoadCombined,
    "roc_auc": ROC_AUC,
    "deletioncurveauc": DeletionCurveAUC,
    "insertioncurveauc": InsertionCurveAUC,
    "infidelity": Infidelity,
    "averagedrop": AverageDrop,
    "coherency": Coherency,
    "complexity": Complexity,
    "mass_iou": Mass_IOU,
    "bce": BCE,
    "l2masknorm": L2MaskNorm,
    "cosinesimilarity": CosineSimilarity,
}
funnybirds_metrics = {
    "accuracy": Accuracy,
    "controlledsyntheticdatacheck": ControlledSyntheticDataCheck,
    "singledeletion": SingleDeletion,
    "preservationcheck": PreservationCheck,
    "deletioncheck": DeletionCheck,
    "targetsensitivity": TargetSensitivity,
    "distractibility": Distractibility,
    "backgroundindependence": BackgroundIndependence,
}
if "funnybirds" in config["metrics"] and config["dataset"] != "funnybirds":
    logger.error("FunnyBirds metrics can only be used with the FunnyBirds dataset.")
    exit(1)

if "funnybirds" in config["metrics"]:
    metrics = [metric() for metric in funnybirds_metrics.values()]

    config["metrics"].remove("funnybirds")
else:
    metrics = []

metrics.extend([metric_map[m]() for m in config["metrics"]])

# If GT metrics are used, ensure we are in the synth dataset
if any(
    m in config["metrics"]
    for m in [
        "roc_auc"
        "mass_iou",
        "bce",
        "l2masknorm",
        "cosinesimilarity",
    ]
) and config["dataset"] not in ["synthetic", "synthetic_small", "synthetic_all"]:
    logger.error(
        "GT-based metrics can only be used with Synthetic datasets."
    )
    exit(1)

logger.info(f"Evaluation metrics initialized: {config['metrics']}")


# Main evaluation loop
logger.info("Starting main evaluation loop.")


#########################################
# LAYERS
#########################################

if config["model"] == "vgg11":
    layers = [
        model.features[20],
        model.features[15],
        model.features[10],
        model.features[5],
    ]
elif config["model"] == "resnet18":
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
elif config["model"] == "resnet50":
    layers = [model.layer4, model.layer3, model.layer2, model.layer1]
elif config["model"] == "vit":
    layers = [model.blocks[11].norm1, model.blocks[10].norm1, model.blocks[9].norm1, model.blocks[8].norm1]
elif config["model"] == "swin":
    layers = [model.layers[3].blocks[-1].norm1, model.layers[2].blocks[-1].norm1, model.layers[1].blocks[-1].norm1, model.layers[0].blocks[-1].norm1]
elif config["model"] == "convnext_tiny" or config["model"] == "convnext_small":
    layers = [model.features[-1][-1], model.features[-3][-1], model.features[-5][-1], model.features[-7][-1]]
else:
    raise ValueError(f"Model {config['model']} not supported.")
layers_names_map = {
    "vgg11": ["features.20", "features.15", "features.10", "features.5"],
    "resnet18": ["layer4", "layer3", "layer2", "layer1"],
    "resnet50": ["layer4", "layer3", "layer2", "layer1"],
    "vit": ["blocks.11", "blocks.10", "blocks.9", "blocks.8"],
    "swin": ["layers.3", "layers.2", "layers.1", "layers.0"],
    "convnext_tiny": ["features.7", "features.5", "features.3", "features.1"],
    "convnext_small": ["features.7", "features.5", "features.3", "features.1"],
}
layers_names = layers_names_map[config["model"]]

if config.get("only_final_layer", False):
    layers = [layers[0]]
    layers_names = [layers_names[0]]
    logger.warning("Only final layer will be used for attributions.")

# Only CAM methods support multiple layers
# if any(["cam" not in m for m in config["attribution_methods"]]) and not config.get("only_final_layer", False):
#     logger.error("Only CAM methods support multiple layers. Set only_final_layer = True.")
#     exit(1)



#######################################
# NUM_SAMPLES
#######################################

if "num_samples" in config:
    num_samples = config["num_samples"]
    if num_samples > len(test_data):
        raise ValueError(
            f"Number of samples {num_samples} exceeds dataset size {len(test_data)}."
        )
    # test_data = torch.utils.data.Subset(test_data, range(num_samples))
    logger.info(f"Using {num_samples} samples for evaluation.")
    INDICES = np.random.choice(
        len(test_data), num_samples, replace=False
    )
else:
    INDICES = np.arange(len(test_data))
    logger.info(f"Using all {len(test_data)} samples for evaluation.")

##########################################
# RESHAPE FUNCTION
##########################################
def reshape_transform(tensor, token_to_remove=0):
    if tensor.size(0) != 1 and tensor.dim() == 3:
        tensor = tensor.unsqueeze(1)
    
    if tensor.dim() == 4:
        result = tensor.transpose(2, 3).transpose(1, 2)
        return result
    
    tensor = tensor[:, token_to_remove:]
    num_elements = tensor.numel()
    height = int((num_elements/tensor.size(2)) ** 0.5)
    width = height
    result = tensor.reshape(tensor.size(0),
                            height, width, tensor.size(2))

    # Bring the channels to the first dimension, like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

from functools import partial
if config["model"] == "vit":
    reshape = partial(reshape_transform, token_to_remove=1)
elif config["model"] == "swin":
    reshape = partial(reshape_transform, token_to_remove=0)
else:
    reshape = lambda x: x

logger.info("Reshape function set for model.")

##########################################
# MIXING METHODS
##########################################
mixing_methods_map = {
    "identitymix": IdentityMix,
    "multipliermix": MultiplierMix,
    "nthrootmultipliermix": NthRootMultiplierMix,
    "logexpmix": LogExpMix,
    "expmeanmix": ExpMeanMix,
}

mixing_methods = [mixing_methods_map[m]() for m in config.get("mixing_methods", [])]
if len(mixing_methods) == 0:
    mixing_methods = [IdentityMix()]  # Default to IdentityMix if none specified
    logger.info("No mixing methods specified. Defaulting to IdentityMix.")

#########################################
# DIFFUSION CURVES
#########################################

if config.get("diffusion_curves", False):
    diffusion_curves = DiffusionCurves()
else:
    diffusion_curves = None

#########################################
# MAIN LOOP
#########################################

results = ResultMetrics(config["output_path"])
for index in tqdm(INDICES):
    if config["dataset"] == "funnybirds":
        funnybirds_test_dataset = FunnyBirdsSubset([index], "/media/data/ldomeniconi/funnybirds/FunnyBirds", "test", get_part_map=True, transform=None)
    else:
        funnybirds_test_dataset = None

    sample = test_data[index]
    if type(sample) == dict:
        images = sample['image'].to(device)
        labels = torch.tensor([sample['class_idx']], dtype=torch.long).to(device)
        mask = None
    elif len(sample) == 2:
        images, labels = sample
        mask = None
    else:
        images, mask, labels = sample
        mask = mask.unsqueeze(0).to(device)

    images = images.unsqueeze(0).to(device)
    labels = torch.tensor([labels], dtype=torch.long).to(device)
    pred_label = model(images).argmax(dim=1)

    if labels.item() != pred_label.item():
        logger.warning(f"Skipping index {index} as it is not correctly classified. True label: {labels.item()}, Predicted label: {pred_label.item()}")
        continue

    if diffusion_curves:
        # Compute the diffusion curves
        curves = diffusion_curves.compute_curves(
            test_images=images,
            saliency_maps=mask,
            metrics=metrics,
            device=device,
            n=20,
            model=model,
            class_idx=labels,
            # attribution_method=attr,
            # layer=layer,
            # upsample_method=upsampler,
            # mixer=mixing_method,
            # previous_attributions=layer_attributions[:-1],
            mask=mask,
            reshape_transform=reshape,
            # explainer=attr,
            test_dataset=funnybirds_test_dataset,
        )
        print(curves)
        exit(0)

    
    for attr in attr_methods:
        try:
            for upsampler in upscale_methods:
                layer_attributions = []
                for layer, layer_name in zip(layers, layers_names):
                    import warnings

                    # Suppress Captum UserWarnings
                    warnings.filterwarnings("ignore", category=UserWarning)
                    attr_map = attr.attribute(input_tensor= images,
                                                model=model,
                                                layer=layer,
                                                target=labels,
                                                reshape_transform=reshape)
                    
                    # If the attribution map is float64, convert to float32
                    if attr_map.dtype == torch.float64:
                        attr_map = attr_map.float()
                    attr_map = upsampler(attribution=attr_map,
                                        image=images,
                                        device=device,
                                        model=model,
                                        layer=layer)

                    if (
                        torch.abs(
                            attr_map.amax((2, 3), keepdim=True)
                            - attr_map.amin((2, 3), keepdim=True)
                        )
                        < 1e-6
                    ).any():
                        logger.warning(
                            f"Skipping constant saliency map at image index {index}"
                        )
                        # continue
                        raise ValueError(
                            f"Constant saliency map detected at image index {index} for method {attr.name}."
                        )

                    attr_map = min_max_normalize(attr_map)
                    if config.get("plotting", False):
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(8, 8))
                        plt.subplot(1, 2, 1)
                        plt.title("Image")
                        plt.axis('off')
                        plt.imshow(images[0].permute(1, 2, 0).cpu().detach().numpy())
                        plt.subplot(1, 2, 2)
                        plt.title(f"Attribution - {attr.name}")
                        plt.imshow(attr_map[0, 0].cpu().detach().numpy())
                        # plt.show()
                        plt.savefig(f"attribution_{config['model']}_{config['dataset']}_{attr.name}_img{index}_{layer_name}.png")
                        plt.close()
                    layer_attributions.append(attr_map)

                    for mixing_method in mixing_methods:
                        mixed_map = mixing_method(layer_attributions)
                        mixed_map = min_max_normalize(mixed_map)

                        for metric in metrics:
                            # for name, map_in_use, mix_method in [
                            #     ("Normal", attr_map, IdentityMix()),
                            #     ("Mixed", mixed_map, mix),
                            # ]:
                                result = metric(
                                    model=model,
                                    test_images=images,
                                    saliency_maps=mixed_map,
                                    class_idx=labels,
                                    attribution_method=attr,
                                    device=device,
                                    layer=layer,
                                    upsample_method=upsampler,
                                    mixer=mixing_method,
                                    previous_attributions=layer_attributions[:-1],
                                    mask=mask,
                                    reshape_transform=reshape,
                                    explainer=attr,
                                    test_dataset=funnybirds_test_dataset,
                                )
                                if isinstance(result, torch.Tensor):
                                    result = result.item()

                                results.add_result(
                                    model=config["model"],
                                    attribution_method=attr.name,
                                    dataset=config["dataset"],
                                    layer=layer_name,
                                    metric=metric.name,
                                    upscale_method=upsampler.name,
                                    mixing_method=mixing_method.name,
                                    value=result,
                                    image_index=index,
                                    label=labels[0].item(),
                                    predicted_label=pred_label[0].item(),
                                )
                                logger.debug(
                                    f"Recorded result: Index={index}, Layer={layer_name}, Attr={attr.name}, "
                                    f"Metric={metric.name}, Mix={mixing_method.name}"
                                )
        except Exception as e:
            logger.error(f"Error processing index {index}: {e}, for method {attr}", exc_info=True)

results.save_results()
logger.info("Results saved successfully.")
