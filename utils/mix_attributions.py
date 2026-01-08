import torch
from typing import List, Literal
from copy import deepcopy


class Mixer:
    def __init__(
        self, name: str, layers_to_combine: Literal["all", "top", "above"] = "all"
    ):
        self.name = name
        self.layers_to_combine = layers_to_combine

    def filter_layers(self, attributions: List[torch.Tensor]):
        # attributions = deepcopy(attributions)
        copy_attributions = []
        for attr in attributions:
            if attr is not None:
                copy_attributions.append(attr.clone())
                copy_attributions[-1].to(attr.device)
        if len(copy_attributions) == 1:
            return copy_attributions

        if self.layers_to_combine == "all":
            return copy_attributions
        elif self.layers_to_combine == "top":
            # Most coarse + most fine-grained
            return [copy_attributions[0], copy_attributions[-1]]
        elif self.layers_to_combine == "above":
            # Second most fine-grained + most fine-grained
            return [copy_attributions[-2], copy_attributions[-1]]

    def __call__(self, attributions: List[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError()


class MultiplierMix(Mixer):
    def __init__(self, layers_to_combine: Literal["all", "top", "above"] = "all"):
        Mixer.__init__(self, "MultiplierMix", layers_to_combine)

    def __call__(self, attributions: List[torch.Tensor]) -> torch.Tensor:
        """
        The attributions are assumed to be ordered from the most coarse to the most fine-grained.
        """
        if len(attributions) == 1:
            return attributions[0]

        attributions = self.filter_layers(attributions)

        result = attributions[0]
        for attr in attributions[1:]:
            result *= attr
        return result


class NthRootMultiplierMix(Mixer):
    def __init__(self, layers_to_combine: Literal["all", "top", "above"] = "all"):
        """
        MultiplierMix with the result raised to the power of 1/n, where n is the number of attributions.
        """
        super().__init__("NthRootMultiplierMix", layers_to_combine)

    def __call__(self, attributions):
        """
        The attributions are assumed to be ordered from the most coarse to the most fine-grained.
        """
        if len(attributions) == 1:
            return attributions[0]

        attributions = self.filter_layers(attributions)

        result = attributions[0]
        for attr in attributions[1:]:
            result *= attr

        return torch.pow(result, 1 / len(attributions))


class LogExpMix(Mixer):
    def __init__(self, layers_to_combine: Literal["all", "top", "above"] = "all"):
        Mixer.__init__(self, "LogExpMix", layers_to_combine)

    def __call__(self, attributions: List[torch.Tensor]) -> torch.Tensor:
        """
        The attributions are assumed to be ordered from the most coarse to the most fine-grained.
        """
        if len(attributions) == 1:
            return attributions[0]

        attributions = self.filter_layers(attributions)

        numerator = (torch.log(torch.tensor([(len(attributions))])) + 1).to(attributions[0].device)
        denominator = torch.log(
            torch.sum(torch.exp(1 / torch.stack(attributions)), dim=0)
        ).to(attributions[0].device)

        return numerator / denominator

class ExpMeanMix(Mixer):
    def __init__(self, layers_to_combine: Literal["all", "top", "above"] = "all"):
        Mixer.__init__(self, "ExpMeanMix", layers_to_combine)

    def __call__(self, attributions: List[torch.Tensor]) -> torch.Tensor:
        """
        The attributions are assumed to be ordered from the most coarse to the most fine-grained.
        """
        if len(attributions) == 1:
            return attributions[0]

        attributions = self.filter_layers(attributions)

        return 1 / torch.log(torch.mean(torch.exp(1 / torch.stack(attributions)), dim=0))


class IdentityMix(Mixer):
    def __init__(self, layers_to_combine="all"):
        super().__init__("IdentityMix", layers_to_combine)

    def __call__(self, attributions):
        return attributions[-1]
