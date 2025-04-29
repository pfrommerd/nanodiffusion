from nanoconfig import config

from nanogen.data import DataPoint, SampleValue, CondValue
from smalldiffusion import Schedule

import torch
import torch.nn as nn
import abc
import typing as ty

class GenerativeModel(abc.ABC, nn.Module):
    @abc.abstractmethod
    def generate(self, sample_structure: SampleValue, cond: CondValue | None = None, **kwargs) -> ty.Iterator[SampleValue]:
        ...

    @abc.abstractmethod
    def loss(self, sample: SampleValue, cond: CondValue | None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...

@config
class ModelConfig(abc.ABC):
    @abc.abstractmethod
    def create(self, datapoint: DataPoint) -> GenerativeModel:
        """
        Create the model from the sample.
        :param sample: A reference sample structure to create the model from.
        :return: The model.
        """
        pass
