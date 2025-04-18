from nanoconfig import config
from smalldiffusion import ModelMixin

import torch.nn as nn
import abc

class DiffusionModel(nn.Module, ModelMixin):
    def __init__(self, sample_shape):
        super().__init__()
        self.input_dims = sample_shape

@config
class ModelConfig(abc.ABC):
    @abc.abstractmethod
    def create(self, sample) -> DiffusionModel:
        """
        Create the model from the sample.
        :param sample: The sample to create the model from.
        :return: The model.
        """
        pass
