import torch.nn as nn
import torch
import typing as ty
import math

from smalldiffusion import ModelMixin, CondEmbedderLabel, get_sigma_embeds
from itertools import pairwise

from nanoconfig import config

from nanodiffusion.diffuser import Diffuser
from . import ModelConfig, DiffusionModel

@config(variant="mlp")
class MlpConfig(ModelConfig):
    hidden_features : ty.Sequence[int] = (64, 64, 128, 128, 64, 64)
    cond_embed_features : int = 4

    def create(self, sample) -> DiffusionModel:
        sample_shape = sample.sample.shape
        cond_shape = sample.cond.shape if sample.cond is not None else None
        return DiffusionMLP(
            sample_shape=sample_shape,
            cond_shape=cond_shape,
            cond_embed_features=self.cond_embed_features,
            hidden_features=self.hidden_features,
            num_classes=sample.num_classes
        )

class DiffusionMLP(DiffusionModel):
    def __init__(self, sample_shape, cond_shape,
                cond_embed_features: int = 64,
                hidden_features=(128,128,256,128,128),
                num_classes=None):
        super().__init__(sample_shape)
        x_features = math.prod(sample_shape)
        cond_features = math.prod(cond_shape) if cond_shape is not None else None
        sigma_features = 2
        features = (x_features + sigma_features + cond_embed_features,) + tuple(hidden_features)
        layers = []
        for in_dim, out_dim in pairwise(features):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(features[-1], x_features))

        self.net = nn.Sequential(*layers)
        if cond_features is not None:
            if num_classes is not None:
                self.cond_embed = CondEmbedderLabel(cond_embed_features, num_classes, 0.1)
            else:
                self.cond_embed = nn.Linear(cond_features, cond_embed_features)
        else:
            self.cond_embed = None

    def forward(self, x, sigma, cond : torch.Tensor | None = None):
        # x     shape: b x dim
        # sigma shape: b x 1 or scalar
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x sigma_dim
        # reshape x into a 1-D tensor
        out_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        nn_input = torch.cat([x, sigma_embeds], dim=1)  # shape: b x (dim + sigma_dim)
        if cond is not None:
            assert self.cond_embed is not None, "cond_embed is None"
            if len(cond.shape) > 2:
                cond = cond.view(cond.shape[0], -1)
            cond_embeds = self.cond_embed(cond)
            nn_input = torch.cat([nn_input, cond_embeds], dim=1)  # shape: b x (dim + sigma_dim)
        x = self.net(nn_input)
        if len(out_shape) > 2:
            x = x.view(out_shape)
        return x
