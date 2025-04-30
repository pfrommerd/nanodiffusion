import torch.nn as nn
import torch
import typing as ty
import math
import itertools
import torch.utils._pytree as pytree

from smalldiffusion import (
    Schedule, get_sigma_embeds
)

from nanogen.data import DiscreteLabel
from nanoconfig import config

from . import Diffuser, DiffuserConfig, CondEmbedder

@config(variant="mlp")
class MlpConfig(DiffuserConfig):
    hidden_features : ty.Sequence[int] = (64, 128, 128, 64, 64)
    embed_features : int = 64

    @ty.override
    def create(self, sample_structure, cond_structure) -> Diffuser:
        return DiffusionMLP(
            sample_structure, cond_structure,
            embed_features=self.embed_features,
            hidden_features=self.hidden_features,
        )

class DiffusionMLP(Diffuser):
    def __init__(self, sample_structure, cond_structure,
                    embed_features: int = 32,
                    hidden_features=(128,128,256,128,128),
                    num_classes=None):
        super().__init__()
        assert isinstance(sample_structure, torch.Tensor), "sample_structure must be a torch.Tensor"
        sample_features = sample_structure.nelement()
        input_features = 2*embed_features if cond_structure is None else 3*embed_features
        features = (input_features,) + tuple(hidden_features)
        layers = []
        for in_dim, out_dim in itertools.pairwise(features):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(features[-1], sample_features))

        self.net = nn.Sequential(*layers)
        self.sigma_embedder = nn.Linear(2, embed_features)
        self.sample_embedder = nn.Linear(sample_features, embed_features)
        self.cond_embedder = CondEmbedder(cond_structure, embed_features)

    def forward(self, x, sigma, cond : torch.Tensor | None = None):
        assert isinstance(x, torch.Tensor), "Input x must be a torch.Tensor"
        out_shape = x.shape
        x_flat = x.flatten(1)
        x_embed = self.sample_embedder(x_flat)
        sigma_embed = self.sigma_embedder(get_sigma_embeds(x_flat.shape[0], sigma.squeeze()))
        if cond is not None:
            cond = self.cond_embedder(cond)
        nn_input = torch.cat([x_embed, sigma_embed] + ([cond] if cond is not None else []), dim=1)
        x_flat = self.net(nn_input)
        x_out = x_flat.reshape(out_shape)
        return x_out
