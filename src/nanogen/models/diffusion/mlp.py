import torch.nn as nn
import torch.nn.functional as F
import torch
import typing as ty
import math
import itertools
import torch.utils._pytree as pytree

from smalldiffusion import (
    Schedule, SigmaEmbedderSinCos, get_sigma_embeds
)

from nanogen.data import DiscreteLabel
from nanoconfig import config

from . import Diffuser, DiffuserConfig, CondEmbedder

@config(variant="mlp")
class MlpConfig(DiffuserConfig):
    hidden_features : ty.Sequence[int] = (64, 128, 128, 64, 64)
    embed_features : int = 64
    simple: bool = False

    @ty.override
    def create(self, sample_structure, cond_structure) -> Diffuser:
        if self.simple:
            return SimpleDiffusionMLP(
                sample_structure, cond_structure,
                embed_features=self.embed_features,
                hidden_features=self.hidden_features,
            )
        else:
            return DiffusionMLP(
                sample_structure, cond_structure,
                embed_features=self.embed_features,
                hidden_features=self.hidden_features,
            )

class DiffusionMLP(Diffuser):
    def __init__(self, sample_structure, cond_structure,
                    embed_features: int = 64,
                    hidden_features=(128,128,256,128,128),
                    num_classes=None):
        super().__init__()
        assert isinstance(sample_structure, torch.Tensor), "sample_structure must be a torch.Tensor"
        sample_features = sample_structure.nelement()

        layers = []
        embed_layers = []
        features = (sample_features,) + tuple(hidden_features)
        for in_dim, out_dim in itertools.pairwise(features):
            layers.append(nn.Linear(in_dim, out_dim))
            embed_layers.append(nn.Linear(embed_features, 2*out_dim))

        self.layers = nn.ModuleList(layers)
        self.embed_layers = nn.ModuleList(embed_layers)
        self.final_layer = nn.Linear(features[-1], sample_features)

        self.sigma_embedder = SigmaEmbedderSinCos(embed_features)
        self.cond_embedder = nn.Sequential(
            CondEmbedder(cond_structure, embed_features),
            nn.GELU(),
            nn.Linear(embed_features, embed_features)
        ) if cond_structure is not None else None

    def forward(self, x, sigma, cond : torch.Tensor | None = None):
        assert isinstance(x, torch.Tensor), "Input x must be a torch.Tensor"
        out_shape = x.shape

        embed = self.sigma_embedder(x.shape[0], sigma)
        if cond is not None:
            assert self.cond_embedder is not None
            embed += self.cond_embedder(cond)

        x = x.flatten(1)
        for (layer, embed_layer) in zip(self.layers, self.embed_layers):
            x = layer(x)
            shift, scale = embed_layer(embed).chunk(2, dim=-1)
            x = F.gelu((scale + 1)*x + shift)
        x = self.final_layer(x)
        return x.reshape(out_shape)

class SimpleDiffusionMLP(Diffuser):
    def __init__(self, sample_structure, cond_structure,
                    embed_features: int = 64,
                    hidden_features=(128,128,256,128,128),
                    num_classes=None):
        super().__init__()
        assert isinstance(sample_structure, torch.Tensor), "sample_structure must be a torch.Tensor"
        sample_features = sample_structure.nelement()

        self.has_cond = cond_structure is not None
        layers = []
        embed_layers = []
        features = (sample_features + (1 if self.has_cond else 0),) + tuple(hidden_features)
        for in_dim, out_dim in itertools.pairwise(features):
            layers.append(nn.Linear(in_dim, out_dim))
            embed_layers.append(nn.Linear(embed_features, 2*out_dim))

        self.layers = nn.ModuleList(layers)
        self.embed_layers = nn.ModuleList(embed_layers)
        self.final_layer = nn.Linear(features[-1], sample_features)

        self.sigma_embedder = SigmaEmbedderSinCos(embed_features)

    def forward(self, x, sigma, cond : torch.Tensor | None = None):
        assert isinstance(x, torch.Tensor), "Input x must be a torch.Tensor"
        out_shape = x.shape

        embed = self.sigma_embedder(x.shape[0], sigma)

        x = x.flatten(1)
        x_in = x
        # sigma = sigma.squeeze()[...,None] * torch.ones_like(cond) # type: ignore
        if self.has_cond:
            x = torch.concatenate([x, cond], dim=-1) # type: ignore
        # for layer in self.layers:
        #     x = layer(x)
        #     x = F.gelu(x)
        for (layer, embed_layer) in zip(self.layers, self.embed_layers):
            x = layer(x)
            shift, scale = embed_layer(embed).chunk(2, dim=-1)
            x = F.gelu((scale + 1)*x + shift)
        x = self.final_layer(x)
        # x = self.final_layer(x)
        # x = (x - x_in)/sigma[...,None]
        return x.reshape(out_shape)
