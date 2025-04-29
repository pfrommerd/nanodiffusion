import torch.nn as nn
import torch
import typing as ty
import math
import itertools
import torch.utils._pytree as pytree

from smalldiffusion import (
    Schedule, CondEmbedderLabel, get_sigma_embeds
)

from nanogen.data import DiscreteLabel
from nanoconfig import config

from . import Diffuser, DiffuserConfig

@config(variant="mlp")
class MlpConfig(DiffuserConfig):
    hidden_features : ty.Sequence[int] = (64, 64, 128, 128, 64, 64)
    cond_embed_features : int = 4

    @ty.override
    def create(self, sample_structure, cond_structure) -> Diffuser:
        return DiffusionMLP(
            sample_structure, cond_structure,
            cond_embed_features=self.cond_embed_features,
            hidden_features=self.hidden_features,
        )

class DiffusionMLP(Diffuser):
    def __init__(self, sample_structure, cond_structure,
                    cond_embed_features: int = 64,
                    hidden_features=(128,128,256,128,128),
                    num_classes=None):
        super().__init__()
        sample_flat : list[torch.Tensor] = pytree.tree_leaves(sample_structure)
        cond_flat : list[torch.Tensor | DiscreteLabel] = pytree.tree_leaves(cond_structure)
        cond_classes = [x.num_classes for x in cond_flat if isinstance(x, DiscreteLabel)]
        cond_features = [x.nelement() for x in cond_flat if not isinstance(x, DiscreteLabel)]
        num_sample_features = sum(x.nelement() for x in sample_flat)

        sigma_features = 2
        features = (num_sample_features + sigma_features + cond_embed_features,) + tuple(hidden_features)
        layers = []
        for in_dim, out_dim in itertools.pairwise(features):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(features[-1], num_sample_features))

        self.net = nn.Sequential(*layers)
        self.cond_embed_features = cond_embed_features
        self.cond_class_embed = nn.ModuleList([
            CondEmbedderLabel(cond_embed_features, num_classes, 0.1)
            for num_classes in cond_classes
        ])
        self.cond_feature_embed = nn.ModuleList([
            nn.Linear(num_cond_features, cond_embed_features)
            for num_cond_features in cond_features
        ])

    def forward(self, x, sigma, cond : torch.Tensor | None = None):
        x_leaves , treedef = pytree.tree_flatten(x)
        x_flat = torch.concatenate([x.reshape(x.shape[0], -1) for x in x_leaves], dim=-1)
        # shape: b x sigma_dim
        sigma_embeds = get_sigma_embeds(x_flat.shape[0], sigma.squeeze())
        # reshape x into a 1-D tensor
        out_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(x.shape[0], -1)
        nn_input = torch.cat([x_flat, sigma_embeds], dim=1)  # shape: b x (dim + sigma_dim)

        cond_labels = [c for c in pytree.tree_leaves(cond) if isinstance(c, DiscreteLabel)]
        cond_features = [c for c in pytree.tree_leaves(cond) if not isinstance(c, DiscreteLabel)]
        if cond_labels or cond_features:
            cond = torch.zeros((x_flat.shape[0], self.cond_embed_features),
                                    device=nn_input.device)
            for label, label_embed in zip(cond_labels, self.cond_class_embed):
                cond += label_embed(label)
            for feature, feature_embed in zip(cond_features, self.cond_feature_embed):
                cond += feature_embed(feature)
            nn_input = torch.cat([nn_input, cond], dim=1) # type: ignore

        x_flat = self.net(nn_input)
        # unflatten the x
        indices = [x.nelement() for x in pytree.tree_leaves(x)]
        x_leaves = [x_flat[s:e].reshape(x.shape) # type: ignore
            for (s, e), x in zip(itertools.pairwise([0] + indices), x_leaves)]
        x_out = pytree.tree_unflatten(x_leaves, treedef)
        return x_out
