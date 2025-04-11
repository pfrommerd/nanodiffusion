import torch.nn as nn
import torch
import typing as ty

from smalldiffusion import ModelMixin, CondEmbedderLabel, get_sigma_embeds
from itertools import pairwise

from nanoconfig import config
from . import ModelConfig

@config(variant="mlp")
class MlpConfig(ModelConfig):
    hidden_features : ty.Sequence[int] = (64, 64, 128, 128, 64, 64)
    cond_features : int = 4

    def create(self, sample):
        x_features = sample.sample.shape[-1]
        sample_cond_features = (
            sample.cond.shape[-1]
            if sample.cond is not None and sample.num_classes is None
            else None
        )
        return DiffusionMLP(
            x_features=x_features,
            sample_cond_features=sample_cond_features,
            cond_features=self.cond_features if sample.cond is not None else None,
            hidden_features=self.hidden_features,
            num_classes=sample.num_classes
        )

class DiffusionMLP(nn.Module, ModelMixin):
    def __init__(self, x_features=2,
                sample_cond_features : int | None = None,
                cond_features: int | None = 4,
                hidden_features=(16,128,256,128,16),
                num_classes=None):
        super().__init__()
        layers = []
        sigma_dim=2

        self.input_dims = (x_features,)
        for in_dim, out_dim in pairwise((x_features + sigma_dim + cond_features,) + tuple(hidden_features)):
            layers.extend([nn.Linear(in_dim, out_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_features[-1], x_features))
        self.net = nn.Sequential(*layers)

        if cond_features is not None:
            if num_classes is not None:
                self.cond_embed = CondEmbedderLabel(cond_features, num_classes, 0.1)
            else:
                self.cond_embed = nn.Linear(sample_cond_features, cond_features)
        else:
            self.cond_embed = None

    def forward(self, x, sigma, cond : torch.Tensor = None):
        # x     shape: b x dim
        # sigma shape: b x 1 or scalar
        sigma_embeds = get_sigma_embeds(x.shape[0], sigma.squeeze()) # shape: b x sigma_dim
        nn_input = torch.cat([x, sigma_embeds], dim=1)  # shape: b x (dim + sigma_dim)
        if cond is not None:
            cond_embeds = self.cond_embed(cond)
            # cond_embeds = torch.zeros((x.shape[0], self.cond_dim), device=x.device)
            nn_input = torch.cat([nn_input, cond_embeds], dim=1)  # shape: b x (dim + sigma_dim)
        return self.net(nn_input)