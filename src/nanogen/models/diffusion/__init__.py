from itertools import pairwise

from smalldiffusion.model import CondEmbedderLabel

from nanogen import utils
from nanogen.data import DataPoint, SampleValue, CondValue, DiscreteLabel
from nanogen.models import ModelConfig, GenerativeModel

from nanoconfig.data.torch import InMemoryDataset, SizedDataset
from nanoconfig import config

from .schedules import ScheduleConfig

import smalldiffusion
import smalldiffusion.diffusion
from smalldiffusion import (
    Schedule, ScheduleLDM, ScheduleDDPM,
    ScheduleLogLinear, ScheduleCosine, ScheduleSigmoid,
)

import torch.utils._pytree as pytree
import typing as ty
import torch.nn as nn
import torch
import abc
import enum

class DiffusionModel(GenerativeModel):
    def __init__(self, sample_structure: ty.Any,
            diffuser: "Diffuser",
            train_noise_schedule: Schedule,
            gen_sigmas: torch.Tensor, gamma: float, mu: float):
        super().__init__()
        self.gamma = gamma
        self.mu = mu
        self.diffuser = diffuser
        self.train_noise_schedule = train_noise_schedule
        self.register_buffer("gen_sigmas", gen_sigmas)

    @torch.no_grad()
    def generate(self, sample_structure: ty.Any, cond: ty.Any = None,
                    gamma: float | None = None, mu: float | None = None,
                    **kwargs) -> ty.Iterator[SampleValue]:
        gamma = gamma or self.gamma # type: ignore
        mu = mu or self.mu # type: ignore
        assert gamma is not None and mu is not None
        was_training = self.training
        self.eval()
        def gen_rand():
            return pytree.tree_map(
                lambda x: torch.randn_like(x, device=self.gen_sigmas.device) # type: ignore
                    if isinstance(x, torch.Tensor) else x,
                sample_structure
            )
        sigma_init = self.gen_sigmas[0] # type: ignore
        xt = pytree.tree_map(
            lambda x: sigma_init*x if isinstance(x, torch.Tensor) else x,
            gen_rand()
        )
        yield xt # type: ignore
        eps = None
        for (sig, sig_prev) in pairwise(self.gen_sigmas): # type: ignore
            if cond is not None: pred = self.diffuser(xt, sig, cond=cond, **kwargs)
            else: pred = self.diffuser(xt, sig, **kwargs)
            eps_prev, eps = eps, pred
            eps_av = (eps * gamma + eps_prev * (1-gamma)) if eps_prev is not None else eps
            sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
            eta = (sig_prev**2 - sig_p**2).sqrt()
            xt = xt - (sig - sig_p) * eps_av + eta * gen_rand()
            yield xt
        if was_training:
            self.train()

    @torch.no_grad()
    def generate_forward(self, sample: ty.Any, sigma: float | torch.Tensor):
        if not isinstance(sigma, torch.Tensor):
            sigma = torch.tensor(sigma).to(self.gen_sigmas.device) # type: ignore
        assert isinstance(sigma, torch.Tensor)
        sigma = sigma[None]
        eps = pytree.tree_map(
            lambda x: torch.randn_like(x) if isinstance(x, torch.Tensor) else x,
            sample
        )
        return pytree.tree_map(
            lambda x, e: x + sigma.reshape((sigma.shape[0],) + (1,)*(len(e.shape) - 1)) * e,
            sample, eps
        )

    def loss(self, sample: ty.Any, cond: ty.Any = None,
                    loss_type = nn.MSELoss):
        schedule = self.train_noise_schedule
        eps = pytree.tree_map(
            lambda x: torch.randn_like(x) if isinstance(x, torch.Tensor) else x,
            sample
        )
        sigma = schedule[torch.randint(0, len(schedule), (utils.axis_size(sample, 0),))].to(sample.device)
        noised = pytree.tree_map(
            lambda x, e: x + sigma.reshape((sigma.shape[0],) + (1,)*(len(e.shape) - 1)) * e,
            sample, eps
        )
        if cond is not None: pred = self.diffuser(noised, sigma, cond=cond)
        else: pred = self.diffuser(noised, sigma)
        return loss_type()(pred, eps), {}

class Diffuser(nn.Module): pass

class IdealDiffuser(Diffuser, smalldiffusion.ModelMixin):
    def __init__(self, samples: torch.Tensor, cond: torch.Tensor | None = None, cond_sigma: float = 0.1):
        super().__init__()
        self.input_dims = samples.shape[1:]
        self.register_buffer("samples", samples)
        self.register_buffer("cond", cond)
        self.cond_sigma = cond_sigma

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        def sq_norm(M, k):
            return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1,k)
        x_flat = x.flatten(start_dim=1)
        d_flat = self.samples.flatten(start_dim=1) # type: ignore
        xb, xr = x_flat.shape
        db, dr = d_flat.shape
        assert xr == dr, 'Input x must have same dimension as data!'
        assert sigma.shape == tuple() or sigma.shape[0] == xb, \
            f'sigma must be singleton or have same batch dimension as x! {sigma.shape}'
        # sq_diffs: ||x - x0||^2
        sq_diffs = sq_norm(x_flat, db).T + sq_norm(d_flat, xb) - 2 * d_flat @ x_flat.T # shape: db x xb

        log_weights = -sq_diffs/2/sigma.squeeze()**2

        # multiply the weights by the condition
        if cond is not None:
            cond_flat = cond.flatten(start_dim=1)
            dc_flat = self.cond.flatten(start_dim=1) # type: ignore
            # cond_sq: ||cond - dc||^2
            cond_sq = sq_norm(cond_flat, db).T + sq_norm(dc_flat, xb) - 2 * dc_flat @ cond_flat.T # shape: db x xb
            cond_weights = -cond_sq/2/(self.cond_sigma**2)
            log_weights += cond_weights

        weights = torch.nn.functional.softmax(log_weights, dim=0)
        x0 = torch.einsum('ij,i...->j...', weights, self.samples)                             # shape: xb x c1 x ... x cn
        sigma = sigma.reshape((sigma.shape[0],) + (1,) * len(x.shape[1:]))
        return (x - x0) / sigma

@config
class DiffuserConfig(abc.ABC):
    @abc.abstractmethod
    def create(self, sample_structure, cond_structure) -> Diffuser:
        pass

@config(variant="diffusion")
class DiffusionModelConfig(ModelConfig):
    nn: DiffuserConfig
    schedule: ScheduleConfig
    sample_timesteps: int
    ideal_denoiser: bool

    sampler_preset: str | None = None
    gamma: float = 1.
    mu: float = 0.

    @ty.override
    def create(self, data: SizedDataset[DataPoint]) -> DiffusionModel:
        sample_structure = data.data_sample.sample
        cond_structure = data.data_sample.cond
        train_noise_schedule = self.schedule.create()
        gen_sigmas = train_noise_schedule.sample_sigmas(self.sample_timesteps) # type: ignore
        if self.ideal_denoiser:
            assert isinstance(data, InMemoryDataset)
            sample, cond = data._data.sample, data._data.cond
            assert isinstance(sample, torch.Tensor) and (isinstance(cond, torch.Tensor) or cond is None)
            diffuser = IdealDiffuser(sample, cond)
        else:
            diffuser = self.nn.create(sample_structure, cond_structure)
        gamma, mu = self.gamma, self.mu
        match self.sampler_preset:
            case "accel": gamma, mu = 2., 0.
            case "ddim": gamma, mu = 1., 0.
            case "ddpm": gamma, mu = 1., 0.5
            case None: ...
            case _: raise ValueError(f"Unknown sampler preset: {self.sampler_preset}")
        return DiffusionModel(
            sample_structure, diffuser, # type: ignore
            train_noise_schedule, gen_sigmas,
            gamma, mu
        )

class CondEmbedder(nn.Module):
    def __init__(self, cond_structure: ty.Any, embed_features: int):
        super().__init__()
        cond_flat : list[torch.Tensor | DiscreteLabel] = pytree.tree_leaves(cond_structure)
        cond_classes = [x.num_classes for x in cond_flat if isinstance(x, DiscreteLabel)]
        cond_features = [x.nelement() for x in cond_flat if isinstance(x, torch.Tensor)]
        self.cond_class_embed = nn.ModuleList([
            CondEmbedderLabel(embed_features, num_classes, 0.1)
            for num_classes in cond_classes
        ])
        self.cond_feature_embed = nn.ModuleList([
            nn.Sequential(
                nn.Linear(num_cond_features, embed_features),
                nn.SiLU(),
                nn.Linear(embed_features, embed_features)
            )
            for num_cond_features in cond_features
        ])

    def forward(self, cond: ty.Any):
        cond_flat : list[torch.Tensor | DiscreteLabel] = pytree.tree_leaves(cond)
        cond_labels = [c for c in cond_flat if isinstance(c, DiscreteLabel)]
        cond_features = [c for c in cond_flat if isinstance(c, torch.Tensor)]
        cond = None
        for label, label_embed in zip(cond_labels, self.cond_class_embed):
            cond = (cond + label_embed(label)) if cond is not None else label_embed(label)
        for feature, feature_embed in zip(cond_features, self.cond_feature_embed):
            cond = (cond + feature_embed(feature)) if cond is not None else feature_embed(feature)
        assert cond is not None, "cond should not be None"
        return cond
