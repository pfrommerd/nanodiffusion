from itertools import pairwise

from smalldiffusion.model import ModelMixin
from nanogen import utils

from nanogen.data import DataPoint, SampleValue, CondValue
from nanogen.models import ModelConfig, GenerativeModel

from nanoconfig import config

from .schedules import ScheduleConfig

from smalldiffusion import (
    Schedule, ScheduleLDM, ScheduleDDPM,
    ScheduleLogLinear, ScheduleCosine, ScheduleSigmoid
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
            gen_sigmas: torch.Tensor):
        super().__init__()
        self.diffuser = diffuser
        self.train_noise_schedule = train_noise_schedule
        self.register_buffer("gen_sigmas", gen_sigmas)

    @torch.no_grad()
    def generate(self, sample_structure: ty.Any, cond: ty.Any = None,
                    gam: float = 1., mu: float = 0.,
                    **kwargs) -> ty.Iterator[SampleValue]:
        was_training = self.training
        self.eval()
        N = utils.axis_size(cond, 0)
        def gen_rand():
            return pytree.tree_map(lambda x: (torch.randn(x.shape, dtype=x.dtype, device=x.device)
                if hasattr(x, "dtype") and hasattr(x, "shape") else x), sample_structure)
        xt = gen_rand()
        yield xt # type: ignore
        eps = None
        for (sig, sig_prev) in pairwise(self.gen_sigmas): # type: ignore
            eps_prev, eps = eps, self.diffuser(xt, sig, cond) # type: ignore
            eps_av = pytree.tree_map(lambda x, y: x*gam + y*(1-gam), eps_prev, eps) \
                if eps_prev is not None else eps
            sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
            eta = (sig_prev**2 - sig_p**2).sqrt()
            xt = xt - (sig - sig_p) * eps_av + eta * gen_rand()
            yield xt
        if was_training:
            self.train()

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

    @ty.override
    def create(self, datapoint: DataPoint) -> DiffusionModel:
        sample_structure = datapoint.sample
        cond_structure = datapoint.cond
        train_noise_schedule = self.schedule.create()
        gen_sigmas = train_noise_schedule.sample_sigmas(self.sample_timesteps) # type: ignore
        diffuser = self.nn.create(sample_structure, cond_structure)
        return DiffusionModel(
            sample_structure,
            diffuser,
            train_noise_schedule,
            gen_sigmas
        )
