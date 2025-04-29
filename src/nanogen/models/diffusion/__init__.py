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
            gen_noise_schedule: Schedule):
        super().__init__()
        self.diffuser = diffuser
        self.sample_structure = sample_structure
        self.train_noise_schedule = train_noise_schedule
        self.gen_noise_schedule = gen_noise_schedule

    @torch.no_grad()
    def generate(self, cond: ty.Any = None,
                    gam: float = 1., mu: float = 0.,
                    **kwargs) -> ty.Iterator[SampleValue]:
        was_training = self.training
        self.eval()
        def gen_rand():
            return pytree.tree_map(lambda x: (torch.randn_like(x)
                if hasattr(x, "dtype") and hasattr(x, "shape") else x),
                    self.sample_structure)

        xt = gen_rand()
        yield xt # type: ignore
        eps = None
        self.gen_noise_schedule
        for (sig, sig_prev) in pairwise(self.gen_noise_schedule.sigmas):
            eps_prev, eps = pred, self(xt, sig, cond) # type: ignore
            eps_av = pytree.tree_map(lambda x, y: x*gam + y*(1-gam), eps_prev, eps) \
                if eps_prev is not None else eps
            sig_p = (sig_prev/sig**mu)**(1/(1-mu)) # sig_prev == sig**mu sig_p**(1-mu)
            eta = (sig_prev**2 - sig_p**2).sqrt()
            xt = xt - (sig - sig_p) * eps_av + eta * gen_rand()
            yield xt
        if was_training:
            self.train()

    def loss(self, sample: ty.Any, cond: ty.Any = None, loss_type = nn.MSELoss):
        is_sample = all(pytree.tree_leaves(pytree.tree_map(
                (lambda x,y: x.shape == y.shape if hasattr(x, "dtype")
                    and hasattr(x, "shape") else True), sample, self.sample_structure)))
        is_batch = all(pytree.tree_leaves(pytree.tree_map(
                (lambda x,y: (len(x.shape) > 1 and x.shape[1:] == y.shape) if hasattr(x, "dtype")
                    and hasattr(x, "shape") else True), sample, self.sample_structure)))
        if not is_sample and not is_batch:
            raise ValueError("Sample does not match the model input shape.")

        schedule = self.train_noise_schedule
        if is_sample: sigma = schedule[torch.randint(0, len(schedule), ())]
        else: sigma = schedule[torch.randint(0, len(schedule),
                            (utils.axis_size(sample, 0),))]
        eps = pytree.tree_map(
            lambda x: torch.randn_like(x) if isinstance(x, torch.Tensor) else x,
            sample
        )
        noised = pytree.tree_map(lambda x: x + sigma * eps, sample)
        if cond is not None: pred = self(noised, sigma, cond=cond)
        else: pred = self(noised, sigma)
        return loss_type()(pred, eps)

class Diffuser(nn.Module, ModelMixin): pass

@config
class DiffuserConfig(abc.ABC):
    @abc.abstractmethod
    def create(self, sample_structure, cond_structure) -> Diffuser:
        pass

@config
class DiffusionModelConfig(ModelConfig):
    nn: DiffuserConfig
    schedule: ScheduleConfig
    sample_timesteps: int

    @ty.override
    def create(self, datapoint: DataPoint) -> DiffusionModel:
        sample_structure = datapoint.sample
        cond_structure = datapoint.cond
        train_noise_schedule = self.schedule.create()
        gen_noise_schedule = Schedule(train_noise_schedule.sample_sigmas(self.sample_timesteps))
        diffuser = self.nn.create(sample_structure, cond_structure)
        return DiffusionModel(
            sample_structure,
            diffuser,
            train_noise_schedule,
            gen_noise_schedule
        )
