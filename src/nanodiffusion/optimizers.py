from nanoconfig import config

import typing as tp
import abc

import torch.optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler, CosineAnnealingLR

@config
class OptimizerConfig(abc.ABC):
    lr: float = 0.001

    @abc.abstractmethod
    def create(self, parameters, iterations) -> tuple[Optimizer, LRScheduler]: ...

@config(variant="adamw")
class AdamwConfig(OptimizerConfig):
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.01

    @tp.override
    def create(self, parameters, iterations) -> tuple[torch.optim.AdamW, LRScheduler]:
        optim = torch.optim.AdamW(
            parameters,
            lr=self.lr,
            betas=self.betas,
            eps=self.eps,
            weight_decay=self.weight_decay
        )
        return optim, CosineAnnealingLR(
            optimizer=optim,
            T_max=iterations,
            eta_min=1e-6
        )