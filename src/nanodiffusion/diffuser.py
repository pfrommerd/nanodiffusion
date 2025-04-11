import logging
import torch
import contextlib
import os
import smalldiffusion
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from accelerate import Accelerator
from ml_collections import ConfigDict
from smalldiffusion import Schedule, ModelMixin
from nanoconfig import config

from .datasets import DataConfig, Sample
from .models import ModelConfig

from .utils import Interval, Iterations
from .optimizers import OptimizerConfig
from .schedules import ScheduleConfig
from . import utils, io

from nanoconfig.experiment import Experiment

from rich.style import Style as RichStyle
from rich.progress import (
    Progress, TextColumn, BarColumn,
    TimeRemainingColumn,
    TimeElapsedColumn,
    TaskProgressColumn
)
from .utils import MofNColumn

@config
class DiffuserConfig:
    optimizer : OptimizerConfig
    schedule : ScheduleConfig
    model: ModelConfig
    data: DataConfig

    batch_size: int = 32
    iterations: int = 10_000
    sample_steps: int = 16

class Diffuser:
    def __init__(self, config: DiffuserConfig,
            model : ModelMixin, schedule : Schedule, optimizer : Optimizer, 
            lr_scheduler: LRScheduler, *,
            train_data : Dataset | None,
            test_data : Dataset | None,
            sample_steps : int = 16,
            ############## Default training parameters ##############
            batch_size : int = 32,
            total_iterations: int | None = None):
        super().__init__()
        self.config = config

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.schedule = schedule
        self.sample_steps = sample_steps

        self.train_data = train_data
        self.test_data = test_data

        self.total_iterations = total_iterations
        self.batch_size = batch_size

    def from_config(config: DiffuserConfig, create_optimizer: bool = True) -> "Diffuser":
        train_data, test_data = config.data.create()
        sample = next(iter(train_data))
        model = config.model.create(sample)
        optimizer, lr_scheduler = config.optimizer.create(model.parameters(), config.iterations) if create_optimizer else None
        schedule = config.schedule.create()
        return Diffuser(
            config, model,
            schedule, optimizer,
            lr_scheduler,

            train_data=train_data,
            test_data=test_data,
            batch_size=config.batch_size,
            sample_steps=config.sample_steps,
            total_iterations=config.iterations,
        )
        

    def to_config(self) -> DiffuserConfig:
        return self.config

    def save(self, file, save_optim_state: bool = False):
        config = self.to_config()
        data = {
            "model": self.model.state_dict(),
            "config": config.to_dict()
        }
        io.save(file, data)

    @staticmethod
    def load(file, load_optim_state: bool = False, device=None):
        data = io.load(file, device=device)
        config = DiffuserConfig.from_dict(data["config"])
        diffuser = Diffuser.from_config(config)
        diffuser.model.load_state_dict(data["model"])
        return diffuser

    def train(self, iterations : Interval | int | None = None, 
              *,
              accelerator : Accelerator | None = None,
              experiment: Experiment | None = None,
              generate_interval : Interval | int | None = Iterations(1000),
              test_interval: Interval | int | None = Iterations(100),
              progress : bool = False
            ):
        iterations = iterations if iterations is not None else self.total_iterations
        iterations_per_epoch = len(self.train_data) // self.batch_size
        iterations = Interval.to_iterations(iterations, iterations_per_epoch)
        total_epochs = (iterations + iterations_per_epoch - 1) // iterations_per_epoch

        generate_interval = Interval.to_iterations(generate_interval, iterations_per_epoch)
        test_interval = Interval.to_iterations(test_interval, iterations_per_epoch)

        accelerator = accelerator if accelerator else Accelerator()
        
        num_workers = os.cpu_count()

        train_loader = DataLoader(self.train_data, shuffle=True,
            num_workers=num_workers, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_data, shuffle=True,
            num_workers=num_workers, batch_size=self.batch_size)

        sample = next(iter(train_loader))
        if sample.cond is not None:
            train_gen_loader = DataLoader(self.train_data, shuffle=True,
                num_workers=num_workers, batch_size=self.batch_size)
            test_gen_loader = DataLoader(self.train_data, shuffle=True,
                num_workers=num_workers, batch_size=self.batch_size)
        else:
            train_gen_loader = None
            test_gen_loader = None

        (
            model, optimizer, lr_scheduler, 
            train_loader, test_loader,
            test_gen_loader, train_gen_loader
        ) = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler,
            train_loader, test_loader,
            test_gen_loader, train_gen_loader
        )
        test_loader = utils.cycle(test_loader)
        if test_gen_loader is not None and train_gen_loader is not None:
            test_gen_loader = utils.cycle(test_gen_loader)
            train_gen_loader = utils.cycle(train_gen_loader)

        if progress:
            pbar = Progress(
                    TextColumn("[progress.description]{task.description}"),
                    BarColumn(finished_style=RichStyle(color="green")),
                    MofNColumn(6),
                    TaskProgressColumn(),
                    TimeRemainingColumn(),
                    TimeElapsedColumn(),
                    refresh_per_second=1
                )
            iterations_task = pbar.add_task(
                "[bold blue]Total Iterations[/bold blue]",
                total=iterations)
            if total_epochs > 1:
                epochs_task = pbar.add_task(
                    "[bold blue]Epochs[/bold blue]",
                    total=total_epochs)
                epoch_iterations_task = pbar.add_task(
                    "[bold blue]Iteration[/bold blue]",
                    total=iterations_per_epoch)
        else:
            pbar = contextlib.nullcontext()

        device = accelerator.device
        with pbar:
            model.train()
            iteration = 0
            for _ in range(total_epochs):
                if progress and total_epochs > 1:
                    pbar.reset(epoch_iterations_task)
                for batch in train_loader:
                    with torch.no_grad():
                        x0, cond = batch.sample, batch.cond
                        x0, cond = x0.to(device), (cond.to(device) if cond is not None else None)
                        x0, sigma, eps, _ = smalldiffusion.diffusion.generate_train_sample(
                            x0, self.schedule
                        )
                    optimizer.zero_grad()
                    loss = model.get_loss(x0, sigma, eps, cond=cond)
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()

                    experiment.log_metric("loss", loss.item(),
                                          series="train", step=iteration)
                    with torch.no_grad():
                        self.model.eval()
                        if iteration % test_interval == 0 and experiment:
                            test_batch = next(test_loader)
                            x0, cond = test_batch.sample, test_batch.cond
                            x0, cond = x0.to(device), (cond.to(device) if cond is not None else None)
                            x0, sigma, eps, _ = smalldiffusion.diffusion.generate_train_sample(
                                x0, self.schedule
                            )
                            loss = model.get_loss(x0, sigma, eps, cond=cond)
                            if experiment:
                                experiment.log_metric("loss", loss.item(),
                                                      series="test", step=iteration)
                        if iteration % generate_interval == 0 and experiment:
                            if sample.cond is not None:
                                train_cond = next(iter(train_gen_loader)).cond.to(device)
                                test_cond = next(iter(test_gen_loader)).cond.to(device)
                                *_, x0_train, = smalldiffusion.diffusion.samples(
                                    self.model, self.schedule.sample_sigmas(self.sample_steps),
                                    batchsize=self.batch_size, cond=train_cond, accelerator=accelerator
                                )
                                *_, x0_test, = smalldiffusion.diffusion.samples(
                                    self.model, self.schedule.sample_sigmas(self.sample_steps),
                                    batchsize=self.batch_size, cond=test_cond, accelerator=accelerator
                                )
                                x0_train = Sample(cond=train_cond, sample=x0_train,
                                    num_classes=sample.num_classes)
                                x0_test = Sample(cond=test_cond, sample=x0_test,
                                    num_classes=sample.num_classes)
                                experiment.log({
                                    "train/samples" : self.train_data.visualize_batch(x0_train),
                                    "test/samples" : self.train_data.visualize_batch(x0_test)
                                }, step=iteration)
                            else:
                                *_, x0_samples, = smalldiffusion.diffusion.samples(
                                    self.model, self.schedule.sample_sigmas(self.sample_steps),
                                    batchsize=self.batch_size, cond=train_cond, accelerator=accelerator
                                )
                                x0_samples = Sample(cond=None, sample=x0_samples,
                                    num_classes=sample.num_classes)
                                experiment.log({
                                    "test/samples" : self.train_data.visualize_batch(x0_samples)
                                }, step=iteration)

                        self.model.train()
                    iteration += 1
                    # check if we are done
                    if progress:
                        pbar.update(iterations_task, advance=1)
                    if progress and total_epochs > 1:
                        pbar.update(epoch_iterations_task, advance=1)
                    if iteration >= iterations:
                        break
                if progress and total_epochs > 1:
                    pbar.update(epochs_task, advance=1)
                if iteration >= iterations:
                    break