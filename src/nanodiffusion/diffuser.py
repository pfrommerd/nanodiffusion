import logging
import accelerate
import torch
import contextlib
import os
import smalldiffusion
import numpy as np
import time

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from accelerate import Accelerator
from smalldiffusion import Schedule, ModelMixin
from nanoconfig import config, Config

from .datasets import DataConfig, Sample, SampleDataset
from .models import DiffusionModel, ModelConfig

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

logger = logging.getLogger(__name__)

@config
class DiffuserConfig(Config):
    optimizer : OptimizerConfig
    schedule : ScheduleConfig
    model: ModelConfig
    data: DataConfig

    batch_size: int = 32
    test_batch_size: int = 64
    gen_batch_size: int = 64

    test_interval: int = 100
    gen_interval: int = 1000

    iterations: int = 10_000
    sample_steps: int = 32

class Diffuser:
    def __init__(self, config: DiffuserConfig,
            model : DiffusionModel, schedule : Schedule,
            optimizer : Optimizer | None = None,
            lr_scheduler: LRScheduler | None = None, *,
            data_sample: Sample,
            train_data : SampleDataset | None,
            test_data : SampleDataset | None,
            sample_steps : int = 32,
            ############## Default training parameters ##############
            batch_size : int = 32,
            test_batch_size : int = 64,
            gen_batch_size : int = 64,
            test_interval : int = 100,
            gen_interval : int = 1000,
            total_iterations: int | None = None):
        super().__init__()
        self.config = config

        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.schedule = schedule
        self.sample_steps = sample_steps

        self.data_sample = data_sample
        self.train_data = train_data
        self.test_data = test_data

        self.test_interval = test_interval
        self.gen_interval = gen_interval

        self.total_iterations = total_iterations
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.gen_batch_size = gen_batch_size

    @staticmethod
    def from_config(config: DiffuserConfig,
                create_optimizer: bool = True,
                experiment: Experiment | None = None) -> "Diffuser":
        train_data, test_data = config.data.create(experiment)
        sample = next(iter(train_data))
        model = config.model.create(sample)
        optimizer, lr_scheduler = config.optimizer.create(model.parameters(), config.iterations) if create_optimizer else (None, None)
        schedule = config.schedule.create()
        return Diffuser(
            config, model,
            schedule, optimizer,
            lr_scheduler,
            train_data=train_data,
            test_data=test_data,
            data_sample=sample,
            batch_size=config.batch_size,
            test_interval=config.test_interval,
            gen_interval=config.gen_interval,
            test_batch_size=config.test_batch_size,
            gen_batch_size=config.gen_batch_size,
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
    def load(file, load_optim_state: bool = False, device="cpu"):
        data = io.load(file, device=device)
        config = DiffuserConfig.from_dict(data["config"])
        diffuser = Diffuser.from_config(config)
        diffuser.model.load_state_dict(data["model"])
        return diffuser

    def sample(self, N: int = 1, cond: torch.Tensor | None = None, seed: int | None = None,
                    accelerator: Accelerator | None = None):
        accelerator = accelerator or Accelerator()
        model = accelerator.prepare(self.model)
        if seed is not None:
            torch.manual_seed(seed)
        if cond is None:
            loader = DataLoader(self.train_data, batch_size=N, shuffle=True) # type: ignore
            cond = next(iter(loader)).cond # type: ignore
            cond = cond.to(accelerator.device) # type: ignore
            N = min(N, cond.shape[0]) # type: ignore
        *_, samples = smalldiffusion.diffusion.samples(
            model, self.schedule.sample_sigmas(self.sample_steps),
            batchsize=N, cond=cond, accelerator=accelerator
        )
        return Sample(cond, samples, num_classes=self.data_sample.num_classes) # type: ignore

    def train(self, iterations : Interval | int | None = None,
              *,
              accelerator : Accelerator | None = None,
              experiment: Experiment | None = None,
              generate_interval : Interval | int | None = None,
              test_interval: Interval | int | None = None,
              progress : bool = False
            ):
        assert self.train_data is not None, "No training data provided"
        assert self.test_data is not None, "No test data provided"
        iterations = iterations if iterations is not None else self.total_iterations
        generate_interval = generate_interval if generate_interval is not None else self.gen_interval
        test_interval = test_interval if test_interval is not None else self.test_interval

        assert iterations is not None, "No iterations provided"
        iterations_per_epoch = (len(self.train_data) + self.batch_size - 1) // self.batch_size
        iterations = Interval.to_iterations(iterations, iterations_per_epoch)
        total_epochs = (iterations + iterations_per_epoch - 1) // iterations_per_epoch

        generate_interval = Interval.to_iterations(generate_interval, iterations_per_epoch)
        test_interval = Interval.to_iterations(test_interval, iterations_per_epoch)

        accelerator = accelerator if accelerator else Accelerator()

        num_workers = os.cpu_count()
        if num_workers is None: num_workers = 0
        if (hasattr(self.train_data, "in_memory") and hasattr(self.test_data, "in_memory")
                and self.train_data.in_memory and self.test_data.in_memory):
            num_workers = 0
        train_loader = DataLoader(self.train_data, shuffle=True,
            num_workers=num_workers, batch_size=self.batch_size)
        test_loader = DataLoader(self.test_data, shuffle=True,
            num_workers=num_workers, batch_size=self.test_batch_size)
        if self.data_sample.cond is not None:
            train_gen_loader = DataLoader(self.train_data, shuffle=True,
                num_workers=num_workers, batch_size=self.gen_batch_size)
            test_gen_loader = DataLoader(self.train_data, shuffle=True,
                num_workers=num_workers, batch_size=self.gen_batch_size)
        else:
            train_gen_loader = None
            test_gen_loader = None

        if experiment:
            sample_batch = next(iter(DataLoader(self.test_data, shuffle=True,
                num_workers=num_workers, batch_size=self.gen_batch_size)))
            experiment.log({
                "samples" : self.train_data.visualize_batch(sample_batch)
            }, series="gt")

        (
            model, optimizer, lr_scheduler,
            # train_loader, test_loader,
            # test_gen_loader, train_gen_loader
        ) = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler,
            # train_loader, test_loader,
            # test_gen_loader, train_gen_loader
        )
        test_loader = utils.cycle(test_loader)
        if test_gen_loader is not None and train_gen_loader is not None:
            test_gen_loader = utils.cycle(test_gen_loader)
            train_gen_loader = utils.cycle(train_gen_loader)

        total_params = sum(param.numel() for param in model.parameters())
        logger.info(f"Total parameters: {total_params}")
        # Test run-through of dataset
        logger.info("Warming dataset...")
        t, i = time.time(), 0
        for data in train_loader:
            i = i + 1
        took = time.time() - t
        logger.info(f"Took {took:.2f} seconds to go through dataset, {i/took:.2f} iter/second")

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

        with pbar:
            model.train()
            iteration = 0
            for _ in range(total_epochs):
                if progress and total_epochs > 1:
                    pbar.reset(epoch_iterations_task) # type: ignore
                for batch in train_loader:
                    with torch.no_grad():
                        x0, cond = batch.sample, batch.cond
                        x0, cond = x0.to(accelerator.device), (cond.to(accelerator.device) if cond is not None else None)
                        x0, sigma, eps, _ = smalldiffusion.diffusion.generate_train_sample(
                            x0, self.schedule
                        )
                    optimizer.zero_grad()
                    loss = model.get_loss(x0, sigma, eps, cond=cond)
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    # check if we are done
                    if progress:
                        pbar.update(iterations_task, advance=1) # type: ignore
                    if progress and total_epochs > 1:
                        pbar.update(epoch_iterations_task, advance=1) # type: ignore

                    if experiment:
                        experiment.log_metric("loss", loss.item(),
                                            series="train", step=iteration)
                        experiment.log_metric("lr", lr_scheduler.get_last_lr()[0],
                                            series="train", step=iteration)
                    with torch.no_grad():
                        self.model.eval()
                        if test_interval and iteration % test_interval == 0 and experiment:
                            test_batch = next(test_loader)
                            x0, cond = test_batch.sample, test_batch.cond
                            x0, cond = x0.to(accelerator.device), (cond.to(accelerator.device) if cond is not None else None)
                            x0, sigma, eps, _ = smalldiffusion.diffusion.generate_train_sample(
                                x0, self.schedule
                            )
                            loss = model.get_loss(x0, sigma, eps, cond=cond)
                            if experiment:
                                experiment.log_metric("loss", loss.item(),
                                                      series="test", step=iteration)
                        if generate_interval and iteration % generate_interval == 0 and experiment:
                            if self.data_sample.cond is not None:
                                train_cond = next(iter(train_gen_loader)).cond # type: ignore
                                test_cond = next(iter(test_gen_loader)).cond # type: ignore
                                train_cond, test_cond = train_cond.to(accelerator.device), test_cond.to(accelerator.device)
                                N_train = train_cond.shape[0]
                                N_test = test_cond.shape[0]
                                *_, x0_train = smalldiffusion.diffusion.samples(
                                    self.model, self.schedule.sample_sigmas(self.sample_steps),
                                    batchsize=N_train, cond=train_cond, accelerator=accelerator
                                )
                                *_, x0_test, = smalldiffusion.diffusion.samples(
                                    self.model, self.schedule.sample_sigmas(self.sample_steps),
                                    batchsize=N_test, cond=test_cond, accelerator=accelerator
                                )
                                x0_train = Sample(cond=train_cond, sample=x0_train, # type: ignore
                                    num_classes=self.data_sample.num_classes)
                                x0_test = Sample(cond=test_cond, sample=x0_test, # type: ignore
                                    num_classes=self.data_sample.num_classes)
                                experiment.log({
                                    "samples" : self.train_data.visualize_batch(x0_train),
                                }, step=iteration, series="train")
                                experiment.log({
                                    "samples" : self.train_data.visualize_batch(x0_test)
                                }, step=iteration, series="test")
                            else:
                                *_, x0_samples, = smalldiffusion.diffusion.samples(
                                    self.model, self.schedule.sample_sigmas(self.sample_steps),
                                    batchsize=self.gen_batch_size, cond=None, accelerator=accelerator
                                )
                                x0_samples = Sample(cond=None, sample=x0_samples, # type: ignore
                                    num_classes=self.data_sample.num_classes)
                                experiment.log({
                                    "samples" : self.train_data.visualize_batch(x0_samples)
                                }, series="test", step=iteration)
                        self.model.train()
                    iteration += 1
                    if iteration >= iterations:
                        break
                if progress and total_epochs > 1:
                    pbar.update(epochs_task, advance=1) # type: ignore
                if iteration >= iterations:
                    break
