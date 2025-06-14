import logging
import accelerate
import torch
import random
import itertools
import contextlib
import smalldiffusion
import numpy as np
import typing as ty
import time

import torch.utils._pytree as pytree

from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from accelerate import Accelerator
from smalldiffusion import Schedule, ModelMixin

from nanoconfig import config, Config
from nanoconfig.data import Data
from nanoconfig.data.source import DataRepository
from nanoconfig.data.torch import SizedDataset, TorchAdapter

from dataclasses import replace

from .data import DataPoint
from .models import GenerativeModel, ModelConfig

from .utils import Interval, Iterations
from .optimizers import OptimizerConfig
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

# Import all of the different models
# so that they can be deserialized

from .models.diffusion import mlp as diffusion_mlp
from .models.diffusion import unet as diffusion_unet

#

logger = logging.getLogger(__name__)

@config
class PipelineConfig(Config):
    model: ModelConfig
    optimizer : OptimizerConfig
    # Aliases or sha of the different datasets to use
    data: str | None = None
    # Override the data type
    data_type: str | None = None
    # Limit the number of training samples to use
    limit_data: int | None = None

    batch_size: int = 32
    test_batch_size: int = 64
    gen_batch_size: int = 64

    test_interval: int = 100
    gen_interval: int = 1000

    iterations: int = 10_000

T = ty.TypeVar("T", bound=DataPoint)

class GenerativePipeline(ty.Generic[T]):
    def __init__(self,
            config: PipelineConfig,
            model : GenerativeModel,
            datapoint: T, *,
            ############## The training informaiton ############
            optimizer : Optimizer | None = None,
            lr_scheduler: LRScheduler | None = None,
            train_data: SizedDataset[T] | None = None,
            test_data: SizedDataset[T] | None = None,
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

        self.train_data = train_data
        self.test_data = test_data
        self.datapoint = datapoint

        self.test_interval = test_interval
        self.gen_interval = gen_interval

        self.total_iterations = total_iterations
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.gen_batch_size = gen_batch_size

    @staticmethod
    def from_config(config: PipelineConfig,
                create_optimizer: bool = True,
                data_adapter: TorchAdapter[T] | None = None,
                data_repo: DataRepository | None = None,
                data: Data | None = None) -> "GenerativePipeline[T]":
        if data is None:
            assert config.data is not None
            if data_repo is None:
                data_repo = DataRepository.default()
            data = data_repo.lookup(config.data)
        if data is None:
            raise ValueError(f"Data not found: {config.data}")
        if data_adapter is None:
            data_adapter = TorchAdapter(override_mime_type=config.data_type)
            from .data import register_types
            register_types(data_adapter) # type: ignore
        # Set the data to sha256 of the data
        config = replace(config, data=data.sha256)
        train_data = data.split("train", data_adapter)
        test_data = data.split("test", data_adapter)
        assert train_data is not None and test_data is not None
        if config.limit_data is not None:
            train_data = train_data.limit(config.limit_data)
        model = config.model.create(train_data) # type: ignore
        if list(model.parameters()):
            optimizer, lr_scheduler = config.optimizer.create(model.parameters(), config.iterations) if create_optimizer else (None, None)
        else:
            logger.warn("Model has no parameters...")
            optimizer, lr_scheduler = None, None
        return GenerativePipeline(
            config, model, train_data.data_sample,
            optimizer=optimizer, lr_scheduler=lr_scheduler,
            train_data=train_data, test_data=test_data,
            batch_size=config.batch_size, test_interval=config.test_interval,
            gen_interval=config.gen_interval, test_batch_size=config.test_batch_size,
            gen_batch_size=config.gen_batch_size, total_iterations=config.iterations,
        )

    def to_config(self) -> PipelineConfig:
        return self.config

    def save(self, file, save_optim_state: bool = False):
        config = self.to_config()
        data = {
            "model": self.model.state_dict(),
            "config": config.to_dict()
        }
        io.save(file, data)

    @staticmethod
    def load(file, load_optim_state: bool = False,
                device: str = "cpu", modify_config: ty.Callable | None = None):
        data = io.load(file, device=device)
        config = PipelineConfig.from_dict(data["config"])
        if modify_config is not None:
            config = modify_config(config)
        pipeline = GenerativePipeline.from_config(config)
        pipeline.model.load_state_dict(data["model"])
        return pipeline

    def generate(self, N: int | None = None, cond: torch.Tensor | None = None,
                    seed: int | None = None, accelerator: Accelerator | None = None,
                    **kwargs) -> T:
        if N is None:
            N = 16 if cond is None else utils.axis_size(cond, 0)
        accelerator = accelerator or Accelerator()
        model : GenerativeModel = accelerator.prepare(self.model)
        model.eval()

        if cond is None and self.datapoint.has_cond:
            loader = self.train_data.loader(batch_size=N, shuffle=True) # type: ignore
            cond = next(iter(loader)).cond # type: ignore
            cond = cond.to(accelerator.device) # type: ignore
            N = min(N, utils.axis_size(cond, 0))
        sample_structure = pytree.tree_map(
            lambda x: x[None].expand(N, *x.shape) if isinstance(x, torch.Tensor) else x,
            self.datapoint.sample
        )
        *_, samples = model.generate(sample_structure, cond, **kwargs)
        return self.datapoint.with_values(samples, cond) # type: ignore

    def train(self, iterations : Interval | int | None = None,
              *,
              accelerator : Accelerator | None = None,
              experiment: Experiment | None = None,
              generate_interval : Interval | int | None = None,
              test_interval: Interval | int | None = None,
              checkpoint_interval: Interval | int | None = None,
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
        checkpoint_interval = Interval.to_iterations(checkpoint_interval, iterations_per_epoch)

        accelerator = accelerator if accelerator else Accelerator()

        train_loader = self.train_data.loader(batch_size=self.batch_size, shuffle=True)
        test_loader = self.test_data.loader(batch_size=self.test_batch_size, shuffle=True)
        if self.datapoint.has_cond:
            train_gen_loader = self.train_data.loader(batch_size=self.gen_batch_size, shuffle=True)
            test_gen_loader = self.test_data.loader(batch_size=self.gen_batch_size, shuffle=True)
            gen_batch = None
        else:
            train_gen_loader, test_gen_loader  = None, None
            gen_batch = pytree.tree_map(
                lambda x: torch.repeat_interleave(x[None], self.gen_batch_size, dim=0)
                    if isinstance(x, torch.Tensor) else x, self.datapoint
            ).to(accelerator.device)
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

        if experiment:
            sample_batch : T = next(test_gen_loader) if test_gen_loader is not None else next(test_loader)
            experiment.log({
                "samples" : sample_batch.to_result()
            }, series="gt")

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
                    if optimizer: optimizer.zero_grad()
                    loss, metrics = model.loss(batch.sample, cond=batch.cond)
                    if optimizer:
                        accelerator.backward(loss)
                        optimizer.step()
                    if lr_scheduler: lr_scheduler.step()
                    # check if we are done
                    if progress:
                        pbar.update(iterations_task, advance=1) # type: ignore
                    if progress and total_epochs > 1:
                        pbar.update(epoch_iterations_task, advance=1) # type: ignore

                    if experiment:
                        experiment.log(metrics, series="train", step=iteration)
                        experiment.log_metric("loss", loss, series="train", step=iteration)
                    if experiment and lr_scheduler:
                        experiment.log_metric("lr", lr_scheduler.get_last_lr()[0],
                                            series="train", step=iteration)
                    with torch.no_grad():
                        model.eval()
                        if test_interval and iteration % test_interval == 0 and experiment:
                            test_batch = next(test_loader)
                            loss, metrics = model.loss(test_batch.sample, cond=test_batch.cond)
                            metrics["loss"] = loss
                            if experiment:
                                experiment.log(metrics, series="test", step=iteration)
                                experiment.log_metric("loss", loss, series="test", step=iteration)
                        if generate_interval and iteration % generate_interval == 0 and experiment:
                            if self.datapoint.has_cond:
                                train_batch = next(train_gen_loader).to(accelerator.device) # type: ignore
                                test_batch = next(test_gen_loader).to(accelerator.device) # type: ignore
                                *_, train_samples = model.generate(train_batch.sample, train_batch.cond)
                                *_, test_samples = model.generate(test_batch.sample, test_batch.cond)
                                train_datapoints : T = train_batch.with_values(train_samples) # type: ignore
                                test_datapoints : T = test_batch.with_values(test_samples) # type: ignore
                                experiment.log({"samples" : train_datapoints.to_result()}, step=iteration, series="train")
                                experiment.log({"samples" : test_datapoints.to_result()}, step=iteration, series="test")
                            else:
                                assert gen_batch is not None
                                *_, train_samples = model.generate(gen_batch.sample)
                                train_datapoints : T = self.datapoint.with_values(train_samples) # type: ignore
                                experiment.log({"samples" : train_datapoints.to_result()}, series="test", step=iteration)
                        if checkpoint_interval and experiment and iteration % checkpoint_interval ==0:
                            with experiment.create_artifact(f"checkpoint_{iteration}", type="model") as builder:
                                with builder.create_file("model.safetensors") as f:
                                    self.save(f)
                    iteration += 1
                    if iteration >= iterations:
                        break
                if progress and total_epochs > 1:
                    pbar.update(epochs_task, advance=1) # type: ignore
                if iteration >= iterations:
                    break

    def distill(self, teacher: "GenerativePipeline",
              iterations : Interval | int | None = None, *,
              accelerator : Accelerator | None = None,
              experiment: Experiment | None = None,
              generate_interval : Interval | int | None = None,
              test_interval: Interval | int | None = None,
              progress : bool = False):
        assert self.train_data is not None, "No train data provided"
        assert self.test_data is not None, "No test data provided"

        iterations = iterations if iterations is not None else self.total_iterations
        generate_interval = generate_interval if generate_interval is not None else self.gen_interval
        test_interval = test_interval if test_interval is not None else self.test_interval
        assert iterations is not None, "No iterations provided"
        iterations = Interval.to_iterations(iterations, None)
        generate_interval = Interval.to_iterations(generate_interval, None)
        test_interval = Interval.to_iterations(test_interval, None)
        accelerator = accelerator if accelerator else Accelerator()

        # Find the upper and lower bounds for the conditioning
        cond_min, cond_max = None, None
        if self.datapoint.has_cond:
            logger.info("Computing conditioning bounds...")
            for data in accelerator.prepare(self.train_data.loader(
                        batch_size=self.test_batch_size)):
                cond = data.cond
                batch_min = pytree.tree_map(lambda x: x.min(dim=0)[0], cond)
                batch_max = pytree.tree_map(lambda x: x.max(dim=0)[0], cond)
                if cond_min is None or cond_max is None:
                    cond_min, cond_max = batch_min, batch_max
                else:
                    cond_min = pytree.tree_map(torch.minimum, cond_min, batch_min)
                    cond_max = pytree.tree_map(torch.maximum, cond_max, batch_max)
            assert cond_min is not None and cond_max is not None
            logger.info("Computed bounds.")

        sample_gen_batch = pytree.tree_map(
            lambda x: x[None].expand(self.gen_batch_size, *x.shape) if isinstance(x, torch.Tensor) else x,
            self.datapoint.sample
        )

        (model, optimizer, lr_scheduler) = accelerator.prepare(
            self.model, self.optimizer, self.lr_scheduler,
        )
        total_params = sum(param.numel() for param in model.parameters())
        if progress:
            pbar = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(finished_style=RichStyle(color="green")),
                MofNColumn(6),
                TaskProgressColumn(), TimeRemainingColumn(),
                TimeElapsedColumn(), refresh_per_second=1
            )
            iterations_task = pbar.add_task(
                "[bold blue]Total Iterations[/bold blue]",
                total=iterations)
        else: pbar = contextlib.nullcontext()

        def data_generator():
            shuffle_buffer = []
            while True:
                if len(shuffle_buffer) < 64:
                    cond = pytree.tree_map(
                        lambda min, max: torch.rand(8*self.batch_size, *min.shape, device=min.device)*(max - min) + min,
                        cond_min, cond_max)
                    samples = teacher.generate(8*self.batch_size, cond=cond, accelerator=accelerator)
                    # Convert big batch into smaller batches
                    samples = pytree.tree_map(
                        lambda x: x.reshape(8, -1, *x.shape[1:]) if isinstance(x, torch.Tensor) else x, samples
                    )
                    batches = [pytree.tree_map(lambda x: x[i] if isinstance(x, torch.Tensor) else x, samples)
                                for i in range(8)]
                    shuffle_buffer.extend(batches)
                else:
                    buffer = shuffle_buffer.pop(torch.randint(len(shuffle_buffer), ()))
                    v = torch.bernoulli(torch.tensor(0.9))
                    if v: shuffle_buffer.append(buffer)
                    yield buffer

        with pbar:
            model.train()
            for iteration, batch in enumerate(itertools.islice(data_generator(), iterations)):
                optimizer.zero_grad()
                loss, metrics = model.loss(batch.sample, cond=batch.cond)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                if experiment:
                    experiment.log_metric("loss", loss.item(),
                                        series="train", step=iteration)
                    experiment.log_metric("lr", lr_scheduler.get_last_lr()[0],
                                        series="train", step=iteration)
                    experiment.log(metrics, series="train", step=iteration)

                    if test_interval and iteration % generate_interval == 0 and experiment:
                        cond_batch = pytree.tree_map(
                            lambda min, max: torch.rand(self.gen_batch_size, *min.shape, device=min.device)*(max - min) + min,
                            cond_min, cond_max)
                        *_, samples = model.generate(sample_gen_batch, cond_batch)
                        samples = self.datapoint.with_values(samples, cond_batch)
                        # log the training samples
                        experiment.log({"samples" : batch.to_result()}, step=iteration, series="gt")
                        experiment.log({"samples" : samples.to_result()}, step=iteration, series="test")

                if progress:
                    pbar.advance(iterations_task) # type: ignore
