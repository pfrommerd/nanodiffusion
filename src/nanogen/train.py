import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import logging


from pathlib import Path
from accelerate import Accelerator

from .utils import setup_logging
from .pipeline import GenerativePipeline, PipelineConfig
from .optimizers import AdamwConfig

from .models.diffusion import DiffusionModelConfig
from .models.diffusion.schedules import LogLinearScheduleConfig

from .models.diffusion import mlp as diffusion_mlp
from .models.diffusion import unet as diffusion_unet

from nanoconfig import config, field
from nanoconfig.options import Options
from nanoconfig.experiment import Experiment, ExperimentConfig

logger = logging.getLogger(__name__)

@config
class TrainConfig:
    pipeline: PipelineConfig = field(flat=True)
    experiment: ExperimentConfig = field(flat=True)
    final_checkpoint: bool = True
    checkpoint_interval: int | None = None
    cpu: bool = False
    distill: bool = False

    def run(self, logger):
        experiment = self.experiment.create(
            logger, _run_experiment, config=self # type: ignore
        )
        return experiment.run()

def _run_experiment(experiment: Experiment):
    # re-setup logging in case we are running
    # on the remote server
    setup_logging()
    config : TrainConfig = experiment.config # type: ignore
    pipeline = GenerativePipeline.from_config(config.pipeline)
    a = Accelerator(cpu=config.cpu)
    logger.info(f"Training {config}")
    pipeline.train(
        progress=True,
        experiment=experiment,
        accelerator=a,
        checkpoint_interval=config.checkpoint_interval
    )
    if config.final_checkpoint:
        with experiment.create_artifact("diffuser", type="model") as builder:
            with builder.create_file("model.safetensors") as f:
                pipeline.save(f)
    # For convenience, run the distillation right away
    if config.distill:
        # essentially makes the next step, step "0"
        experiment.step_offset(experiment.step + 1)
        student = GenerativePipeline.from_config(config.pipeline)
        student.distill(pipeline, progress=True,
            experiment=experiment, accelerator=a)
        if config.final_checkpoint:
            with experiment.create_artifact("distilled", type="model") as builder:
                with builder.create_file("model.safetensors") as f:
                    student.save(f)
    return pipeline

def main():
    setup_logging()
    default = TrainConfig(
        pipeline=PipelineConfig(
            model=DiffusionModelConfig(
                nn=diffusion_mlp.MlpConfig(
                    hidden_features=(256, 256, 256, 256),
                    embed_features=128
                ),
                schedule=LogLinearScheduleConfig(
                    timesteps=512, sigma_min=5e-4, sigma_max=5
                ),
                sample_timesteps=32,
                ideal_denoiser=False
            ),
            optimizer=AdamwConfig(
                lr=5e-4, weight_decay=1e-4,
                betas=(0.9, 0.999), eps=1e-8
            ),
            data="single-maze-trajectory",
            gen_batch_size=512,
            batch_size=512,
            test_batch_size=512,
            iterations=50_000
        ),
        experiment=ExperimentConfig(
            project="nanogen",
            console=True,
            console_intervals={
                "train": 100,
                "test": 100
            }
        )
    )
    opts = Options.as_options(TrainConfig, default=default)
    config = opts.from_parsed(opts.parse())

    experiment = config.experiment.create(
        logger,
        _run_experiment, config=config
    )
    experiment.run()
