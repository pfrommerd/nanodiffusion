from nanoconfig import config, field, Config, MISSING
from nanoconfig.experiment import Experiment, ExperimentConfig
from nanoconfig.options import Options

from nanogen.models.diffusion import DiffusionModel, DiffusionModelConfig
from nanogen.pipeline import GenerativePipeline

from .utils import setup_logging

import logging
logger = logging.getLogger(__name__)

@config
class DistillConfig(Config):
    teacher_artifact: str
    experiment: ExperimentConfig = field(flat=True)
    final_checkpoint: bool = field(default=False)

    sampler_preset: str | None = None

    batch_size: int | None = None
    iterations: int | None = None

def _run(experiment: Experiment):
    assert experiment.config is not None
    config : DistillConfig = experiment.config # type: ignore
    logger.info("Starting distillation...")
    if not ":" in config.teacher_artifact:
        raise ValueError("teacher_artifact must be in the format 'name:version'")
    name, version = config.teacher_artifact.split(":")
    artifact_id = experiment.find_artifact(name, version, "model")
    if artifact_id is None:
        raise ValueError(f"Artifact {name}:{version} not found")
    artifact = experiment.use_artifact(artifact_id)
    assert artifact is not None

    with artifact.open_file("model.safetensors") as f:
        teacher = GenerativePipeline.load(f)

    logger.info("Loaded teacher model")
    student = GenerativePipeline.from_config(teacher.config)
    # Override batch size if specified
    if config.batch_size is not None:
        student.batch_size = config.batch_size

    if config.sampler_preset is not None:
        assert isinstance(teacher.model, DiffusionModel)
        match config.sampler_preset:
            case "ddim":
                teacher.model.gamma = 1.0
                teacher.model.mu = 0.0
            case "ddpm":
                teacher.model.gamma = 1.0
                teacher.model.mu = 0.5
            case "accel":
                teacher.model.gamma = 2.0
                teacher.model.mu = 0.0
            case _:
                raise ValueError(f"Unknown sampler preset {config.sampler_preset}")

    student.distill(
        teacher, config.iterations,
        experiment=experiment,
        progress=True
    )
    if config.final_checkpoint:
        with experiment.create_artifact("distilled", type="model") as builder:
            with builder.create_file("model.safetensors") as f:
                student.save(f)

def main():
    setup_logging()
    default = DistillConfig(
        teacher_artifact=MISSING, # type: ignore
        final_checkpoint=True,
        experiment=ExperimentConfig(
            project="nanogen",
            console=True,
            wandb=True,
            console_intervals={
                "train": 100,
                "test": 100
            }
        )
    )
    opts = Options.as_options(DistillConfig, default)
    config : DistillConfig = opts.from_parsed(opts.parse())
    experiment = config.experiment.create(
        logger=logger,
        config=config, main=_run
    )
    experiment.run()
