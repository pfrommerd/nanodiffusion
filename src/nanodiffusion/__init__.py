import warnings

from numpy import c_
warnings.filterwarnings("ignore", category=SyntaxWarning)

import logging

from accelerate import Accelerator
from pathlib import Path

from .utils import setup_logging

from .datasets.tree import TreeDataConfig
from .datasets.trajectory import TrajectoryDataConfig

from .diffuser import Diffuser, DiffuserConfig
from .optimizers import AdamwConfig
from .schedules import LogLinearScheduleConfig

from nanoconfig import config, field
from nanoconfig.options import Options
from nanoconfig.experiment import Experiment, ExperimentConfig

from .datasets import perlin
from .models import mlp, unet1d

logger = logging.getLogger(__name__)

@config
class TrainConfig:
    diffuser: DiffuserConfig = field(flat=True)
    experiment: ExperimentConfig = field(flat=True)
    cpu: bool = False

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
    diffuser = Diffuser.from_config(config.diffuser)
    a = Accelerator(cpu=config.cpu)
    diffuser.train(
        progress=True,
        experiment=experiment,
        accelerator=a
    )
    return diffuser

def train():
    setup_logging()
    logger.info("Training...")
    default = TrainConfig(
        diffuser=DiffuserConfig(
            schedule=LogLinearScheduleConfig(
                timesteps=256,
                sigma_min=0.001,
                sigma_max=20
            ),
            optimizer=AdamwConfig(
                lr=1e-4,
                weight_decay=1e-2,
                betas=(0.9, 0.999),
                eps=1e-8
            ),
            data=TreeDataConfig(),
            model=mlp.MlpConfig(
                hidden_features=(64, 64, 128, 128, 64, 64),
                cond_embed_features=64
            ),
            gen_batch_size=512,
            batch_size=512,
            test_batch_size=512,
            iterations=50_000
        ),
        experiment=ExperimentConfig(
            project="nanodiffusion",
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
