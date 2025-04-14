import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import logging

from accelerate import Accelerator
from pathlib import Path

from .utils import setup_logging

from .diffuser import Diffuser, DiffuserConfig
from .optimizers import AdamwConfig
from .schedules import LogLinearScheduleConfig

from nanoconfig import config, field
from nanoconfig.options import Options
from nanoconfig.experiment import Experiment, ExperimentConfig

from .datasets import perlin
from .models import mlp

logger = logging.getLogger(__name__)

@config
class TrainConfig:
    diffuser: DiffuserConfig = field(flat=True)
    experiment: ExperimentConfig = field(flat=True)

def run(experiment: Experiment):
    # re-setup logging in case we are running
    # on the remote server
    setup_logging()
    config : TrainConfig = experiment.config # type: ignore
    diffuser = Diffuser.from_config(config.diffuser)
    diffuser.train(
        progress=True,
        experiment=experiment
    )

def train():
    setup_logging()
    default = TrainConfig(
        diffuser=DiffuserConfig(
            schedule=LogLinearScheduleConfig(
                timesteps=1000,
                sigma_min=0.0001,
                sigma_max=1
            ),
            optimizer=AdamwConfig(
                lr=1e-4,
                weight_decay=1e-2,
                betas=(0.9, 0.999),
                eps=1e-8
            ),
            data=perlin.PerlinDataConfig(),
            model=mlp.MlpConfig(
                hidden_features=(64, 64, 128, 128, 64, 64),
                cond_features=4
            )
        ),
        experiment=ExperimentConfig(
            project="nanodiffusion",
            console=True,
            clearml=True,
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
        run, config=config
    )
    experiment.run()
