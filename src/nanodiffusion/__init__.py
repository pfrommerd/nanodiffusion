import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

import logging

from accelerate import Accelerator
from .utils import setup_logging

from .diffuser import Diffuser, DiffuserConfig
from .optimizers import AdamwConfig
from .schedules import LogLinearScheduleConfig

from nanoconfig import config, field, options
from nanoconfig.experiment import ExperimentConfig

from .datasets import tree_dataset
from .models import mlp
logger = logging.getLogger(__name__)

@config
class TrainConfig:
    diffuser: DiffuserConfig = field(flat=True)
    experiment: ExperimentConfig = field(flat=True)

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
            data=tree_dataset.TreeDataConfig(),
            model=mlp.MlpConfig(
                hidden_features=(64, 64, 128, 128, 64, 64),
                cond_features=4
            )
        ),
        experiment=ExperimentConfig(
            project="nanodiffusion",
            console=True,
            console_intervals={
                "train": 10,
                "test": 100
            }
        )
    )
    opts = options.as_options(TrainConfig, default=default)
    parsed = options.parse_cli_options(opts)
    config = options.from_parsed_options(parsed, TrainConfig, default=default)

    diffuser = Diffuser.from_config(config.diffuser)

    experiment = config.experiment.create(logger)
    diffuser.train(
        progress=True,
        experiment=experiment
    )
