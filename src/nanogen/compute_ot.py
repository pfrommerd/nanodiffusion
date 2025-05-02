from nanoconfig.options import Options
from nanoconfig import config, Config, field, MISSING

from nanoconfig.experiment import ExperimentConfig, Experiment

import logging
import ot

logger = logging.getLogger(__name__)

@config
class OTConfig(Config):
    artifact: str
    batch_size: int = field(default=128)
    experiment : ExperimentConfig = field(flat=True)

def _run(experiment: Experiment):
    pass

def main():
    default = OTConfig(
        artifact=MISSING,
        batch_size=128,
        experiment=ExperimentConfig(
            wandb=True
        )
    )
    options = Options.as_options(OTConfig, default)
    config : OTConfig = options.from_parsed(options.parse())
    config.experiment.create(
        logger=logger,
        config=config,
        main=_run
    )
