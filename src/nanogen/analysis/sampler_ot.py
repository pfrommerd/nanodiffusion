from nanoconfig.options import Options
from nanoconfig import config, Config, field, MISSING

from nanoconfig.experiment import ExperimentConfig, Experiment

from ..pipeline import GenerativePipeline

from accelerate import Accelerator

import itertools
import torch.utils._pytree as pytree
import torch
import logging
import ot
import pandas as pd
import numpy as np
import rich.progress

logger = logging.getLogger(__name__)

@config
class OTConfig(Config):
    artifact: str
    num_samples: int
    batch_size: int
    experiment : ExperimentConfig = field(flat=True)

def data_generator(pipeline, accelerator, samples_per_cond):
    logger.info("Computing conditioning bounds...")
    assert pipeline.test_data is not None
    cond_min, cond_max = None, None
    for data in accelerator.prepare(pipeline.test_data.loader(
                batch_size=pipeline.test_batch_size)):
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

    while True:
        cond = pytree.tree_map(
            lambda min, max: (torch.rand(min.shape,
                device=min.device)*(max - min) + min),
            cond_min, cond_max)
        cond_ex = cond.reshape(-1)[None].expand(samples_per_cond, -1)
        ddpm_samples = pipeline.generate(samples_per_cond,
                    cond=cond_ex, accelerator=accelerator,
                    gamma=1.0, mu=0.5).sample
        ddim_samples = pipeline.generate(samples_per_cond,
                    cond=cond_ex, accelerator=accelerator,
                    gamma=1.0, mu=0.).sample
        accel_samples = pipeline.generate(samples_per_cond,
                    cond=cond_ex, accelerator=accelerator,
                    gamma=2.0, mu=0.).sample
        yield cond, ddpm_samples, ddim_samples, accel_samples

def distances(samples_A, samples_B):
    samples_A = samples_A.reshape(samples_A.shape[0], -1)
    samples_B = samples_B.reshape(samples_B.shape[0], -1)
    return torch.cdist(samples_A, samples_B)

def _run(experiment: Experiment):
    assert experiment.config is not None
    config: OTConfig = experiment.config # type: ignore
    accelerator = Accelerator()
    logger.info("Starting OT calculations...")
    if not ":" in config.artifact:
        raise ValueError("teacher_artifact must be in the format 'name:version'")
    name, version = config.artifact.split(":")
    artifact_id = experiment.find_artifact(name, version, "model")
    if artifact_id is None:
        raise ValueError(f"Artifact {name}:{version} not found")
    artifact = experiment.use_artifact(artifact_id)
    assert artifact is not None
    with artifact.open_file("model.safetensors") as f:
        pipeline = GenerativePipeline.load(f)

    rows = []
    for cond, ddpm_samples, ddim_samples, accel_samples in rich.progress.track(itertools.islice(
                data_generator(pipeline, accelerator, config.batch_size), config.num_samples
            ), total=config.num_samples):
        # Process the generated samples here
        # For example, save them to disk or perform some analysis
        C = distances(ddpm_samples, ddim_samples).cpu().numpy()
        ddpm_ddim_dist = np.sum(ot.emd([], [], C)*C) # type: ignore
        C = distances(ddpm_samples, accel_samples).cpu().numpy()
        ddpm_accel_dist = np.sum(ot.emd([], [], C)*C) # type: ignore
        C = distances(ddim_samples, accel_samples).cpu().numpy()
        ddim_accel_dist = np.sum(ot.emd([], [], C)*C) # type: ignore
        rows.append({
            "condition": cond.cpu().numpy(),
            "ddpm_ddim_dist": ddpm_ddim_dist,
            "ddpm_accel_dist": ddpm_accel_dist,
            "ddim_accel_dist": ddim_accel_dist
        })
    df = pd.DataFrame(rows)
    with experiment.create_artifact("ot_results", type="results") as artifact:
        with artifact.create_file("distances.csv") as f:
            df.to_csv(f, index=False)
    experiment.log_table("distances", df)

def main():
    default = OTConfig(
        artifact=MISSING,
        batch_size=256,
        num_samples=10_000,
        experiment=ExperimentConfig(
            wandb=True
        )
    )
    options = Options.as_options(OTConfig, default)
    config : OTConfig = options.from_parsed(options.parse())
    experiment = config.experiment.create(
        logger=logger,
        config=config,
        main=_run
    )
    experiment.run()

if __name__ == "__main__":
    main()
