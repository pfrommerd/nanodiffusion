from nanoconfig.options import Options
from nanoconfig import config, Config, field, MISSING

from nanoconfig.experiment import ExperimentConfig, Experiment

from nanogen.models.diffusion import DiffusionModel

from ..pipeline import GenerativePipeline
from ..utils import setup_logging

from accelerate import Accelerator

import itertools
import torch.utils._pytree as pytree
import torch
import logging
import ot
import pandas as pd
import numpy as np
import rich.progress
import torch.func

logger = logging.getLogger(__name__)

@config
class MetricsConfig(Config):
    artifact: str
    divergence_dims: int
    num_samples: int
    batch_size: int
    timesteps: int
    eval_timesteps_interval: int
    experiment : ExperimentConfig = field(flat=True)

def sq_norm(M, k):
    return (torch.norm(M, dim=1)**2).unsqueeze(1).repeat(1,k)

def sum_except_dim(x, dim):
    dims = list(range(x.ndim))
    dims.remove(dim)
    return torch.sum(x, dim=dims, keepdim=True)

def divergence_diff(nd, a, b, x) -> torch.Tensor:
    if b is not None:
        assert a.shape == b.shape
    assert a.ndim == 2
    # do a "randomized" divergence by picking a random coordinate
    # per element in the batch
    ests = []
    for j in range(nd):
        i = torch.randint(0, a.shape[1], (a.shape[0],), device=a.device)
        a_elems = torch.gather(a, dim=1, index=i.unsqueeze(1))
        all_par_a = torch.autograd.grad(a_elems, x, torch.ones_like(a_elems), retain_graph=True)[0]
        div_a = torch.gather(all_par_a, dim=1, index=i.unsqueeze(1)).squeeze(1)
        if b is not None:
            b_elems = torch.gather(b, dim=1, index=i.unsqueeze(1))
            all_par_b = torch.autograd.grad(b_elems, x, torch.ones_like(b_elems), retain_graph=True)[0]
            div_b = torch.gather(all_par_b, dim=1, index=i.unsqueeze(1)).squeeze(1)
            ests.append((div_a - div_b)*a.shape[1])
        else:
            ests.append(div_a*a.shape[1])
    ests = torch.stack(ests, -1)
    ests = torch.mean(ests, -1)
    return ests

def measure_si(model, cond, final_samples, sigma, nd) -> torch.Tensor:
    inter_samples = model.generate_forward(final_samples, sigma)

    orig_shape = inter_samples.shape
    inter_samples_flat = inter_samples.reshape(inter_samples.shape[0], -1)
    inter_samples_flat = inter_samples_flat.detach().requires_grad_()
    pred = model.diffuser(inter_samples_flat.reshape(*orig_shape), sigma, cond=cond)
    pred = pred.reshape(inter_samples.shape[0], -1)

    x_flat = inter_samples_flat
    d_flat = final_samples.flatten(start_dim=1) # type: ignore
    (xb, xr), (db, dr) = x_flat.shape, d_flat.shape
    sq_diffs = sq_norm(x_flat, db).T + sq_norm(d_flat, xb) - 2 * d_flat @ x_flat.T # shape: db x xb
    log_weights = -sq_diffs/2/sigma.squeeze()**2
    weights = torch.nn.functional.softmax(log_weights, dim=0)
    x0_flat = torch.einsum('ij,i...->j...', weights, d_flat)
    true_eps = (x_flat - x0_flat) / sigma
    # use zeros for the comparison since true_eps
    # comes from nabla log p
    div = divergence_diff(nd, pred, None, inter_samples_flat).detach()
    pred = pred.reshape(*orig_shape).detach()
    true_eps = true_eps.reshape(*orig_shape).detach()

    # we scale the div comp by sigma to match the scaling of the dot_comp
    # note that by the manner in which we have parameterized "v"
    div_comp = sigma * div
    dot_comp = torch.sum(true_eps.reshape(true_eps.shape[0], -1) *
            (true_eps - pred).reshape(pred.shape[0], -1), dim=1)
    si = div_comp + dot_comp
    si = si.abs().mean()
    return si

def measure_si_traj(model, cond, final_samples, sigmas, nd):
    # return torch.zeros(len(inter_samples), device=final_samples.device)
    return torch.stack([
        measure_si(model, cond, final_samples, sigma, nd)
        for sigma in sigmas
    ], dim=0)

def data_generator(pipeline, accelerator, samples_per_cond, timesteps_interval,
                            divergence_dims):
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
    model = accelerator.prepare(pipeline.model)

    while True:
        cond = pytree.tree_map(
            lambda min, max: (torch.rand(min.shape,
                device=min.device)*(max - min) + min),
            cond_min, cond_max)
        cond_ex = pytree.tree_map(
            lambda x: x.reshape(-1)[None].expand(samples_per_cond, -1), cond
        )
        sample_structure = pytree.tree_map(
            lambda x: x[None].expand(samples_per_cond, *x.shape) if isinstance(x, torch.Tensor) else x,
            pipeline.datapoint.sample
        )
        ddpm_samples = list(model.generate(sample_structure,
                    cond=cond_ex, gamma=1.0, mu=0.5))
        ddpm_si = measure_si_traj(model, cond_ex, ddpm_samples[-1], model.gen_sigmas[::timesteps_interval], divergence_dims)
        ddpm_samples = ddpm_samples[-1]

        ddim_samples = list(model.generate(sample_structure,
                    cond=cond_ex, gamma=1.0, mu=0.))
        ddim_si = measure_si_traj(model, cond_ex, ddim_samples[-1], model.gen_sigmas[::timesteps_interval], divergence_dims)
        ddim_samples = ddim_samples[-1]

        accel_samples = list(model.generate(sample_structure,
                    cond=cond_ex, gamma=2.0, mu=0.))
        accel_si = measure_si_traj(model, cond_ex, accel_samples[-1], model.gen_sigmas[::timesteps_interval], divergence_dims)
        accel_samples = accel_samples[-1]
        yield (cond, (ddpm_samples, ddpm_si),
            (ddim_samples, ddim_si), (accel_samples, accel_si)
        )

def distances(samples_A, samples_B):
    samples_A = samples_A.reshape(samples_A.shape[0], -1)
    samples_B = samples_B.reshape(samples_B.shape[0], -1)
    return torch.cdist(samples_A, samples_B)

def _run(experiment: Experiment):
    assert experiment.config is not None
    config: MetricsConfig = experiment.config # type: ignore
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

    assert isinstance(pipeline.model, DiffusionModel)
    pipeline.model.gen_sigmas = pipeline.model.train_noise_schedule.sample_sigmas(config.timesteps)

    rows = []
    for i, (cond, (ddpm_samples, ddpm_si), (ddim_samples, ddim_si),
                    (accel_samples, accel_si)) in enumerate(rich.progress.track(itertools.islice(
                data_generator(pipeline, accelerator, config.batch_size, config.eval_timesteps_interval, config.divergence_dims), config.num_samples
            ), total=config.num_samples)):
        cond = pytree.tree_map(
            lambda x: x.cpu().numpy() if isinstance(x, torch.Tensor) else x,
            cond
        )
        ddpm_si, ddim_si, accel_si = (
            ddpm_si.cpu().numpy(), ddim_si.cpu().numpy(),
            accel_si.cpu().numpy()
        )
        # Process the generated samples here
        # For example, save them to disk or perform some analysis
        C = distances(ddpm_samples, ddim_samples).cpu().numpy()
        ddpm_ddim_dist = np.sum(ot.emd([], [], C)*C) # type: ignore
        C = distances(ddpm_samples, accel_samples).cpu().numpy()
        ddpm_accel_dist = np.sum(ot.emd([], [], C)*C) # type: ignore
        C = distances(ddim_samples, accel_samples).cpu().numpy()
        ddim_accel_dist = np.sum(ot.emd([], [], C)*C) # type: ignore

        row = {}
        if isinstance(cond, (torch.Tensor,np.ndarray)):
            row.update({
                f"condition/{i}": cond[i] for i in range(cond.shape[-1])
            })
        else:
            for key, value in cond.items():
                row.update({
                    f"condition/{key}/{i}": cond[key][i] for i in range(cond[key].shape[-1])
                })
        row.update({f"ddpm_si/{t}": ddpm_si[t] for t in range(ddpm_si.shape[0])})
        row.update({f"ddim_si/{t}": ddim_si[t] for t in range(ddim_si.shape[0])})
        row.update({f"accel_si/{t}": accel_si[t] for t in range(accel_si.shape[0])})
        row.update({
            "ddpm_ddim_dist": ddpm_ddim_dist,
            "ddpm_accel_dist": ddpm_accel_dist,
            "ddim_accel_dist": ddim_accel_dist
        })
        rows.append(row)
        if i % 100 == 0:
            for (k, v) in row.items():
                logger.info(f"{i}: {k}: {v}")
    df = pd.DataFrame(rows)
    with experiment.create_artifact("results", type="results") as artifact:
        with artifact.create_file("metrics.csv") as f:
            df.to_csv(f, index=False)
    experiment.log_table("distances", df)

def main():
    setup_logging()
    default = MetricsConfig(
        artifact=MISSING,
        batch_size=64,
        num_samples=100,
        divergence_dims=1,
        experiment=ExperimentConfig(
            wandb=True
        ),
        timesteps=64,
        eval_timesteps_interval=8
    )
    options = Options.as_options(MetricsConfig, default)
    config : MetricsConfig = options.from_parsed(options.parse())
    experiment = config.experiment.create(
        logger=logger,
        config=config,
        main=_run
    )
    experiment.run()

if __name__ == "__main__":
    main()
