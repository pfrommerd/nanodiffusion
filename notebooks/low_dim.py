

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from nanogen.pipeline import GenerativePipeline, PipelineConfig
    from nanogen.models.diffusion import DiffusionModelConfig
    from nanogen.models.diffusion import mlp as diffusion_mlp
    from nanogen.models.diffusion.schedules import LogLinearScheduleConfig
    from nanogen.optimizers import AdamwConfig

    from nanoconfig.experiment import ExperimentConfig
    from nanoconfig.data import Data, SplitInfo

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import seaborn as sns
    import pyarrow.dataset as ds
    import pyarrow as pa
    import numpy.random as nr
    import hashlib

    sns.set_theme()
    plt.rcParams["font.family"] = "serif"


    import nanogen.utils as utils
    import logging

    from accelerate import Accelerator

    accelerator = Accelerator()


    logger = logging.getLogger("nanogen.notebooks")
    utils.setup_logging()
    return (
        AdamwConfig,
        Data,
        DiffusionModelConfig,
        ExperimentConfig,
        GenerativePipeline,
        LogLinearScheduleConfig,
        PipelineConfig,
        SplitInfo,
        accelerator,
        diffusion_mlp,
        ds,
        hashlib,
        logger,
        np,
        nr,
        pa,
        plt,
        torch,
    )


@app.cell(hide_code=True)
def _(Data, SplitInfo, ds, hashlib):
    class InMemoryData(Data):
        def __init__(self, splits : dict[str, ds.Dataset]):
            self.splits = splits
        def aux(self):
            raise NotImplementedError()
        def split_info(self, split):
            return {
                k: SplitInfo(k, v.count_rows(), 0, v.schema) for (k, v) in self.splits
            }
        def split(self, name, adapter = None):
            data = self.splits[name]
            if adapter is not None:
                data = adapter(data)
            return data
        def split_infos(self):
            return {k: self.split_info(k) for k in self.splits.keys()}
        def splits(self, adapter = None):
            return {k: self.split(k, adapter) for k in self.splits.keys()}

        def sha256(self):
            return hashlib.sha256(b"foobar").hexdigest()

    return (InMemoryData,)


@app.cell
def _(ds, nr, pa):
    def create_ds(name):
        N = 100_000
        if name == "two_modes":
            cond = nr.choice([0, 1], size=(N,))
            value = nr.choice([1, 2], size=(N,))*cond - 1
            value = value + nr.normal(loc=0, scale=0.08,size=(N,))
            # cond = cond + 0.001*nr.uniform(-1,1,size=(N,))
        elif name == "continuous_modes":
            cond = nr.choice([0, 1], size=(N,))
            value = nr.choice([1, 2], size=(N,))*cond - 1
            value = value + nr.normal(loc=0, scale=0.08,size=(N,))        
            cond = cond + 0.08*nr.normal(size=(N,))
        return ds.dataset(pa.table([cond,value],
            schema=pa.schema([
                pa.field("x", pa.float32()),
                pa.field("y", pa.float32())
            ], metadata={"mime_type": "data/cond_point"})))
    return (create_ds,)


@app.cell
def _(
    AdamwConfig,
    DiffusionModelConfig,
    ExperimentConfig,
    GenerativePipeline,
    InMemoryData,
    LogLinearScheduleConfig,
    PipelineConfig,
    create_ds,
    diffusion_mlp,
    logger,
):
    config = PipelineConfig(
        model=DiffusionModelConfig(
            nn=diffusion_mlp.MlpConfig(
                hidden_features=(64, 64, 64, 64),
                embed_features=64, simple=True
            ),
            schedule=LogLinearScheduleConfig(
                timesteps=1024, sigma_min=8e-3, sigma_max=10
            ),
            sampler_preset="ddpm",
            sample_timesteps=128,
            ideal_denoiser=False
        ),
        optimizer=AdamwConfig(
            lr=4e-3, weight_decay=1e-2,
            betas=(0.9, 0.999), eps=1e-8
        ),
        gen_batch_size=512,
        batch_size=128,
        test_batch_size=512,
        iterations=10_000
    )
    def train_pipeline(data_name):
        data = InMemoryData({"test": create_ds(data_name), "train": create_ds(data_name)})
        pipeline = GenerativePipeline.from_config(config, data=data)
        experiment = ExperimentConfig(wandb=False, console_intervals={"train": 500, "test": 500}).create(logger=logger)
        pipeline.train(progress=True, experiment=experiment)
        return pipeline

    pipelines = {
        "two_modes": train_pipeline("two_modes"),
        "continuous_modes": train_pipeline("continuous_modes")
    }
    return config, pipelines


@app.cell
def _(GenerativePipeline, InMemoryData, config, create_ds, pipelines, torch):
    from nanogen.models.diffusion import IdealDiffuser
    from torch import nn

    class ClosedFormDiffuser(nn.Module):
        def __init__(self):
            super().__init__()
            self.sigma = 0.1

        def left_flow(self, x, s):
            combined_var = self.sigma**2 + s**2
            # compute the score at the sigma**2 + 0.1**2 noise level
            score = -(x + 1)/combined_var
            # convert from score to expectation x0
            # given the noise level using tweedie's
            x0 = x + s**2 * score
            x1 = (x - x0)/s
            return x1

        def right_flow(self, x, s):
            combined_var = self.sigma**2 + s**2

            a_log_prob = -(x - 1)**2/(2*combined_var)
            b_log_prob = -x**2/(2*combined_var)
            tot_log_prob = torch.logaddexp(a_log_prob, b_log_prob)
            a_log_prob = a_log_prob - tot_log_prob
            b_log_prob = b_log_prob - tot_log_prob

            a_weight = torch.exp(a_log_prob)
            b_weight = torch.exp(b_log_prob)

            a_score = -(x - 1)/combined_var
            b_score = -x/combined_var

            score = a_score*a_weight + b_score*b_weight
            # convert from score to expectation x0
            # given the noise level using tweedie's
            x0 = x + s**2 * score
            x1 = (x - x0)/s
            return x1

    class LinearDiffuser(ClosedFormDiffuser):
        def forward(self, x, sigma, cond):
            return self.left_flow(x, sigma)*(1-cond) + self.right_flow(x,sigma)*cond

    class SinusoidalDiffuser(ClosedFormDiffuser):
        def forward(self, x, t, cond):
            c = 4
            g = lambda s: 1.5/(1+(c*s)**2)**(3/2) - 1.5/((1 + c**2)**(3/2))
            l = g(cond)
            r = g(1-cond)
            total = (l + r)
            l /= total
            r /= total
            return self.left_flow(x, t)*l + self.right_flow(x,t)*r

    data_ = InMemoryData({"test": create_ds("two_modes"), "train": create_ds("two_modes")})

    linear_pipeline = GenerativePipeline.from_config(config, data=data_)
    linear_pipeline.model.diffuser = LinearDiffuser()
    sinusoid_pipeline = GenerativePipeline.from_config(config, data=data_)
    sinusoid_pipeline.model.diffuser = SinusoidalDiffuser()
    all_pipelines = dict(pipelines)
    all_pipelines.update({
        "linear": linear_pipeline,
        "fourier": sinusoid_pipeline
    })
    return (all_pipelines,)


@app.cell
def _(GenerativePipeline, accelerator, np, pipelines, torch):
    from nanogen.analysis.sampler_metrics import measure_si_traj
    SIGMAS = 64
    SAMPLES = 2_000
    eval_points = np.arange(0, 1.01, 0.01, dtype=np.float32)
    def gen_si(pipeline: GenerativePipeline):
        model = accelerator.prepare(pipeline.model)
        pipeline.model.gen_sigmas = pipeline.model.train_noise_schedule.sample_sigmas(SIGMAS)
        inconsistencies = []
        for c in eval_points:
            conds = torch.tensor(c, device=accelerator.device)[None,None].repeat(SAMPLES, 1)
            s = pipeline.generate(cond=conds, accelerator=accelerator).sample
            inconsistencies.append(measure_si_traj(model, conds, s, model.gen_sigmas))
        inconsistencies = torch.stack(inconsistencies, 1).cpu().numpy()
        return inconsistencies

    sis = {n: gen_si(p) for n, p in pipelines.items()}
    #sis = {n: torch.zeros((SIGMAS, 101)) for n in pipelines }
    return eval_points, gen_si, sis


@app.cell
def _(all_pipelines, gen_si, sis):
    all_sis = {n: gen_si(p) for n, p in all_pipelines.items() if n not in sis}
    all_sis.update(sis)
    return (all_sis,)


@app.cell
def _(all_pipelines, plt):
    def _():
        fig, axs = plt.subplots(ncols=1, nrows=2, figsize=(3, 5.5))
        fig.subplots_adjust(hspace=0.5)
        cond = all_pipelines["two_modes"].train_data.head(1024).cond.cpu().numpy().squeeze()
        value = all_pipelines["two_modes"].train_data.head(1024).sample.cpu().numpy().squeeze()
        axs[0].scatter(cond, value, alpha=0.05, color="Brown", rasterized=True, s=3)

        cond = all_pipelines["continuous_modes"].train_data.head(1024).cond.cpu().numpy().squeeze()
        value = all_pipelines["continuous_modes"].train_data.head(1024).sample.cpu().numpy().squeeze()
        axs[1].scatter(cond, value,alpha=0.1, color="Brown", rasterized=True, s=3)

        axs[0].set_xlim(-0.25, 1.25)
        axs[1].set_xlim(-0.25, 1.25)
        axs[0].set_xticks([0.0, 0.5, 1.0])
        axs[1].set_xticks([0.0, 0.5, 1.0])
        axs[0].set_yticks([-1, 0., 1.0])
        axs[1].set_yticks([-1, 0., 1.0])
        axs[0].set_title("Discrete-Support Dataset")
        axs[1].set_title("Continuous-Support Dataset")
        axs[0].set_xlabel("Conditioning Value", labelpad=4)
        axs[1].set_xlabel("Conditioning Value", labelpad=4)
        axs[0].set_ylabel("Generated Value", labelpad=0)
        axs[1].set_ylabel("Generated Value", labelpad=0)
        for ax in axs:
            ax.tick_params(axis='both', pad=0)
        return fig
    _fig = _()
    _fig
    return


@app.cell
def _(all_pipelines, all_sis, eval_points, np, plt, torch):
    _cond = torch.tensor(eval_points)[:,None].repeat(1, 10_000).reshape(-1, 1).to("cuda")
    def _plot_gen(ax, pipeline, cmap="Blues"):
        values = pipeline.generate(cond=_cond).sample # .clip(-2, 2)
        x_bins = eval_points
        y_bins = np.linspace(-1.2, 1.2, 100)
        v = torch.concatenate((_cond,values), -1).cpu().numpy()
        ax.hist2d(v[:,0], v[:,1], bins=[x_bins, y_bins], cmap=cmap, rasterized=True)   

    _fig, _axs = plt.subplots(ncols=6,nrows=2,figsize=(16,4.5), sharex="col",
                             gridspec_kw={'width_ratios': [0.4, -0.13, 0.4, 0.4, -0.13, 0.4]})
    _fig.subplots_adjust(wspace=0.52)
    _axs[0,1].set_axis_off()
    _axs[1,1].set_axis_off()
    _axs[0,4].set_axis_off()
    _axs[1,4].set_axis_off()

    _plot_gen(_axs[0,0], all_pipelines["two_modes"])
    _plot_gen(_axs[1,0], all_pipelines["continuous_modes"])
    _plot_gen(_axs[0,3], all_pipelines["linear"],
             cmap="Blues")
    _plot_gen(_axs[1,3], all_pipelines["fourier"],
             cmap="Blues")

    _axs[0,0].set_xlim([0, 1])
    _axs[1,0].set_xlim([0, 1])
    _axs[0,3].set_xlim([0, 1])
    _axs[1,3].set_xlim([0, 1])


    _c1 = _axs[0,2].imshow(all_sis["two_modes"], extent=[0, 1, 0, 1], cmap="Purples", aspect='auto',
                          rasterized=True, vmin=0., vmax=0.5)
    _c2 = _axs[1,2].imshow(all_sis["continuous_modes"], extent=[0, 1, 0, 1], cmap="Purples", aspect='auto',
                          rasterized=True, vmin=0., vmax=0.5)
    _c1 = _axs[0,5].imshow(all_sis["linear"], extent=[0, 1, 0, 1], cmap="Purples", aspect='auto',
                        rasterized=True, vmin=0., vmax=0.5)
    _c2 = _axs[1,5].imshow(all_sis["fourier"], extent=[0, 1, 0, 1], cmap="Purples", aspect='auto',
                          rasterized=True, vmin=0., vmax=0.5)
    _axs[0,2].grid(False)
    _axs[1,2].grid(False)
    _axs[0,5].grid(False)
    _axs[1,5].grid(False)

    _fig.colorbar(_c2, ax=_axs.ravel().tolist(), label="Schedule Deviation", fraction=0.08, pad=0.01,
                  shrink=0.75, ticks=[0,0.1,0.2,0.3,0.4,0.5])

    _lp = 1
    _axs[1,0].set_xlabel("Conditioning Value", labelpad=_lp)
    _axs[1,2].set_xlabel("Conditioning Value", labelpad=_lp)
    _axs[1,3].set_xlabel("Conditioning Value", labelpad=_lp)
    _axs[1,5].set_xlabel("Conditioning Value", labelpad=_lp)

    _axs[0,0].set_ylabel("Generated Value", labelpad=_lp)
    _axs[1,0].set_ylabel("Generated Value", labelpad=_lp)
    _axs[0,3].set_ylabel("Generated Value", labelpad=_lp)
    _axs[1,3].set_ylabel("Generated Value", labelpad=_lp)

    _axs[0,2].set_ylabel("Timestep", labelpad=_lp)
    _axs[1,2].set_ylabel("Timestep", labelpad=_lp)
    _axs[0,5].set_ylabel("Timestep", labelpad=_lp)
    _axs[1,5].set_ylabel("Timestep", labelpad=_lp)

    # _axs[0,2].set_ylabel("Total Schedule Deviation")
    # _axs[1,2].set_ylabel("Total Schedule Deviation")

    _axs[0,0].set_title("Neural Network Diffusion")
    _axs[0,2].set_title("Neural Network SD")
    _axs[0,3].set_title("Closed-Form Diffusion")
    _axs[0,5].set_title("Closed-Form SD")

    for _a in _axs:
        for _ax in _a:
            _ax.tick_params(axis='both', pad=0)
    _fig
    return


@app.cell
def _():
    return


@app.cell
def _():
    import marimo as mo
    return


if __name__ == "__main__":
    app.run()
