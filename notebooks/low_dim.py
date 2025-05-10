

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

    import nanogen.utils as utils
    import logging

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
            value = value + nr.normal(loc=0, scale=0.1,size=(N,))
            # cond = cond + 0.001*nr.uniform(-1,1,size=(N,))
        elif name == "continuous_modes":
            cond = nr.choice([0, 1], size=(N,))
            value = nr.choice([1, 2], size=(N,))*cond - 1
            value = value + nr.normal(loc=0, scale=0.1,size=(N,))        
            cond = cond + 0.1*nr.normal(size=(N,))
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
                timesteps=512, sigma_min=1e-3, sigma_max=20
            ),
            sampler_preset="ddpm",
            sample_timesteps=32,
            ideal_denoiser=False
        ),
        optimizer=AdamwConfig(
            lr=4e-3, weight_decay=1e-2,
            betas=(0.9, 0.999), eps=1e-8
        ),
        gen_batch_size=512,
        batch_size=64,
        test_batch_size=512,
        iterations=2_000
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


app._unparsable_cell(
    r"""
    _fig, _axs = plt.subplots(ncols=len(pipelines),nrows=1)
    for _ax, _v in zip(_axs, pipelines.values()):

    _fig
    """,
    name="_"
)


@app.cell
def _(GenerativePipeline, InMemoryData, config, create_ds, pipelines, torch):
    from nanogen.models.diffusion import IdealDiffuser
    from torch import nn

    class LinearDiffuser(nn.Module):
        def __init__(self):
            super().__init__()
            self.ideal_left = IdealDiffuser(torch.tensor([-1], dtype=torch.float32).unsqueeze(1))
            self.ideal_right = IdealDiffuser(torch.tensor([0., 1.0], dtype=torch.float32).unsqueeze(1))

        def forward(self, x, t, cond):
            return self.ideal_left(x, t + 0.1)*(1-cond) + self.ideal_right(x,t + 0.1)*cond

    class SinusoidalDiffuser(nn.Module):
        def __init__(self):
            super().__init__()
            self.ideal_left = IdealDiffuser(torch.tensor([-1], dtype=torch.float32).unsqueeze(1))
            self.ideal_right = IdealDiffuser(torch.tensor([0., 1.0], dtype=torch.float32).unsqueeze(1))

        def forward(self, x, t, cond):
            l = torch.exp(-cond**2/(2*0.05)) + 0.1
            r = torch.exp(-(cond-1)**2/(2*0.05)) + 0.1
            total = (l + r)
            l /= total
            r /= total
            return self.ideal_left(x, t + 0.1)*l + self.ideal_right(x,t + 0.1)*r

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
def _(all_pipelines, np, plt, torch):

    def _plot_gen(ax, pipeline):
        cond = torch.tensor(np.random.uniform(size=(1_000_000,1),low=0,high=1).astype(np.float32)).to("cuda")
        values = pipeline.generate(cond=cond).sample # .clip(-2, 2)
        v = torch.concatenate((cond,values), -1).cpu().numpy()
        ax.hist2d(v[:,0], v[:,1], bins=[128, 128], cmap="Blues")   
    
    _fig, _axs = plt.subplots(ncols=3,nrows=2,figsize=(15,10))

    _cond = all_pipelines["two_modes"].train_data.head(1024).cond.cpu().numpy().squeeze()
    _value = all_pipelines["two_modes"].train_data.head(1024).sample.cpu().numpy().squeeze()
    _axs[0,0].scatter(_cond, _value, alpha=0.3)

    _cond = all_pipelines["continuous_modes"].train_data.head(1024).cond.cpu().numpy().squeeze()
    _value = all_pipelines["continuous_modes"].train_data.head(1024).sample.cpu().numpy().squeeze()
    _axs[1,0].scatter(_cond, _value,alpha=0.3)

    _plot_gen(_axs[0,1], all_pipelines["two_modes"])
    _plot_gen(_axs[1,1], all_pipelines["continuous_modes"])
    _plot_gen(_axs[0,2], all_pipelines["linear"])
    _plot_gen(_axs[1,2], all_pipelines["fourier"])
    _fig
    return


@app.cell
def _(GenerativePipeline, all_pipelines, torch):
    from nanogen.analysis.sampler_metrics import measure_si_traj
    from accelerate import Accelerator

    accelerator = Accelerator()

    def gen_si(pipeline: GenerativePipeline):
        model = accelerator.prepare(pipeline.model)
        for c in range(0, 1, 128):
            conds = torch.tensor(c, device=accelerator.device)[None,None].repeat(128, 1)
            s = pipeline.generate(cond=conds, accelerator=accelerator).sample
            measure_si_traj(model, s, model.gen_sigmas)

    sis = {n: gen_si(p) for n, p in all_pipelines.items()}
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
