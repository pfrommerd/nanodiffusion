

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

    import numpy as np
    import torch
    import pyarrow.dataset as ds
    import pyarrow as pa
    import numpy.random as nr
    import hashlib

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
        torch,
    )


@app.cell
def _(Data, SplitInfo, ds, hashlib, nr, pa):
    def create_ds():
        cond = nr.choice([-1, 1], size=(2048,))
        value = nr.choice([0.3, 0.8], size=(2048,))*(cond + 1)/2 + 0.2
        value = value + nr.normal(loc=0, scale=0.03,size=(2048,))
        return ds.dataset(pa.table([cond,value],
            schema=pa.schema([
                pa.field("x", pa.float32()),
                pa.field("y", pa.float32())
            ], metadata={"mime_type": "data/cond_point"})))


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

    data = InMemoryData({"test": create_ds(), "train": create_ds()})
    return (data,)


@app.cell
def _(
    AdamwConfig,
    DiffusionModelConfig,
    ExperimentConfig,
    GenerativePipeline,
    LogLinearScheduleConfig,
    PipelineConfig,
    data,
    diffusion_mlp,
    logger,
):
    config = PipelineConfig(
        model=DiffusionModelConfig(
            nn=diffusion_mlp.MlpConfig(
                hidden_features=(128, 128, 128, 128),
                embed_features=64
            ),
            schedule=LogLinearScheduleConfig(
                timesteps=512, sigma_min=3e-2, sigma_max=5
            ),
            sample_timesteps=32,
            ideal_denoiser=False
        ),
        optimizer=AdamwConfig(
            lr=5e-4, weight_decay=1e-4,
            betas=(0.9, 0.999), eps=1e-8
        ),
        gen_batch_size=512,
        batch_size=512,
        test_batch_size=512,
        iterations=3_000
    )
    pipeline = GenerativePipeline.from_config(config, data=data)
    experiment = ExperimentConfig(
        wandb=True,
    ).create(logger=logger)

    pipeline.train(progress=True, experiment=experiment)
    return (pipeline,)


@app.cell
def _(np, pipeline, torch):
    # Generate intermediate samples
    def gen_samples():
        cond = torch.tensor(np.random.uniform(size=(1024,1),low=-1,high=1).astype(np.float32)).to("cuda")
        values = pipeline.generate(cond=cond).sample
        return torch.concatenate((cond,values), -1)
    samples = gen_samples().cpu().numpy()
    return (samples,)


@app.cell
def _(samples):
    import matplotlib.pyplot as plt
    plt.scatter(samples[:,0],samples[:,1])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
