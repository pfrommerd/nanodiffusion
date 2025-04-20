import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import nanodiffusion
    from nanodiffusion import TrainConfig, DiffuserConfig, ExperimentConfig
    import logging
    return (
        DiffuserConfig,
        ExperimentConfig,
        TrainConfig,
        logging,
        nanodiffusion,
    )


@app.cell
def _():
    from nanodiffusion.optimizers import AdamwConfig
    from nanodiffusion.schedules import LogLinearScheduleConfig
    from nanodiffusion.models.mlp import MlpConfig
    from nanodiffusion.datasets.tree import TreeDataConfig
    return AdamwConfig, LogLinearScheduleConfig, MlpConfig, TreeDataConfig


@app.cell
def _(logging):
    logger = logging.getLogger("nanodiffusion")
    return (logger,)


@app.cell
def _(
    AdamwConfig,
    DiffuserConfig,
    ExperimentConfig,
    LogLinearScheduleConfig,
    MlpConfig,
    TrainConfig,
    TreeDataConfig,
    logger,
    nanodiffusion,
):
    config = TrainConfig(
        experiment=ExperimentConfig(clearml=True),
        diffuser=DiffuserConfig(
            optimizer=AdamwConfig(lr=4e-3),
            schedule=LogLinearScheduleConfig(
                sigma_min=1e-3, sigma_max=10, timesteps=512
            ),
            model=MlpConfig(),
            data=TreeDataConfig(),
            iterations=5_000,
            batch_size=1024
        )
    )
    nanodiffusion.setup_logging()
    model = config.run(logger)
    return config, model


@app.cell
def _(model):
    samples = model.sample(1024, seed=1)
    model.test_data.visualize_batch(samples)
    return (samples,)


if __name__ == "__main__":
    app.run()
