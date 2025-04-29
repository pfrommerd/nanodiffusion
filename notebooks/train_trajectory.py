import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    import nanogen
    from nanogen import TrainConfig, DiffuserConfig, ExperimentConfig
    from torch.utils.data import DataLoader
    import logging
    import marimo as mo
    return (
        DataLoader,
        DiffuserConfig,
        ExperimentConfig,
        TrainConfig,
        logging,
        mo,
        nanogen,
    )


@app.cell
def _():
    from nanogen.optimizers import AdamwConfig
    from nanogen.schedules import LogLinearScheduleConfig
    from nanogen.models.mlp import MlpConfig
    from nanogen.datasets.trajectory import TrajectoryDataConfig
    return (
        AdamwConfig,
        LogLinearScheduleConfig,
        MlpConfig,
        TrajectoryDataConfig,
    )


@app.cell
def _(logging):
    logger = logging.getLogger("nanogen")
    return (logger,)


@app.cell
def _(
    AdamwConfig,
    DataLoader,
    DiffuserConfig,
    ExperimentConfig,
    LogLinearScheduleConfig,
    MlpConfig,
    TrainConfig,
    TrajectoryDataConfig,
):
    config = TrainConfig(
        experiment=ExperimentConfig(clearml=True),
        diffuser=DiffuserConfig(
            optimizer=AdamwConfig(lr=4e-3),
            schedule=LogLinearScheduleConfig(
                sigma_min=1e-3, sigma_max=10, timesteps=512
            ),
            model=MlpConfig(),
            data=TrajectoryDataConfig(),
            iterations=200_000,
            batch_size=1024
        )
    )
    _, test_data = config.diffuser.data.create()
    test_samples = next(iter(DataLoader(test_data, batch_size=5)))
    test_data.visualize_batch(test_samples)
    return config, test_data, test_samples


@app.cell
def _(config, logger, nanogen):
    nanogen.setup_logging()
    model = config.run(logger)
    samples = model.sample(16, seed=1)
    model.test_data.visualize_batch(samples)
    return model, samples


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
