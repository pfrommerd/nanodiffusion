

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

    config = PipelineConfig(
        model=DiffusionModelConfig(
            nn=diffusion_mlp.MlpConfig(
                hidden_features=(256, 256, 256, 256),
                embed_features=128
            ),
            schedule=LogLinearScheduleConfig(
                timesteps=512, sigma_min=5e-4, sigma_max=5
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
        iterations=50_000
    )

    pipeline = GenerativePipeline.from_config(config)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
