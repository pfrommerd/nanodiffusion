

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from nanoconfig.experiment.wandb import WandbExperiment
    return (WandbExperiment,)


@app.cell
def _(WandbExperiment):
    experiment = WandbExperiment(
        project_name="nanodiffusion"
    )
    return (experiment,)


@app.cell
def _(experiment):
    artifact = experiment.find_artifact("diffuser:latest", type="model")
    artifact = experiment.use_artifact(artifact)
    return (artifact,)


@app.cell
def _(artifact):
    from nanodiffusion.diffuser import Diffuser

    with artifact.open_file("model.safetensors") as f:
        diffuser = Diffuser.load(f)
    return (diffuser,)


@app.cell
def _(diffuser):
    samples = diffuser.sample(8, seed=1)
    diffuser.test_data.visualize_batch(samples)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
