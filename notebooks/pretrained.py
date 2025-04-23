

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from nanoconfig.experiment.wandb import WandbExperiment
    import nanodiffusion.models.unet1d
    import nanodiffusion.datasets.trajectory
    return (WandbExperiment,)


@app.cell
def _(WandbExperiment):
    experiment = WandbExperiment(
        project_name="nanodiffusion"
    )
    return (experiment,)


@app.cell
def _(experiment):
    artifact_id = experiment.find_artifact("distilled", version="v0", type="model")
    artifact = experiment.use_artifact(artifact_id)
    return (artifact,)


@app.cell
def _(artifact):
    from nanodiffusion.diffuser import Diffuser

    with artifact.open_file("model.safetensors") as f:
        diffuser = Diffuser.load(f)
    return (diffuser,)


@app.cell
def _(diffuser):
    import torch
    def visualize(cond):
        cond = torch.tensor([cond]).repeat(16, 1, 1)
        samples = diffuser.sample(16, cond=cond, seed=1)
        return diffuser.test_data.visualize_batch(samples)
    visualize([[-0.5, 0.5], [0.9, -0.9]])
    return (visualize,)


@app.cell
def _(visualize):
    visualize([[-0.2, -0.15], [0.9, -0.9]])
    return


@app.cell
def _(visualize):
    visualize([[-0.2, -0.1], [0.9, -0.9]])
    return


@app.cell
def _(visualize):
    visualize([[-0.2, -0.05], [0.9, -0.9]])
    return


if __name__ == "__main__":
    app.run()
