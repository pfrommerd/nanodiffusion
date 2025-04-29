

import marimo

__generated_with = "0.13.0"
app = marimo.App(widtp@h="medium")


@app.cell
def _():
    from nanoconfig.experiment.wandb import WandbExperiment
    from nanogen.diffuser import Diffuser

    import nanogen.models.unet1d
    import nanogen.datasets.trajectory
    return Diffuser, WandbExperiment


@app.cell
def _(WandbExperiment):
    experiment = WandbExperiment(
        project_name="nanogen"
    )
    return (experiment,)


@app.cell
def _(Diffuser, experiment):
    artifact_id = experiment.find_artifact("distilled", version="v3", type="model")
    artifact = experiment.use_artifact(artifact_id)
    with artifact.open_file("model.safetensors") as f:
        distilled = Diffuser.load(f)
    artifact_id = experiment.find_artifact("diffuser", version="v5", type="model")
    artifact = experiment.use_artifact(artifact_id)
    with artifact.open_file("model.safetensors") as f:
        expert = Diffuser.load(f)
    return (expert,)


@app.cell
def _(expert):
    import torch
    def visualize(cond):
        cond = torch.tensor([cond]).repeat(16, 1, 1)
        samples = expert.sample(16, cond=cond, seed=1)
        return expert.test_data.visualize_batch(samples)
    visualize([[-0.9, 0.8], [0.9, -0.9]])
    return (visualize,)

@app.cell
def _(visualize):
    visualize([[-0.2, -0.15], [0.9, -0.9]])
    return


@app.cell
def _(visualize):
    visualize([[-0.2, -0.3], [0.9, -0.9]])
    return


@app.cell
def _(visualize):
    visualize([[-0.2, 0.05], [0.9, -0.9]])
    return


if __name__ == "__main__":
    app.run()
