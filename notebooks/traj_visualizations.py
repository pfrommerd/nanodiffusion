

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import json
    import wandb
    import scipy
    import pandas as pd
    import itertools
    import os
    import seaborn as sns

    sns.set_theme()
    plt.rcParams["font.family"] = "serif"

    from pathlib import Path
    return Path, np, pd, plt, scipy, wandb


@app.cell
def _(wandb):
    api = wandb.Api()
    return (api,)


@app.cell
def _(Path, api, pd):
    sweep = api.sweep("dpfrommer-projects/nanogen_trajectory/8iyi4w9r")
    artifacts = list(list(a for a in run.logged_artifacts() if a.type == "results")[0]
                   for run in sweep.runs if list(a for a in run.logged_artifacts() if a.type == "results"))
    # run = wandb.init()
    # artifacts = [run.use_artifact("dpfrommer-projects/nanogen-.venv_bin/results:v3")]
    def load_data(artifacts):
        data = []
        for artifact in artifacts:
            run = artifact.logged_by()
            input_artifact = list(a for a in run.used_artifacts() if a.type == "model")[0]
            samples = input_artifact.logged_by().config["pipeline"]["limit_data"]
            path = Path(artifact.download()) / "metrics.csv"
            df = pd.read_csv(path)
            df["samples"] = samples
            df["ddpm_si"] =  df["ddpm_si/0"] + df["ddpm_si/1"] + df["ddpm_si/2"]
            df["ddim_si"] =  df["ddim_si/0"] + df["ddim_si/1"] + df["ddim_si/2"]
            df["accel_si"] =  df["accel_si/0"] + df["accel_si/1"] + df["accel_si/2"]
            data.append(df)
        return pd.concat(data)
    data = load_data(artifacts)
    return (data,)


@app.cell
def _(np):
    from nanoconfig.data.source import DataRepository
    import nanoconfig.data.utils as data_utils

    traj_data = DataRepository.default().lookup("trajectory")
    traj_data = traj_data.split("test")
    traj_data = np.concatenate(list(data_utils.as_numpy(batch["start"])
            for batch in traj_data.to_batches(columns=["start","end"])))
    return (traj_data,)


@app.cell
def _(np, scipy, traj_data):
    def calc_density(sample_points):
        dists = -np.sum(np.square(sample_points[:,None, :] - traj_data[None,:, :]), axis=-1)
        dists = 100*dists
        log_pdfs = scipy.special.logsumexp(dists, axis=1)
        log_pdfs -= np.log(10)
        # log_pdfs = log_pdfs - scipy.special.logsumexp(log_pdfs)
        return np.exp(log_pdfs)

    def smooth(sample_points, df):
        dists = -np.sum(np.square(sample_points[:,None, :] - sample_points[None,:, :]), axis=-1)
        dists = 100*dists
        log_pdfs = dists - scipy.special.logsumexp(dists, axis=1)[:,None]
        del dists
        df = df.copy(deep=False)
        for n in df.columns:
            df[n] = np.exp(scipy.special.logsumexp(log_pdfs, b=df[n], axis=1))
        return df
    return (calc_density,)


@app.cell
def _(calc_density, data, np):
    def transform_densities(data):
        print("Calculating density...")
        # new_data = []
        # for _, group in data.groupby("samples"):
        #     cond = np.stack((group["condition/0"], group["condition/1"]), axis=-1)
        #     new_group = smooth(cond, group[["ddpm_si", "ddpm_ddim_dist",  "ddpm_accel_dist", "ddim_accel_dist"]])
        #     new_group["condition/0"] = group["condition/0"]
        #     new_group["condition/1"] = group["condition/1"]
        #     new_group["samples"] = group["samples"]
        #     new_data.append(new_group)
        # new_data = pd.concat(new_data)
        new_data = data
        cond = np.stack((new_data["condition/0"], new_data["condition/1"]), axis=-1)
        new_data["density"] = calc_density(cond)
        return new_data
    transformed_data = transform_densities(data)
    return (transformed_data,)


@app.cell
def _(calc_density, np, plt):
    def _():
        fig, ax = plt.subplots()
        sample_points = np.random.uniform(size=(5000, 2,), low=-1, high=1)
        sample_points_density = calc_density(sample_points)
        s = ax.scatter(sample_points[:,0], sample_points[:,1], c=sample_points_density)
        fig.colorbar(s)
        return fig
    _()
    return


@app.cell
def _(plt, transformed_data):
    def _():
        data = transformed_data[transformed_data["samples"] == 1000]
        fig, ax = plt.subplots()
        ax.scatter(data["condition/0"], data["condition/1"], c=data["ddpm_accel_dist"])
        return fig
    _()
    return


@app.cell
def _(plt, transformed_data):
    def _():
        data = transformed_data[transformed_data["samples"] == 1000]
        fig, ax = plt.subplots()
        ax.scatter(data["condition/0"], data["condition/1"], c=data["ddpm_si"])
        return fig
    _()
    return


@app.cell
def _(plt, transformed_data):
    for s, g in transformed_data.groupby("samples"):
        plt.scatter(g["ddpm_si"], g["ddpm_ddim_dist"], s=2, label=f"N={s}")
    plt.legend(loc="upper right")
    plt.show()
    return


@app.cell
def _(data, plt):
    plt.scatter(data["condition/0"], data["condition/1"], c=data["ddpm_si"], s=4)
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(data, plt):
    plt.scatter(data["condition/0"], data["condition/1"], c=data["ddpm_ddim_dist"], s=4)
    plt.colorbar()
    plt.show()
    return


@app.cell
def _(np, plt, scipy, transformed_data):
    def heatmaps(column_name, column_label):
        from matplotlib.colors import LogNorm
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 3))
        cbars = []

        grid_y, grid_x = np.mgrid[-1:1:100j, -1:1:100j]
        xs, ys = grid_x[0,:], grid_y[:,0]
        for i, ((samples, sub_data), ax) in enumerate(zip(transformed_data.groupby("samples"), axs)):
            # evaluate on a grid
            cond = np.stack((sub_data["condition/0"], sub_data["condition/1"]), axis=-1)
            values = scipy.interpolate.griddata(cond, sub_data[column_name].to_numpy(),
                                                (grid_x, grid_y), method='nearest')[::-1,:]
            m = ax.imshow(values,cmap="binary", extent=[-1, 1, -1, 1])
            ax.grid(False)
            cbars.append(m)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_xlabel("Start Y Location")
            ax.set_title(f"Trajectories, N={samples}")
            if i == 0:
                ax.set_ylabel("Start Y Location")
        fig.colorbar(cbars[0], ax=axs, label=column_label)
        return fig
    _fig = heatmaps("ddpm_ddim_dist", "DDPM/DDIM OT Distance")
    _fig.savefig("figures/ddpm_ddim_dist.pdf", bbox_inches="tight")
    _fig
    return (heatmaps,)


@app.cell
def _(heatmaps):
    _fig = heatmaps("ddim_accel_dist", "DDIM/GE OT Distance")
    _fig.savefig("figures/ddim_idop_dist.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(heatmaps):
    _fig = heatmaps("ddpm_accel_dist", "DDPM/GE OT Distance")
    _fig.savefig("figures/ddpm_idop_dist.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(plt, transformed_data):
    plt.hist(transformed_data.density, bins=50)
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
