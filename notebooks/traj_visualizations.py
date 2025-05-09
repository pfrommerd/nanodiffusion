

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
    sweep = api.sweep("dpfrommer-projects/nanogen_trajectory/cdlg8gkw")
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
            data.append(df)
        return pd.concat(data)
    data = load_data(artifacts)
    data
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
    return (calc_density,)


@app.cell
def _(calc_density, data, np):
    def transform_densities(data):
        new_data = data.copy(deep=False)
        print("Calculating density...")
        cond = np.stack((data["condition/0"], data["condition/1"]), axis=-1)
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
def _(calc_density, np, pd, plt, transformed_data):
    def _():
        fig, (ax, ax_density) = plt.subplots(nrows=1, ncols=2, figsize=(15, 3.5))
        for samples, samples_data in transformed_data.groupby("samples"):
            if samples > 50_000:
                continue
            samples_data = samples_data.copy(deep=False)
            samples_data = samples_data[samples_data["density"] > 0.1]
            density = samples_data["density"].to_numpy()
            labels, bins = pd.cut(density, 12, retbins=True)
            bins = (bins[:-1] + bins[1:])/2
            samples_data["bins"] = labels
            data = samples_data.groupby("bins", observed=True).median().reset_index()
            data_upper = samples_data.groupby("bins", observed=True).quantile(0.75).reset_index()
            data_lower = samples_data.groupby("bins", observed=True).quantile(0.25).reset_index()
            bin_values = data.bins.apply(lambda x: (x.left + x.right)/2).to_numpy()
            ax.plot(bin_values, data.ddpm_ddim_dist, label=f"N={samples}")
            ax.fill_between(bin_values, data_lower.ddpm_ddim_dist, data_upper.ddpm_ddim_dist, alpha=0.2)
        ax.legend(loc="upper right")
        #ax.set_xlim([-0.2, 1])
        ax.set_xlabel("Conditional Log Density")
        ax.set_ylabel("DDPM/DDIM Transport Distance")

        grid_y, grid_x = np.mgrid[-100:100:100j, -100:100:100j]
        xs, ys = grid_x[0,:], grid_y[:,0]
        density = calc_density(np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), -1))
        density = density.reshape(grid_x.shape)/bin_values.max()

        m = ax_density.imshow(density[::-1,:],cmap="Blues", extent=[-100, 100, -100, 100])
        ax_density.grid(False)
        ax_density.set_ylabel("t-SNE Second Component")
        ax_density.set_xlabel("t-SNE First Component")
        fig.colorbar(m, ax=(ax, ax_density), label="Conditional Density")
        return fig
    _fig = _()
    _fig.savefig("figures/mnist_density_transport.pdf")
    _fig
    return


@app.cell
def _(np, pd, plt, scipy, transformed_data):
    def _():
        fig, ((ax_lines, ax_hm), (ax_scat, ax_scat_comp)) = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))
        cols = [f"ddpm_si/{i}" for i in range(5)]
        main_col = cols[1]
        for samples, samples_data in transformed_data.groupby("samples"):
            samples_data = samples_data.copy(deep=False)

            samples_data = samples_data[samples_data["density"] > 0.1]
            density = np.log(samples_data["density"].to_numpy())
            labels, bins = pd.cut(density, 10, retbins=True)
            bins = (bins[:-1] + bins[1:])/2
            samples_data["bins"] = labels
            samples_data["si"] = samples_data[main_col] #samples_data[cols].mean(axis=1)
            data = samples_data.groupby("bins", observed=True).median().reset_index()
            data_upper = samples_data.groupby("bins", observed=True).quantile(0.75).reset_index()
            data_lower = samples_data.groupby("bins", observed=True).quantile(0.25).reset_index()
            bin_values = data.bins.apply(lambda x: (x.left + x.right)/2).to_numpy()

            ax_lines.plot(bin_values, data.si, label=f"N={samples}")
            ax_lines.fill_between(bin_values, data_lower.si, data_upper.si, alpha=0.2)
        ax_lines.legend(loc="upper right")
        #ax1.set_xlim([-1, 2.3])
        ax_lines.set_xlabel("Conditional Log Density")
        ax_lines.set_ylabel("DDPM Schedule Inconsistency")
        sub_data = transformed_data[transformed_data["samples"] == 8000]
        ax_scat.scatter(sub_data["density"], sub_data["ddpm_si/1"], alpha=0.2, s=1)

        ax_scat_comp.scatter(sub_data["ddpm_si/1"], sub_data["ddpm_ddim_dist"], alpha=0.2, s=1)
        ax_scat_comp.set_xlabel("Schedule Inconsistency")
        ax_scat_comp.set_ylabel("DDPM/DDIM")

        grid_y, grid_x = np.mgrid[-100:100:100j, -100:100:100j]
        xs, ys = grid_x[0,:], grid_y[:,0]
        cond = np.stack((sub_data["condition/0"], sub_data["condition/1"]), axis=-1)
        values = scipy.interpolate.griddata(cond, sub_data["ddpm_si/1"].to_numpy(),
                                            (grid_x, grid_y), method='cubic')[::-1,:]
        m = ax_hm.imshow(values[:,:],cmap="Blues", extent=[-100, 100, -100, 100])
        ax_hm.grid(False)
        ax_hm.set_ylabel("t-SNE Second Component")
        ax_hm.set_xlabel("t-SNE First Component")
        fig.colorbar(m, ax=(ax_hm), use_gridspec=True, label="Schedule Inconsistency")
        return fig
    _fig = _()
    _fig.savefig("figures/mnist_density_transport.pdf")
    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
