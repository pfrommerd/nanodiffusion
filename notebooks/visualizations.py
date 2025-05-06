

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
    sweep = api.sweep("dpfrommer-projects/nanogen_mnist/8bgc69t0")
    return (sweep,)


@app.cell
def _(Path, np, parse_row, pd, sweep):
    def load_data():
        data = []
        for samples, run in zip([10000, 20000, 30000, 40000, 50000], sweep.runs):
            artifact = list(a for a in run.logged_artifacts() if a.type == "results")[0]
            path = Path(artifact.download()) / "metrics.csv"
            with open(path) as f:
                print(f.read())
            df = pd.read_csv(path)
            cond = df["condition"]
            cond = np.stack(list(parse_row(c) for c in cond))
            df["condition_x"] = cond[:,0]
            df["condition_y"] = cond[:,1]
            df["samples"] = samples
            del df["condition"]
            data.append(df)
        return pd.concat(data)
    data = load_data()
    data
    return (data,)


@app.cell
def _(np):
    from nanoconfig.data.source import DataRepository
    import nanoconfig.data.utils as data_utils

    mnist_data = DataRepository.default().lookup("mnist")
    mnist_data = mnist_data.split("test")
    mnist_labels = np.concatenate(list(data_utils.as_numpy(batch["class"]) for batch in mnist_data.to_batches(columns=["class"])))
    mnist_data = np.concatenate(list(data_utils.as_numpy(batch["tsne"]) for batch in mnist_data.to_batches(columns=["tsne"])))
    return mnist_data, mnist_labels


@app.cell
def _(mnist_data, np, scipy):
    def calc_density(sample_points):
        dists = -np.sum(np.square(sample_points[:,None, :] - mnist_data[None,:, :]), axis=-1)
        dists = dists/20
        log_pdfs = scipy.special.logsumexp(dists, axis=1)
        log_pdfs -= np.log(10)
        # log_pdfs = log_pdfs - scipy.special.logsumexp(log_pdfs)
        return np.exp(log_pdfs)

    def calc_smoothed(cond, values):
        dists = -np.sum(np.square(cond[:,None, :] - cond[None,:, :]), axis=-1)
        dists = dists/20
        return scipy.special.logsumexp(dists, axis=1, b=values)
    return (calc_density,)


@app.cell
def _(calc_density, data, np):
    def transform_densities(data):
        new_data = data.copy(deep=False)
        print("Calculating density...")
        cond = np.stack((data["condition_x"], data["condition_y"]), axis=-1)
        new_data["density"] = calc_density(cond)
        return new_data
    transformed_data = transform_densities(data)
    return (transformed_data,)


@app.cell
def _(calc_density, np, plt):
    def _():
        fig, ax = plt.subplots()
        sample_points = np.random.uniform(size=(5000, 2,), low=-110, high=110)
        sample_points_density = calc_density(sample_points)
        s = ax.scatter(sample_points[:,0], sample_points[:,1], c=sample_points_density)
        fig.colorbar(s)
        return fig
    _()
    return


@app.cell
def _(mnist_data, mnist_labels, np, plt, scipy, transformed_data):
    def heatmaps(column_name, column_label):
        from matplotlib.colors import LogNorm
        colors = [
            "#4c72b0",  # blue
            "#dd8452",  # orange
            "#55a868",  # green
            "#c44e52",  # red
            "#8172b2",  # purple
            "#937860",  # brown
            "#da8bc3",  # pink
            "#b07aa1",  # magenta
            "#ccb974",  # khaki
            "#64b5cd",  # cyan
        ]
        fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(15, 3))
        cbars = []

        grid_y, grid_x = np.mgrid[-100:100:100j, -100:100:100j]
        xs, ys = grid_x[0,:], grid_y[:,0]
        for i, ((samples, sub_data), ax) in enumerate(zip(transformed_data.groupby("samples"), axs)):
            # evaluate on a grid
            cond = np.stack((sub_data["condition_x"], sub_data["condition_y"]), axis=-1)
            values = scipy.interpolate.griddata(cond, sub_data[column_name].to_numpy(),
                                                (grid_x, grid_y), method='cubic')[::-1,:]
            m = ax.imshow(values,cmap="binary", extent=[-100, 100, -100, 100],
                         vmin=1, vmax=7)
            ax.scatter(mnist_data[::5,0], mnist_data[::5,1], c=[colors[l] for l in mnist_labels[::5]], s=1)
            #m = ax.scatter(sub_data["condition_x"], sub_data["condition_y"], c=sub_data["ddpm_ddim_dist"],
            #               cmap="binary", s=2)
            ax.grid(False)
            cbars.append(m)
            ax.set_xlim([-100, 100])
            ax.set_ylim([-100, 100])
            ax.set_xlabel("t-SNE First Component")
            ax.set_title(f"Conditional MNIST, N={samples}")
            if i == 0:
                ax.set_ylabel("t-SNE Second Component")
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
def _(pd, plt, transformed_data):
    def _():
        fig, ax = plt.subplots()
        for samples, samples_data in transformed_data.groupby("samples"):
            if samples > 50_000:
                continue
            samples_data = samples_data.copy(deep=False)
            samples_data = samples_data[samples_data["density"] > 0.5]
            density = samples_data["density"].to_numpy()
            min_density, max_density = density.min(), density.max()
            labels, bins = pd.cut(samples_data["density"], 10, retbins=True)
            bins = (bins[:-1] + bins[1:])/2
            samples_data["bins"] = labels
            data = samples_data.groupby("bins", observed=True).median().reset_index()
            data_upper = samples_data.groupby("bins", observed=True).quantile(0.90).reset_index()
            data_lower = samples_data.groupby("bins", observed=True).quantile(0.25).reset_index()
            data.bins = data.bins.apply(lambda x: (x.left + x.right)/2)
            ax.plot(data.bins, data.ddpm_ddim_dist, label=f"N={samples}")
            ax.fill_between(data.bins, data_lower.ddpm_ddim_dist, data_upper.ddpm_ddim_dist, alpha=0.2)
        ax.set_xlabel("Density")
        ax.set_ylabel("DDPM/DDIM Transport Distance")
        fig.legend(loc="upper left")
        return fig
    _()
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
