

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
    sweep = api.sweep("dpfrommer-projects/nanogen_fashion_mnist/a6py7u6u")
    return api, sweep


@app.cell
def _(Path, pd, sweep):
    def load_data():
        data = []
        for run in sweep.runs:
            artifact = list(a for a in run.logged_artifacts() if a.type == "results")
            if not artifact:
                continue
            input_artifact = list(a for a in run.used_artifacts() if a.type == "model")[0]
            samples = input_artifact.logged_by().config["pipeline"]["limit_data"]
            artifact = artifact[0]
            path = Path(artifact.download()) / "metrics.csv"
            df = pd.read_csv(path)
            df["samples"] = samples
            df["ddpm_si"] =  df["ddpm_si/0"] + df["ddpm_si/1"] + df["ddpm_si/2"]
            df["ddim_si"] =  df["ddim_si/0"] + df["ddim_si/1"] + df["ddim_si/2"]
            df["accel_si"] =  df["accel_si/0"] + df["accel_si/1"] + df["accel_si/2"]
            data.append(df)
        return pd.concat(data)
    data = load_data()
    return (data,)


@app.cell
def _(np):
    from nanoconfig.data.source import DataRepository
    import nanoconfig.data.utils as data_utils

    mnist_data = DataRepository.default().lookup("fashion-mnist")
    mnist_data = mnist_data.split("test")
    mnist_labels = np.concatenate(list(data_utils.as_numpy(batch["class"]) for batch in mnist_data.to_batches(columns=["class"])))
    mnist_data = np.concatenate(list(data_utils.as_numpy(batch["tsne"]) for batch in mnist_data.to_batches(columns=["tsne"])))
    return mnist_data, mnist_labels


@app.cell
def _(mnist_data, np, scipy):
    def calc_density(sample_points):
        dists = -np.sum(np.square(sample_points[:,None, :] - mnist_data[None,:, :]), axis=-1)
        dists = dists/100
        log_pdfs = scipy.special.logsumexp(dists, axis=1)
        log_pdfs -= np.log(10000) # approximately the right normalizatin constant for the square domain
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
def _(transformed_data):
    import pickle
    pickle.dump(transformed_data, open("figures/fashion_mnist_data.pkl", "wb"))
    return


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
    def heatmaps(fig, axs, column, label, vmin = None, vmax = None, normalize=False):
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
        cbars = []
        grid_y, grid_x = np.mgrid[-100:100:100j, -100:100:100j]
        xs, ys = grid_x[0,:], grid_y[:,0]
        groups = [(s, d) for s, d in transformed_data.groupby("samples") if s in [10_000, 30_000, 60_000]]
        for i, ((samples, sub_data), ax) in enumerate(zip(groups, axs)):
            # evaluate on a grid
            cond = np.stack((sub_data["condition/0"], sub_data["condition/1"]), axis=-1)
            values = scipy.interpolate.griddata(cond, sub_data[column].to_numpy(),
                                                (grid_x, grid_y), method='nearest')[::-1,:]
            if i == 0:
                values = values.clip(0, 150)

            if normalize:
                values = values / values.max()
                vmin = vmin or values.min()
                vmax = vmax or 1.
            cbar = ax.imshow(values,cmap="binary", extent=[-100, 100, -100, 100],
                                                vmin=vmin, vmax=vmax)
            ax.scatter(mnist_data[::5,0], mnist_data[::5,1],
                             c=[colors[l] for l in mnist_labels[::5]], s=1.5, alpha=0.8)
            ax.grid(False)
            cbars.append(cbar)
            ax.set_xlim([-100, 100])
            ax.set_ylim([-100, 100])
            if i == 0:
                ax.set_ylabel("t-SNE Second Component")
            #ax.set_xlabel("t-SNE First Component")
        fig.colorbar(cbars[0], ax=axs, label=label,fraction=0.03, pad=0.01, shrink=0.7)
        return fig
    _fig, _axs = plt.subplots(ncols=3, nrows=2, figsize=(15,9), sharex="col", sharey="row")
    _fig.subplots_adjust(hspace=0.06, wspace=0.06)
    #_fig.tight_layout()
    _fig = heatmaps(_fig, _axs[0], "ddpm_ddim_dist", "DDPM/DDIM OT Distance", 0.5, 7)
    _fig = heatmaps(_fig, _axs[1], "ddpm_si", "DDPM Schedule Deviation", 50, 310)
    _axs[-1][0].set_xlabel("t-SNE First Component")
    _axs[-1][1].set_xlabel("t-SNE First Component")
    _axs[-1][2].set_xlabel("t-SNE First Component")

    _axs[0][0].set_title("Conditional Fashion-MNIST N=10000")
    _axs[0][1].set_title("Conditional Fashion-MNIST N=30000")
    _axs[0][2].set_title("Conditional Fashion-MNIST N=60000")

    _fig.savefig("figures/fashion_mnist_ddpm_ddim_dist.pdf", bbox_inches="tight")
    _fig
    return (heatmaps,)


@app.cell
def _(heatmaps, plt):
    _fig, _axs = plt.subplots(ncols=3, nrows=6, figsize=(12,25), sharex="col", sharey="row")
    _fig.subplots_adjust(hspace=-0.2, wspace=0.05)
    _fig = heatmaps(_fig, _axs[0], "ddpm_ddim_dist", "DDPM/DDIM OT Distance", 0.5, 7)
    _fig = heatmaps(_fig, _axs[1], "ddpm_accel_dist", "DDPM/GE OT Distance", 0.5, 7)
    _fig = heatmaps(_fig, _axs[2], "ddim_accel_dist", "DDIM/GE OT Distance", 0.5, 7)
    _fig = heatmaps(_fig, _axs[3], "ddpm_si", "DDPM Schedule Deviation", 50, 300)
    _fig = heatmaps(_fig, _axs[4], "ddim_si", "DDIM Schedule Deviation", 40, 400)
    _fig = heatmaps(_fig, _axs[5], "accel_si", "GE Schedule Deviation", 40, 450)
    for _ax in _axs[-1]:
        _ax.set_xlabel("t-SNE First Component")
        _ax.set_xticks([-75, -25, 25, 75])
    for _ax in _axs:
        _ax[0].set_yticks([-75, -25, 25, 75])
    _fig.savefig("figures/fashion_mnist_heatmaps.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _():
    from matplotlib.legend_handler import HandlerPatch
    from matplotlib import patches as mpatches

    class HandlerSquare(HandlerPatch):
        def create_artists(self, legend, orig_handle,
                           xdescent, ydescent, width, height, fontsize, trans):
            center = xdescent + 0.5 * (width - height), ydescent
            p = mpatches.Rectangle(xy=center, width=height,
                    height=height, angle=0.0,
                    facecolor=orig_handle.get_facecolor()[0],
                    linewidth=0)
            #self.update_prop(p, orig_handle, legend)
            p.set_transform(trans)
            return [p] 
    return (HandlerSquare,)


@app.cell
def _(api, pd):
    def load_capacity_train_history():
        sweep = api.sweep("dpfrommer-projects/nanogen_mnist/pjz731bz")
        data = []
        for run in sweep.runs:
            loss_history = run.history(keys=["loss/test", "loss/train"], samples=4000)
            channels = run.config["pipeline"]["model"]["nn"]["base_channels"]
            loss_history["channels"] = channels
            data.append(loss_history)
        return pd.concat(data)

    capacity_history_data = load_capacity_train_history()
    return


@app.cell
def _(mnist_data, mnist_labels, np, pd, plt, transformed_data):
    def _():
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    
        td = transformed_data.copy()
        mean_data = td.groupby("samples").median().reset_index()
        high_data = td.groupby("samples").quantile(0.75).reset_index()
        low_data = td.groupby("samples").quantile(0.25).reset_index()
        axs[0].plot(mean_data["samples"],mean_data["ddpm_si"])
        axs[0].fill_between(mean_data["samples"], low_data["ddpm_si"], high_data["ddpm_si"],
                           alpha=0.3)

        # The violin plot
        sub_data = transformed_data[transformed_data["samples"] == 60_000]
        cond = np.stack((sub_data["condition/0"], sub_data["condition/1"]), axis=-1)
        mnist_conds = mnist_data[::10]
        idxs = np.argmin(np.linalg.norm(mnist_conds[:,None,:] - cond[None,:,:], axis=-1),axis=1)
        sis = sub_data["ddpm_si"].to_numpy()[idxs]
        ots = sub_data["ddpm_ddim_dist"].to_numpy()[idxs]

        df = pd.DataFrame({
            "label": mnist_labels[::10], "ddpm_si": sis,
            "ddpm_ddim_dist": ots
        })
        groups = [g["ddpm_si"].to_numpy() for l, g in df.groupby("label")]
        axs[1].violinplot(groups, np.arange(10))
        axs[1].set_xticks(np.arange(10), ["T-shirt", "Trouser","Pullover"," Dress", " Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"], rotation=90)
        axs[0].set_xlim([10_000, 60_000])

        axs[0].set_title("Fashion-MNIST Training Samples vs SD")
        axs[0].set_xlabel("Training Samples")
        axs[0].set_ylabel("DDPM Schedule Deviation")

        axs[1].set_title("Fashion-MNIST SD by Class")
        axs[1].set_ylabel("DDPM Schedule Deviation")
        return fig
    _fig = _()
    _fig.savefig("figures/fashion_mnist_expanded.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(HandlerSquare, plt, transformed_data):
    def _():
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,3))
        scatters = []
        for samples, samples_data in transformed_data.groupby("samples"):
            if samples not in [10_000, 30_000, 60_000]:
                continue
            samples_data = samples_data.copy(deep=False)
            samples_data = samples_data[samples_data["density"] > 0.01]
            scatters.append(ax.scatter(samples_data["ddpm_si"],
                           samples_data["ddpm_ddim_dist"],
                           s=3, label=f"N={samples}", alpha=0.7))
        ax.legend(loc="lower right", handler_map={s: HandlerSquare() for s in scatters})
        ax.set_title("MNIST SD vs OT Distance")
        ax.set_xlabel("DDPM Schedule Deviation")
        ax.set_ylabel("DDPM/DDIM OT Distance")
        return fig
    _fig = _()
    _fig
    return


@app.cell
def _(HandlerSquare, mnist_data, mnist_labels, plt):
    _colors = [
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
    _scatters = []
    plt.title("Fashion-MNIST Conditioning")
    for _i, (_c, _n) in enumerate(zip(_colors, ["T-shirt", "Trouser","Pullover"," Dress", " Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"])):
        _d = mnist_data[::5][mnist_labels[::5] == _i]
        _s = plt.scatter(_d[:,0],_d[:,1], color=_c, s=10, alpha=0.8, rasterized=True,
                   label=f"{_n}")
        _scatters.append(_s)
    plt.legend(loc="right", handler_map={s: HandlerSquare() for s in _scatters},
              bbox_to_anchor=(1.3, 0.5))
    plt.savefig("figures/fashion_mnist_examples.pdf", bbox_inches="tight")
    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
