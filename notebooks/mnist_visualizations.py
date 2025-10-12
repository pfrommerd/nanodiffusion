

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
    # OLD: dpfrommer-projects/nanogen_mnist/nqlofeyh
    # NEW: dpfrommer-projects/nanogen_mnist/cylpdnas
    sweep = api.sweep("dpfrommer-projects/nanogen_mnist/t3c0i7vz")
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
            idxs = range(0,9)
            df["ddpm_si"] =  sum(df[f"ddpm_si/{i}"] for i in idxs)/len(idxs)
            df["ddim_si"] =  sum(df[f"ddim_si/{i}"] for i in idxs)/len(idxs)
            df["accel_si"] =  sum(df[f"accel_si/{i}"] for i in idxs)/len(idxs)
            data.append(df)
        return pd.concat(data)
    data = load_data()
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
    pickle.dump(transformed_data, open("figures/mnist_data.pkl", "wb"))
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
    def interpolate(cond, data, grid_x, grid_y):
        grid = np.stack((grid_x.reshape(-1), grid_y.reshape(-1)), axis=-1)
        diff = cond[None, :, :] - grid[:, None, :]
        weight = np.sum(np.square(diff), axis=-1)
        weight = scipy.special.softmax(-0.1*weight, axis=-1)
        interpolated = weight @ data
        return interpolated.reshape(grid_x.shape)

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
            # values = scipy.interpolate.griddata(cond, sub_data[column].to_numpy(),
            #                                     (grid_x, grid_y), method='nearest')[::-1,:]
            values = interpolate(cond, sub_data[column].to_numpy(), grid_x, grid_y)[::-1,:]
            if False:
                values = values / values.max()
                vmin = vmin or values.min()
                vmax = vmax or 1.
            cbar = ax.imshow(values,cmap="binary", extent=[-100, 100, -100, 100],
                                                vmin=vmin, vmax=vmax, rasterized=True)
            ax.scatter(mnist_data[::5,0], mnist_data[::5,1],
                             c=[colors[l] for l in mnist_labels[::5]], s=2, alpha=0.8,
                             rasterized=True)
            ax.grid(False)
            cbars.append(cbar)
            ax.set_xlim([-100, 100])
            ax.set_ylim([-100, 100])
            if i == 0:
                ax.set_ylabel("t-SNE Second Component")
            #ax.set_xlabel("t-SNE First Component")
        fig.colorbar(cbars[0], ax=axs, label=label,fraction=0.03, pad=0.01, shrink=0.7)
        return fig
    _fig, _axs = plt.subplots(ncols=3, nrows=2, figsize=(10,7), sharex="col", sharey="row")
    _fig.subplots_adjust(hspace=-0.1, wspace=0.18)
    #_fig.tight_layout()
    _fig = heatmaps(_fig, _axs[0], "ddpm_ddim_dist", "DDPM/DDIM OT Distance", 0.5, 8)
    _fig = heatmaps(_fig, _axs[1], "ddpm_si", "DDPM Schedule Deviation", 50, 140)
    _axs[-1][0].set_xlabel("t-SNE First Component")
    _axs[-1][1].set_xlabel("t-SNE First Component")
    _axs[-1][2].set_xlabel("t-SNE First Component")

    _axs[0][0].set_title("MNIST N=10000")
    _axs[0][1].set_title("MNIST N=30000")
    _axs[0][2].set_title("MNIST N=60000")

    _fig.savefig("figures/mnist_ddpm_ddim_dist.pdf", bbox_inches="tight")
    _fig
    return (heatmaps,)


@app.cell
def _(heatmaps, plt):
    _fig, _axs = plt.subplots(ncols=3, nrows=6, figsize=(12,25), sharex="col", sharey="row")
    _fig.subplots_adjust(hspace=-0.2, wspace=0.05)
    _fig = heatmaps(_fig, _axs[0], "ddpm_ddim_dist", "DDPM/DDIM OT Distance", 0.5, 8)
    _fig = heatmaps(_fig, _axs[1], "ddpm_accel_dist", "DDPM/GE OT Distance", 0.5, 8)
    _fig = heatmaps(_fig, _axs[2], "ddim_accel_dist", "DDIM/GE OT Distance", 0.5, 8)
    _fig = heatmaps(_fig, _axs[3], "ddpm_si", "DDPM Schedule Deviation", 50, 140)
    _fig = heatmaps(_fig, _axs[4], "ddim_si", "DDIM Schedule Deviation", 50, 140)
    _fig = heatmaps(_fig, _axs[5], "accel_si", "GE Schedule Deviation",  50, 140)
    for _ax in _axs[-1]:
        _ax.set_xlabel("t-SNE First Component")
        _ax.set_xticks([-75, -25, 25, 75])
    for _ax in _axs:
        _ax[0].set_yticks([-75, -25, 25, 75])
    _fig.savefig("figures/mnist_heatmaps.pdf", bbox_inches="tight")
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
def _(Path, api, pd):
    def load_capacity_sweep():
        sweep = api.sweep("dpfrommer-projects/nanogen_mnist/1y7jm61s")
        artifacts = list(list(a for a in run.logged_artifacts() if a.type == "results")[0]
                       for run in sweep.runs if list(a for a in run.logged_artifacts() if a.type == "results"))
        data = []
        for artifact in artifacts:
            run = artifact.logged_by()
            input_artifact = list(a for a in run.used_artifacts() if a.type == "model")[0]
            cfg = input_artifact.logged_by().config
            if "checkpoint" in input_artifact.name:
                iteration = int(input_artifact.name[len("checkpoint_"):].split(":")[0])
            else:
                iteration = 300_000
            channels = cfg["pipeline"]["model"]["nn"]["base_channels"]
            path = Path(artifact.download()) / "metrics.csv"
            df = pd.read_csv(path)
            df["channels"] = channels
            df["iteration"] = iteration
            idxs = range(2, 7) # don't include the very beginning or end
            df["ddpm_si"] =  sum(df[f"ddpm_si/{i}"] for i in idxs)/len(idxs)
            df["ddim_si"] =  sum(df[f"ddim_si/{i}"] for i in idxs)/len(idxs)
            df["accel_si"] =  sum(df[f"accel_si/{i}"] for i in idxs)/len(idxs)
            data.append(df)
        return pd.concat(data)
    capacity_data = load_capacity_sweep()
    return (capacity_data,)


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
    return (capacity_history_data,)


@app.cell
def _(
    capacity_data,
    capacity_history_data,
    mnist_data,
    mnist_labels,
    np,
    pd,
    plt,
    transformed_data,
):
    import matplotlib.ticker as ticker

    def _():
        fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(15,3))
        fig.subplots_adjust(wspace=0.3)
        label = {64: "5.9M", 96: "13.3M", 128: "23.5M", 160: "36.8M"}
        for (channels, g), color in zip(capacity_history_data.groupby("channels"),
                                        plt.rcParams['axes.prop_cycle'].by_key()['color']):
            g = g[g["_step"] >= 30_000].copy()
            g["step_bin"] = pd.cut(g["_step"], 20)
            g_step = g.groupby("step_bin", observed=True).max().reset_index()["_step"]
            g_mean = g.groupby("step_bin", observed=True).median().reset_index()
            g_upper = g.groupby("step_bin", observed=True).quantile(0.7).reset_index()
            g_lower = g.groupby("step_bin", observed=True).quantile(0.3).reset_index()
            if channels <= 64: continue
            axs[0].plot(g_step, g_mean["loss/test"], alpha=0.7, label=label[channels] + " parameters")#, color=color)
            axs[0].fill_between(g_step, g_lower["loss/test"], g_upper["loss/test"], alpha=0.3)#, color=color)
            #axs[0].plot(g_step, g_mean["loss/train"], alpha=0.7, label=label[channels] + " parameters", color=color)
            #axs[0].fill_between(g_step, g_lower["loss/train"], g_upper["loss/train"], alpha=0.3, color=color)

        axs[0].legend(loc="upper right")
        axs[0].set_xlim([50_000, 300_000])
        axs[0].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3f}"))
        axs[0].set_title("MNIST Test Loss")
        axs[0].set_xlabel("Training Iteration")
        axs[0].set_ylabel("Test Loss")

        cd = capacity_data.groupby(["channels","iteration"])[["ddpm_si"]].median().reset_index()
        cd_upper = capacity_data.groupby(["channels","iteration"])[["ddpm_si"]].quantile(0.7).reset_index()
        cd_lower = capacity_data.groupby(["channels","iteration"])[["ddpm_si"]].quantile(0.3).reset_index()

        for (channels, g), (_, g_upper), (_, g_lower) in zip(cd.groupby("channels"),
                                               cd_upper.groupby("channels"), cd_lower.groupby("channels")):
            g = g[g["iteration"] > 0]
            g_upper = g_upper[g_upper["iteration"] > 0] 
            g_lower = g_lower[g_lower["iteration"] > 0]
            if channels <= 64: continue
            axs[1].plot(g["iteration"], g["ddpm_si"], label=label[channels] + " parameters")
            axs[1].fill_between(g["iteration"], g_lower["ddpm_si"], g_upper["ddpm_si"], alpha=0.3)

        axs[1].legend(loc="upper right")
        axs[1].set_xlim([50_000, 300_000])
        axs[1].set_title("MNIST SD vs Model Size")
        axs[1].set_xlabel("Training Iteration")
        axs[1].set_ylabel("DDPM Schedule Deviation")


        td = transformed_data.copy()
        mean_data = td.groupby("samples").median().reset_index()
        high_data = td.groupby("samples").quantile(0.7).reset_index()
        low_data = td.groupby("samples").quantile(0.3).reset_index()
        axs[2].plot(mean_data["samples"],mean_data["ddpm_si"])
        axs[2].fill_between(mean_data["samples"], low_data["ddpm_si"], high_data["ddpm_si"],
                           alpha=0.3)
        axs[2].set_xlim([10_000, 60_000])
        axs[2].set_title("MNIST Training Samples vs SD")
        axs[2].set_xlabel("Training Dataset Size")
        #axs[2].set_ylabel("DDPM Schedule Deviation")

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
        axs[3].violinplot(groups, np.arange(10))
        axs[3].set_xticks(np.arange(10))
        axs[3].set_title("MNIST SD by Class")
        axs[3].set_xlabel("Digit Label")
        #axs[3].set_ylabel("DDPM Schedule Deviation")

        axs[0].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x//1_000)}k"))
        axs[1].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x//1_000)}k"))
        axs[2].xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x//1_000)}k"))

        return fig
    _fig = _()
    _fig.savefig("figures/mnist_expanded.pdf", bbox_inches="tight")
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
    plt.title("MNIST Conditioning")
    for _i, _c in enumerate(_colors):
        _d = mnist_data[::5][mnist_labels[::5] == _i]
        _s = plt.scatter(_d[:,0],_d[:,1], color=_c, s=10, alpha=0.8, rasterized=True,
                   label=f"{_i}")
        _scatters.append(_s)
    plt.legend(loc="right", handler_map={s: HandlerSquare() for s in _scatters},
              bbox_to_anchor=(1.15, 0.5))
    plt.savefig("figures/mnist_examples.pdf")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
