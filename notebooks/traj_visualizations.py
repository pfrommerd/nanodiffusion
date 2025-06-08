

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
    sweep = api.sweep("dpfrommer-projects/nanogen_trajectory/4ipspumr")
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
            idxs = range(2, 12)
            df["ddpm_si"] =  sum(df[f"ddpm_si/{i}"] for i in idxs)
            df["ddim_si"] =  sum(df[f"ddim_si/{i}"] for i in idxs)
            df["accel_si"] =  sum(df[f"accel_si/{i}"] for i in idxs)
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
    return DataRepository, data_utils, traj_data


@app.cell
def _(DataRepository):
    from nanogen.data import register_types
    from nanoconfig.data.torch import TorchAdapter
    import plotly.io
    _traj_data = DataRepository.default().lookup("trajectory")
    _adapter = TorchAdapter()
    register_types(_adapter)
    _traj_data = _traj_data.split("test", _adapter)
    figure = _traj_data.head(32).to_result()
    figure.figure.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    figure.figure.write_image("figures/trajectories_examples.pdf")
    return


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
        dists = 200*dists
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
        #     new_group = smooth(cond, group[["ddpm_si/10", "ddpm_ddim_dist",  "ddpm_accel_dist", "ddim_accel_dist"]])
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
def _(HandlerSquare, plt, transformed_data):
    def _():
        import pickle
        mnist_data = pickle.load(open("figures/mnist_data.pkl", "rb"))
        fashion_mnist_data = pickle.load(open("figures/fashion_mnist_data.pkl", "rb"))

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(11,3))
        scatters = []
        for samples, samples_data in mnist_data.groupby("samples"):
            if samples not in [10_000, 30_000, 60_000]:
                continue
            scatters.append(axs[0].scatter(samples_data["ddpm_si"],
                           samples_data["ddpm_ddim_dist"], rasterized=True,
                           s=1, label=f"N={samples}", alpha=0.7))
        axs[0].legend(loc="lower right", handler_map={s: HandlerSquare() for s in scatters})
        axs[0].set_title("MNIST")
        axs[0].set_xlabel("DDPM Schedule Deviation")
        axs[0].set_ylabel("DDPM/DDIM OT Distance")
        axs[0].set_yticks([0, 2, 4, 6, 8])
        axs[0].set_xlim([40, 200])

        scatters = []
        for samples, g in fashion_mnist_data.groupby("samples"):
            if samples not in [10_000, 30_000, 60_000]:
                continue
            scatters.append(axs[1].scatter(g["ddpm_si"], g["ddpm_ddim_dist"],
                        rasterized=True, s=1, label=f"N={samples}"))
        axs[1].legend(loc="lower right",
            handler_map={s: HandlerSquare() for s in scatters})
        axs[1].set_title("Fashion-MNIST")
        axs[1].set_xlabel("DDPM Schedule Deviation")
        # axs[1].set_ylabel("DDPM/DDIM OT Distance")
        axs[1].set_xlim([70, 380])
        axs[1].set_yticks([0, 2, 4, 6])

        scatters = []
        for samples, g in transformed_data.groupby("samples"):
            if samples not in [2_000, 4_000, 8_000]:
                continue
            scatters.append(axs[2].scatter(g["ddpm_si"], g["ddpm_ddim_dist"],
                            rasterized=True, s=1, label=f"N={samples}"))
        axs[2].legend(loc="lower right",
            handler_map={s: HandlerSquare() for s in scatters})
        axs[2].set_ylim([0,0.45])
        axs[2].set_title("Maze Solutions")
        axs[2].set_xlabel("DDPM Schedule Deviation")
        # axs[2].set_ylabel("DDPM/DDIM OT Distance")


        return fig
    _fig = _()
    _fig.subplots_adjust(wspace=0.15, hspace=0.01)
    _fig.savefig("figures/distances_scatters.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(HandlerSquare, plt, transformed_data):
    def _():
        import pickle
        mnist_data = pickle.load(open("figures/mnist_data.pkl", "rb"))
        fashion_mnist_data = pickle.load(open("figures/fashion_mnist_data.pkl", "rb"))

        fig, all_axs = plt.subplots(nrows=3, ncols=3, figsize=(12,11))
        for axs, (yc, yl, xc, xl) in zip(all_axs, [
            ("ddpm_ddim_dist", "DDIM/DDIM OT Distance", "ddpm_si", "DDPM Schedule Deviation"),
            ("ddpm_accel_dist", "DDPM/GE OT Distance", "accel_si", "GE Schedule Deviation"),
            ("ddim_accel_dist", "DDIM/GE OT Distance", "ddim_si", "DDIM Schedule Deviation")
        ]):
            scatters = []
            for samples, samples_data in mnist_data.groupby("samples"):
                if samples not in [10_000, 30_000, 60_000]:
                    continue
                scatters.append(axs[0].scatter(samples_data[xc],
                               samples_data[yc], rasterized=True,
                               s=1, label=f"N={samples}", alpha=0.7))
            axs[0].legend(loc="lower right", handler_map={s: HandlerSquare() for s in scatters})
            axs[0].set_title("MNIST")
            axs[0].set_xlabel(xl)
            axs[0].set_ylabel(yl)
            axs[0].set_yticks([0, 2, 4, 6, 8])
            axs[0].set_xlim([40, 200])
    
            scatters = []
            for samples, g in fashion_mnist_data.groupby("samples"):
                if samples not in [10_000, 30_000, 60_000]:
                    continue
                scatters.append(axs[1].scatter(g[xc], g[yc],
                            rasterized=True, s=1, label=f"N={samples}"))
            axs[1].legend(loc="lower right",
                handler_map={s: HandlerSquare() for s in scatters})
            axs[1].set_title("Fashion-MNIST")
            axs[1].set_xlabel(xl)
            # axs[1].set_ylabel("DDPM/DDIM OT Distance")
            axs[1].set_xlim([70, 370])
            axs[1].set_yticks([0, 2, 4, 6])
    
            scatters = []
            for samples, g in transformed_data.groupby("samples"):
                if samples not in [2_000, 4_000, 8_000]:
                    continue
                scatters.append(axs[2].scatter(g[xc], g[yc],
                                rasterized=True, s=1, label=f"N={samples}"))
            axs[2].legend(loc="lower right",
                handler_map={s: HandlerSquare() for s in scatters})
            axs[2].set_ylim([0,0.45])
            axs[2].set_title("Maze Solutions")
            axs[2].set_xlabel(xl)
        # axs[2].set_ylabel("DDPM/DDIM OT Distance")

        all_axs[1][1].set_xlim([70, 460])
        all_axs[2][1].set_xlim([70, 410])

        return fig
    _fig = _()
    _fig.subplots_adjust(wspace=0.15, hspace=0.45)
    _fig.savefig("figures/all_distances_scatters.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(plt, transformed_data):
    plt.scatter(transformed_data["condition/0"], transformed_data["condition/1"], c=transformed_data["ddpm_si/8"], s=4)
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
    def heatmaps(fig, axs, column, label, vmin = None, vmax = None, show_samples=[2_000,4_000,8_000],
                    cbar_fraction=0.03, cbar_shrink=0.9):
        from matplotlib.colors import LogNorm
        cbars = []
        grid_y, grid_x = np.mgrid[-1:1:100j, -1:1:100j]
        xs, ys = grid_x[0,:], grid_y[:,0]
        groups = [(s, d) for s, d in transformed_data.groupby("samples") if s in show_samples]
        for i, ((samples, sub_data), ax) in enumerate(zip(groups, axs)):
            # evaluate on a grid
            cond = np.stack((sub_data["condition/0"], sub_data["condition/1"]), axis=-1)
            values = scipy.interpolate.griddata(cond, sub_data[column].to_numpy(),
                                                (grid_x, grid_y), method='nearest')[::-1,:]
            cbar = ax.imshow(values,cmap="binary", extent=[-1, 1, -1, 1],
                                                vmin=vmin, vmax=vmax)
            ax.grid(False)
            cbars.append(cbar)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            #ax.set_xlabel("t-SNE First Component")
        fig.colorbar(cbars[0], ax=axs, label=label,fraction=cbar_fraction, pad=0.01, shrink=cbar_shrink)
        return fig

    _fig, _axs = plt.subplots(ncols=3, nrows=2, figsize=(15,9), sharex="col", sharey="row")
    _fig.subplots_adjust(hspace=0.06, wspace=0.06)
    #_fig.tight_layout()
    _fig = heatmaps(_fig, _axs[0], "ddpm_ddim_dist", "DDPM/DDIM OT Distance", 0, 1.1)
    _fig = heatmaps(_fig, _axs[1], "ddpm_si", "DDPM Schedule Deviation", 60, 130)

    _axs[-1][0].set_xlabel("Start Point X")
    _axs[-1][1].set_xlabel("Start Point X")
    _axs[-1][2].set_xlabel("Start Point X")

    _axs[0][0].set_title("Trajectory N=1000")
    _axs[0][1].set_title("Trajectory N=2000")
    _axs[0][2].set_title("Trajectory N=4000")

    _fig.savefig("figures/traj_ddpm_ddim_dist.pdf", bbox_inches="tight")
    _fig
    return (heatmaps,)


@app.cell
def _(heatmaps, plt):
    _fig, _axs = plt.subplots(ncols=3, nrows=2, figsize=(15,9), sharex="col", sharey="row")
    _fig.subplots_adjust(hspace=0.06, wspace=0.06)
    #_fig.tight_layout()
    _fig = heatmaps(_fig, _axs[0], "ddpm_ddim_dist", "DDPM/GE OT Distance", 0, 1.1)
    _fig = heatmaps(_fig, _axs[1], "ddpm_si", "DDPM Schedule Deviation (Normalized)")

    _axs[-1][0].set_xlabel("t-SNE First Component")
    _axs[-1][1].set_xlabel("t-SNE First Component")
    _axs[-1][2].set_xlabel("t-SNE First Component")

    _axs[0][0].set_title("Trajectory N=1000")
    _axs[0][1].set_title("Trajectory N=2000")
    _axs[0][2].set_title("Trajectory N=4000")

    _fig.savefig("figures/traj_ddpm_accel_dist.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(heatmaps, plt):
    _fig, _axs = plt.subplots(ncols=3, nrows=6, figsize=(12,25), sharex="col", sharey="row")
    _fig.subplots_adjust(hspace=-0.2, wspace=0.05)
    _fig = heatmaps(_fig, _axs[0], "ddpm_ddim_dist", "DDPM/DDIM OT Distance", 0, 1.1, cbar_shrink=0.7)
    _fig = heatmaps(_fig, _axs[1], "ddpm_accel_dist", "DDPM/GE OT Distance", 0, 1.1, cbar_shrink=0.7)
    _fig = heatmaps(_fig, _axs[2], "ddim_accel_dist", "DDIM/GE OT Distance", 0, 1.1, cbar_shrink=0.7)
    _fig = heatmaps(_fig, _axs[3], "ddpm_si", "DDPM Schedule Deviation", 50, 130, cbar_shrink=0.7)
    _fig = heatmaps(_fig, _axs[4], "ddim_si", "DDIM Schedule Deviation", 50, 130, cbar_shrink=0.7)
    _fig = heatmaps(_fig, _axs[5], "accel_si", "GE Schedule Deviation", 50, 130, cbar_shrink=0.7)
    for _ax in _axs[-1]:
        _ax.set_xlabel("Start Point X")
        _ax.set_xticks([-0.75, -0.25, 0.25, 0.75])
    for _ax in _axs:
        _ax[0].set_ylabel("Start Point Y")
        _ax[0].set_yticks([-0.75, -0.25, 0.25, 0.75])
    _fig.savefig("figures/traj_heatmaps.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(DataRepository, data_utils, np):
    fashion_mnist_data = DataRepository.default().lookup("fashion-mnist")
    fashion_mnist_data = fashion_mnist_data.split("test")
    fashion_mnist_labels = np.concatenate(list(data_utils.as_numpy(batch["class"]) for batch in fashion_mnist_data.to_batches(columns=["class"])))
    fashion_mnist_data = np.concatenate(list(data_utils.as_numpy(batch["tsne"]) for batch in fashion_mnist_data.to_batches(columns=["tsne"])))
    return fashion_mnist_data, fashion_mnist_labels


@app.cell
def _(fashion_mnist_data, fashion_mnist_labels, np, scipy):
    def fashion_mnist_heatmaps(fig, axs, column, label, vmin = None, vmax = None, show_samples=[30_000,60_000]):
        import pickle
        fashion_mnist_df = pickle.load(open("figures/fashion_mnist_data.pkl", "rb"))
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
        groups = [(s, d) for s, d in fashion_mnist_df.groupby("samples") if s in show_samples]
        for i, ((samples, sub_data), ax) in enumerate(zip(groups, axs)):
            # evaluate on a grid
            cond = np.stack((sub_data["condition/0"], sub_data["condition/1"]), axis=-1)
            values = scipy.interpolate.griddata(cond, sub_data[column].to_numpy(),
                                                (grid_x, grid_y), method='nearest')[::-1,:]
            cbar = ax.imshow(values,cmap="binary", extent=[-100, 100, -100, 100],
                                                vmin=vmin, vmax=vmax, rasterized=True)
            ax.scatter(fashion_mnist_data[::5,0], fashion_mnist_data[::5,1],
                             c=[colors[l] for l in fashion_mnist_labels[::5]], s=1.5, alpha=0.8,
                             rasterized=True)
            ax.grid(False)
            cbars.append(cbar)
            ax.set_xlim([-100, 100])
            ax.set_ylim([-100, 100])
        fig.colorbar(cbars[0], ax=axs, label=label,fraction=0.03, pad=0.01, shrink=0.75)
        return fig
    return (fashion_mnist_heatmaps,)


@app.cell
def _(fashion_mnist_heatmaps, heatmaps, plt):
    _fig, _axs = plt.subplots(ncols=4, nrows=2, figsize=(11,7), sharex="col",
                        gridspec_kw={'width_ratios': [0.4, 0.12, 0.4, 0.4]})
    _axs[0,1].set_axis_off()
    _axs[1,1].set_axis_off()
    _fig.subplots_adjust(hspace=-0.18, wspace=0.05)


    fashion_mnist_heatmaps(_fig, _axs[0,2:], "ddpm_ddim_dist", "DDPM/DDIM OT Distance", 0, 9)
    fashion_mnist_heatmaps(_fig, _axs[1,2:], "ddpm_si", "DDPM Schedule Deviation", 60, 340)


    heatmaps(_fig, _axs[0,:1], "ddpm_ddim_dist", "DDPM/DDIM OT Distance", 0, 1.1,show_samples=[4_000],
             cbar_fraction=0.08, cbar_shrink=0.67)
    heatmaps(_fig, _axs[1,:1], "ddpm_si", "DDPM Schedule Deviation", 60, 130,show_samples=[4_000],
             cbar_fraction=0.08, cbar_shrink=0.67)

    for _ax in _axs.reshape(-1):
        _ax.xaxis.set_ticklabels([])
        _ax.yaxis.set_ticklabels([])

    _axs[0][0].set_ylabel("Start Point Y", labelpad=-5)
    _axs[1][0].set_ylabel("Start Point Y", labelpad=-5)
    _axs[1][0].set_xlabel("Start Point X", labelpad=-5)

    _axs[0][2].set_ylabel("t-SNE Second Component", labelpad=-5)
    _axs[1][2].set_ylabel("t-SNE Second Component", labelpad=-5)
    _axs[1][2].set_xlabel("t-SNE First Component", labelpad=-5)
    _axs[1][3].set_xlabel("t-SNE First Component", labelpad=-5)

    _axs[0][0].set_title("Trajectory N=4000")
    _axs[0][2].set_title("Fashion-MNIST N=30000")
    _axs[0][3].set_title("Fashion-MNIST N=60000")

    _fig.savefig("figures/heatmaps_expanded.pdf", bbox_inches="tight")

    _fig
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
