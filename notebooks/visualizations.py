

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

    from pathlib import Path
    return Path, np, pd, plt, scipy, wandb


@app.cell
def _(wandb):
    api = wandb.Api()
    sweep = api.sweep("dpfrommer-projects/nanogen_mnist/hsnrxhw0")
    return (sweep,)


@app.cell
def _(Path, np, pd, sweep):
    def parse_row(r):
        r = r.strip("[]")
        return list(float(v) for v in r.strip("[]").split(r" ") if v)

    def load_data():
        data = []
        for samples, run in zip([10000, 30000, 45000, 60000], sweep.runs):
            artifact = list(a for a in run.logged_artifacts() if a.type == "results")[0]
            path = Path(artifact.download()) / "distances.csv"
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
    mnist_data = np.concatenate(list(data_utils.as_numpy(batch["tsne"]) for batch in mnist_data.to_batches(columns=["tsne"])))
    return (mnist_data,)


@app.cell
def _(mnist_data, np, scipy):
    def calc_density(sample_points):
        dists = -np.sum(np.square(sample_points[:,None, :] - mnist_data[None,:, :]), axis=-1)
        dists = dists/50
        log_pdfs = scipy.special.logsumexp(dists, axis=1)
        log_pdfs -= np.log(10)
        # log_pdfs = log_pdfs - scipy.special.logsumexp(log_pdfs)
        return np.exp(log_pdfs)

    def calc_smoothed(cond, values):
        dists = -np.sum(np.square(cond[:,None, :] - cond[None,:, :]), axis=-1)
        return scipy.special.logsumexp(dists, axis=1, b=values)
    return calc_density, calc_smoothed


@app.cell
def _(calc_density, calc_smoothed, data, np):
    def transform_densities(data):
        new_data = data.copy(deep=False)
        cond = np.stack((data["condition_x"], data["condition_y"]), axis=-1)
        for c in ["ddpm_ddim_dist"]:
            print("Smoothing", c)
            new_data[c] = calc_smoothed(cond, new_data[c].to_numpy())
        print("Calculating density...")
        new_data["density"] = calc_density(cond)
        # average distances over conditioning points
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
def _(plt, transformed_data):
    plt.scatter(transformed_data["condition_x"], transformed_data["condition_y"], c=transformed_data["ddpm_ddim_dist"])
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
