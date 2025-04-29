

import marimo

__generated_with = "0.13.0"
app = marimo.App(width="medium")


@app.cell
def _():
    from torch.utils.data import DataLoader
    def visualize(config, N):
        dataset, _ = config.create()
        _batch = next(iter(DataLoader(dataset, shuffle=True,
                                      batch_size=N)))
        return dataset.visualize_batch(_batch)
    return (visualize,)


@app.cell
def _(visualize):
    from nanogen.datasets.tree  import TreeDataConfig
    visualize(TreeDataConfig(), 2048)
    return


@app.cell
def _(visualize):
    from nanogen.datasets.trajectory import TrajectoryDataConfig
    visualize(TrajectoryDataConfig(), 32)
    return


@app.cell
def _(visualize):
    from nanogen.datasets.oneway import OnewayDataConfig
    visualize(OnewayDataConfig(), 2048)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
