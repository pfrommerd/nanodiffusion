import marimo

__generated_with = "0.12.10"
app = marimo.App(width="medium")


@app.cell
def _():
    from nanodiffusion.datasets.trajectory import TrajectoryDataConfig
    from nanodiffusion.datasets.tree  import TreeDataConfig
    # dataset, _ = TrajectoryDataConfig().create()
    dataset, _ = TreeDataConfig().create()
    return TrajectoryDataConfig, TreeDataConfig, dataset


@app.cell
def _(dataset):
    from torch.utils.data import DataLoader

    _batch = next(iter(DataLoader(dataset, shuffle=True, batch_size=2048)))
    dataset.visualize_batch(_batch)
    return (DataLoader,)


@app.cell
def _(DataLoader, dataset):
    loader = DataLoader(dataset, batch_size=512, shuffle=True)
    import time
    t = time.time()
    iteration = 0
    for e in range(100):
        for batch in loader:
            batch.to("cuda")
            iteration += 1
    took = time.time() - t
    print(iteration / took, "iter/second", "total", iteration, took, "seconds")
    return batch, e, iteration, loader, t, time, took


if __name__ == "__main__":
    app.run()
