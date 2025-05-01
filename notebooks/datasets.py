

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from nanoconfig.data.source import DataRepository
    from nanoconfig.data.torch import TorchAdapter
    from nanogen.data import register_types

    repo = DataRepository.default()
    trajectory = repo.lookup("trajectory")

    adapter = TorchAdapter()
    register_types(adapter)
    return adapter, trajectory


@app.cell
def _(adapter, trajectory):
    dataset = trajectory.split("train", adapter)
    return (dataset,)


@app.cell
def _(dataset):
    import torch.utils._pytree as pytree

    data = pytree.tree_map(lambda x: x[:16], dataset._data)
    data.to_result()
    return


@app.cell
def _():
    from mazelib.maze import Maze, Cell
    from mazelib.generator import DepthFirstGenerator

    import numpy.random
    import plotly.graph_objects as go

    maze = Maze(8, 8)
    gen = DepthFirstGenerator()
    maze = gen.generate(numpy.random.default_rng(), maze)
    go.Figure([maze.render_plotly()], layout=dict(xaxis_range=[-0.5,8.5],yaxis_range=[-8.5,0.5]))
    return go, maze, numpy


@app.cell
def _(go, maze, numpy):
    from mazelib.solver import DjikstraSolver

    solver = DjikstraSolver()
    path = solver.solve(numpy.random.default_rng(), maze, maze.grid[0][0], maze.grid[-1][-1])

    go.Figure([maze.render_plotly(),
        go.Scatter(x=[p.col + 0.5 for p in path], y=[-1*p.row - 0.5 for p in path])
    ], layout=dict(xaxis_range=[-0.5,8.5],yaxis_range=[-8.5,0.5]))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
