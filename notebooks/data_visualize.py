

import marimo

__generated_with = "0.13.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from nanoconfig.data.source import DataRepository
    from nanogen.data import register_types
    from nanoconfig.data.torch import TorchAdapter


    adapter = TorchAdapter()
    register_types(adapter)

    traj_data = DataRepository.default().lookup("trajectory")
    traj_data = traj_data.split("test", adapter)
    traj_data.head(128).to_result()
    return


@app.cell
def _():
    from mazelib.maze import Maze
    from mazelib.generator import DepthFirstGenerator
    from mazelib.solver import DjikstraSolver

    import numpy as np
    import numpy.random
    rng = numpy.random.default_rng(seed=42)

    generator = DepthFirstGenerator()
    maze = Maze(8,8)
    maze = generator.generate(rng, maze)
    cells = maze.cells
    for i in rng.integers(0, len(cells), size=(3,)):
        nr = rng.integers(0, len(cells[i].unreachable_neighbors))
        cells[i].remove_walls(cells[i].unreachable_neighbors[nr])
    return DjikstraSolver, maze, np, rng


@app.cell
def _(DjikstraSolver, maze, np, rng):
    solver = DjikstraSolver()
    solutions = solver.solve(rng, maze, maze.grid[1][1], maze.grid[4][4], all_paths=True)
    solutions = [np.array([(c.col + 0.5, -c.row - 0.5) for c in path]) for path in solutions]
    return (solutions,)


@app.cell
def _(maze, solutions):
    import plotly.graph_objects as go
    go.Figure([
        maze.render_plotly()
    ] + [
        go.Scatter(x=solutions[i][:,0],y=solutions[i][:,1]) for i in range(len(solutions))
    ])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
