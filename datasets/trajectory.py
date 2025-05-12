#!/usr/bin/env python3

import hashlib
import argparse
import sys
import scipy
import scipy.stats
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=2*4096)
parser.add_argument("--num-mazes", type=int, default=1)
parser.add_argument("--trajectory-length", type=int, default=64)
parser.add_argument("--maze-size", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sha256", action="store_true")
parser.add_argument("--output-dir", type=str, default=None)
args = parser.parse_args()

VERSION = "0.0.1"
NUM_TRAJECTORIES = args.num
NUM_MAZES = args.num_mazes
MAZE_SIZE = args.maze_size
TRAJECTORY_LENGTH = args.trajectory_length
SEED = args.seed

SHA = hashlib.sha256((f"trajectory-{VERSION}" +
    "-".join(f"{k}={v}" for k, v in {
        "num": NUM_TRAJECTORIES,
        "num_mazes": NUM_MAZES,
        "maze_size": MAZE_SIZE,
        "trajectory_length": TRAJECTORY_LENGTH,
        "seed": SEED
    }.items())
).encode()).hexdigest()

if args.sha256:
    print(SHA)
    sys.exit()
if not args.output_dir:
    print("No output directory specified!")
    sys.exit(1)

from nanoconfig.data.fs import FsDataWriter
from nanoconfig.data import utils as data_utils
from mazelib.maze import Maze
from mazelib.generator import DepthFirstGenerator
from mazelib.solver import DjikstraSolver

from numpy.random import Generator as Rng
from scipy.interpolate import CubicSpline
from scipy.special import comb

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import random
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger("trajectory_generator")

writer = FsDataWriter(args.output_dir, SHA)
maze_generator = DepthFirstGenerator()
maze_solver = DjikstraSolver()

schema = pa.schema([
    pa.field("start", pa.list_(pa.float32(), 2),
        metadata={"mime_type": "point"}),
    pa.field("end", pa.list_(pa.float32(), 2),
        metadata={"mime_type": "point"}),
    pa.field("trajectory",
        pa.list_(pa.list_(pa.float32(),2), TRAJECTORY_LENGTH),
        metadata={"mime_type": "trajectory"}
    ),
    pa.field("maze",
        pa.list_(pa.list_(
            pa.list_(pa.bool_(),4),
            MAZE_SIZE),MAZE_SIZE),
        metadata={"mime_type": "maze/edges_one_hot"}
    ),
], metadata={"mime_type": "data/trajectory+maze"})

def generate_maze(rng):
    maze = Maze(MAZE_SIZE, MAZE_SIZE)
    maze = maze_generator.generate(rng, maze)
    # pick four random walls to knock down in the mizzle
    cells = maze.cells
    for i in rng.integers(0, len(cells), size=(3,)):
        nr = rng.integers(0, len(cells[i].unreachable_neighbors))
        cells[i].remove_walls(cells[i].unreachable_neighbors[nr])
    return maze

def get_bezier_parameters(X, Y, degree=3):
    if degree < 1:
        raise ValueError('degree must be 1 or greater.')
    if len(X) != len(Y):
        raise ValueError('X and Y must be of the same length.')
    if len(X) < degree + 1:
        raise ValueError(f'Need at least {degree + 1} points for degree {degree} curve. Got {len(X)}.')
    T = np.linspace(0, 1, len(X))
    M = np.matrix([[
        t**k * (1-t)**(degree-k) * comb(degree, k)
        for k in range(degree + 1)
    ] for t in T])
    # Perform least square fit
    points = np.array(list(zip(X, Y)))
    M_ = np.linalg.pinv(M)
    final = (M_ * points).tolist()
    # Set endpoints to match input
    final[0] = [X[0], Y[0]]
    final[-1] = [X[-1], Y[-1]]
    return final

def bezier_curve(points, N):
    T = len(points)
    x, y = points.T
    t = np.linspace(0.0, 1.0, N)
    polynomial_array = np.array([
        comb(T-1, i) * ((1-t)**(T-1-i)) * t**i
        for i in range(T)
    ])
    xvals = np.dot(x, polynomial_array)
    yvals = np.dot(y, polynomial_array)
    return np.stack([xvals, yvals], axis=-1)

def lin_interpolate(trajectory):
    trajectory_expanded = np.repeat(trajectory, 2, axis=0)
    trajectory_shifted = np.roll(trajectory_expanded, 1, axis=0)
    trajectory_shifted[0] = trajectory[0]
    trajectory = (trajectory_expanded + trajectory_shifted) / 2
    return trajectory

def generate_trajectories(rng: Rng, maze: Maze, trajectory_length: int):
    col_weights = np.exp(-0.5*np.arange(maze.num_cols)) + np.exp(-0.5*np.arange(maze.num_cols))[::-1]
    col_weights = col_weights / col_weights.sum()
    while True:
        start_row = rng.integers(0, maze.num_rows, ())
        start_col = rng.choice(maze.num_cols, (), p=col_weights)
        # generate the end column randomly and the start column weighted toward the left or right side
        end_row, end_col = maze.num_rows // 2, maze.num_cols // 2

        start, end = maze.grid[start_row][start_col], maze.grid[end_row][end_col]
        solutions = maze_solver.solve(rng, maze, start, end, all_paths=True)
        weights = np.array([len(s) for s in solutions])
        weights = np.exp(-(weights - weights.min()))
        weights = weights / weights.sum()

        solution = rng.choice(len(solutions), p=weights)
        solution = solutions[solution]

        trajectory = np.array([(c.col + 0.5, c.row + 0.5) for c in solution], dtype=np.float32)
        trajectory += rng.normal(0, 0.2, trajectory.shape).astype(np.float32).clip(-0.3, 0.3)
        trajectory = lin_interpolate(lin_interpolate(trajectory))
        trajectory = lin_interpolate(trajectory)
        trajectory = bezier_curve(trajectory, TRAJECTORY_LENGTH + 4)[2:-2]
        yield trajectory / np.array([maze.num_cols, maze.num_rows]) * 2 - 1

rng = np.random.default_rng(seed=SEED)
mazes = [generate_maze(rng) for mn in range(NUM_MAZES)]

train_mazes = mazes[:(NUM_MAZES + 1) // 2]
test_mazes = mazes[NUM_MAZES // 2:]

for (split, split_mazes) in zip(["train", "test"], [train_mazes, test_mazes]):
    with writer.split(split) as split_writer:
        logger.info(f"Generating split: {split}")
        for mn, maze in enumerate(split_mazes):
            trajectories = []
            generator = generate_trajectories(rng, maze, TRAJECTORY_LENGTH)
            for i, trajectory in zip(range(mn, NUM_TRAJECTORIES, len(split_mazes)), generator):
                logger.info(f"Generating trajectory: {(i - mn) // NUM_MAZES}")
                trajectories.append(trajectory)
            trajectories = np.array(trajectories, dtype=np.float32)
            maze = data_utils.as_arrow_array(
                maze.to_numpy_onehot()[None].repeat(len(trajectories), 0)
            )
            start = data_utils.as_arrow_array(trajectories[:,0])
            end = data_utils.as_arrow_array(trajectories[:,-1])
            trajectories = data_utils.as_arrow_array(trajectories)
            table = pa.table([start, end, trajectories, maze], schema=schema)
            for batch in table.to_batches():
                split_writer.write_batch(batch)
