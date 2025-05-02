#!/usr/bin/env python3

import hashlib
import argparse
import sys
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
        comb(T-1, i) * (t**(T-1-i)) * (1-t)**i
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

def generate_trajectory(rng: Rng, maze: Maze, trajectory_length: int) -> np.ndarray:
    reachable_cells = maze.cells
    start_idx, end_idx = rng.choice(len(reachable_cells), 2, replace=False)
    start, end = reachable_cells[start_idx], reachable_cells[end_idx]
    path = maze_solver.solve(rng, maze, start, end)
    trajectory = np.array([(c.col + 0.5, c.row + 0.5) for c in path], dtype=np.float32)
    trajectory += rng.normal(0, 0.2, trajectory.shape).astype(np.float32).clip(-0.3, 0.3)
    trajectory = lin_interpolate(lin_interpolate(trajectory))
    trajectory += rng.normal(0, 0.05, trajectory.shape).astype(np.float32).clip(-0.1, 0.1)
    trajectory = lin_interpolate(trajectory)
    trajectory = bezier_curve(trajectory, TRAJECTORY_LENGTH)
    return trajectory / np.array([maze.num_cols, maze.num_rows]) * 2 - 1

rng = np.random.default_rng(seed=SEED)
mazes = [generate_maze(rng) for mn in range(NUM_MAZES)]

train_mazes = mazes[:(NUM_MAZES + 1) // 2]
test_mazes = mazes[NUM_MAZES // 2:]

for (split, split_mazes) in zip(["train", "test"], [train_mazes, test_mazes]):
    with writer.split(split) as split_writer:
        logger.info(f"Generating split: {split}")
        for mn, maze in enumerate(split_mazes):
            trajectories = []
            for i in range(mn, NUM_TRAJECTORIES, len(split_mazes)):
                logger.info(f"Generating trajectory: {(i - mn) // NUM_MAZES}")
                trajectory = generate_trajectory(rng, maze, TRAJECTORY_LENGTH)
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
