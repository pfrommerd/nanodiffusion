#!/usr/bin/env python3

import hashlib
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=4096)
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
from mazelib.maze import Maze, CellType
from mazelib.generator import DepthFirstGenerator
from mazelib.solver import DjikstraSolver

from numpy.random import Generator as Rng
from scipy.interpolate import CubicSpline

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
    pa.field("start", pa.list_(pa.float32(), 2)),
    pa.field("end", pa.list_(pa.float32(), 2)),
    pa.field("trajectory", pa.list_(pa.list_(pa.float32(),2), TRAJECTORY_LENGTH)),
], metadata={"mime_type": "parquet/trajectory"})

def generate_maze(rng):
    maze = Maze(MAZE_SIZE, MAZE_SIZE)
    maze = maze_generator.generate(rng, maze)
    return maze

def generate_trajectory(rng: Rng, maze: Maze, trajectory_length: int) -> np.ndarray:
    reachable_cells = maze.type_cells[CellType.REACHABLE]
    start_idx, end_idx = rng.choice(len(reachable_cells), 2, replace=False)
    start, end = reachable_cells[start_idx], reachable_cells[end_idx]
    path = maze_solver.solve(rng, maze, start, end)
    trajectory = np.array([(c.row + 0.5, c.col + 0.5) for c in path], dtype=np.float32)
    # perturb the trajectory by clipped, gaussian noise
    perturb = rng.normal(0, 0.1, trajectory.shape).astype(np.float32).clip(-0.25, 0.25)
    trajectory += perturb
    spline = CubicSpline(np.arange(len(trajectory)) / len(trajectory), trajectory)
    # Evaluate the spline
    trajectory = spline(np.linspace(0, 1, trajectory_length))
    # Fit spline to trajectory
    return trajectory

rng = np.random.default_rng(seed=SEED)
for split in {"train", "test"}:
    with writer.split(split) as split_writer:
        logger.info(f"Generating split: {split}")
        for mn in range(NUM_MAZES):
            logger.info(f"Generating maze: {mn}")
            maze = generate_maze(rng)
            start = []
            end = []
            trajectories = []
            for i in range(mn, NUM_TRAJECTORIES, NUM_MAZES):
                logger.info(f"Generating trajectory: {(i - mn) // NUM_MAZES}")
                trajectory = generate_trajectory(rng, maze, TRAJECTORY_LENGTH)
                start.append([trajectory[0,0], trajectory[0,1]])
                end.append([trajectory[-1,0], trajectory[-1,1]])
                trajectories.append([[trajectory[i,0].item(), trajectory[i,1].item()] for i in range(len(trajectory))])
            table = pa.table([start, end, trajectories], schema=schema)
            for batch in table.to_batches():
                split_writer.write_batch(batch)
