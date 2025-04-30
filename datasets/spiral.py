#!/usr/bin/env python3

import hashlib
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num", type=int, default=4096)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--sha256", action="store_true")
parser.add_argument("--output-dir", type=str, default=None)
args = parser.parse_args()

VERSION = "0.0.1"
NUM_POINTS = args.num
SEED = args.seed
SHA = hashlib.sha256((f"spiral-{VERSION}" +
    "-".join(f"{k}={v}" for k, v in {
        "num_points": NUM_POINTS,
        "seed": SEED,
    }.items())
).encode()).hexdigest()

if args.sha256:
    print(SHA)
    sys.exit()
if not args.output_dir:
    print("No output directory specified!")
    sys.exit(1)

from nanoconfig.data.fs import FsDataWriter
from nanoconfig.data.utils import as_array

from numpy.random import Generator as Rng
from scipy.interpolate import CubicSpline

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import random
import logging

writer = FsDataWriter(args.output_dir, SHA)

schema = pa.schema([
    pa.field("x", pa.float32()),
    pa.field("y", pa.float32()),
], metadata={"mime_type": "parquet/planar"})

rng = np.random.default_rng(seed=SEED)
for split in {"train", "test"}:
    with writer.split(split) as split_writer:
        print(f"Generating split: {split}")
        s = rng.uniform(0, 4*np.pi, (NUM_POINTS,))
        points = s[:,None]*np.stack((np.cos(s), np.sin(s)), axis=-1) / (4*np.pi)
        points = points + rng.normal(scale=0.02, size=points.shape)
        x, y = points[:,0], points[:,1]
        for batch in pa.table([x, y], schema=schema).to_batches():
            split_writer.write_batch(batch)
