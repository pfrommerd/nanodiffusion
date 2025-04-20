import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import plotly.tools as tls

from . import Sample, DataConfig, SampleDataset

from nanoconfig.experiment import Figure

from nanoconfig import config, experiment

@config(variant="tree")
class TreeDataConfig(DataConfig):
    branching_factor: int = 4
    depth: int = 3
    num_samples_per_path: int = 60

    def create(self) -> tuple[SampleDataset, SampleDataset]:
        """
        Create the tree dataset from the config.
        :return: The dataset.
        """
        train_data = TreeDataset(
            branching_factor=self.branching_factor,
            depth=self.depth,
            num_samples_per_path=self.num_samples_per_path
        )
        test_data = TreeDataset(
            branching_factor=self.branching_factor,
            depth=self.depth,
            num_samples_per_path=self.num_samples_per_path
        )
        return train_data, test_data

def interpolate_polyline(points, num_samples):
    """
    Given a list of 2D points defining a polyline,
    sample num_samples points uniformly along its arc length.
    """
    points = np.array(points)
    # Compute distances between consecutive points
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumdist = np.concatenate(([0], np.cumsum(dists)))
    total_length = cumdist[-1]
    # Equally spaced arc-length values
    sample_dists = np.linspace(0, total_length, num_samples)
    samples = []
    for d in sample_dists:
        # Find which segment d falls in
        seg = np.searchsorted(cumdist, d, side='right') - 1
        seg = min(seg, len(dists) - 1)
        # Compute local interpolation parameter
        t = (d - cumdist[seg]) / dists[seg] if dists[seg] > 0 else 0
        sample = (1 - t) * points[seg] + t * points[seg + 1]
        samples.append(sample)
    return np.array(samples)

class TreeDataset(SampleDataset):
    def __init__(self, branching_factor=4, depth=3, num_samples_per_path=60):
        """
        Initializes a tree dataset where each leaf of the tree lies on the
        circle of radius 1. The tree is constructed with the given branching_factor
        and depth. Each leaf’s path is sampled uniformly, and each sampled point
        is given the label of the leaf.

        Parameters:
         - branching_factor (int): number of branches at each node.
         - depth (int): number of branchings (excluding the root).
                        Total leaves = branching_factor ** depth.
         - num_samples_per_path (int): number of points sampled along each path.
        """
        self.branching_factor = branching_factor
        self.depth = depth
        self.num_samples_per_path = num_samples_per_path
        self.total_leaves = branching_factor ** depth

        # Iterate over each leaf index
        samples = []
        labels = []
        for i in range(self.total_leaves):
            # Build the sequence of nodes along the path from the root to this leaf.
            # Start with the root at (0, 0)
            path_points = [np.array([0.0, 0.0])]

            # For each level l (1 to depth), compute the branch node.
            for l in range(1, depth + 1):
                # Group size for this level (leaves per branch node)
                group_size = branching_factor ** (depth - l)  # For l == depth, group_size == 1.
                # A_l is the branch index for level l
                A_l = i // group_size
                # Compute the average index for all leaves under this branch node
                avg_index = A_l * group_size + (group_size - 1) / 2.0
                # Compute angular coordinate (all leaves are uniformly spaced on the circle)
                theta = avg_index * (2 * np.pi / self.total_leaves)
                # Set radius proportional to the level (leaf at level==depth has r==1)
                r = l / depth
                p = np.array([r * np.cos(theta), r * np.sin(theta)])
                path_points.append(p)

            # Sample points uniformly along the polyline defined by the path
            samples.extend(interpolate_polyline(path_points, num_samples_per_path))
            labels.extend([i] * num_samples_per_path)
        samples = np.array(samples)
        labels = np.array(labels)
        self.samples = torch.tensor(samples, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.int)

    @property
    def in_memory(self) -> bool:
        return True

    def visualize_batch(self, samples: Sample):
        cond = samples.cond.cpu().numpy() if samples.cond is not None else None
        cond = cond.astype(float) if cond is not None else None
        sample = samples.sample.cpu().numpy()
        fig, ax = plt.subplots()
        ax.scatter(sample[:, 0], sample[:, 1], c=cond, cmap='viridis', s=5)
        figure = tls.mpl_to_plotly(fig)
        plt.close(fig)
        return Figure(figure)

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        cond = self.labels[idx]
        sample = self.samples[idx]
        return Sample(cond=cond, sample=sample,
                      num_classes=self.total_leaves)
