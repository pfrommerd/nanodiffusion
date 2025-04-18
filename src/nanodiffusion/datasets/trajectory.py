# import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import random
import heapq
import torch.utils.data
from nanoconfig import config
from . import Sample, DataConfig, SampleDataset

from nanoconfig.experiment import Figure

class TrajectoryDataset(SampleDataset):
    def __init__(self, num_trajectories=50, points_per_trajectory=64, grid_size=6, margin=0.1,
                 allowed_cells=None, seed=0):
        """
        Args:
            num_trajectories: Number of trajectories in the dataset
            points_per_trajectory: Number of points to sample from each trajectory
            grid_size: Size of the grid (M x M)
            margin: Margin to shrink allowed regions by
            allowed_cells: List of (i, j) tuples representing allowed cells
        """
        self.T = num_trajectories
        self.N = points_per_trajectory
        self.M = grid_size
        self.margin = margin
        random.seed(seed)
        np.random.seed(seed)

        # Define allowed cells
        if allowed_cells is None:
            maze = [
                'ooxooo',
                'ooxoxo',
                'xoooox',
                'xxoxoo',
                'ooooxo',
                'ooxooo',
            ]
            self.allowed_cells = [(i, j) for i, row in enumerate(maze)
                                for j, c in enumerate(row)
                                if c == 'o']
        else:
            self.allowed_cells = allowed_cells

        # Generate trajectories
        self.trajectories = []  # Will store 2xN tensors
        self.labels = []        # Will store 2x2 tensors (start and end cell coordinates)

        self._generate_dataset()

    def __len__(self):
        return self.T

    def __getitem__(self, idx):
        return Sample(sample=self.trajectories[idx], cond=self.labels[idx], num_classes=None)

    def _get_cell_bounds(self, i, j):
        """Get the bounds of cell (i, j) in the [-1, 1] x [-1, 1] space"""
        cell_width = 2.0 / self.M
        x_min = -1.0 + j * cell_width
        x_max = x_min + cell_width
        y_min = 1.0 - (i + 1) * cell_width
        y_max = y_min + cell_width
        return x_min, x_max, y_min, y_max

    def _get_allowed_region(self, i, j):
        """Get the allowed region for cell (i, j) with margins applied"""
        x_min, x_max, y_min, y_max = self._get_cell_bounds(i, j)

        # Apply margins
        if (i, j-1) not in self.allowed_cells:  # Left neighbor not allowed
            x_min += self.margin
        if (i, j+1) not in self.allowed_cells:  # Right neighbor not allowed
            x_max -= self.margin
        if (i-1, j) not in self.allowed_cells:  # Top neighbor not allowed
            y_max -= self.margin
        if (i+1, j) not in self.allowed_cells:  # Bottom neighbor not allowed
            y_min += self.margin

        return x_min, x_max, y_min, y_max

    def _sample_point_in_cell(self, i, j):
        """Sample a random point within the allowed region of cell (i, j)"""
        x_min, x_max, y_min, y_max = self._get_allowed_region(i, j)
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return np.array([x, y])

    def _get_cell_from_point(self, point):
        """Get the cell indices (i, j) for a given point"""
        x, y = point
        cell_width = 2.0 / self.M

        j = int((x + 1.0) / cell_width)
        i = int((1.0 - y) / cell_width)

        # Clamp to valid indices
        i = max(0, min(i, self.M - 1))
        j = max(0, min(j, self.M - 1))

        return i, j

    def _find_shortest_path(self, start_cell, end_cell):
        """Find shortest path between two cells using A* algorithm"""
        # Define heuristic (Manhattan distance)
        def heuristic(cell1, cell2):
            return abs(cell1[0] - cell2[0]) + abs(cell1[1] - cell2[1])

        open_set = []
        heapq.heappush(open_set, (0, start_cell))
        came_from = {}
        g_score = {start_cell: 0}
        f_score = {start_cell: heuristic(start_cell, end_cell)}

        open_set_hash = {start_cell}

        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)

            if current == end_cell:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_cell)
                return path[::-1]

            # Check neighbors
            neighbors = []
            i, j = current
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                next_i, next_j = i + di, j + dj
                if (next_i, next_j) in self.allowed_cells:
                    neighbors.append((next_i, next_j))

            for neighbor in neighbors:
                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, end_cell)
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)

        # No path found
        return None

    def _bezier_curve(self, points, num_points=100):
        """Generate points along a Bezier curve defined by control points"""
        n = len(points) - 1
        t = np.linspace(0, 1, num_points)

        result = np.zeros((num_points, 2))
        for i, point in enumerate(points):
            bernstein = comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
            result += np.outer(bernstein, point)

        return result

    def _generate_smooth_trajectory(self, start_point, end_point, path):
        """Generate a smooth trajectory using piecewise Bezier curves"""
        if len(path) <= 2:
            # If path is just start and end, use a simple cubic Bezier
            control1 = start_point + 0.3 * (end_point - start_point) + np.random.normal(0, 0.1, 2)
            control2 = start_point + 0.7 * (end_point - start_point) + np.random.normal(0, 0.1, 2)
            bezier_points = self._bezier_curve([start_point, control1, control2, end_point], num_points=self.N)
            return bezier_points

        # Otherwise, create a piecewise Bezier curve
        intermediate_points = [self._sample_point_in_cell(i, j) for i, j in path[1:-1]]
        all_points = [start_point] + intermediate_points + [end_point]

        # Number of segments
        num_segments = len(all_points) - 1
        points_per_segment = max(self.N // num_segments, 10)

        # Generate segmented Bezier curves
        trajectory_points = []

        for i in range(num_segments):
            p0 = all_points[i]
            p3 = all_points[i + 1]

            # Direction vectors for tangents
            if i == 0:
                v_prev = p3 - p0
            else:
                v_prev = all_points[i] - all_points[i - 1]

            if i == num_segments - 1:
                v_next = p3 - p0
            else:
                v_next = all_points[i + 2] - all_points[i + 1]

            # Control points that ensure G1 continuity (tangent continuity)
            scale = 0.3 * np.linalg.norm(p3 - p0)
            p1 = p0 + scale * v_prev / (np.linalg.norm(v_prev) + 1e-8)
            p2 = p3 - scale * v_next / (np.linalg.norm(v_next) + 1e-8)

            # Generate Bezier curve for this segment
            segment_points = self._bezier_curve([p0, p1, p2, p3], num_points=points_per_segment)

            # Add points to trajectory, avoiding duplicates
            if i > 0:
                segment_points = segment_points[1:]  # Skip first point (it's the last of previous segment)

            trajectory_points.append(segment_points)

        # Concatenate all segments
        full_trajectory = np.vstack(trajectory_points)

        # Resample to get exactly N points
        if len(full_trajectory) >= self.N:
            indices = np.linspace(0, len(full_trajectory) - 1, self.N, dtype=int)
            return full_trajectory[indices]
        else:
            # If we have fewer points than needed, use interpolation
            t_old = np.linspace(0, 1, len(full_trajectory))
            t_new = np.linspace(0, 1, self.N)
            x_interp = np.interp(t_new, t_old, full_trajectory[:, 0])
            y_interp = np.interp(t_new, t_old, full_trajectory[:, 1])
            return np.column_stack((x_interp, y_interp))

    def _is_valid_trajectory(self, trajectory):
        """Check if a trajectory lies entirely within allowed regions"""
        for point in trajectory:
            i, j = self._get_cell_from_point(point)
            if (i, j) not in self.allowed_cells:
                return False

            # Check if point is within margins
            x_min, x_max, y_min, y_max = self._get_allowed_region(i, j)
            x, y = point
            if x < x_min or x > x_max or y < y_min or y > y_max:
                return False

        return True

    def _generate_dataset(self):
        """Generate the full dataset of trajectories"""
        while len(self.trajectories) < self.T:
            # Select random start and end cells
            start_cell = random.choice(self.allowed_cells)
            end_cell = random.choice(self.allowed_cells)

            if start_cell == end_cell:
                continue  # Skip if start and end are the same

            # Find shortest path between cells
            path = self._find_shortest_path(start_cell, end_cell)
            if path is None:
                continue  # Skip if no path found

            # Sample start and end points within the cells
            start_point = self._sample_point_in_cell(*start_cell)
            end_point = self._sample_point_in_cell(*end_cell)

            # Generate trajectory
            trajectory = self._generate_smooth_trajectory(start_point, end_point, path)

            # Validate trajectory
            if self._is_valid_trajectory(trajectory):
                # Convert to tensor and store
                traj_tensor = torch.tensor(trajectory, dtype=torch.float32).T
                label_tensor = torch.tensor([start_cell, end_cell], dtype=torch.float32)
                self.trajectories.append(traj_tensor)
                self.labels.append(label_tensor)

    def create_base_plot(self, ax):
        # Draw grid
        cell_width = 2.0 / self.M
        for i in range(self.M + 1):
            ax.axhline(1.0 - i * cell_width, color='gray', linestyle='-', alpha=0.3)
            ax.axvline(-1.0 + i * cell_width, color='gray', linestyle='-', alpha=0.3)

        # Highlight allowed cells
        for i, j in self.allowed_cells:
            x_min, x_max, y_min, y_max = self._get_cell_bounds(i, j)
            width = x_max - x_min
            height = y_max - y_min
            rect = plt.Rectangle((x_min, y_min), width, height, # type: ignore
                    color='lightgray', alpha=0.3)
            ax.add_patch(rect)

            # Show allowed region with margins
            x_min, x_max, y_min, y_max = self._get_allowed_region(i, j)
            width = x_max - x_min
            height = y_max - y_min
            rect = plt.Rectangle((x_min, y_min), width, height, # type: ignore
                color='lightblue', alpha=0.3)
            ax.add_patch(rect)

        # Set limits and labels
        ax.set_xlim((-1.05, 1.05))
        ax.set_ylim((-1.05, 1.05))
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Trajectories on 2D Grid')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        return ax


    def visualize_batch(self, samples : Sample):
        """Visualize trajectories on the grid"""
        fig, ax = plt.subplots()
        self.create_base_plot(ax)
        # Draw trajectories
        for i in range(samples.sample.shape[0]):
            traj = self.trajectories[i].numpy().T
            ax.plot(traj[:, 0], traj[:, 1], '-', lw=2, alpha=0.7)
            ax.plot(traj[0, 0], traj[0, 1], 'go', markersize=5)  # Start point
            ax.plot(traj[-1, 0], traj[-1, 1], 'ro', markersize=5)  # End point
        fig.tight_layout()
        return Figure(fig)

@config(variant="trajectory")
class TrajectoryDataConfig(DataConfig):
    num_trajectories: int = 4096
    points_per_trajectory: int = 64
    grid_size: int = 6
    margin: float = 0.1
    allowed_cells: list[tuple[int, int]] | None = None
    seed: int = 0

    def create(self) -> tuple[SampleDataset, SampleDataset]:
        test_data = TrajectoryDataset(
            num_trajectories=self.num_trajectories // 4,
            points_per_trajectory=self.points_per_trajectory,
            grid_size=self.grid_size,
            margin=self.margin,
            allowed_cells=self.allowed_cells,
            seed=self.seed
        )
        train_data = TrajectoryDataset(
            num_trajectories=self.num_trajectories,
            points_per_trajectory=self.points_per_trajectory,
            grid_size=self.grid_size,
            margin=self.margin,
            allowed_cells=self.allowed_cells,
            seed=self.seed
        )
        return train_data, test_data
