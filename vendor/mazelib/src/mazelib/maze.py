import itertools
import enum
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from rich.segment import Segment

class Walls:
    def __init__(self, top: bool = True, right: bool = True, bottom: bool = True, left: bool = True):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

    def copy(self):
        return Walls(self.top, self.right, self.bottom, self.left)

    def __repr__(self):
        return f"Walls(top={self.top}, right={self.right}, bottom={self.bottom}, left={self.left})"

class Cell:
    """Class for representing a cell in a 2D grid.
        Attributes:
            row (int): The row that this cell belongs to
            col (int): The column that this cell belongs to
            visited (bool): True if this cell has been visited by an algorithm
            active (bool):
            type (str): Type of cell.
            walls (list):
            neighbours (list):
    """
    def __init__(self, maze: "Maze", row: int , col: int):
        self.maze = maze
        self.row = row
        self.col = col
        self.walls = Walls()
        self.neighbors = list()

    @property
    def loc(self) -> tuple[int, int]:
        return (self.col, self.row)

    @property
    def all_neighbors(self):
        def _generate():
            if self.row > 0:
                yield self.maze.grid[self.row - 1][self.col]
            if self.row < self.maze.num_rows - 1:
                yield self.maze.grid[self.row + 1][self.col]
            if self.col > 0:
                yield self.maze.grid[self.row][self.col - 1]
            if self.col < self.maze.num_cols - 1:
                yield self.maze.grid[self.row][self.col + 1]
        return list(_generate())

    def is_neighbor(self, neighbor: "Cell"):
        """Function that checks if there are walls between self and a neighbour cell.
        Returns true if there are walls between. Otherwise returns False.

        Args:
            neighbour The cell to check between
        Return:
            True: If there are walls in between self and neighbor
            False: If there are no walls in between the neighbors and self
        """
        return (neighbor.maze is self.maze) and neighbor in self.neighbors

    def remove_walls(self, neighbor: "Cell"):
        """Function that removes walls between neighbor cell given by indices in grid.
        """
        assert neighbor.maze is self.maze

        nr, nc = neighbor.row, neighbor.col
        if self.row == nr + 1 and self.col == nc and self.walls.top:
            self.walls.top = False
            neighbor.walls.bottom = False
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)
            return True
        elif self.row == nr - 1 and self.col == nc and self.walls.bottom:
            self.walls.bottom = False
            neighbor.walls.top = False
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)
            return True
        elif self.row == nr and self.col == nc + 1 and self.walls.left:
            self.walls.left = False
            neighbor.walls.right = False
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)
            return True
        elif self.row == nr and self.col == nc - 1 and self.walls.right:
            self.walls.right = False
            neighbor.walls.left = False
            self.neighbors.append(neighbor)
            neighbor.neighbors.append(self)
            return True
        return False

    # def _remove_walls_entry_exit(self):
    #     if self.row == 0:
    #         self.walls.top = False
    #     elif self.row == self.maze.num_rows - 1:
    #         self.walls.bottom = False
    #     elif self.col == 0:
    #         self.walls.left = False
    #     elif self.col == self.maze.num_cols - 1:
    #         self.walls.right = False

    # def set_type(self, type: CellType):
    #     if self.type == type:
    #         return
    #     if type == CellType.ENTRY or type == CellType.EXIT:
    #         self._remove_walls_entry_exit()
    #     # Remove from old type list
    #     self.maze.type_cells[self.type].remove(self)
    #     self.type = type
    #     self.maze.type_cells[self.type].append(self)

    def __hash__(self):
        return hash((id(self.maze), self.row, self.col))

    def __eq__(self, other):
        return (self.row == other.row and self.col == other.col
                    and self.maze is other.maze)

class Maze:
    def __init__(self, num_rows, num_cols):
        """Creates a Maze with all walls filled in."""
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.grid_size = num_rows*num_cols
        self.generation_path = []
        self.solution_path = None
        self.grid = [
            [Cell(self, i, j) for j in range(num_cols)]
            for i in range(num_rows)
        ]

    @property
    def cells(self) -> list[Cell]:
        return list(itertools.chain.from_iterable(self.grid))

    @property
    def total_cells(self) -> int:
        return self.num_rows * self.num_cols

    @property
    def edge_cells(self) -> list[Cell]:
        edge_cells = []
        for c in self.grid[0]:
            edge_cells.append(c)
        for c in self.grid[self.num_rows - 1]:
            edge_cells.append(c)
        for r in range(1, self.num_rows - 1):
            edge_cells.append(self.grid[r][0])
        for r in range(1, self.num_rows - 1):
            edge_cells.append(self.grid[r][self.num_cols - 1])
        return edge_cells

    def copy(self) -> "Maze":
        new_maze = Maze(self.num_rows, self.num_cols)
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                cell = self.grid[i][j]
                new_cell = new_maze.grid[i][j]
                new_cell.walls = cell.walls.copy()
                new_cell.neighbors = [
                    new_maze.grid[n.row][n.col] for n in cell.neighbors
                ]
        return new_maze

    def render_matplotlib(self, ax):
        pass

    def render_plotly(self,
                scale : tuple[int,int] | np.ndarray | None = None,
                mean: tuple[int,int] | np.ndarray | None = None) -> go.Trace:
        segments = []
        for c in self.grid[0]:
            loc = c.loc*np.array([1, -1])
            if c.walls.top: segments.append([loc, loc + np.array([1, 0])])
        for (c, *_) in self.grid:
            loc = c.loc*np.array([1, -1])
            if c.walls.left: segments.append([loc, loc - np.array([0, 1])])
        for c in self.cells:
            loc = c.loc*np.array([1, -1])
            if c.walls.bottom:
                segments.append([loc + np.array([0, -1]), loc + np.array([1, -1])])
            if c.walls.right:
                segments.append([loc + np.array([1, 0]), loc + np.array([1, -1])])
        segments = np.array(segments)
        if scale is not None:
            segments = (segments / np.array([self.num_cols, self.num_rows]))*np.array(scale)
        if mean is not None:
            segments = segments - np.array([1,-1])*np.array(scale)/2 + np.array(mean)
        aug = np.tile(np.array(None)[None,None,None],
                (segments.shape[0], 1, segments.shape[2]))
        segments = np.concatenate((segments, aug), axis=1, dtype=object)
        xs = list(segments[:, :, 0].flatten())
        ys = list(segments[:, :, 1].flatten())
        return go.Scatter(x=xs, y=ys,
            mode="lines", line=dict(color='black')) # type: ignore

    def to_numpy_onehot(self) -> np.ndarray:
        left = [[c.walls.left for c in row] for row in self.grid]
        right = [[c.walls.right for c in row] for row in self.grid]
        top = [[c.walls.top for c in row] for row in self.grid]
        bottom = [[c.walls.bottom for c in row] for row in self.grid]
        return np.stack([left, right, top, bottom], axis=-1)

    @staticmethod
    def from_numpy_onehot(array: np.ndarray) -> "Maze":
        assert array.ndim == 3
        assert array.shape[2] == 4
        num_rows, num_cols = array.shape[:2]
        maze = Maze(num_rows, num_cols)
        for i in range(num_rows):
            for j in range(num_cols):
                cell = maze.grid[i][j]
                cell.walls.left = array[i, j, 0]
                cell.walls.right = array[i, j, 1]
                cell.walls.top = array[i, j, 2]
                cell.walls.bottom = array[i, j, 3]
        return maze
