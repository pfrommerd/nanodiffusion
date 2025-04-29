import itertools
import enum

class CellType(enum.Enum):
    ENTRY = "entry"
    EXIT = "exit"
    UNREACHABLE = "unreachable"
    REACHABLE = "reachable"

class Walls:
    def __init__(self, top: bool = True, right: bool = True, bottom: bool = True, left: bool = True):
        self.top = top
        self.right = right
        self.bottom = bottom
        self.left = left

    def copy(self):
        return Walls(self.top, self.right, self.bottom, self.left)

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
        self.type = CellType.UNREACHABLE
        self.walls = Walls()
        self.neighbors = list()

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

    def _remove_walls_entry_exit(self):
        if self.row == 0:
            self.walls.top = False
        elif self.row == self.maze.num_rows - 1:
            self.walls.bottom = False
        elif self.col == 0:
            self.walls.left = False
        elif self.col == self.maze.num_cols - 1:
            self.walls.right = False

    def set_type(self, type: CellType):
        if self.type == type:
            return
        if type == CellType.ENTRY or type == CellType.EXIT:
            self._remove_walls_entry_exit()
        # Remove from old type list
        self.maze.type_cells[self.type].remove(self)
        self.type = type
        self.maze.type_cells[self.type].append(self)

    def __hash__(self):
        return hash((id(self.maze), self.row, self.col))

    def __eq__(self, other):
        return (self.row == other.row and self.col == other.col
            and self.type == other.type and self.maze is other.maze)

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
        self.type_cells = {
            CellType.UNREACHABLE: list(itertools.chain.from_iterable(self.grid)),
            CellType.REACHABLE: [], CellType.ENTRY: [], CellType.EXIT: []
        }

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
                new_cell.type = cell.type
                new_cell.walls = cell.walls.copy()
                new_cell.neighbors = [
                    new_maze.grid[n.row][n.col] for n in cell.neighbors
                ]
        return new_maze
