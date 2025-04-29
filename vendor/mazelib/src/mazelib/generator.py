import abc
import logging
import time

from numpy.random import Generator as Rng

from .maze import Maze, Cell, CellType

logger = logging.getLogger(__name__)

class MazeGenerator(abc.ABC):
    @abc.abstractmethod
    def generate(self, rng : Rng, initial_maze: Maze) -> Maze:
        pass

class DepthFirstGenerator(MazeGenerator):
    def generate(self, rng : Rng, initial_maze: Maze) -> Maze:
        maze = initial_maze.copy()
        visited : set[Cell] = set()
        visited_stack : list[Cell] = list()

        logger.info("Generating the maze with depth-first search...")
        time_start = time.time()
        # pick an initial, reachable cell to start from
        # if the maze is partially generated
        if maze.type_cells[CellType.REACHABLE]:
            initial_cells = maze.type_cells[CellType.REACHABLE]
        else:
            # If there are no existing reachable cells, pick any cell
            initial_cells = maze.cells
        current_cell = initial_cells[rng.choice(len(initial_cells))]
        visited.add(current_cell)

        iteration = 0
        while len(visited) < maze.total_cells:
            iteration = iteration + 1
            # All unreachable, unvisited neighbors of this cell
            new_neighbors = list(n for n in current_cell.all_neighbors
                if n not in visited and n.type == CellType.UNREACHABLE)
            if new_neighbors:
                visited_stack.append(current_cell)              # Add current cell to stack
                neighbor_idx = rng.choice(len(new_neighbors))     # Choose random neighbour
                next_cell = new_neighbors[neighbor_idx]
                next_cell.set_type(CellType.REACHABLE)
                visited.add(next_cell)
                current_cell.remove_walls(next_cell)
                current_cell = next_cell
            elif len(visited_stack) > 0:
                # Go up on the stack
                current_cell = visited_stack.pop()
            if iteration > maze.total_cells*maze.total_cells:
                raise TimeoutError("Maze generation timed out")
        gen_time = time.time() - time_start
        # Generate start and end cells
        if not maze.type_cells[CellType.ENTRY] and not maze.type_cells[CellType.EXIT]:
            edge_cells = maze.edge_cells
            entry_idx, exit_idx = rng.choice(len(edge_cells), 2, replace=False)
            entry_cell = edge_cells[entry_idx]
            exit_cell = edge_cells[exit_idx]
            entry_cell.set_type(CellType.ENTRY)
            exit_cell.set_type(CellType.EXIT)
        logger.info(f"Maze generation took: {gen_time:.4f}")
        return maze
