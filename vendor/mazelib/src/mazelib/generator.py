import abc
import logging
import time

from numpy.random import Generator as Rng

from .maze import Maze, Cell

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
        initial_cells = maze.cells
        current_cell = initial_cells[rng.choice(len(initial_cells))]
        visited.add(current_cell)

        iteration = 0
        while len(visited) < maze.total_cells:
            iteration = iteration + 1
            # All unreachable, unvisited neighbors of this cell
            new_neighbors = list(n for n in current_cell.all_neighbors if n not in visited)
            if new_neighbors:
                visited_stack.append(current_cell)              # Add current cell to stack
                neighbor_idx = rng.choice(len(new_neighbors))     # Choose random neighbour
                next_cell = new_neighbors[neighbor_idx]
                visited.add(next_cell)
                current_cell.remove_walls(next_cell)
                current_cell = next_cell
            elif len(visited_stack) > 0:
                # Go up on the stack
                current_cell = visited_stack.pop()
            if iteration > maze.total_cells*maze.total_cells:
                raise TimeoutError("Maze generation timed out")
        gen_time = time.time() - time_start
        logger.info(f"Maze generation took: {gen_time:.4f}")
        return maze
