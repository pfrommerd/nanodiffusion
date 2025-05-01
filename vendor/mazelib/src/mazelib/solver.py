import logging
import abc
import heapq

from dataclasses import dataclass, field
from .maze import Maze, Cell
from numpy.random import Generator as Rng

class Solver(abc.ABC):
    @abc.abstractmethod
    def solve(self, rng: Rng, maze: Maze,
                    start: Cell, end: Cell) -> list[Cell]:
        pass

@dataclass(order=True)
class QueueItem:
    distance: int
    parent: Cell = field(compare=False)
    cell: Cell = field(compare=False)

class DjikstraSolver(Solver):
    def solve(self, rng: Rng, maze: Maze, start: Cell,
                            end: Cell) -> list[Cell]:
        assert start is not None and end is not None
        visited = set()

        best_distances : dict[Cell, int] = {}
        # The best cell from which to reach a given cell
        best_from : dict[Cell, Cell] = {}

        best_distances[start] = 0
        queue : list[QueueItem] = [
            QueueItem(1, start, n) for n in start.neighbors
        ]
        heapq.heapify(queue)

        while not end in best_distances:
            best = heapq.heappop(queue)
            # If this is a cell we have already found
            # a better path to, skip it
            if best.cell in best_distances:
                continue
            best_distances[best.cell] = best.distance
            best_from[best.cell] = best.parent
            for n in best.cell.neighbors:
                if n in best_distances:
                    continue
                heapq.heappush(queue,
                    QueueItem(best.distance + 1, best.cell, n)
                )
        # Reconstruct the path
        path = []
        cell = end
        while cell != start:
            path.append(cell)
            cell = best_from[cell]
        path.append(start)
        path.reverse()
        return path
