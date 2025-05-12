import logging
import abc
import heapq

from dataclasses import dataclass, field
from .maze import Maze, Cell
from numpy.random import Generator as Rng

class Solver(abc.ABC):
    @abc.abstractmethod
    def solve(self, rng: Rng, maze: Maze,
                    start: Cell, end: Cell) -> list[list[Cell]]:
        pass

@dataclass(order=True)
class QueueItem:
    distance: int
    path: list[Cell] = field(compare=False)
    cell: Cell = field(compare=False)

class DjikstraSolver(Solver):
    def solve(self, rng: Rng, maze: Maze, start: Cell,
                end: Cell, all_paths: bool = False) -> list[list[Cell]]:
        assert start is not None and end is not None
        if start == end:
            return [[start]]
        visited: set[Cell] = set()
        visited.add(start)
        queue : list[QueueItem] = [
            QueueItem(1, [start], n) for n in start.reachable_neighbors
        ]
        heapq.heapify(queue)

        solutions = []
        while queue:
            best = heapq.heappop(queue)
            if not all_paths:
                if best.cell in visited:
                    continue
                visited.add(best.cell)
                proceed = lambda x: x not in visited
            else:
                if best.cell in best.path:
                    continue
                proceed = lambda n: n not in best.path
            # If we don't want all solutions, terminate as soon as a solution is found
            path = list(best.path)
            path.append(best.cell)
            if best.cell == end:
                solutions.append(path)
            if not all_paths and solutions:
                break
            # Put the neighbors onto the queue
            for n in best.cell.reachable_neighbors:
                if not proceed(n):
                    continue
                heapq.heappush(queue, QueueItem(best.distance + 1, path, n))
        return solutions
