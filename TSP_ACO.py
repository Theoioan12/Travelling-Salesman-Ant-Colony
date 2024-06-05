import random

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from itertools import product, pairwise
from typing import List


class TSPFileReader:
    HEADER = 0
    NODE_COORD_SECTION = 1

    def __init__(self, filename: str):
        self.filename = filename
        self.points = {}
        self.state = self.HEADER
        self.dist_matrix = None

    def read(self):
        """
        Read the file and store the points in a dictionary
        """
        with open(self.filename, 'r') as f:
            for line in f:
                if self.state == self.HEADER:
                    if line.startswith("NODE_COORD_SECTION"):
                        self.state = self.NODE_COORD_SECTION
                elif self.state == self.NODE_COORD_SECTION:
                    if not line.startswith("EOF"):
                        fields = line.split()
                        self.points[int(fields[0]) - 1] = (float(fields[1]), float(fields[2]))

        assert self.state == self.NODE_COORD_SECTION, "File format error"
        self._calculate_distance_matrix()

    def calculate_distance(self, i: int, j: int) -> float:
        """
        Calculate the distance between two points
        Args:
            i, j: the index of the points
        """
        assert self.state == self.NODE_COORD_SECTION, "Please read the file first"

        return ((self.points[i][0] - self.points[j][0]) ** 2 + (self.points[i][1] - self.points[j][1]) ** 2) ** 0.5

    def path_distance(self, path: List[int]) -> float:
        """
        Calculate the distance of a circular path. The last point will be connected to the first point
        Args:
            path: a list of points, the last point will be connected to the first point
        """
        assert self.state == self.NODE_COORD_SECTION, "Please read the file first"

        mask = np.array(list(pairwise(path + [path[0]])))
        return self.dist_matrix[mask[:, 0], mask[:, 1]].sum()

    def n_points(self) -> int:
        """
        Return the number of points
        """
        assert self.state == self.NODE_COORD_SECTION, "Please read the file first"
        return len(self.points)

    def draw_path(self, path: List[int], axs: plt.Axes):
        """
        Draw the path using networkx
        """
        assert self.state == self.NODE_COORD_SECTION, "Please read the file first"

        G = nx.DiGraph()
        G.add_nodes_from(self.points.keys())
        for i in range(len(path) - 1):
            G.add_edge(path[i], path[i + 1])
        G.add_edge(path[-1], path[0])

        nx.set_node_attributes(G, "RoyalBlue", "color")
        G.nodes[0]["color"] = "Crimson"
        nx.draw(G, pos=self.points, with_labels=False, node_size=10, node_color=[G.nodes[i]["color"] for i in G.nodes],
                ax=axs)

    def get_points_ids(self) -> List[int]:
        """
        Return the ids of the points
        """
        assert self.state == self.NODE_COORD_SECTION, "Please read the file first"
        return list(self.points.keys())

    def get_distance_matrix(self) -> np.array:
        """
        Return the distance matrix
        """
        assert self.state == self.NODE_COORD_SECTION, "Please read the file first"
        return self.dist_matrix

    def _calculate_distance_matrix(self):
        """
        Calculate the distance matrix
        """
        self.dist_matrix = (np.array([self.calculate_distance(i, j) for i, j in product(self.points.keys(), repeat=2)]).
                            reshape(self.n_points(), self.n_points()))


class GreedySolver:

    def __init__(self, tsp: TSPFileReader) -> None:
        self.tsp = tsp

    def solve(self) -> List[int]:
        solution = [0]
        dist_matrix = self.tsp.get_distance_matrix()

        for _ in range(self.tsp.n_points() - 1):
            last_point = solution[-1]
            mask = np.array([i for i in range(self.tsp.n_points()) if i not in solution])
            distances = dist_matrix[last_point, mask]
            next_point = mask[np.argmin(distances)]
            solution.append(next_point)

        return solution


tsp = TSPFileReader("rat195.tsp")
tsp.read()

solver = GreedySolver(tsp)
path = solver.solve()

fig, axs = plt.subplots(figsize=(10, 10))
axs.set_title(f"Path length {tsp.path_distance(path)}")
tsp.draw_path(path, axs=axs)


class AntColonyOptimizer:
    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)
        for i in range(self.n_iterations):
            all_paths = self.construct_paths()
            self.spread_pheromone(all_paths, self.n_best, shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone *= self.decay
        return all_time_shortest_path

    def construct_paths(self):
        all_paths = []
        for _ in range(self.n_ants):
            path = [random.randint(0, len(self.distances) - 1)]
            while len(path) < len(self.distances):
                move = self.probabilistic_next_step(path)
                path.append(move)
            path.append(path[0])  # Completing the circuit
            all_paths.append((path, self.path_cost(path)))
        return all_paths

    def probabilistic_next_step(self, current_path):
        current_loc = current_path[-1]
        choices = list(set(self.all_inds) - set(current_path))
        probs = [self.pheromone[current_loc][next_loc] * (1.0 / self.distances[current_loc][next_loc]) ** self.beta
                 for next_loc in choices]
        probs = probs / np.sum(probs)
        next_step = np.random.choice(choices, p=probs)
        return next_step

    def spread_pheromone(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in zip(path[:-1], path[1:]):
                self.pheromone[move] += 1.0 / self.distances[move]

    def path_cost(self, path):
        return sum([self.distances[path[i]][path[i + 1]] for i in range(len(path) - 1)])


# Example usage:
tsp = TSPFileReader("rat195.tsp")
tsp.read()
aco = AntColonyOptimizer(distances=tsp.get_distance_matrix(), n_ants=100, n_best=5, n_iterations=100, decay=0.95,
                         alpha=-1, beta=2)
best_path = aco.run()
print(f"Best path: {best_path[0]} with distance: {best_path[1]}")
