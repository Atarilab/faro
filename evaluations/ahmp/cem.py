"""Discrete part of AHMP CEM-MD; see UPSTREAM_LICENSE.

Adapted from hucebot/ahmp src/cem/algo.py at
f6972a298c843ffd3e0c3f6f921af007e787ad3c (Copyright 2025 Inria).
There are deliberately no Gaussian variables, smoothing, or elite injection.
"""
from dataclasses import dataclass
from numbers import Integral

import numpy as np

UPSTREAM_COMMIT = "f6972a298c843ffd3e0c3f6f921af007e787ad3c"


@dataclass(frozen=True)
class CEMConfig:
    population_size: int = 8
    elite_size: int = 4
    min_prob: float = 0.05
    seed: int = 0

    def __post_init__(self):
        for name in ("population_size", "elite_size", "seed"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative")
        if not 1 <= self.elite_size <= self.population_size:
            raise ValueError("Require 1 <= elite_size <= population_size")
        if not np.isfinite(self.min_prob) or self.min_prob < 0:
            raise ValueError("min_prob must be finite and nonnegative")


class DiscreteCEM:
    """Independent categorical distribution per phase, as in AHMP.

    Arrays are (phase, individual), matching upstream. The probability offset
    is added *before* normalization; it is not a lower probability bound.
    """

    def __init__(self, n_values, config: CEMConfig):
        self.n_values = tuple(n_values)
        if not self.n_values or any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1
                                    for n in self.n_values):
            raise ValueError("Every phase needs at least one category")
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.probs = [np.full(n, 1.0 / n) for n in self.n_values]

    def sample(self) -> np.ndarray:
        population = np.empty((len(self.n_values), self.config.population_size), dtype=int)
        # Preserve upstream's individual-major order of random draws.
        cumulative = [np.cumsum(p) for p in self.probs]
        for i in range(self.config.population_size):
            for j, cdf in enumerate(cumulative):
                population[j, i] = min(
                    np.searchsorted(cdf, self.rng.random(), side="right"), len(cdf) - 1)
        return population

    def update(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        if population.shape != (len(self.n_values), self.config.population_size):
            raise ValueError("Unexpected population shape")
        fitness = np.asarray(fitness, dtype=float)
        if fitness.shape != (self.config.population_size,) or np.isnan(fitness).any():
            raise ValueError("Expected one non-NaN fitness per individual")
        if not np.issubdtype(population.dtype, np.integer) or any(
                np.any((population[j] < 0) | (population[j] >= n))
                for j, n in enumerate(self.n_values)):
            raise ValueError("Population contains invalid category IDs")
        # Pruned candidates are absent from selection, even when fewer than
        # elite_size candidates survive. Preserve upstream's finite-score ties.
        ranked = np.argsort(fitness)[::-1]
        elites = ranked[np.isfinite(fitness[ranked])][:self.config.elite_size]
        if not len(elites):
            return elites
        for j, n in enumerate(self.n_values):
            counts = np.bincount(population[j, elites], minlength=n)
            probabilities = counts / len(elites) + self.config.min_prob
            self.probs[j] = probabilities / probabilities.sum()
        return elites
