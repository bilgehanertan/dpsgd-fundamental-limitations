# coding=utf-8
# Copyright 2025 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file is a modified version of the batch_selection.py file from the JAX Privacy library.
https://github.com/google-deepmind/jax_privacy/blob/main/jax_privacy/batch_selection.py
"""

import abc
import dataclasses
import enum
from typing import Iterator

import numpy as np

RngType = np.random.Generator | int | None


def pad_to_multiple_of(indices: np.ndarray, multiple: int) -> np.ndarray:
    """Pads the last dimension of indices to a multiple of multiple.

    Example Usage:
      >>> indices = np.arange(10)
      >>> pad_to_multiple_of(indices, multiple=4)
      array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, -1, -1])

    Args:
      indices: A 1D array of batch indices.
      multiple: A positive integer. The input batch will be padded to a multiple
        of this value.

    Returns:
      A new 1D array of indices padded with -1.
    """
    if indices.ndim > 1:
        raise ValueError("pad_to_multiple_of currently expects 1D indices.")
    curr_size = indices.shape[0]
    # Important corner case: if curr_size == 0 (maybe under Poisson sampling),
    # we still need a non-empty padded batch so downstream compiled code sees a
    # fixed shape.
    pad_size = multiple if curr_size == 0 else (multiple - curr_size) % multiple
    new_indices = np.full(curr_size + pad_size, -1, dtype=indices.dtype)
    new_indices[:curr_size] = indices
    return new_indices


# Exact copy from the JAX Privacy library
class BatchSelectionStrategy(abc.ABC):
    """Abstract base class for batch selection strategies.

    A batch selection strategy is a function that takes a random number generator
    and returns an iterator of batches of data indices. The strategy can
    either be deterministic or random, it may produce equal-sized batches or
    variable-sized batches. Note that the batches of indices, which
    specify which examples contribute in which iterations, are generally
    considered sensitive, and should not be inspected directly.

    This API does not prescribe a specific dataset format, but it is expected
    that the format used supports efficient random access to individual examples.
    """

    @abc.abstractmethod
    def batch_iterator(
        self, num_examples: int, rng: RngType = None
    ) -> Iterator[np.ndarray]:
        """Yields 1D batches of data indices."""


@dataclasses.dataclass(frozen=True)
class PoissonSubsampling(BatchSelectionStrategy):
    sampling_prob: float  # q in (0,1)
    iterations: int  # M rounds

    def batch_iterator(
        self, num_examples: int, rng: RngType = None
    ) -> Iterator[np.ndarray]:
        if not (0.0 < self.sampling_prob < 1.0):
            raise ValueError("sampling_prob must be in (0,1)")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if num_examples <= 0:
            raise ValueError("num_examples must be positive")

        g = np.random.default_rng(rng)
        dtype = np.min_scalar_type(-num_examples)

        for _ in range(self.iterations):
            mask = g.random(num_examples) < self.sampling_prob
            yield np.nonzero(mask)[0].astype(dtype)


@dataclasses.dataclass(frozen=True)
class FixedBatchShufflingSampling(BatchSelectionStrategy):
    batch_size: int
    epochs: int

    def batch_iterator(self, num_examples: int, rng: RngType = None):
        g = np.random.default_rng(rng)
        dtype = np.min_scalar_type(-num_examples)
        usable = (num_examples // self.batch_size) * self.batch_size
        if usable == 0:
            raise ValueError(
                f"batch_size={self.batch_size} > num_examples={num_examples}"
            )
        for _ in range(self.epochs):
            perm = g.permutation(num_examples).astype(dtype)[:usable]
            for i in range(0, usable, self.batch_size):
                yield perm[i : i + self.batch_size]
