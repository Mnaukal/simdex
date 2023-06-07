# Replay buffer by Milan Straka (NPFL122 course)

# We use a custom implementation instead of `collections.deque`, which has
# linear complexity of indexing (it is a two-way linked list). The following
# implementation has similar runtime performance as a numpy array of objects,
# but it has unnecessary memory overhead (hundreds of MBs for 1M elements).
# Using five numpy arrays (for state, action, reward, done, and next state)
# would provide minimal memory overhead, but it is not so flexible.

import numpy as np


class ReplayBuffer:
    """Simple replay buffer with possibly limited capacity."""
    def __init__(self, max_length=None):
        self._max_length = max_length
        self._data = []
        self._offset = 0

    def __len__(self):
        return len(self._data)

    @property
    def max_length(self):
        return self._max_length

    def append(self, item):
        if self._max_length is not None and len(self._data) >= self._max_length:
            self._data[self._offset] = item
            self._offset = (self._offset + 1) % self._max_length
        else:
            self._data.append(item)

    def extend(self, items):
        if self._max_length is None:
            self._data.extend(items)
        else:
            for item in items:
                if len(self._data) >= self._max_length:
                    self._data[self._offset] = item
                    self._offset = (self._offset + 1) % self._max_length
                else:
                    self._data.append(item)

    def __getitem__(self, index):
        assert -len(self._data) <= index < len(self._data)
        return self._data[(self._offset + index) % len(self._data)]

    def sample(self, size):
        # The same element can be sampled multiple times. However, making sure the samples
        # are unique is costly, and we do not mind the duplicites much during training.
        return [self._data[index] for index in np.random.randint(len(self._data), size=size)]
