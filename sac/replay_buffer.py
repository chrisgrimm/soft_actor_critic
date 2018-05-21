from collections import deque

import numpy as np


class RollingBuffer:
    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.rolling_buffer = [None for _ in range(maxlen)]
        self.pos = 0
        self.full = False
        self.empty = True

    def append(self, x):
        self.empty = False
        self.rolling_buffer[self.pos] = x
        self.pos += 1
        if self.pos >= self.maxlen:
            self.full = True
            self.pos = 0

    def extend(self, xs):
        for x in xs:
            self.append(x)

    def sample(self, batch_size):
        top_pos = self.maxlen if self.full else self.pos
        indices = np.random.randint(0, top_pos, size=batch_size)
        samples = []
        for idx in indices:
            sample = self.rolling_buffer[idx]
            samples.append(sample)
        return samples

    def __len__(self):
        return self.maxlen if self.full else self.pos


class ReplayBuffer(RollingBuffer):
    def sample(self, batch_size):
        sample = super().sample(batch_size)
        return tuple(map(list, zip(*sample)))
