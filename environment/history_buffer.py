from collections import deque

import numpy as np


class HistoryBuffer(object):
    def __init__(self, history_len):
        self.shapes = None
        self._buffers = None
        self._history_len = history_len

    def update(self, *args):
        if self.shapes is None:
            self.shapes = [np.shape(arg) for arg in args]
            self.reset()

        assert len(args) == len(self._buffers) == len(self.shapes)
        for arg, shape in zip(args, self.shapes):
            assert arg.shape == shape
        for arg, buffer in zip(args, self._buffers):
            buffer += [arg]

    def reset(self):
        if self.shapes is not None:
            def initialize(shape):
                return deque(
                    [np.zeros(shape) for _ in range(self._history_len)],
                    maxlen=self._history_len)

            self._buffers = list(map(initialize, self.shapes))

    def get(self):
        if self.shapes is None:
            raise RuntimeError(
                "Shapes not specified. Call `update` before calling `get`.")
        return tuple(
            np.concatenate(buffer, axis=-1) for buffer in self._buffers)