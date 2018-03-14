import numpy as np
from collections import deque

class ReplayBuffer(object):

    def __init__(self, maxlen):
        self.S1 = deque(maxlen=maxlen)
        self.S2 = deque(maxlen=maxlen)
        self.A = deque(maxlen=maxlen)
        self.R = deque(maxlen=maxlen)
        self.T = deque(maxlen=maxlen)

    def append(self, s1, a, r, s2, t):
        self.S1.append(s1)
        self.A.append(a)
        self.R.append(r)
        self.S2.append(s2)
        self.T.append(t)

    def sample(self, batch_size):
        indices = np.random.randint(0, len(self.S1), size=batch_size)
        S1_sample = []
        A_sample = []
        R_sample = []
        S2_sample = []
        T_sample = []
        for idx in list(indices):
            S1_sample.append(self.S1[idx])
            A_sample.append(self.A[idx])
            R_sample.append(self.R[idx])
            S2_sample.append(self.S2[idx])
            T_sample.append(self.T[idx])
        return S1_sample, A_sample, R_sample, S2_sample, T_sample

    def __len__(self):
        return len(self.S1)


class RollingBuffer(object):

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.rolling_buffer = [None for _ in range(maxlen)]
        self.pos = 0
        self.full = False

    def append(self, x):
        self.rolling_buffer[self.pos] = x
        self.pos += 1
        if self.pos >= self.maxlen:
            self.full = True
            self.pos = 0

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


class ReplayBuffer2(object):

    def __init__(self, maxlen):
        self.buffer = RollingBuffer(maxlen)

    def append(self, s1, a, r, s2, t):
        self.buffer.append((s1, a, r, s2, t))

    def sample(self, batch_size):
        sample = self.buffer.sample(batch_size)
        S1 = [x[0] for x in sample]
        A = [x[1] for x in sample]
        R = [x[2] for x in sample]
        S2 = [x[3] for x in sample]
        T = [x[4] for x in sample]
        return S1, A, R, S2, T

    def __len__(self):
        return len(self.buffer)


