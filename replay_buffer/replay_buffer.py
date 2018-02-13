import numpy as np
from collections import deque

class ReplayBuffer(object):

    def __init__(self, state_shape, action_shape, maxlen):
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