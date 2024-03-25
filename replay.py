import math
from collections import deque
import numpy as np

class ExperienceReplay:

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.max_reward = float('-inf')

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, random=True):
        buffer_size = len(self.buffer)
        if random:
            indexes = np.random.choice(np.arange(buffer_size),
                                     size=batch_size,
                                     replace=False)
        else:
            indexes = np.arange(buffer_size)

        sample = [self.buffer[i] for i in indexes]

        imp_w = np.ones((batch_size, 1), dtype=np.float32)

        return indexes, sample, imp_w

    def buffer_len(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer.clear()

    def reset_size(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def update(self, idx, error):
        None


class ExperienceReplay_v2:

    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)
        self.max_reward = float('-inf')

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size, random=True):
        buffer_size = len(self.buffer)
        if random:
            indexes = np.random.choice(np.arange(buffer_size),
                                     size=batch_size,
                                     replace=False)
        else:
            indexes = np.arange(buffer_size)

        sample1 = [self.buffer[i][0][0] for i in indexes]
        sample2 = [self.buffer[i][0][1] for i in indexes]

        imp_w = np.ones((batch_size, 1), dtype=np.float32)

        return indexes, sample1, sample2, imp_w

    def buffer_len(self):
        return len(self.buffer)

    def reset_buffer(self):
        self.buffer.clear()

    def reset_size(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def update(self, idx, error):
        None