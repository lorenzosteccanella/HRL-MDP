from collections import deque
from typing import Tuple, List

import numpy as np

class ExperienceReplay:
    """
    Experience Replay buffer, optimized for the specific HRL-MDP task.
    Stores pairs of consecutive states.
    """

    def __init__(self, max_size: int):
        """
        Initializes the Experience Replay v2 buffer.

        Args:
            max_size: Maximum number of experiences to store in the buffer.
        """
        self.buffer: deque = deque(maxlen=max_size)
        self.max_reward: float = float('-inf')  # Used? Not used

    def add(self, experience: Tuple[Tuple[np.ndarray, np.ndarray], int]) -> None:
        """
        Adds an experience (state pair and associated action) to the buffer.

        Args:
            experience: A tuple containing a pair of consecutive states (NumPy arrays) and the action taken.
        """
        self.buffer.append(experience)

    def sample(self, batch_size: int, random_sample: bool = True) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray]:
        """
        Samples a batch of experiences (state pairs) from the buffer.

        Args:
            batch_size: The number of experiences to sample.
            random_sample: Whether to sample randomly or sequentially.

        Returns:
            A tuple containing:
            - Indexes of the sampled experiences.
            - A list of the first states in each pair.
            - A list of the second states in each pair.
            - Importance weights (currently all ones).
        """
        buffer_size: int = len(self.buffer)
        if random_sample:
            indexes: np.ndarray = np.random.choice(np.arange(buffer_size), size=batch_size, replace=False)
        else:
            indexes: np.ndarray = np.arange(buffer_size)

        # Extract state pairs efficiently
        sample1: List[np.ndarray] = [self.buffer[i][0][0] for i in indexes]
        sample2: List[np.ndarray] = [self.buffer[i][0][1] for i in indexes]

        imp_w: np.ndarray = np.ones((batch_size, 1), dtype=np.float32)  # Importance weights (not used)

        return indexes, sample1, sample2, imp_w

    def buffer_len(self) -> int:
        """Returns the current length of the buffer."""
        return len(self.buffer)

    def reset_buffer(self) -> None:
        """Clears the buffer."""
        self.buffer.clear()

    def reset_size(self, max_size: int) -> None:
        """Resets the maximum size of the buffer."""
        self.buffer: deque = deque(maxlen=max_size)

    def update(self, idx: np.ndarray, error: np.ndarray) -> None:
        """
        Updates the priority of the experiences. (Not used in this implementation)

        Args:
            idx: Indexes of the experiences to update.
            error: The error associated with each experience.
        """
        pass  # Not implemented