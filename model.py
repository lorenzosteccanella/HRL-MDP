import torch
from torch import nn
import numpy as np
import random


class SoftClusterNetwork(nn.Module):
    """
    A neural network that learns a soft clustering of the environment's states.
    It can handle both dense (position-based) and image-based observations.
    """

    def __init__(self, out_shape: int, w: int = None, h: int = None, device: torch.device = None):
        """
        Initializes the SoftClusterNetwork.

        Args:
            out_shape: The number of abstract states (clusters).
            w: Width of the input image (if image-based observations are used).
            h: Height of the input image (if image-based observations are used).
            device: The device to use for computation (CPU or GPU). Defaults to GPU if available, else CPU.
        """
        super().__init__()

        self.n_abstract_state: int = out_shape  # Number of abstract states
        self.device: torch.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu") # Auto select device
        if w is not None and h is not None:
            self.image_init(out_shape, w, h)
        else:
            self.dense_init(out_shape, w, h)  # Initialize layers for dense input

        self.to(self.device)  # Move the entire model to the specified device


    def dense_init(self, out_shape: int, w: int = None, h: int = None) -> None:
        """Initializes layers for dense (position-based) input."""
        self.fc1 = nn.Linear(2, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, out_shape)


    def image_init(self, out_shape: int, w: int, h: int) -> None:
        """Initializes convolutional layers for image-based input."""
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        # Calculate the output size of the convolutional layers
        def conv2d_size_out(size: int, kernel_size: int = 1, stride: int = 1, padding: int = 0, dilation: int = 1) -> int:
            return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

        convw: int = conv2d_size_out(conv2d_size_out(conv2d_size_out(w, kernel_size=1), kernel_size=5, padding=1), kernel_size=3, padding=1)
        convh: int = conv2d_size_out(conv2d_size_out(conv2d_size_out(h, kernel_size=1), kernel_size=5, padding=1), kernel_size=3, padding=1)
        linear_input_size: int = convw * convh * 32

        self.head = nn.Linear(linear_input_size, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, out_shape)


    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the network.

        Args:
            obs: The input observation (image or dense vector).  Shape: (batch_size, channels, height, width) for images or (batch_size, feature_dim) for dense vectors.

        Returns:
            The logits (unnormalized probabilities) for each abstract state. Shape: (batch_size, n_abstract_states)
        """
        if len(obs.shape) == 4:  # Assuming image input
            return self.image_forward(obs)
        else:  # Assuming dense input
            return self.dense_forward(obs)


    def dense_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for dense input."""
        obs = torch.flatten(obs, start_dim=1)
        x = torch.selu(self.bn1(self.fc1(obs)))
        x = torch.selu(self.bn2(self.fc2(x)))
        x = torch.selu(self.bn3(self.fc3(x)))
        logits = self.fc4(x)
        return logits


    def image_forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass for image input."""
        obs = obs.permute(0, 3, 1, 2)  # Change to (B, C, H, W)
        x = torch.selu(self.bn1(self.conv1(obs)))
        x = torch.selu(self.bn2(self.conv2(x)))
        x = torch.selu(self.bn3(self.conv3(x)))
        x = torch.selu(self.bn4(self.head(x.reshape(x.size(0), -1))))
        return self.out(x)


    def prob(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Calculates the softmax probability distribution over abstract states."""
        return torch.softmax(logits / temperature, dim=-1)


    def gumbel_prob(self, logits: torch.Tensor, temperature: float) -> torch.Tensor:
        """Calculates the Gumbel-softmax distribution over abstract states."""
        return nn.functional.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)


    def pred(self, obs: torch.Tensor, temperature: float) -> torch.Tensor:
        """Predicts the probability distribution over abstract states given an observation."""
        obs = obs.to(self.device) #Move obs to device
        logits: torch.Tensor = self.forward(obs)
        return torch.clamp(torch.softmax(logits / temperature, dim=-1), 1e-9, 1 - (self.n_abstract_state * 1e-9))


    def gumbel_pred(self, obs: torch.Tensor, temperature: float) -> torch.Tensor:
        """Predicts the Gumbel-softmax distribution over abstract states given an observation."""
        obs = obs.to(self.device) #Move obs to device
        logits: torch.Tensor = self.forward(obs)
        return nn.functional.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)


    def save(self, path: str) -> None:
        """Saves the model's state dictionary to the given path."""
        torch.save(self.state_dict(), path)


    def load(self, path: str) -> None:
        """Loads the model's state dictionary from the given path."""
        self.load_state_dict(torch.load(path, map_location=self.device)) #Load on the correct device