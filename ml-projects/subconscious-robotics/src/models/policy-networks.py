"""
Policy Network Architectures for Stable Baselines3.

Implements custom MLP and CNN feature extractors compatible with SB3.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

if TYPE_CHECKING:
    from omegaconf import DictConfig


class MLPExtractor(BaseFeaturesExtractor):
    """
    Custom MLP feature extractor for SB3.

    Processes observation through configurable dense layers.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        features_dim: int = 256,
        net_arch: list[int] | None = None,
        activation: str = "tanh",
    ):
        """
        Initialize MLP extractor.

        Args:
            observation_space: Gymnasium observation space.
            features_dim: Output dimension of the feature extractor.
            net_arch: List of hidden layer sizes.
            activation: Activation function ('tanh', 'relu', 'elu').
        """
        super().__init__(observation_space, features_dim)

        if net_arch is None:
            net_arch = [256, 256, 128]

        activation_fn = self._get_activation(activation)

        # Build network
        layers = []
        in_dim = gym.spaces.utils.flatdim(observation_space)

        for hidden_dim in net_arch:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation_fn())
            in_dim = hidden_dim

        # Final layer to features_dim
        layers.append(nn.Linear(in_dim, features_dim))
        layers.append(activation_fn())

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> type[nn.Module]:
        """Get activation function by name."""
        activations = {
            "tanh": nn.Tanh,
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
            "gelu": nn.GELU,
        }
        return activations.get(name.lower(), nn.Tanh)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        return self.network(observations)


class CNNExtractor(BaseFeaturesExtractor):
    """
    Custom CNN feature extractor for image-based observations.

    Compatible with SB3 policies for visual input processing.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 256,
    ):
        """
        Initialize CNN extractor.

        Args:
            observation_space: Gymnasium Box observation space (image).
            features_dim: Output dimension of the feature extractor.
        """
        super().__init__(observation_space, features_dim)

        # Assume input is (C, H, W) or (H, W, C)
        n_input_channels = observation_space.shape[0]
        if observation_space.shape[0] > 4:
            # Channels last format
            n_input_channels = observation_space.shape[-1]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            if observation_space.shape[0] > 4:
                # Convert from (H, W, C) to (C, H, W)
                sample = sample.permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN and linear layers."""
        # Handle channels-last format
        if observations.shape[1] > 4:
            observations = observations.permute(0, 3, 1, 2)

        return self.linear(self.cnn(observations.float() / 255.0))


def get_policy_kwargs(config: DictConfig) -> dict:
    """
    Build policy keyword arguments from configuration.

    Args:
        config: Hydra configuration with agent settings.

    Returns:
        Dictionary of policy kwargs for SB3.
    """
    agent_cfg = config.agent

    if agent_cfg.policy_type == "cnn":
        return {
            "features_extractor_class": CNNExtractor,
            "features_extractor_kwargs": {
                "features_dim": agent_cfg.cnn.features_dim,
            },
            "net_arch": list(agent_cfg.cnn.net_arch),
        }
    else:  # MLP
        return {
            "features_extractor_class": MLPExtractor,
            "features_extractor_kwargs": {
                "features_dim": 256,
                "net_arch": list(agent_cfg.mlp.net_arch.pi),
                "activation": agent_cfg.mlp.activation,
            },
            "net_arch": dict(
                pi=list(agent_cfg.mlp.net_arch.pi),
                vf=list(agent_cfg.mlp.net_arch.vf),
            ),
        }


class NatureCNN(BaseFeaturesExtractor):
    """
    CNN from DQN Nature paper.

    Standard architecture for Atari-like inputs.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
    ):
        """
        Initialize Nature CNN.

        Args:
            observation_space: Gymnasium Box observation space.
            features_dim: Output dimension of features.
        """
        assert isinstance(observation_space, gym.spaces.Box)
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened size
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.linear(self.cnn(observations))
