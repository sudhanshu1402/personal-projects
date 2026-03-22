"""
Multi-Objective Reward Shaper for Subconscious Robot Training.

Implements configurable reward composition with normalization,
clipping, and curriculum learning support.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class RewardComponents:
    """Container for individual reward components."""

    task_completion: float = 0.0
    distance_to_goal: float = 0.0
    energy_penalty: float = 0.0
    stability: float = 0.0
    smoothness: float = 0.0
    collision_penalty: float = 0.0
    time_penalty: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "task_completion": self.task_completion,
            "distance_to_goal": self.distance_to_goal,
            "energy_penalty": self.energy_penalty,
            "stability": self.stability,
            "smoothness": self.smoothness,
            "collision_penalty": self.collision_penalty,
            "time_penalty": self.time_penalty,
        }


@dataclass
class RewardStats:
    """Running statistics for reward normalization."""

    mean: float = 0.0
    var: float = 1.0
    count: int = 0

    def update(self, value: float) -> None:
        """Update running statistics using Welford's algorithm."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.var += delta * delta2

    @property
    def std(self) -> float:
        """Get standard deviation."""
        if self.count < 2:
            return 1.0
        return np.sqrt(self.var / (self.count - 1)) + 1e-8


class RewardShaper:
    """
    Multi-objective reward shaping with configurable weights.

    Features:
    - Weighted combination of reward components
    - Optional normalization and clipping
    - Curriculum learning support
    - Reward component tracking for analysis
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the reward shaper.

        Args:
            config: Hydra configuration containing reward parameters.
        """
        self.config = config.reward
        self.weights = dict(self.config.weights)
        self.shaping = self.config.shaping

        # Statistics for normalization
        self._stats = RewardStats()
        self._component_stats: dict[str, RewardStats] = {}

        # Previous action for smoothness calculation
        self._prev_action: np.ndarray | None = None

        # Custom reward functions
        self._custom_rewards: dict[str, Callable] = {}

        logger.info(f"RewardShaper initialized with weights: {self.weights}")

    def compute(
        self,
        end_effector_pos: np.ndarray,
        goal_pos: np.ndarray,
        action: np.ndarray,
        collision: bool = False,
        is_terminal: bool = False,
        orientation_error: float = 0.0,
    ) -> tuple[float, RewardComponents]:
        """
        Compute the shaped reward.

        Args:
            end_effector_pos: Current end-effector position.
            goal_pos: Target goal position.
            action: Current action taken.
            collision: Whether a collision occurred.
            is_terminal: Whether episode terminated.
            orientation_error: Error in end-effector orientation.

        Returns:
            Tuple of (total_reward, reward_components).
        """
        components = RewardComponents()

        # Distance to goal (negative distance as reward)
        distance = np.linalg.norm(end_effector_pos - goal_pos)
        components.distance_to_goal = -distance

        # Task completion bonus
        if distance < self.config.goal.success_threshold:
            components.task_completion = self.config.goal.success_bonus

        # Energy penalty (action magnitude)
        components.energy_penalty = -np.sum(np.square(action))

        # Stability (penalize orientation error)
        components.stability = -orientation_error

        # Smoothness (penalize action changes)
        if self._prev_action is not None:
            action_diff = np.linalg.norm(action - self._prev_action)
            components.smoothness = -action_diff
        self._prev_action = action.copy()

        # Collision penalty
        if collision:
            components.collision_penalty = -1.0

        # Time penalty (encourage efficiency)
        components.time_penalty = -1.0

        # Compute weighted sum
        total_reward = self._weighted_sum(components)

        # Apply shaping (normalization, clipping)
        total_reward = self._apply_shaping(total_reward)

        return total_reward, components

    def _weighted_sum(self, components: RewardComponents) -> float:
        """Compute weighted sum of reward components."""
        total = 0.0
        comp_dict = components.to_dict()

        for name, value in comp_dict.items():
            weight = self.weights.get(name, 0.0)
            total += weight * value

        return total

    def _apply_shaping(self, reward: float) -> float:
        """Apply normalization and clipping to reward."""
        # Update statistics
        self._stats.update(reward)

        # Normalize
        if self.shaping.normalize and self._stats.count > 100:
            reward = (reward - self._stats.mean) / self._stats.std

        # Scale
        reward *= self.shaping.scale

        # Clip
        reward = np.clip(reward, self.shaping.clip_min, self.shaping.clip_max)

        return float(reward)

    def compute_sparse(
        self,
        distance: float,
        success: bool,
    ) -> float:
        """
        Compute sparse reward (only at success/failure).

        Args:
            distance: Distance to goal.
            success: Whether task was completed.

        Returns:
            Sparse reward value.
        """
        if success:
            return self.config.goal.success_bonus
        return 0.0

    def register_custom_reward(
        self,
        name: str,
        fn: Callable[..., float],
        weight: float = 1.0,
    ) -> None:
        """
        Register a custom reward function.

        Args:
            name: Name of the reward component.
            fn: Function that computes the reward value.
            weight: Weight for this component.
        """
        self._custom_rewards[name] = fn
        self.weights[name] = weight
        logger.info(f"Registered custom reward: {name} (weight={weight})")

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """
        Update reward weights (e.g., for curriculum learning).

        Args:
            new_weights: Dictionary of weight updates.
        """
        self.weights.update(new_weights)
        logger.info(f"Updated reward weights: {new_weights}")

    def reset(self) -> None:
        """Reset per-episode state."""
        self._prev_action = None

    def get_stats(self) -> dict[str, float]:
        """Get reward statistics for logging."""
        return {
            "reward_mean": self._stats.mean,
            "reward_std": self._stats.std,
            "reward_count": self._stats.count,
        }


class CurriculumScheduler:
    """
    Curriculum learning scheduler for progressive difficulty.

    Adjusts environment and reward parameters based on training progress.
    """

    def __init__(self, config: DictConfig):
        """
        Initialize curriculum scheduler.

        Args:
            config: Curriculum configuration from Hydra.
        """
        self.config = config.reward.curriculum
        self.enabled = self.config.enabled
        self.stages = list(self.config.stages) if self.enabled else []
        self.current_stage = 0
        self.current_difficulty = 0.3 if self.stages else 1.0

    def update(self, timesteps: int) -> float:
        """
        Update curriculum based on training progress.

        Args:
            timesteps: Current total timesteps.

        Returns:
            Current difficulty level (0.0 to 1.0).
        """
        if not self.enabled or not self.stages:
            return 1.0

        # Find current stage
        cumulative = 0
        for i, stage in enumerate(self.stages):
            if stage.timesteps is None:
                # Final stage (infinite)
                self.current_stage = i
                self.current_difficulty = stage.difficulty
                break

            cumulative += stage.timesteps
            if timesteps < cumulative:
                self.current_stage = i
                self.current_difficulty = stage.difficulty
                break

        return self.current_difficulty

    def get_goal_distance_modifier(self) -> float:
        """
        Get goal distance modifier based on difficulty.

        Returns:
            Multiplier for goal distance (lower = easier).
        """
        # Start with close goals, increase distance as difficulty rises
        return 0.3 + 0.7 * self.current_difficulty

    def get_randomization_scale(self) -> float:
        """
        Get domain randomization scale based on difficulty.

        Returns:
            Multiplier for randomization ranges.
        """
        # Start with less randomization, increase over time
        return self.current_difficulty
