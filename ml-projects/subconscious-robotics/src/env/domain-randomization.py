"""
Domain Randomization Module for Sim-to-Real Transfer.

Implements physics parameter randomization to bridge the reality gap
between simulation and real-world deployment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pybullet as p

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclass
class RandomizationState:
    """Stores current randomization state for reproducibility."""

    mass_scale: float = 1.0
    friction_lateral: float = 1.0
    friction_spinning: float = 0.5
    friction_rolling: float = 0.005
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    joint_damping_scale: float = 1.0


class DomainRandomizer:
    """
    Domain Randomization for bridging the sim-to-real gap.

    Randomizes physical parameters including:
    - Link masses
    - Contact friction coefficients
    - Gravity vector
    - Joint damping
    - Observation noise
    """

    def __init__(
        self,
        config: DictConfig,
        physics_client: int = 0,
        seed: int | None = None,
    ):
        """
        Initialize the domain randomizer.

        Args:
            config: Hydra configuration containing randomization parameters.
            physics_client: PyBullet physics client ID.
            seed: Random seed for reproducibility.
        """
        self.config = config
        self.physics_client = physics_client
        self.rng = np.random.default_rng(seed)
        self.state = RandomizationState()

        # Cache for original dynamics
        self._original_dynamics: dict[tuple[int, int], dict] = {}

    def randomize_all(self, body_id: int) -> RandomizationState:
        """
        Apply all enabled randomizations.

        Args:
            body_id: PyBullet body ID to randomize.

        Returns:
            Current randomization state.
        """
        dr_config = self.config.domain_randomization

        if not dr_config.enabled:
            return self.state

        # Cache original dynamics on first call
        if not self._original_dynamics:
            self._cache_original_dynamics(body_id)

        if dr_config.mass.enabled:
            self._randomize_mass(body_id, dr_config.mass.range)

        if dr_config.friction.enabled:
            self._randomize_friction(
                body_id,
                dr_config.friction.lateral_range,
                dr_config.friction.spinning_range,
                dr_config.friction.rolling_range,
            )

        if dr_config.gravity.enabled:
            self._randomize_gravity(dr_config.gravity.range)

        if dr_config.joint_damping.enabled:
            self._randomize_joint_damping(body_id, dr_config.joint_damping.range)

        return self.state

    def _cache_original_dynamics(self, body_id: int) -> None:
        """Cache original dynamics info for restoration."""
        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)

        # Base link
        dynamics = p.getDynamicsInfo(body_id, -1, physicsClientId=self.physics_client)
        self._original_dynamics[(body_id, -1)] = {
            "mass": dynamics[0],
            "lateral_friction": dynamics[1],
            "local_inertia_diagonal": dynamics[2],
        }

        # All other links
        for link_idx in range(num_joints):
            dynamics = p.getDynamicsInfo(
                body_id, link_idx, physicsClientId=self.physics_client
            )
            self._original_dynamics[(body_id, link_idx)] = {
                "mass": dynamics[0],
                "lateral_friction": dynamics[1],
                "local_inertia_diagonal": dynamics[2],
            }

    def _randomize_mass(
        self,
        body_id: int,
        mass_range: list[float],
    ) -> None:
        """
        Randomize link masses by applying a scale factor.

        Args:
            body_id: PyBullet body ID.
            mass_range: [min_scale, max_scale] multiplier range.
        """
        scale = self.rng.uniform(mass_range[0], mass_range[1])
        self.state.mass_scale = scale

        for (bid, link_idx), orig in self._original_dynamics.items():
            if bid != body_id:
                continue

            new_mass = orig["mass"] * scale
            if new_mass > 0:
                p.changeDynamics(
                    body_id,
                    link_idx,
                    mass=new_mass,
                    physicsClientId=self.physics_client,
                )

        logger.debug(f"Mass scale: {scale:.3f}")

    def _randomize_friction(
        self,
        body_id: int,
        lateral_range: list[float],
        spinning_range: list[float],
        rolling_range: list[float],
    ) -> None:
        """
        Randomize contact friction coefficients.

        Args:
            body_id: PyBullet body ID.
            lateral_range: Range for lateral friction coefficient.
            spinning_range: Range for spinning friction.
            rolling_range: Range for rolling friction.
        """
        lateral = self.rng.uniform(lateral_range[0], lateral_range[1])
        spinning = self.rng.uniform(spinning_range[0], spinning_range[1])
        rolling = self.rng.uniform(rolling_range[0], rolling_range[1])

        self.state.friction_lateral = lateral
        self.state.friction_spinning = spinning
        self.state.friction_rolling = rolling

        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)

        for link_idx in range(-1, num_joints):
            p.changeDynamics(
                body_id,
                link_idx,
                lateralFriction=lateral,
                spinningFriction=spinning,
                rollingFriction=rolling,
                physicsClientId=self.physics_client,
            )

        logger.debug(
            f"Friction - lateral: {lateral:.3f}, "
            f"spinning: {spinning:.3f}, rolling: {rolling:.4f}"
        )

    def _randomize_gravity(self, gravity_range: list[float]) -> None:
        """
        Randomize gravity magnitude.

        Args:
            gravity_range: [min_z, max_z] for gravity z-component.
        """
        gz = self.rng.uniform(gravity_range[0], gravity_range[1])
        gravity = (0.0, 0.0, gz)
        self.state.gravity = gravity

        p.setGravity(*gravity, physicsClientId=self.physics_client)
        logger.debug(f"Gravity: {gravity}")

    def _randomize_joint_damping(
        self,
        body_id: int,
        damping_range: list[float],
    ) -> None:
        """
        Randomize joint damping coefficients.

        Args:
            body_id: PyBullet body ID.
            damping_range: [min_scale, max_scale] for damping multiplier.
        """
        scale = self.rng.uniform(damping_range[0], damping_range[1])
        self.state.joint_damping_scale = scale

        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)

        for joint_idx in range(num_joints):
            joint_info = p.getJointInfo(
                body_id, joint_idx, physicsClientId=self.physics_client
            )
            # Joint damping is at index 6
            original_damping = joint_info[6]
            new_damping = original_damping * scale

            p.changeDynamics(
                body_id,
                joint_idx,
                jointDamping=new_damping,
                physicsClientId=self.physics_client,
            )

        logger.debug(f"Joint damping scale: {scale:.3f}")

    def add_observation_noise(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        """
        Add noise to observations for robustness.

        Args:
            observation: Raw observation array.

        Returns:
            Noisy observation.
        """
        dr_config = self.config.domain_randomization

        if not dr_config.enabled or not dr_config.observation_noise.enabled:
            return observation

        # Assume first half is positions, second half is velocities
        n = len(observation)
        position_noise = self.rng.normal(
            0, dr_config.observation_noise.position_std, n // 2
        )
        velocity_noise = self.rng.normal(
            0, dr_config.observation_noise.velocity_std, n - n // 2
        )

        noise = np.concatenate([position_noise, velocity_noise])
        return observation + noise

    def reset(self, body_id: int) -> None:
        """
        Reset all dynamics to original values.

        Args:
            body_id: PyBullet body ID.
        """
        for (bid, link_idx), orig in self._original_dynamics.items():
            if bid != body_id:
                continue

            p.changeDynamics(
                body_id,
                link_idx,
                mass=orig["mass"],
                lateralFriction=orig["lateral_friction"],
                physicsClientId=self.physics_client,
            )

        # Reset gravity
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)

        self.state = RandomizationState()
        logger.debug("Domain randomization reset to defaults")
