"""
Quadruped Walker Gymnasium Environment for Subconscious Robot Training.

Implements an 8-DoF quadruped robot with life-like locomotion training.
Features domain randomization and multi-objective reward shaping.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

if TYPE_CHECKING:
    from omegaconf import DictConfig

from src.env.domain_randomization import DomainRandomizer

logger = logging.getLogger(__name__)


class QuadrupedEnv(gym.Env):
    """
    8-DoF Quadruped Walker Environment.

    Observation Space (26 dims):
    - Base orientation (roll, pitch, yaw): 3
    - Base angular velocity: 3
    - Base linear velocity: 3
    - Joint positions: 8
    - Joint velocities: 8
    - Previous actions: 8 (for smoothness)
    Total: 26 (excluding previous actions) or 34 (with prev actions)

    Action Space (8 dims):
    - Target joint positions (normalized [-1, 1])
    - Mapped to actual joint limits

    Reward Components:
    1. Forward velocity (primary)
    2. Stability (penalize tilt)
    3. Energy efficiency (penalize torque)
    4. Survival bonus (staying upright)
    5. Smoothness (penalize jerky motions)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # Joint configuration - ordered for consistent action mapping
    JOINT_NAMES = [
        "fl_hip_joint", "fl_knee_joint",  # Front-left
        "fr_hip_joint", "fr_knee_joint",  # Front-right
        "rl_hip_joint", "rl_knee_joint",  # Rear-left
        "rr_hip_joint", "rr_knee_joint",  # Rear-right
    ]

    # Foot links for contact detection
    FOOT_LINKS = ["fl_foot", "fr_foot", "rl_foot", "rr_foot"]

    # Physical parameters
    TORSO_HEIGHT_THRESHOLD = 0.15  # Minimum torso height for survival
    MAX_TILT_ANGLE = 0.8  # Radians (~45 degrees)

    def __init__(
        self,
        config: DictConfig | None = None,
        render_mode: str | None = None,
        urdf_path: str | None = None,
    ):
        """
        Initialize the quadruped environment.

        Args:
            config: Hydra configuration.
            render_mode: "human" for GUI, "rgb_array" for pixel output.
            urdf_path: Override path to URDF file.
        """
        super().__init__()

        self.config = config
        self.render_mode = render_mode

        # URDF path
        if urdf_path:
            self.urdf_path = Path(urdf_path)
        elif config and hasattr(config, "env") and hasattr(config.env, "urdf_path"):
            self.urdf_path = Path(config.env.urdf_path)
        else:
            self.urdf_path = Path(__file__).parent.parent.parent / "assets" / "quadruped.urdf"

        # Physics setup
        self._setup_physics()

        # Load robot
        self.robot_id = self._load_robot()
        self._setup_joints()

        # Domain randomization
        if config is not None:
            self.domain_randomizer = DomainRandomizer(
                config, self.physics_client, seed=config.seed
            )
        else:
            self.domain_randomizer = None

        # Define spaces
        self._define_spaces()

        # State tracking
        self.current_step = 0
        self.total_timesteps = 0
        self.prev_action = np.zeros(8, dtype=np.float32)
        self.prev_torso_pos = np.zeros(3, dtype=np.float32)

        # Reward coefficients
        self.reward_weights = {
            "forward_velocity": 2.0,
            "stability": -1.0,
            "energy": -0.005,
            "survival": 0.1,
            "smoothness": -0.1,
            "foot_contact": 0.05,
            "height_bonus": 0.2,
        }

        # Episode config
        self.max_episode_steps = config.env.max_episode_steps if config else 1000
        self.time_step = config.env.time_step if config else 1.0 / 240.0

        logger.info(f"QuadrupedEnv initialized with {len(self.joint_indices)} joints")

    def _setup_physics(self) -> None:
        """Initialize PyBullet physics engine."""
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.2],
            )
        else:
            self.physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.physics_client)
        p.setTimeStep(self.time_step if hasattr(self, "time_step") else 1 / 240)

        # High-fidelity physics
        p.setPhysicsEngineParameter(
            numSolverIterations=50,
            numSubSteps=4,
            contactBreakingThreshold=0.02,
            enableConeFriction=True,
            physicsClientId=self.physics_client,
        )

        # Load ground
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.physics_client)

        # Set ground friction
        p.changeDynamics(
            self.plane_id,
            -1,
            lateralFriction=1.0,
            spinningFriction=0.3,
            rollingFriction=0.01,
            physicsClientId=self.physics_client,
        )

    def _load_robot(self) -> int:
        """Load the quadruped URDF."""
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")

        # Spawn robot at standing height
        initial_height = 0.35  # Slightly above ground
        robot_id = p.loadURDF(
            str(self.urdf_path),
            basePosition=[0, 0, initial_height],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=False,
            physicsClientId=self.physics_client,
        )

        return robot_id

    def _setup_joints(self) -> None:
        """Map joint names to indices and extract limits."""
        self.joint_indices = []
        self.joint_limits_lower = []
        self.joint_limits_upper = []
        self.joint_max_forces = []
        self.joint_max_velocities = []
        self.link_name_to_index = {}

        num_joints = p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)

        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i, physicsClientId=self.physics_client)
            joint_name = joint_info[1].decode("utf-8")
            link_name = joint_info[12].decode("utf-8")
            self.link_name_to_index[link_name] = i

            if joint_name in self.JOINT_NAMES:
                idx = self.JOINT_NAMES.index(joint_name)
                self.joint_indices.append((idx, i))  # (order, pybullet_idx)
                self.joint_limits_lower.append(joint_info[8])
                self.joint_limits_upper.append(joint_info[9])
                self.joint_max_forces.append(joint_info[10])
                self.joint_max_velocities.append(joint_info[11])

        # Sort by order to match action vector
        self.joint_indices.sort(key=lambda x: x[0])
        self.joint_indices = [idx for _, idx in self.joint_indices]

        self.joint_limits_lower = np.array(self.joint_limits_lower, dtype=np.float32)
        self.joint_limits_upper = np.array(self.joint_limits_upper, dtype=np.float32)
        self.joint_max_forces = np.array(self.joint_max_forces, dtype=np.float32)

        # Foot link indices for contact detection
        self.foot_indices = [self.link_name_to_index.get(f, -1) for f in self.FOOT_LINKS]

    def _define_spaces(self) -> None:
        """Define observation and action spaces."""
        # Observation: orientation(3) + ang_vel(3) + lin_vel(3) + joint_pos(8) + joint_vel(8) = 25
        obs_dim = 25

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Action: 8 joint position targets (normalized [-1, 1])
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)

        # Reset robot pose
        initial_height = 0.35
        p.resetBasePositionAndOrientation(
            self.robot_id,
            [0, 0, initial_height],
            p.getQuaternionFromEuler([0, 0, 0]),
            physicsClientId=self.physics_client,
        )
        p.resetBaseVelocity(
            self.robot_id,
            [0, 0, 0],
            [0, 0, 0],
            physicsClientId=self.physics_client,
        )

        # Reset joints to initial stance
        initial_positions = self._get_initial_stance()
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(
                self.robot_id,
                joint_idx,
                targetValue=initial_positions[i],
                targetVelocity=0.0,
                physicsClientId=self.physics_client,
            )

        # Apply domain randomization (±15% leg mass variation)
        if self.domain_randomizer:
            self.domain_randomizer.randomize_all(self.robot_id)

        # Let robot settle
        for _ in range(50):
            p.stepSimulation(physicsClientId=self.physics_client)

        # Reset state
        self.current_step = 0
        self.prev_action = np.zeros(8, dtype=np.float32)
        pos, _ = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        self.prev_torso_pos = np.array(pos, dtype=np.float32)

        observation = self._get_observation()
        info = {"initial_height": initial_height}

        return observation, info

    def _get_initial_stance(self) -> np.ndarray:
        """Get initial joint positions for a stable standing pose."""
        # Hip joints near neutral, knees slightly bent
        stance = np.array([
            0.0, -0.5,  # FL hip, knee
            0.0, -0.5,  # FR hip, knee
            0.0, -0.5,  # RL hip, knee
            0.0, -0.5,  # RR hip, knee
        ], dtype=np.float32)
        return stance

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simulation step."""
        self.current_step += 1
        self.total_timesteps += 1

        # Clip and scale actions to joint limits
        action = np.clip(action, -1.0, 1.0)
        target_positions = self._scale_action_to_joints(action)

        # Apply position control with PD gains
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=target_positions[i],
                force=self.joint_max_forces[i],
                maxVelocity=10.0,
                positionGain=0.5,
                velocityGain=0.1,
                physicsClientId=self.physics_client,
            )

        # Step simulation (multiple substeps for stability)
        for _ in range(4):
            p.stepSimulation(physicsClientId=self.physics_client)

        # Get new state
        observation = self._get_observation()

        # Get torso state
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        euler = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )

        # Compute reward
        reward, reward_info = self._compute_reward(
            pos, euler, lin_vel, action, target_positions
        )

        # Check termination
        terminated = self._check_termination(pos, euler)

        # Truncation
        truncated = self.current_step >= self.max_episode_steps

        # Update state for next step
        self.prev_action = action.copy()
        self.prev_torso_pos = np.array(pos, dtype=np.float32)

        info = {
            "position": pos,
            "orientation": euler,
            "velocity": lin_vel,
            "reward_components": reward_info,
            "step": self.current_step,
        }

        return observation, float(reward), terminated, truncated, info

    def _scale_action_to_joints(self, action: np.ndarray) -> np.ndarray:
        """Scale normalized actions to joint position limits."""
        mid = (self.joint_limits_upper + self.joint_limits_lower) / 2
        rng = (self.joint_limits_upper - self.joint_limits_lower) / 2
        return mid + action * rng

    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        # Base state
        pos, orn = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self.physics_client
        )
        euler = np.array(p.getEulerFromQuaternion(orn), dtype=np.float32)
        lin_vel, ang_vel = p.getBaseVelocity(
            self.robot_id, physicsClientId=self.physics_client
        )

        # Joint states
        joint_positions = []
        joint_velocities = []
        for joint_idx in self.joint_indices:
            state = p.getJointState(
                self.robot_id, joint_idx, physicsClientId=self.physics_client
            )
            joint_positions.append(state[0])
            joint_velocities.append(state[1])

        observation = np.concatenate([
            euler,  # 3: roll, pitch, yaw
            np.array(ang_vel, dtype=np.float32),  # 3
            np.array(lin_vel, dtype=np.float32),  # 3
            np.array(joint_positions, dtype=np.float32),  # 8
            np.array(joint_velocities, dtype=np.float32),  # 8
        ])

        # Add observation noise if domain randomization enabled
        if self.domain_randomizer:
            observation = self.domain_randomizer.add_observation_noise(observation)

        return observation.astype(np.float32)

    def _compute_reward(
        self,
        pos: tuple,
        euler: tuple,
        lin_vel: tuple,
        action: np.ndarray,
        target_positions: np.ndarray,
    ) -> tuple[float, dict]:
        """
        Compute multi-objective reward for life-like locomotion.

        Components:
        1. Forward velocity - primary objective
        2. Stability - penalize excessive tilting
        3. Energy - penalize high torques
        4. Survival - bonus for staying upright
        5. Smoothness - penalize jerky motions
        """
        reward_components = {}

        # 1. Forward velocity reward (x-direction)
        forward_vel = lin_vel[0]
        reward_components["forward_velocity"] = forward_vel

        # 2. Stability penalty (penalize roll and pitch)
        roll, pitch, _ = euler
        tilt_penalty = abs(roll) ** 2 + abs(pitch) ** 2
        reward_components["stability"] = tilt_penalty

        # 3. Energy penalty (sum of squared actions ~ torque)
        joint_torques = []
        for joint_idx in self.joint_indices:
            state = p.getJointState(
                self.robot_id, joint_idx, physicsClientId=self.physics_client
            )
            joint_torques.append(state[3])  # Motor torque

        energy = np.sum(np.square(joint_torques))
        reward_components["energy"] = energy

        # 4. Survival bonus (torso above threshold)
        height = pos[2]
        survival = 1.0 if height > self.TORSO_HEIGHT_THRESHOLD else 0.0
        reward_components["survival"] = survival

        # 5. Smoothness (penalize action changes)
        action_diff = np.sum(np.square(action - self.prev_action))
        reward_components["smoothness"] = action_diff

        # 6. Foot contact reward (encourage ground contact)
        num_feet_contact = self._count_foot_contacts()
        reward_components["foot_contact"] = num_feet_contact

        # 7. Height maintenance bonus
        target_height = 0.25
        height_error = abs(height - target_height)
        reward_components["height_bonus"] = max(0, 1.0 - height_error * 5)

        # Weighted sum
        total_reward = 0.0
        for key, value in reward_components.items():
            weight = self.reward_weights.get(key, 0.0)
            total_reward += weight * value

        return total_reward, reward_components

    def _count_foot_contacts(self) -> int:
        """Count number of feet in contact with ground."""
        count = 0
        for foot_idx in self.foot_indices:
            if foot_idx < 0:
                continue
            contacts = p.getContactPoints(
                self.robot_id,
                self.plane_id,
                foot_idx,
                -1,
                physicsClientId=self.physics_client,
            )
            if contacts:
                count += 1
        return count

    def _check_termination(self, pos: tuple, euler: tuple) -> bool:
        """Check if episode should terminate."""
        height = pos[2]
        roll, pitch, _ = euler

        # Fallen over (torso too low)
        if height < self.TORSO_HEIGHT_THRESHOLD:
            return True

        # Excessive tilt
        if abs(roll) > self.MAX_TILT_ANGLE or abs(pitch) > self.MAX_TILT_ANGLE:
            return True

        return False

    def render(self) -> np.ndarray | None:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            width, height = 640, 480

            # Camera following robot
            pos, _ = p.getBasePositionAndOrientation(
                self.robot_id, physicsClientId=self.physics_client
            )

            view_matrix = p.computeViewMatrix(
                cameraEyePosition=[pos[0] - 1.0, pos[1] - 1.0, pos[2] + 0.5],
                cameraTargetPosition=pos,
                cameraUpVector=[0, 0, 1],
            )
            projection_matrix = p.computeProjectionMatrixFOV(
                fov=60, aspect=width / height, nearVal=0.1, farVal=10.0
            )
            _, _, rgb, _, _ = p.getCameraImage(
                width,
                height,
                viewMatrix=view_matrix,
                projectionMatrix=projection_matrix,
                physicsClientId=self.physics_client,
            )
            return np.array(rgb[:, :, :3], dtype=np.uint8)
        return None

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self, "physics_client"):
            p.disconnect(physicsClientId=self.physics_client)
        logger.info("QuadrupedEnv closed")


# ============================================================================
# Legacy compatibility - original SubconsciousEnv
# ============================================================================

class SubconsciousEnv(QuadrupedEnv):
    """
    Alias for QuadrupedEnv for backward compatibility.

    This class can be reconfigured to use different URDFs via config.
    """

    def __init__(
        self,
        config: DictConfig | None = None,
        render_mode: str | None = None,
    ):
        # Determine URDF from config
        if config and hasattr(config, "env") and hasattr(config.env, "urdf_path"):
            urdf = config.env.urdf_path
        else:
            urdf = None

        super().__init__(config=config, render_mode=render_mode, urdf_path=urdf)


def make_env(config: DictConfig, rank: int = 0, seed: int = 0):
    """Factory function for creating vectorized environments."""
    def _init():
        env = QuadrupedEnv(config=config, render_mode=None)
        env.reset(seed=seed + rank)
        return env
    return _init
