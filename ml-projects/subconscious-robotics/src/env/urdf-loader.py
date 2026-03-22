"""
URDF Loader Utility for Subconscious Robotics Framework.

Provides utilities for loading and inspecting robot URDF files in PyBullet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pybullet as p

if TYPE_CHECKING:
    from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class JointInfo:
    """Container for joint information extracted from URDF."""

    index: int
    name: str
    joint_type: int
    lower_limit: float
    upper_limit: float
    max_force: float
    max_velocity: float
    link_name: str
    parent_index: int


@dataclass
class LinkInfo:
    """Container for link information."""

    index: int
    name: str
    mass: float
    local_inertia_diagonal: tuple[float, float, float]


@dataclass
class RobotDescription:
    """Complete robot description loaded from URDF."""

    body_id: int
    urdf_path: Path
    base_position: tuple[float, float, float]
    base_orientation: tuple[float, float, float, float]
    num_joints: int
    joints: list[JointInfo] = field(default_factory=list)
    links: list[LinkInfo] = field(default_factory=list)
    controllable_joints: list[int] = field(default_factory=list)


class URDFLoader:
    """
    URDF Loader for PyBullet simulation.

    Handles loading robot descriptions from URDF files and provides
    utilities for joint/link enumeration and dynamics inspection.
    """

    # PyBullet joint type mapping
    JOINT_TYPES = {
        p.JOINT_REVOLUTE: "REVOLUTE",
        p.JOINT_PRISMATIC: "PRISMATIC",
        p.JOINT_SPHERICAL: "SPHERICAL",
        p.JOINT_PLANAR: "PLANAR",
        p.JOINT_FIXED: "FIXED",
    }

    def __init__(self, physics_client: int | None = None):
        """
        Initialize URDF Loader.

        Args:
            physics_client: PyBullet physics client ID. If None, uses default.
        """
        self.physics_client = physics_client if physics_client is not None else 0

    def load(
        self,
        urdf_path: str | Path,
        base_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        base_orientation: tuple[float, float, float, float] | None = None,
        use_fixed_base: bool = True,
        global_scaling: float = 1.0,
    ) -> RobotDescription:
        """
        Load a robot from URDF file.

        Args:
            urdf_path: Path to the URDF file.
            base_position: Initial position (x, y, z).
            base_orientation: Initial orientation as quaternion (x, y, z, w).
            use_fixed_base: Whether to fix the robot base to the world.
            global_scaling: Scale factor for the robot.

        Returns:
            RobotDescription containing all robot information.

        Raises:
            FileNotFoundError: If URDF file doesn't exist.
            RuntimeError: If PyBullet fails to load the URDF.
        """
        urdf_path = Path(urdf_path)
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF file not found: {urdf_path}")

        if base_orientation is None:
            base_orientation = p.getQuaternionFromEuler([0, 0, 0])

        logger.info(f"Loading URDF: {urdf_path}")

        try:
            body_id = p.loadURDF(
                str(urdf_path),
                basePosition=base_position,
                baseOrientation=base_orientation,
                useFixedBase=use_fixed_base,
                globalScaling=global_scaling,
                physicsClientId=self.physics_client,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load URDF: {e}") from e

        robot = RobotDescription(
            body_id=body_id,
            urdf_path=urdf_path,
            base_position=base_position,
            base_orientation=base_orientation,
            num_joints=p.getNumJoints(body_id, physicsClientId=self.physics_client),
        )

        # Extract joint and link information
        robot.joints = self._extract_joints(body_id)
        robot.links = self._extract_links(body_id)
        robot.controllable_joints = self._get_controllable_joints(robot.joints)

        logger.info(
            f"Loaded robot with {robot.num_joints} joints, "
            f"{len(robot.controllable_joints)} controllable"
        )

        return robot

    def _extract_joints(self, body_id: int) -> list[JointInfo]:
        """Extract all joint information from loaded robot."""
        joints = []
        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)

        for i in range(num_joints):
            info = p.getJointInfo(body_id, i, physicsClientId=self.physics_client)
            joints.append(
                JointInfo(
                    index=info[0],
                    name=info[1].decode("utf-8"),
                    joint_type=info[2],
                    lower_limit=info[8],
                    upper_limit=info[9],
                    max_force=info[10],
                    max_velocity=info[11],
                    link_name=info[12].decode("utf-8"),
                    parent_index=info[16],
                )
            )

        return joints

    def _extract_links(self, body_id: int) -> list[LinkInfo]:
        """Extract all link information from loaded robot."""
        links = []
        num_joints = p.getNumJoints(body_id, physicsClientId=self.physics_client)

        # Base link (index -1)
        base_dynamics = p.getDynamicsInfo(body_id, -1, physicsClientId=self.physics_client)
        links.append(
            LinkInfo(
                index=-1,
                name="base_link",
                mass=base_dynamics[0],
                local_inertia_diagonal=base_dynamics[2],
            )
        )

        # Other links
        for i in range(num_joints):
            dynamics = p.getDynamicsInfo(body_id, i, physicsClientId=self.physics_client)
            joint_info = p.getJointInfo(body_id, i, physicsClientId=self.physics_client)
            links.append(
                LinkInfo(
                    index=i,
                    name=joint_info[12].decode("utf-8"),
                    mass=dynamics[0],
                    local_inertia_diagonal=dynamics[2],
                )
            )

        return links

    def _get_controllable_joints(self, joints: list[JointInfo]) -> list[int]:
        """Get indices of controllable (non-fixed) joints."""
        return [j.index for j in joints if j.joint_type != p.JOINT_FIXED]

    def get_joint_limits(
        self, robot: RobotDescription
    ) -> tuple[list[float], list[float]]:
        """
        Get joint position limits.

        Args:
            robot: Robot description.

        Returns:
            Tuple of (lower_limits, upper_limits) for controllable joints.
        """
        lower = []
        upper = []
        for idx in robot.controllable_joints:
            joint = robot.joints[idx]
            lower.append(joint.lower_limit)
            upper.append(joint.upper_limit)
        return lower, upper

    def print_robot_info(self, robot: RobotDescription) -> None:
        """Print detailed robot information for debugging."""
        print(f"\n{'='*60}")
        print(f"Robot: {robot.urdf_path.name}")
        print(f"{'='*60}")
        print(f"Body ID: {robot.body_id}")
        print(f"Total Joints: {robot.num_joints}")
        print(f"Controllable Joints: {len(robot.controllable_joints)}")
        print(f"\nJoints:")
        print("-" * 60)
        for j in robot.joints:
            jtype = self.JOINT_TYPES.get(j.joint_type, "UNKNOWN")
            print(f"  [{j.index}] {j.name:20s} | {jtype:10s} | "
                  f"limits: [{j.lower_limit:.2f}, {j.upper_limit:.2f}]")
        print(f"\nLinks:")
        print("-" * 60)
        for link in robot.links:
            print(f"  [{link.index:2d}] {link.name:20s} | mass: {link.mass:.4f} kg")
        print("=" * 60)
