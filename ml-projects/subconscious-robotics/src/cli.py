"""
CLI Interface for Subconscious Robot Training.

Provides commands for training, evaluation, export, and system info.
Includes --watch flag for live GUI observation.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import click

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Project root
PROJECT_ROOT = Path(__file__).parent.parent

# Rich console
console = Console() if RICH_AVAILABLE else None


def print_banner() -> None:
    """Print CLI banner."""
    if RICH_AVAILABLE and console:
        console.print(Panel.fit(
            "[bold cyan]🦿 SUBCONSCIOUS ROBOT TRAINING[/bold cyan]\n"
            "[dim]Production Sim-to-Real Framework[/dim]",
            border_style="cyan",
        ))
    else:
        click.echo("=" * 60)
        click.echo("🦿 SUBCONSCIOUS ROBOT TRAINING")
        click.echo("=" * 60)


@click.group()
@click.version_option(version="1.0.0", prog_name="robot-train")
def main():
    """
    Subconscious Robot Training Framework.

    Production-ready sim-to-real framework for robot training
    using PyBullet simulation with domain randomization.

    Commands:
        train    - Train a robot policy
        eval     - Evaluate trained model
        export   - Export model to ONNX
        info     - System information
    """
    pass


@main.command()
@click.option(
    "--config", "-c",
    type=str,
    default="config.yaml",
    help="Hydra configuration file name",
)
@click.option(
    "--algorithm", "-a",
    type=click.Choice(["ppo", "sac"]),
    default=None,
    help="Training algorithm (overrides config)",
)
@click.option(
    "--timesteps", "-t",
    type=int,
    default=None,
    help="Total training timesteps",
)
@click.option(
    "--envs", "-n",
    type=int,
    default=None,
    help="Number of parallel environments",
)
@click.option(
    "--seed", "-s",
    type=int,
    default=None,
    help="Random seed",
)
@click.option(
    "--watch", "-w",
    is_flag=True,
    default=False,
    help="🔴 LIVE MODE: Watch robot learn in real-time GUI (1x speed)",
)
def train(
    config: str,
    algorithm: str | None,
    timesteps: int | None,
    envs: int | None,
    seed: int | None,
    watch: bool,
):
    """
    Start Subconscious Training mode.

    Train a robot policy using PPO or SAC with domain randomization.

    \b
    Examples:
        robot-train train                      # Default training
        robot-train train --watch              # Live GUI observation
        robot-train train -a sac -t 500000     # SAC with 500k steps
        robot-train train -n 8 --timesteps 1M  # 8 parallel envs
    """
    print_banner()

    if watch:
        click.echo("")
        click.echo("👁️  WATCH MODE ENABLED")
        click.echo("    Robot will learn in real-time GUI visualization")
        click.echo("    (Training speed reduced to 1x real-time)")
        click.echo("")
    else:
        click.echo("")
        click.echo("🚀 FAST MODE: Maximum training speed (headless)")
        click.echo("")

    # Build Hydra overrides
    overrides = []
    if algorithm:
        overrides.append(f"training.algorithm={algorithm}")
    if timesteps:
        overrides.append(f"training.total_timesteps={timesteps}")
    if envs and not watch:  # Watch mode forces single env
        overrides.append(f"training.n_envs={envs}")
    if seed:
        overrides.append(f"seed={seed}")

    # Run training with watch mode env var
    cmd = [sys.executable, str(PROJECT_ROOT / "src" / "train.py")]
    cmd.extend(overrides)

    click.echo(f"Running: {' '.join(cmd)}")
    if watch:
        click.echo("(Press Ctrl+C to stop training)")
    click.echo("-" * 60)

    # Prepare environment variables
    env = os.environ.copy()
    if watch:
        env["ROBOT_WATCH_MODE"] = "1"

    try:
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT), env=env)
    except subprocess.CalledProcessError as e:
        click.echo(f"Training failed with exit code {e.returncode}", err=True)
        sys.exit(e.returncode)
    except KeyboardInterrupt:
        click.echo("\nTraining interrupted by user")
        sys.exit(0)


@main.command()
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model (.zip file)",
)
@click.option(
    "--episodes", "-n",
    type=int,
    default=10,
    help="Number of evaluation episodes",
)
@click.option(
    "--render/--no-render",
    default=True,
    help="Enable/disable visual rendering",
)
@click.option(
    "--deterministic/--stochastic",
    default=True,
    help="Use deterministic or stochastic actions",
)
@click.option(
    "--watch", "-w",
    is_flag=True,
    default=True,
    help="Watch robot in GUI (default: True)",
)
def eval(
    model: str,
    episodes: int,
    render: bool,
    deterministic: bool,
    watch: bool,
):
    """
    Visual Evaluation mode.

    Test learned behaviors with visualization and statistics.

    \b
    Examples:
        robot-train eval --model outputs/latest/model.zip
        robot-train eval -m model.zip -n 20 --no-render
    """
    print_banner()
    click.echo("👁️ Visual Evaluation Mode")
    click.echo("-" * 60)

    from src.eval import evaluate, load_model
    from src.env.base_env import QuadrupedEnv

    # Load model
    click.echo(f"Loading model: {model}")
    loaded_model = load_model(model)

    # Create environment
    render_mode = "human" if (render or watch) else None
    env = QuadrupedEnv(config=None, render_mode=render_mode)

    try:
        stats = evaluate(
            model=loaded_model,
            env=env,
            n_episodes=episodes,
            deterministic=deterministic,
            render=render,
            verbose=True,
        )

        click.echo("\n" + "=" * 60)
        click.echo("✅ Evaluation Complete!")
        if "success_rate" in stats:
            click.echo(f"   Success Rate: {stats['success_rate'] * 100:.1f}%")
        if "mean_reward" in stats:
            click.echo(f"   Mean Reward: {stats['mean_reward']:.2f}")
        click.echo("=" * 60)

    finally:
        env.close()


@main.command()
@click.option(
    "--model", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to trained model (.zip file)",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output path for ONNX file",
)
@click.option(
    "--format", "-f",
    type=click.Choice(["onnx"]),
    default="onnx",
    help="Export format",
)
@click.option(
    "--optimize",
    is_flag=True,
    help="Apply ONNX optimization",
)
def export(
    model: str,
    output: str | None,
    format: str,
    optimize: bool,
):
    """
    Export trained model for hardware deployment.

    Convert SB3 model to ONNX format for edge deployment.

    \b
    Examples:
        robot-train export --model outputs/latest/model.zip
        robot-train export -m model.zip -o robot_policy.onnx --optimize
    """
    print_banner()
    click.echo("📦 Model Export")
    click.echo("-" * 60)

    from export.onnx_export import export_to_onnx, optimize_onnx

    # Set default output
    if output is None:
        model_path = Path(model)
        output = str(model_path.with_suffix(".onnx"))

    # Export
    click.echo(f"Exporting: {model} → {output}")
    onnx_path = export_to_onnx(
        model_path=model,
        output_path=output,
    )

    # Optimize if requested
    if optimize:
        click.echo("Applying optimization...")
        optimize_onnx(onnx_path)

    click.echo("\n" + "=" * 60)
    click.echo("✅ Export Complete!")
    click.echo(f"   ONNX model: {onnx_path}")
    click.echo("=" * 60)


@main.command()
def info():
    """
    Display system and device information.

    Shows PyTorch device availability, MPS status, and configuration.
    """
    print_banner()
    click.echo("📊 System Information")
    click.echo("-" * 60)

    from src.models.device_utils import print_device_info, check_mps_compatibility

    print_device_info()

    # Additional info
    import platform

    if RICH_AVAILABLE and console:
        table = Table(show_header=False, box=None)
        table.add_column(style="cyan")
        table.add_column()
        table.add_row("OS", f"{platform.system()} {platform.release()}")
        table.add_row("Machine", platform.machine())
        table.add_row("Python", platform.python_version())
        console.print(table)
    else:
        click.echo(f"\nSystem:")
        click.echo(f"  OS: {platform.system()} {platform.release()}")
        click.echo(f"  Machine: {platform.machine()}")
        click.echo(f"  Python: {platform.python_version()}")


@main.command()
@click.option(
    "--urdf", "-u",
    type=click.Path(exists=True),
    required=True,
    help="Path to URDF file to inspect",
)
def inspect_robot(urdf: str):
    """
    Inspect a robot URDF file.

    Display detailed joint and link information.

    \b
    Examples:
        robot-train inspect-robot --urdf assets/quadruped.urdf
        robot-train inspect-robot -u assets/simple_robot.urdf
    """
    print_banner()
    click.echo("🔬 Robot Inspection")
    click.echo("-" * 60)

    import time
    import pybullet as p
    import pybullet_data

    from src.env.urdf_loader import URDFLoader

    # Start PyBullet in GUI mode
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)

    # Load ground plane
    p.loadURDF("plane.urdf")

    try:
        loader = URDFLoader(physics_client)
        # Load robot dynamically above ground
        robot = loader.load(urdf, use_fixed_base=False, base_position=(0, 0, 0.5))
        loader.print_robot_info(robot)

        click.echo("\n🎮 Interactive Mode Enabled")
        click.echo("Use the sliders in the GUI to control each joint.")
        click.echo("Use mouse to rotate/zoom camera.")
        click.echo("\n⌨️  Keyboard Shortcuts:")
        click.echo("    [Space]  Jump (Apply Upward Force)")
        click.echo("    [Arrows] Push Robot (Apply Horizontal Force)")
        click.echo("    [Q]      Quit")

        # Create sliders for all movable joints
        joint_ids = []
        param_ids = []
        
        for joint in robot.joints:
            # If joint is revolute or prismatic
            if joint.joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                joint_ids.append(joint.index)
                lower_limit = joint.lower_limit
                upper_limit = joint.upper_limit
                
                # Handle unbounded joints (usually 0 to -1 means unlimited or equal)
                if lower_limit >= upper_limit:
                    lower_limit = -3.14
                    upper_limit = 3.14
                
                # Add slider
                param = p.addUserDebugParameter(joint.name, lower_limit, upper_limit, 0, physicsClientId=physics_client)
                param_ids.append(param)

        while p.isConnected(physicsClientId=physics_client):
            try:
                keys = p.getKeyboardEvents(physicsClientId=physics_client)

                # Quit
                if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                    break

                # Update joints based on sliders
                for j_id, p_id in zip(joint_ids, param_ids):
                    target_pos = p.readUserDebugParameter(p_id, physicsClientId=physics_client)
                    p.setJointMotorControl2(
                        robot.body_id,
                        j_id,
                        p.POSITION_CONTROL,
                        targetPosition=target_pos,
                        force=100.0,
                        physicsClientId=physics_client
                    )
                
                # Keyboard Control (External Forces)
                # Space to Jump
                if p.B3G_SPACE in keys and keys[p.B3G_SPACE] & p.KEY_WAS_TRIGGERED:
                    # Apply explicit impulse to base
                    p.applyExternalForce(robot.body_id, -1, [0, 0, 500], [0, 0, 0], p.LINK_FRAME, physicsClientId=physics_client)

                # Arrow Keys for lateral movement (Push)
                force_mag = 40.0
                if p.B3G_UP_ARROW in keys and keys[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:
                    p.applyExternalForce(robot.body_id, -1, [force_mag, 0, 0], [0, 0, 0], p.LINK_FRAME, physicsClientId=physics_client)
                if p.B3G_DOWN_ARROW in keys and keys[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:
                    p.applyExternalForce(robot.body_id, -1, [-force_mag, 0, 0], [0, 0, 0], p.LINK_FRAME, physicsClientId=physics_client)
                if p.B3G_LEFT_ARROW in keys and keys[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:
                    p.applyExternalForce(robot.body_id, -1, [0, force_mag, 0], [0, 0, 0], p.LINK_FRAME, physicsClientId=physics_client)
                if p.B3G_RIGHT_ARROW in keys and keys[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:
                    p.applyExternalForce(robot.body_id, -1, [0, -force_mag, 0], [0, 0, 0], p.LINK_FRAME, physicsClientId=physics_client)
                
                p.stepSimulation(physicsClientId=physics_client)
                time.sleep(1.0 / 240.0)

            except p.error as e:
                # Only print if it's an unexpected error (not just window closing)
                # Usually window close raises "Not connected to physics server" or similar
                click.echo(f"\n⚠️  Simulation loop stopped: {e}")
                break

    except KeyboardInterrupt:
        click.echo("\nExiting inspection.")
    finally:
        if p.isConnected():
            p.disconnect(physics_client)


if __name__ == "__main__":
    main()
