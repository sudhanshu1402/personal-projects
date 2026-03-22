"""
Training Script for Subconscious Robot Training (Quadruped Walker).

Features:
- Video recording every 50k timesteps (headless via getCameraImage)
- --watch flag for live GUI viewing at 1x real-time
- MPS (Apple Silicon) optimization
- Evolution Logger with rich terminal UI
- Domain randomization for sim-to-real transfer
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from src.env.base_env import QuadrupedEnv, SubconsciousEnv
from src.models.device_utils import (
    create_device_sync,
    get_device,
    get_device_string,
    print_device_info,
    warmup_device,
)
from src.models.policy_networks import get_policy_kwargs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Rich console for corporate-ready output
console = Console() if RICH_AVAILABLE else None

# Watch mode from environment variable (set by CLI)
WATCH_MODE = os.environ.get("ROBOT_WATCH_MODE", "0") == "1"


class VideoRecorderCallback(BaseCallback):
    """
    Custom callback for recording training videos using PyBullet's getCameraImage.

    Records 10-second MP4 clips every N timesteps in headless mode.
    """

    def __init__(
        self,
        video_folder: str | Path,
        record_freq: int = 50000,
        video_length: int = 240,  # ~10 seconds at 24 fps
        fps: int = 24,
        verbose: int = 1,
    ):
        """
        Initialize video recorder.

        Args:
            video_folder: Directory to save videos.
            record_freq: Record every N timesteps.
            video_length: Number of frames per video.
            fps: Frames per second for output video.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.video_folder = Path(video_folder)
        self.video_folder.mkdir(parents=True, exist_ok=True)
        self.record_freq = record_freq
        self.video_length = video_length
        self.fps = fps
        self._recording = False
        self._frames = []
        self._last_record_step = 0

    def _on_step(self) -> bool:
        # Check if we should start recording
        if not self._recording and self.num_timesteps - self._last_record_step >= self.record_freq:
            self._start_recording()

        # Capture frame if recording
        if self._recording:
            self._capture_frame()

            if len(self._frames) >= self.video_length:
                self._save_video()

        return True

    def _start_recording(self) -> None:
        """Start a new recording session."""
        self._recording = True
        self._frames = []
        self._last_record_step = self.num_timesteps
        if self.verbose > 0:
            logger.info(f"📹 Starting video recording at step {self.num_timesteps:,}")

    def _capture_frame(self) -> None:
        """Capture a frame from the environment."""
        try:
            # Get the first environment for rendering
            env = self.training_env.envs[0] if hasattr(self.training_env, "envs") else self.training_env
            
            # Try to get frame via render
            if hasattr(env, "render"):
                frame = env.render()
                if frame is not None:
                    self._frames.append(frame)
                    return

            # Fallback: try to access PyBullet directly
            if hasattr(env, "physics_client") and hasattr(env, "robot_id"):
                import pybullet as p
                
                pos, _ = p.getBasePositionAndOrientation(
                    env.robot_id, physicsClientId=env.physics_client
                )
                
                view_matrix = p.computeViewMatrix(
                    cameraEyePosition=[pos[0] - 1.5, pos[1] - 1.5, pos[2] + 0.8],
                    cameraTargetPosition=pos,
                    cameraUpVector=[0, 0, 1],
                )
                projection_matrix = p.computeProjectionMatrixFOV(
                    fov=60, aspect=640 / 480, nearVal=0.1, farVal=10.0
                )
                
                _, _, rgb, _, _ = p.getCameraImage(
                    640, 480,
                    viewMatrix=view_matrix,
                    projectionMatrix=projection_matrix,
                    renderer=p.ER_TINY_RENDERER,
                    physicsClientId=env.physics_client,
                )
                
                frame = np.array(rgb[:, :, :3], dtype=np.uint8)
                self._frames.append(frame)

        except Exception as e:
            if self.verbose > 0:
                logger.debug(f"Frame capture failed: {e}")

    def _save_video(self) -> None:
        """Save recorded frames to MP4."""
        if not self._frames:
            self._recording = False
            return

        try:
            import imageio

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_path = self.video_folder / f"quadruped_step_{self.num_timesteps}_{timestamp}.mp4"

            # Write video
            writer = imageio.get_writer(
                str(video_path),
                fps=self.fps,
                codec="libx264",
                quality=8,
            )

            for frame in self._frames:
                writer.append_data(frame)

            writer.close()

            if self.verbose > 0:
                logger.info(f"💾 Video saved: {video_path.name} ({len(self._frames)} frames)")

        except ImportError:
            logger.warning("imageio not installed, skipping video save")
        except Exception as e:
            logger.warning(f"Failed to save video: {e}")

        self._recording = False
        self._frames = []


class EvolutionLoggerCallback(BaseCallback):
    """
    Corporate-ready evolution logger with rich terminal UI.

    Prints status reports every N steps with:
    - Current mean reward
    - Distance traveled
    - Health (torso height)
    """

    def __init__(
        self,
        log_freq: int = 5000,
        verbose: int = 1,
    ):
        """
        Initialize evolution logger.

        Args:
            log_freq: Log every N timesteps.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        self._episode_rewards = []
        self._episode_lengths = []
        self._distances = []
        self._heights = []
        self._last_log_step = 0
        self._start_time = None

    def _on_training_start(self) -> None:
        self._start_time = time.time()

    def _on_step(self) -> bool:
        # Collect episode info
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])
                self._episode_lengths.append(info["episode"]["l"])
            if "position" in info:
                self._distances.append(info["position"][0])  # X distance
            if "position" in info:
                self._heights.append(info["position"][2])  # Z height

        # Log status every log_freq steps
        if self.num_timesteps - self._last_log_step >= self.log_freq:
            self._log_status()
            self._last_log_step = self.num_timesteps

        return True

    def _log_status(self) -> None:
        """Print evolution status report."""
        elapsed = time.time() - self._start_time if self._start_time else 0
        fps = self.num_timesteps / elapsed if elapsed > 0 else 0

        # Compute metrics
        mean_reward = np.mean(self._episode_rewards[-100:]) if self._episode_rewards else 0
        mean_length = np.mean(self._episode_lengths[-100:]) if self._episode_lengths else 0
        mean_distance = np.mean(self._distances[-100:]) if self._distances else 0
        mean_height = np.mean(self._heights[-100:]) if self._heights else 0

        if RICH_AVAILABLE and console:
            self._log_rich(mean_reward, mean_distance, mean_height, fps, elapsed)
        else:
            self._log_simple(mean_reward, mean_distance, mean_height, fps, elapsed)

    def _log_rich(
        self,
        mean_reward: float,
        mean_distance: float,
        mean_height: float,
        fps: float,
        elapsed: float,
    ) -> None:
        """Rich terminal output."""
        # Create status table
        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column(style="cyan bold")
        table.add_column(style="white")

        table.add_row("📊 Step", f"{self.num_timesteps:,}")
        table.add_row("⏱️  Time", f"{elapsed / 60:.1f} min")
        table.add_row("🚀 FPS", f"{fps:.0f}")
        table.add_row("", "")
        table.add_row("🎯 Mean Reward", f"{mean_reward:.2f}")
        table.add_row("📏 Distance", f"{mean_distance:.3f} m")
        table.add_row("💚 Health", f"{mean_height:.3f} m")

        # Health indicator
        health_status = "🟢 STABLE" if mean_height > 0.2 else "🟡 LOW" if mean_height > 0.1 else "🔴 CRITICAL"
        table.add_row("", "")
        table.add_row("Status", health_status)

        panel = Panel(
            table,
            title=f"[bold cyan]🦿 EVOLUTION STATUS[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

        console.print(panel)

    def _log_simple(
        self,
        mean_reward: float,
        mean_distance: float,
        mean_height: float,
        fps: float,
        elapsed: float,
    ) -> None:
        """Simple terminal output fallback."""
        print("\n" + "=" * 50)
        print("🦿 EVOLUTION STATUS")
        print("=" * 50)
        print(f"  Step:        {self.num_timesteps:,}")
        print(f"  Time:        {elapsed / 60:.1f} min")
        print(f"  FPS:         {fps:.0f}")
        print(f"  Mean Reward: {mean_reward:.2f}")
        print(f"  Distance:    {mean_distance:.3f} m")
        print(f"  Health:      {mean_height:.3f} m")
        health = "STABLE" if mean_height > 0.2 else "LOW" if mean_height > 0.1 else "CRITICAL"
        print(f"  Status:      {health}")
        print("=" * 50 + "\n")


class RewardLoggingCallback(BaseCallback):
    """Callback for logging reward components to TensorBoard."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "reward_components" in info and info["reward_components"]:
                for key, value in info["reward_components"].items():
                    self.logger.record(f"reward/{key}", value)
        return True


def make_env_fn(config: DictConfig, rank: int, seed: int = 0, watch: bool = False):
    """Create environment factory function."""
    def _init():
        render_mode = "human" if watch and rank == 0 else None
        env = QuadrupedEnv(config=config, render_mode=render_mode)
        env.reset(seed=seed + rank)
        return env
    return _init


def create_callbacks(
    config: DictConfig,
    log_dir: Path,
    enable_video: bool = True,
) -> CallbackList:
    """Create all training callbacks."""
    callbacks = []

    # Checkpoint callback - save every 50k steps
    checkpoint_dir = log_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    save_freq_per_env = config.training.save_freq // config.training.n_envs

    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq_per_env,
        save_path=str(checkpoint_dir),
        name_prefix="quadruped_model",
        save_replay_buffer=config.training.algorithm == "sac",
        save_vecnormalize=True,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)

    # Video recording callback (every 50k steps)
    if enable_video:
        video_folder = log_dir / "videos"
        video_callback = VideoRecorderCallback(
            video_folder=video_folder,
            record_freq=50000,
            video_length=240,  # 10 seconds at 24 fps
            fps=24,
            verbose=1,
        )
        callbacks.append(video_callback)
        logger.info(f"📹 Video recording enabled: {video_folder}")

    # Evolution logger (every 5k steps)
    evolution_callback = EvolutionLoggerCallback(log_freq=5000, verbose=1)
    callbacks.append(evolution_callback)

    # Reward logging callback
    reward_callback = RewardLoggingCallback()
    callbacks.append(reward_callback)

    # Evaluation callback
    if config.training.n_envs >= 2 and not WATCH_MODE:
        eval_env = QuadrupedEnv(config=config, render_mode=None)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(log_dir / "best_model"),
            log_path=str(log_dir / "eval_logs"),
            eval_freq=save_freq_per_env,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1,
        )
        callbacks.append(eval_callback)

    return CallbackList(callbacks)


def get_algorithm_class(algorithm: str):
    """Get SB3 algorithm class by name."""
    algorithms = {"ppo": PPO, "sac": SAC}
    algo_class = algorithms.get(algorithm.lower())
    if algo_class is None:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    return algo_class


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def train(config: DictConfig, watch: bool = False) -> None:
    """
    Main training function for Quadruped Walker.

    Args:
        config: Hydra configuration.
        watch: If True, run in GUI mode at 1x real-time.
    """
    global WATCH_MODE
    WATCH_MODE = watch

    # Print banner
    if RICH_AVAILABLE and console:
        console.print(Panel.fit(
            "[bold cyan]🦿 QUADRUPED WALKER[/bold cyan]\n"
            "[dim]Subconscious Robot Training System[/dim]",
            border_style="cyan",
        ))
    else:
        print("=" * 60)
        print("🦿 QUADRUPED WALKER - Subconscious Robot Training")
        print("=" * 60)

    # Print device info
    print_device_info()

    # Warmup device
    device = get_device(force_mps=True)
    warmup_device(device)

    # Create device sync
    device_sync = create_device_sync(device)

    # Print configuration
    if watch:
        logger.info("👁️ WATCH MODE: Running with GUI at 1x real-time speed")
    else:
        logger.info("🚀 FAST MODE: Running headless at maximum speed")

    logger.info(f"\n{OmegaConf.to_yaml(config)}")

    # Set random seed
    seed = config.seed

    # Create directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"quadruped_{config.training.algorithm}_{timestamp}"
    log_dir = Path(config.training.log_dir) / run_name
    log_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(config, log_dir / "config.yaml")

    # Create environment
    n_envs = 1 if watch else config.training.n_envs
    logger.info(f"Creating {n_envs} environment(s)...")

    if n_envs > 1:
        env = SubprocVecEnv(
            [make_env_fn(config, i, seed, watch) for i in range(n_envs)],
            start_method="spawn",
        )
    else:
        env = DummyVecEnv([make_env_fn(config, 0, seed, watch)])

    env = VecMonitor(env, str(log_dir / "monitor"))

    # Get device string for SB3
    device_str = get_device_string(force_mps=True)

    # Policy architecture
    policy_kwargs = {
        "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
        "activation_fn": torch.nn.Tanh,
    }

    # Create model
    logger.info(f"Initializing {config.training.algorithm.upper()} on {device_str}...")

    if config.training.algorithm == "ppo":
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir / "tensorboard"),
            verbose=1,
            seed=seed,
            device=device_str,
        )
    else:
        model = SAC(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            buffer_size=1000000,
            learning_starts=10000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(log_dir / "tensorboard"),
            verbose=1,
            seed=seed,
            device=device_str,
        )

    # Create callbacks (disable video in watch mode)
    callbacks = create_callbacks(config, log_dir, enable_video=not watch)

    # Training info
    logger.info("=" * 60)
    logger.info(f"🚀 Starting training for {config.training.total_timesteps:,} timesteps...")
    logger.info(f"📊 TensorBoard: tensorboard --logdir {log_dir / 'tensorboard'}")
    logger.info(f"📹 Videos: {log_dir / 'videos'}")
    logger.info("=" * 60)

    try:
        model.learn(
            total_timesteps=config.training.total_timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.warning("⚠️ Training interrupted")
    finally:
        # Sync device
        device_sync.sync()

        # Save model
        model.save(str(log_dir / "final_model.zip"))
        logger.info(f"✅ Model saved: {log_dir / 'final_model.zip'}")

        # Save to latest
        latest_dir = Path(config.training.checkpoint_dir) / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)
        model.save(str(latest_dir / "model.zip"))

    env.close()

    if RICH_AVAILABLE and console:
        console.print(Panel.fit(
            "[bold green]🏁 Training Complete![/bold green]",
            border_style="green",
        ))
    else:
        print("\n🏁 Training Complete!")


if __name__ == "__main__":
    train()
