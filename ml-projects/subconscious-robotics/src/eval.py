"""
Evaluation Script for Subconscious Robot Training.

Tests learned behaviors with visual rendering and statistics reporting.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3 import PPO, SAC

from src.env.base_env import SubconsciousEnv

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(model_path: str) -> BaseAlgorithm:
    """
    Load a trained model from checkpoint.

    Args:
        model_path: Path to the model .zip file.

    Returns:
        Loaded SB3 model.

    Raises:
        FileNotFoundError: If model file doesn't exist.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Try PPO first, then SAC
    try:
        model = PPO.load(str(model_path))
        logger.info(f"Loaded PPO model from: {model_path}")
    except Exception:
        try:
            model = SAC.load(str(model_path))
            logger.info(f"Loaded SAC model from: {model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}") from e

    return model


def evaluate(
    model: BaseAlgorithm,
    env: SubconsciousEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
    render: bool = True,
    verbose: bool = True,
) -> dict[str, float]:
    """
    Evaluate a trained model.

    Args:
        model: Trained SB3 model.
        env: Evaluation environment.
        n_episodes: Number of evaluation episodes.
        deterministic: Whether to use deterministic actions.
        render: Whether to render the environment.
        verbose: Whether to print episode statistics.

    Returns:
        Dictionary with evaluation statistics.
    """
    episode_rewards = []
    episode_lengths = []
    successes = []
    final_distances = []

    logger.info(f"Starting evaluation for {n_episodes} episodes...")
    logger.info("-" * 60)

    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1

            if render:
                env.render()
                time.sleep(1.0 / 60.0)  # 60 FPS

        # Record statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(info.get("is_success", False))
        final_distances.append(info.get("distance", float("inf")))

        if verbose:
            status = "✓ SUCCESS" if info.get("is_success", False) else "✗ FAILED"
            logger.info(
                f"Episode {episode + 1:3d}/{n_episodes} | "
                f"Reward: {episode_reward:8.2f} | "
                f"Length: {episode_length:4d} | "
                f"Dist: {info.get('distance', 0):.4f} | "
                f"{status}"
            )

    # Compute statistics
    stats = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "min_reward": float(np.min(episode_rewards)),
        "max_reward": float(np.max(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(successes)),
        "mean_final_distance": float(np.mean(final_distances)),
    }

    # Print summary
    logger.info("-" * 60)
    logger.info("Evaluation Summary:")
    logger.info(f"  Mean Reward:      {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}")
    logger.info(f"  Mean Episode Len: {stats['mean_length']:.1f}")
    logger.info(f"  Success Rate:     {stats['success_rate'] * 100:.1f}%")
    logger.info(f"  Mean Final Dist:  {stats['mean_final_distance']:.4f}")
    logger.info("-" * 60)

    return stats


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Subconscious Robot model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the trained model (.zip file)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        default=True,
        help="Enable visual rendering",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Disable visual rendering",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="Use deterministic actions",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic actions",
    )

    args = parser.parse_args()

    # Handle render flag
    render = args.render and not args.no_render
    deterministic = args.deterministic and not args.stochastic

    # Load model
    model = load_model(args.model)

    # Create environment
    render_mode = "human" if render else None
    env = SubconsciousEnv(config=None, render_mode=render_mode)

    try:
        # Run evaluation
        stats = evaluate(
            model=model,
            env=env,
            n_episodes=args.episodes,
            deterministic=deterministic,
            render=render,
            verbose=True,
        )
    finally:
        env.close()

    return stats


if __name__ == "__main__":
    main()
