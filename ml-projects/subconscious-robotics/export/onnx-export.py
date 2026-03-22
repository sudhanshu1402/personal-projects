"""
ONNX Export Functionality for Hardware Deployment.

Converts trained Stable Baselines3 models to ONNX format for
deployment on edge devices and robotics hardware.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from stable_baselines3.common.base_class import BaseAlgorithm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


class PolicyWrapper(nn.Module):
    """
    Wrapper to extract the policy network for ONNX export.

    Extracts just the action prediction from the SB3 policy.
    """

    def __init__(self, policy, observation_space_shape: tuple[int, ...]):
        """
        Initialize policy wrapper.

        Args:
            policy: SB3 policy object.
            observation_space_shape: Shape of observation space.
        """
        super().__init__()
        self.policy = policy
        self.obs_shape = observation_space_shape

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning deterministic action.

        Args:
            observation: Observation tensor.

        Returns:
            Action tensor.
        """
        # Get features
        features = self.policy.extract_features(observation)
        if hasattr(self.policy, "mlp_extractor"):
            latent_pi, _ = self.policy.mlp_extractor(features)
        else:
            latent_pi = features

        # Get action
        mean_actions = self.policy.action_net(latent_pi)

        # For SAC, we want the mean action (deterministic)
        # For PPO, this is already the action
        return mean_actions


def export_to_onnx(
    model_path: str,
    output_path: str,
    opset_version: int = 14,
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    dynamic_axes: dict | None = None,
) -> Path:
    """
    Export SB3 model to ONNX format.

    Args:
        model_path: Path to the trained SB3 model (.zip).
        output_path: Path for output ONNX file.
        opset_version: ONNX opset version.
        input_names: Names for input tensors.
        output_names: Names for output tensors.
        dynamic_axes: Dynamic axes configuration.

    Returns:
        Path to the exported ONNX file.
    """
    from stable_baselines3 import PPO, SAC

    model_path = Path(model_path)
    output_path = Path(output_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load model
    logger.info(f"Loading model from: {model_path}")
    try:
        model = PPO.load(str(model_path))
        algo_name = "PPO"
    except Exception:
        model = SAC.load(str(model_path))
        algo_name = "SAC"

    logger.info(f"Loaded {algo_name} model")

    # Get observation space shape
    obs_shape = model.observation_space.shape
    logger.info(f"Observation shape: {obs_shape}")

    # Create wrapper
    policy = model.policy
    policy.eval()

    wrapped_policy = PolicyWrapper(policy, obs_shape)
    wrapped_policy.eval()

    # Create dummy input
    dummy_input = torch.randn(1, *obs_shape)

    # Set default names
    if input_names is None:
        input_names = ["observation"]
    if output_names is None:
        output_names = ["action"]
    if dynamic_axes is None:
        dynamic_axes = {
            "observation": {0: "batch_size"},
            "action": {0: "batch_size"},
        }

    # Export to ONNX
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to ONNX: {output_path}")

    torch.onnx.export(
        wrapped_policy,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    logger.info("✓ ONNX export successful")

    # Verify the exported model
    verify_onnx(output_path, obs_shape)

    return output_path


def verify_onnx(onnx_path: Path, obs_shape: tuple[int, ...]) -> bool:
    """
    Verify exported ONNX model.

    Args:
        onnx_path: Path to ONNX file.
        obs_shape: Expected observation shape.

    Returns:
        True if verification passes.
    """
    try:
        import onnx
        import onnxruntime as ort

        # Check ONNX model
        logger.info("Verifying ONNX model...")
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        logger.info("✓ ONNX model check passed")

        # Test inference
        logger.info("Testing ONNX inference...")
        session = ort.InferenceSession(str(onnx_path))

        # Create test input
        test_input = np.random.randn(1, *obs_shape).astype(np.float32)
        input_name = session.get_inputs()[0].name

        # Run inference
        result = session.run(None, {input_name: test_input})

        logger.info(f"✓ Inference test passed")
        logger.info(f"  Input shape: {test_input.shape}")
        logger.info(f"  Output shape: {result[0].shape}")

        # Model info
        logger.info("\nONNX Model Info:")
        logger.info(f"  File size: {onnx_path.stat().st_size / 1024:.1f} KB")
        logger.info(f"  Input: {session.get_inputs()[0].name} - {session.get_inputs()[0].shape}")
        logger.info(f"  Output: {session.get_outputs()[0].name} - {session.get_outputs()[0].shape}")

        return True

    except ImportError as e:
        logger.warning(f"ONNX verification skipped (missing dependency): {e}")
        return True
    except Exception as e:
        logger.error(f"ONNX verification failed: {e}")
        return False


def optimize_onnx(onnx_path: Path, output_path: Path | None = None) -> Path:
    """
    Optimize ONNX model for inference.

    Args:
        onnx_path: Path to input ONNX file.
        output_path: Path for optimized output. Defaults to <name>_optimized.onnx.

    Returns:
        Path to optimized ONNX file.
    """
    try:
        import onnx
        from onnxruntime.transformers import optimizer

        logger.info("Optimizing ONNX model...")

        if output_path is None:
            output_path = onnx_path.with_stem(f"{onnx_path.stem}_optimized")

        # Basic optimization using ONNX Runtime
        optimized = optimizer.optimize_model(str(onnx_path))
        optimized.save_model_to_file(str(output_path))

        logger.info(f"✓ Optimized model saved to: {output_path}")
        logger.info(
            f"  Size reduction: "
            f"{onnx_path.stat().st_size / 1024:.1f} KB → "
            f"{output_path.stat().st_size / 1024:.1f} KB"
        )

        return output_path

    except ImportError:
        logger.warning("ONNX optimization skipped (onnxruntime-transformers not installed)")
        return onnx_path


def main():
    """Main ONNX export entry point."""
    parser = argparse.ArgumentParser(
        description="Export trained model to ONNX format",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to trained SB3 model (.zip)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for ONNX file",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version",
    )
    parser.add_argument(
        "--optimize",
        action="store_true",
        help="Apply ONNX optimization",
    )

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        model_path = Path(args.model)
        args.output = str(model_path.with_suffix(".onnx"))

    # Export
    onnx_path = export_to_onnx(
        model_path=args.model,
        output_path=args.output,
        opset_version=args.opset,
    )

    # Optionally optimize
    if args.optimize:
        optimize_onnx(onnx_path)

    logger.info("\n" + "=" * 50)
    logger.info("Export complete!")
    logger.info(f"ONNX model: {onnx_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
