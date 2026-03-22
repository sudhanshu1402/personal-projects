"""
Device Utilities for Hardware Acceleration (Apple Silicon Optimized).

Provides device detection with MPS priority for M-series chips,
including synchronization utilities for CPU-GPU coordination.
"""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from typing import Generator

logger = logging.getLogger(__name__)


class DeviceSync:
    """
    Synchronization utility for CPU-GPU coordination.

    Ensures PyBullet physics steps (CPU) stay synchronized with
    Neural Network updates (GPU/MPS).
    """

    def __init__(self, device: torch.device):
        """
        Initialize sync utility.

        Args:
            device: PyTorch device being used.
        """
        self.device = device
        self.is_mps = device.type == "mps"
        self.is_cuda = device.type == "cuda"
        self._step_count = 0
        self._last_sync_time = time.perf_counter()

    def sync(self) -> None:
        """
        Synchronize CPU and GPU/MPS operations.

        Call this after physics steps to ensure tensors are ready
        before network forward pass.
        """
        if self.is_mps:
            # MPS synchronization - ensure all MPS operations complete
            torch.mps.synchronize()
        elif self.is_cuda:
            # CUDA synchronization
            torch.cuda.synchronize()
        # CPU doesn't need sync

    def sync_if_needed(self, step_interval: int = 100) -> None:
        """
        Conditional sync based on step interval.

        Syncs every N steps to balance performance with accuracy.

        Args:
            step_interval: Number of steps between syncs.
        """
        self._step_count += 1
        if self._step_count % step_interval == 0:
            self.sync()

    @contextmanager
    def timed_sync(self) -> Generator[None, None, None]:
        """
        Context manager for timed GPU operations.

        Yields:
            None, but ensures sync after block completion.
        """
        start = time.perf_counter()
        yield
        self.sync()
        elapsed = time.perf_counter() - start
        if elapsed > 0.1:  # Log slow operations
            logger.debug(f"GPU operation took {elapsed:.3f}s")

    def get_sync_stats(self) -> dict[str, float]:
        """Get synchronization statistics."""
        current_time = time.perf_counter()
        elapsed = current_time - self._last_sync_time
        self._last_sync_time = current_time
        return {
            "steps": self._step_count,
            "time_since_last_sync": elapsed,
        }


@lru_cache(maxsize=1)
def get_device(force_mps: bool = True) -> torch.device:
    """
    Get the best available device for PyTorch operations.

    Priority (when force_mps=True):
    1. Apple Metal Performance Shaders (MPS) - for M-series chips
    2. CUDA - for NVIDIA GPUs
    3. CPU - fallback

    Args:
        force_mps: If True, prioritize MPS over CUDA.

    Returns:
        torch.device: The selected device.
    """
    device = torch.device("cpu")
    device_name = "CPU"

    # Check for Apple MPS (Metal Performance Shaders) - PRIORITY for M-series
    if force_mps and hasattr(torch.backends, "mps"):
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            device = torch.device("mps")
            device_name = "Apple MPS (Metal)"
            logger.info("🍎 Apple Metal Performance Shaders (MPS) activated")
            logger.info("   Optimized for M1/M2/M3 Neural Engine")

            # Verify MPS is working
            try:
                test_tensor = torch.zeros(1, device=device)
                del test_tensor
                logger.info("   ✓ MPS tensor allocation verified")
            except Exception as e:
                logger.warning(f"   ⚠ MPS test failed, falling back to CPU: {e}")
                device = torch.device("cpu")
                device_name = "CPU (MPS fallback)"

            return device

    # Check for CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = f"CUDA ({torch.cuda.get_device_name(0)})"
        logger.info(f"🖥️ CUDA available: {torch.cuda.get_device_name(0)}")
        return device

    # CPU fallback
    logger.info("💻 Using CPU (no GPU acceleration available)")
    return device


def get_device_string(force_mps: bool = True) -> str:
    """
    Get device as string for Stable Baselines3.

    SB3 expects device as string: 'cuda', 'mps', 'cpu', or 'auto'.

    Args:
        force_mps: If True, prioritize MPS.

    Returns:
        Device string compatible with SB3.
    """
    device = get_device(force_mps=force_mps)
    return str(device.type)


def check_mps_compatibility() -> dict[str, bool]:
    """
    Check MPS compatibility details for Apple Silicon.

    Returns:
        Dictionary with compatibility information.
    """
    info = {
        "mps_available": False,
        "mps_built": False,
        "mps_working": False,
        "cuda_available": False,
        "cpu_available": True,
        "recommended_device": "cpu",
    }

    # Check MPS
    if hasattr(torch.backends, "mps"):
        info["mps_available"] = torch.backends.mps.is_available()
        info["mps_built"] = torch.backends.mps.is_built()

        if info["mps_available"] and info["mps_built"]:
            try:
                test_tensor = torch.zeros(1, device="mps")
                del test_tensor
                info["mps_working"] = True
                info["recommended_device"] = "mps"
            except Exception:
                info["mps_working"] = False

    # Check CUDA
    info["cuda_available"] = torch.cuda.is_available()
    if info["cuda_available"] and not info["mps_working"]:
        info["recommended_device"] = "cuda"

    return info


def create_device_sync(device: torch.device | str | None = None) -> DeviceSync:
    """
    Create a DeviceSync instance for the given device.

    Args:
        device: Device to sync. If None, uses auto-detected device.

    Returns:
        DeviceSync instance.
    """
    if device is None:
        device = get_device()
    elif isinstance(device, str):
        device = torch.device(device)

    return DeviceSync(device)


def print_device_info() -> None:
    """Print detailed device information with corporate-ready formatting."""
    print("\n" + "=" * 60)
    print("⚡ PyTorch Device Configuration")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")

    compat = check_mps_compatibility()

    print(f"\n📊 Device Availability:")
    print(f"   CPU: ✓ Always available")
    print(f"   CUDA: {'✓' if compat['cuda_available'] else '✗'}")
    print(f"   MPS (Apple Metal): {'✓' if compat['mps_available'] else '✗'}")

    if compat["mps_available"]:
        print(f"   MPS Built: {'✓' if compat['mps_built'] else '✗'}")
        print(f"   MPS Working: {'✓' if compat['mps_working'] else '✗'}")

    device = get_device()
    print(f"\n🎯 Selected device: {device}")
    print(f"   Recommended: {compat['recommended_device']}")

    # Memory info for CUDA
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"\n🖥️ GPU Memory: {props.total_memory / 1e9:.1f} GB")

    print("=" * 60 + "\n")


def warmup_device(device: torch.device | None = None) -> None:
    """
    Warm up the device with initial allocations.

    This helps avoid initialization latency during training.

    Args:
        device: Device to warm up. If None, uses auto-detected device.
    """
    if device is None:
        device = get_device()

    logger.info(f"Warming up {device}...")

    try:
        # Small tensor operations to initialize device
        for size in [100, 1000, 10000]:
            x = torch.randn(size, size, device=device)
            y = torch.mm(x, x)
            del x, y

        # Sync to ensure warmup completes
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

        # Clear cache
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

        logger.info(f"✓ {device} warmup complete")
    except Exception as e:
        logger.warning(f"Device warmup failed: {e}")


if __name__ == "__main__":
    print_device_info()
    warmup_device()
