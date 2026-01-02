"""Hardware detection and configuration for optimal performance."""

import platform
from dataclasses import dataclass
from enum import Enum
from typing import Any

from loguru import logger


class HardwareType(str, Enum):
    """Hardware acceleration types."""

    MPS = "mps"  # Apple Metal Performance Shaders
    CUDA = "cuda"  # NVIDIA CUDA
    CPU = "cpu"  # CPU only


@dataclass
class HardwareConfig:
    """Hardware configuration with optimal settings."""

    device_type: HardwareType
    device: Any  # torch.device
    batch_size: int
    use_fp16: bool
    num_workers: int
    additional_config: dict[str, Any]


class HardwareDetector:
    """Detect available hardware and return optimal configuration."""

    @staticmethod
    def detect() -> HardwareConfig:
        """
        Detect available hardware and return optimal configuration.

        Returns:
            HardwareConfig with device and optimal settings
        """
        try:
            import torch
        except ImportError:
            logger.warning("PyTorch not installed, defaulting to CPU")
            return HardwareDetector._get_cpu_config()

        # Check for MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            logger.info("Detected Apple Silicon with MPS support")
            return HardwareDetector._get_mps_config(torch)

        # Check for CUDA (NVIDIA GPU)
        if torch.cuda.is_available():
            logger.info(f"Detected CUDA GPU: {torch.cuda.get_device_name(0)}")
            return HardwareDetector._get_cuda_config(torch)

        # Fallback to CPU
        logger.info("No GPU detected, using CPU")
        return HardwareDetector._get_cpu_config(torch)

    @staticmethod
    def _get_mps_config(torch) -> HardwareConfig:
        """Get MPS configuration for Apple Silicon."""
        return HardwareConfig(
            device_type=HardwareType.MPS,
            device=torch.device("mps"),
            batch_size=8,
            use_fp16=False,  # MPS doesn't fully support fp16 yet
            num_workers=4,
            additional_config={
                "use_mlx": True,  # Use MLX for VLM pipeline
                "platform": "darwin",
                "chip": HardwareDetector._get_apple_chip(),
            },
        )

    @staticmethod
    def _get_cuda_config(torch) -> HardwareConfig:
        """Get CUDA configuration for NVIDIA GPUs."""
        # Get GPU memory to determine batch size
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        # Scale batch size based on GPU memory
        if gpu_memory_gb >= 24:  # High-end GPU (A100, RTX 4090, etc.)
            batch_size = 16
        elif gpu_memory_gb >= 16:  # Mid-range GPU (V100, RTX 3090, etc.)
            batch_size = 12
        elif gpu_memory_gb >= 8:  # Entry-level GPU (RTX 3060, etc.)
            batch_size = 8
        else:
            batch_size = 4

        return HardwareConfig(
            device_type=HardwareType.CUDA,
            device=torch.device("cuda:0"),
            batch_size=batch_size,
            use_fp16=True,  # Use mixed precision on CUDA
            num_workers=8,
            additional_config={
                "use_mlx": False,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_gb": gpu_memory_gb,
                "cuda_version": torch.version.cuda,
            },
        )

    @staticmethod
    def _get_cpu_config(torch=None) -> HardwareConfig:
        """Get CPU-only configuration."""
        device = torch.device("cpu") if torch else "cpu"

        # Detect CPU cores
        try:
            import os

            cpu_count = os.cpu_count() or 4
        except Exception:
            cpu_count = 4

        return HardwareConfig(
            device_type=HardwareType.CPU,
            device=device,
            batch_size=4,  # Smaller batch size for CPU
            use_fp16=False,
            num_workers=min(cpu_count, 8),
            additional_config={
                "use_mlx": False,
                "cpu_count": cpu_count,
            },
        )

    @staticmethod
    def _get_apple_chip() -> str:
        """Detect Apple Silicon chip type."""
        if platform.system() != "Darwin":
            return "unknown"

        try:
            import subprocess

            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            chip_info = result.stdout.strip()

            # Parse chip type (M1, M2, M3, etc.)
            if "Apple M" in chip_info:
                # Extract M1, M2, M3, etc.
                chip_type = chip_info.split("Apple ")[1].split()[0]
                return chip_type

            return chip_info
        except Exception as e:
            logger.warning(f"Could not detect Apple chip: {e}")
            return "unknown"

    @staticmethod
    def get_device_info() -> dict[str, Any]:
        """
        Get detailed hardware information.

        Returns:
            Dictionary with hardware details
        """
        config = HardwareDetector.detect()

        info = {
            "device_type": config.device_type.value,
            "batch_size": config.batch_size,
            "use_fp16": config.use_fp16,
            "num_workers": config.num_workers,
            **config.additional_config,
        }

        return info
