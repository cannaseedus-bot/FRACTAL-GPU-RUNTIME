#!/usr/bin/env python3
"""
ASXR-GPU CUDA Engine
Python GPU acceleration with PyTorch/CUDA for K'UHUL operations.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asxr-gpu-engine")


@dataclass
class GPUInfo:
    device: str
    initialized: bool
    cuda_available: bool
    operations_registered: int
    gpu_memory_used: int
    cuda_version: Optional[str] = None
    gpu_name: Optional[str] = None
    total_memory_gb: Optional[float] = None
    memory_allocated_gb: Optional[float] = None
    memory_reserved_gb: Optional[float] = None


class ASXRGPUEngine:
    """Main GPU engine for ASXR-GPU runtime."""

    def __init__(self, device: Optional[str] = None) -> None:
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.gpu_memory: Dict[str, torch.Tensor] = {}
        self.operations: Dict[str, Any] = {}
        self.initialized = False
        self.cuda_version = torch.version.cuda if torch.cuda.is_available() else None

        logger.info("🚀 ASXR-GPU Engine initialized on %s", self.device)
        if torch.cuda.is_available():
            logger.info("  CUDA Version: %s", self.cuda_version)
            logger.info("  GPU: %s", torch.cuda.get_device_name(0))
            logger.info(
                "  Memory: %.1f GB",
                torch.cuda.get_device_properties(0).total_memory / 1e9,
            )

    def initialize(self) -> None:
        """Initialize engine components."""
        try:
            if self.device.type == "cuda":
                warmup_tensor = torch.randn(512, 512, device=self.device)
                _ = warmup_tensor @ warmup_tensor.T
                torch.cuda.synchronize()

            self.register_kuhul_operations()
            self.initialized = True
            logger.info("✅ ASXR-GPU Engine ready for K'UHUL operations")
        except Exception as exc:
            logger.error("❌ GPU initialization failed: %s", exc)
            raise

    def register_kuhul_operations(self) -> None:
        self.operations = {
            "(⤍)": self.vector_encrypt,
            "(⤎)": self.vector_decrypt,
            "(⤏)": self.path_key_derivation,
            "(⤐)": self.bezier_cryptography,
            "(↻)": self.rotational_compression,
            "(↔)": self.symmetrical_compression,
            "(⤒)": self.hierarchical_compression,
            "(⤓)": self.progressive_detail,
            "(⟲)": self.spherical_loop,
            "(⤦)": self.vector_conditional,
            "(⤧)": self.path_iteration,
            "(⤨)": self.gradient_flow_control,
            "(⟿)": self.neural_path_generation,
            "(⤂)": self.weight_vector_application,
            "(⤃)": self.activation_shape_morph,
            "(⤄)": self.gradient_backpropagation,
        }
        logger.info("✅ Registered %s K'UHUL operations", len(self.operations))

    def execute_kuhul(self, operation: str, *args: Any, **kwargs: Any) -> Any:
        if not self.initialized:
            raise RuntimeError("GPU engine not initialized")

        if operation not in self.operations:
            raise ValueError(f"Unknown K'UHUL operation: {operation}")

        start_time = time.time()
        try:
            result = self.operations[operation](*args, **kwargs)
            exec_time = time.time() - start_time
            logger.debug("✅ %s executed in %.3fs", operation, exec_time)
            return result
        except Exception as exc:
            logger.error("❌ %s failed: %s", operation, exc)
            raise

    def vector_encrypt(self, data: Any, path_key: Optional[str] = None) -> np.ndarray:
        logger.info("🔐 GPU Vector Encryption: %s", type(data))
        data_tensor = self._to_tensor(data)
        key_tensor = (
            self._path_to_key(path_key, data_tensor.shape)
            if path_key
            else torch.randn_like(data_tensor, device=self.device)
        )
        encrypted = torch.bitwise_xor(data_tensor.long(), key_tensor.long())
        self.gpu_memory["encrypted"] = encrypted
        return encrypted.cpu().numpy()

    def vector_decrypt(self, encrypted_data: Any, path_key: str) -> np.ndarray:
        logger.info("🔓 GPU Vector Decryption")
        encrypted_tensor = self._to_tensor(encrypted_data)
        key_tensor = self._path_to_key(path_key, encrypted_tensor.shape)
        decrypted = torch.bitwise_xor(encrypted_tensor.long(), key_tensor.long())
        self.gpu_memory["decrypted"] = decrypted
        return decrypted.cpu().numpy()

    def path_key_derivation(self, path: str) -> str:
        logger.info("🧭 Path key derivation")
        return hashlib.sha256(path.encode()).hexdigest()[:24]

    def bezier_cryptography(self, path: str) -> str:
        logger.info("➿ Bezier cryptography")
        return hashlib.md5(path.encode()).hexdigest()[:24]

    def rotational_compression(self, geometry: Any, angle: float) -> Dict[str, Any]:
        logger.info("🗜️ GPU Rotational Compression: %s°", angle)
        geometry_tensor = self._to_tensor(geometry).float()
        angle_rad = torch.tensor(angle * np.pi / 180, device=self.device)
        cos_a = torch.cos(angle_rad)
        sin_a = torch.sin(angle_rad)

        if geometry_tensor.ndim >= 2 and geometry_tensor.shape[-1] >= 2:
            rot_matrix = torch.stack(
                [torch.stack([cos_a, -sin_a]), torch.stack([sin_a, cos_a])]
            ).to(self.device)
            rotated = torch.matmul(geometry_tensor[..., :2], rot_matrix)
            remainder = geometry_tensor[..., 2:]
            result = (
                torch.cat([rotated, remainder], dim=-1)
                if remainder.numel() > 0
                else rotated
            )
        else:
            result = geometry_tensor

        compression_ratio = torch.abs(torch.sin(angle_rad))
        compressed = result * compression_ratio

        return {
            "original_shape": tuple(geometry_tensor.shape),
            "compressed_shape": tuple(compressed.shape),
            "ratio": compression_ratio.item(),
            "data": compressed.cpu().numpy(),
        }

    def symmetrical_compression(self, geometry: Any, plane: str = "vertical") -> Dict[str, Any]:
        logger.info("🪞 GPU Symmetrical Compression: %s", plane)
        geometry_tensor = self._to_tensor(geometry)
        compressed = geometry_tensor[..., ::2]
        return {
            "plane": plane,
            "compressed_shape": tuple(compressed.shape),
            "data": compressed.cpu().numpy(),
        }

    def hierarchical_compression(self, geometry: Any, levels: int = 2) -> Dict[str, Any]:
        logger.info("🧱 GPU Hierarchical Compression: %s", levels)
        geometry_tensor = self._to_tensor(geometry)
        compressed = geometry_tensor
        for _ in range(max(levels, 1)):
            compressed = compressed[..., ::2]
        return {
            "levels": levels,
            "compressed_shape": tuple(compressed.shape),
            "data": compressed.cpu().numpy(),
        }

    def progressive_detail(self, geometry: Any, detail: float = 0.5) -> Dict[str, Any]:
        logger.info("🧩 GPU Progressive Detail: %s", detail)
        geometry_tensor = self._to_tensor(geometry)
        scale = torch.tensor(detail, device=self.device)
        refined = geometry_tensor.float() * scale
        return {
            "detail": detail,
            "data": refined.cpu().numpy(),
        }

    def neural_path_generation(self, input_data: Any, network_params: Optional[Dict] = None) -> Dict[str, Any]:
        logger.info("🧠 GPU Neural Path Generation")
        if network_params is None:
            network_params = {"hidden_size": 256, "num_layers": 3, "dropout": 0.1}

        input_tensor = self._to_tensor(input_data).float()
        if input_tensor.ndim == 1:
            input_tensor = input_tensor.unsqueeze(0)

        hidden_size = network_params.get("hidden_size", 256)
        layers = []
        in_features = input_tensor.shape[-1]
        for i in range(network_params.get("num_layers", 3)):
            out_features = hidden_size // (2**i) if i > 0 else hidden_size
            layers.append(torch.nn.Linear(in_features, out_features).to(self.device))
            layers.append(torch.nn.ReLU())
            dropout = network_params.get("dropout", 0)
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            in_features = out_features
        layers.append(torch.nn.Linear(in_features, 3))
        model = torch.nn.Sequential(*layers)

        with torch.no_grad():
            output = model(input_tensor)

        points = output.cpu().numpy()
        path_d = "M" + " L".join([f"{p[0]},{p[1]}" for p in points[0]])
        return {"path": path_d, "points": points, "network_params": network_params}

    def spherical_loop(self, radius: float, degrees: float = 360) -> Dict[str, Any]:
        logger.info("🌀 GPU Spherical Loop: r=%s, θ=%s°", radius, degrees)
        steps = int(degrees / 15)
        theta = torch.linspace(0, degrees * np.pi / 180, steps, device=self.device)
        phi = torch.linspace(0, 2 * np.pi, 12, device=self.device)
        theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing="ij")
        x = radius * torch.sin(theta_grid) * torch.cos(phi_grid)
        y = radius * torch.sin(theta_grid) * torch.sin(phi_grid)
        z = radius * torch.cos(theta_grid)
        points = torch.stack([x, y, z, theta_grid, phi_grid], dim=-1)
        return {"radius": radius, "degrees": degrees, "points": points.cpu().numpy()}

    def vector_conditional(self, condition: bool, on_true: Any, on_false: Any) -> Any:
        logger.info("🔀 Vector Conditional")
        return on_true if condition else on_false

    def path_iteration(self, path: str) -> str:
        logger.info("🧵 Path Iteration")
        return path

    def gradient_flow_control(self, gradient: Any, control: Any) -> Dict[str, Any]:
        logger.info("🌈 Gradient Flow Control")
        return {"gradient": gradient, "control": control}

    def weight_vector_application(self, weights: Any, geometry: Any) -> Dict[str, Any]:
        logger.info("⚖️ Weight Vector Application")
        return {"weights": weights, "geometry": geometry}

    def activation_shape_morph(self, shapes: Any, activation: Any) -> Dict[str, Any]:
        logger.info("🔺 Activation Shape Morph")
        return {"shapes": shapes, "activation": activation}

    def gradient_backpropagation(self, gradients: Any, learning_rate: float) -> Dict[str, Any]:
        logger.info("📉 Gradient Backpropagation")
        return {"gradients": gradients, "learning_rate": learning_rate}

    def get_gpu_info(self) -> GPUInfo:
        info = GPUInfo(
            device=str(self.device),
            initialized=self.initialized,
            cuda_available=torch.cuda.is_available(),
            operations_registered=len(self.operations),
            gpu_memory_used=len(self.gpu_memory),
        )
        if torch.cuda.is_available():
            info.cuda_version = self.cuda_version
            info.gpu_name = torch.cuda.get_device_name(0)
            info.total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            info.memory_allocated_gb = torch.cuda.memory_allocated(0) / 1e9
            info.memory_reserved_gb = torch.cuda.memory_reserved(0) / 1e9
        return info

    def clear_gpu_memory(self) -> None:
        self.gpu_memory.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 GPU memory cleared")

    def _to_tensor(self, data: Any) -> torch.Tensor:
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        if isinstance(data, str):
            return torch.tensor([ord(c) for c in data], device=self.device)
        if isinstance(data, np.ndarray):
            return torch.tensor(data, device=self.device)
        if isinstance(data, list):
            return torch.tensor(data, device=self.device)
        return torch.tensor(data, device=self.device)

    def _path_to_key(self, path: str, shape: torch.Size) -> torch.Tensor:
        path_hash = hashlib.sha256(path.encode()).hexdigest()
        seed = int(path_hash[:8], 16)
        torch.manual_seed(seed)
        return torch.randint(0, 255, shape, device=self.device)


_gpu_engine: Optional[ASXRGPUEngine] = None


def get_gpu_engine(device: Optional[str] = None) -> ASXRGPUEngine:
    global _gpu_engine
    if _gpu_engine is None:
        _gpu_engine = ASXRGPUEngine(device)
        _gpu_engine.initialize()
    return _gpu_engine
