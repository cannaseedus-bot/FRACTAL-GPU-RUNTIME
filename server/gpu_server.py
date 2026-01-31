#!/usr/bin/env python3
"""
ASXR-GPU REST API Server
Provides HTTP endpoints for GPU-accelerated K'UHUL operations.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from typing import Any

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

from gpu.cuda_engine import get_gpu_engine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("asxr-gpu-server")

app = Flask(__name__)
CORS(app)

gpu_engine = None


def initialize_gpu_engine() -> None:
    global gpu_engine
    try:
        gpu_engine = get_gpu_engine()
        logger.info("✅ GPU engine initialized")
    except Exception as exc:
        logger.error("❌ GPU engine initialization failed: %s", exc)
        gpu_engine = None


init_thread = threading.Thread(target=initialize_gpu_engine, daemon=True)
init_thread.start()


def now() -> str:
    return datetime.utcnow().isoformat() + "Z"


@app.route("/api/gpu/health", methods=["GET"])
def gpu_health():
    if gpu_engine is None:
        return jsonify(
            {
                "ok": False,
                "service": "ASXR-GPU Server",
                "version": "4.0.0",
                "error": "GPU engine not initialized",
                "timestamp": now(),
            }
        )

    gpu_info = gpu_engine.get_gpu_info()
    return jsonify(
        {
            "ok": True,
            "service": "ASXR-GPU Server",
            "version": "4.0.0",
            "gpu": gpu_info.__dict__,
            "timestamp": now(),
        }
    )


@app.route("/api/gpu/info", methods=["GET"])
def gpu_info():
    if gpu_engine is None:
        return jsonify({"error": "GPU engine not initialized"}), 503
    return jsonify(gpu_engine.get_gpu_info().__dict__)


@app.route("/api/kuhul/execute", methods=["POST"])
def kuhul_execute():
    if gpu_engine is None:
        return jsonify({"error": "GPU engine not initialized"}), 503

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No JSON data provided"}), 400

    operation = data.get("operation")
    args = data.get("args", [])
    kwargs = data.get("kwargs", {})

    if not operation:
        return jsonify({"error": "Operation is required"}), 400

    try:
        start_time = time.time()
        result = gpu_engine.execute_kuhul(operation, *args, **kwargs)
        exec_time = time.time() - start_time
        result = convert_numpy_to_list(result)
        return jsonify(
            {
                "success": True,
                "operation": operation,
                "result": result,
                "execution_time": exec_time,
                "timestamp": now(),
            }
        )
    except Exception as exc:
        logger.error("K'UHUL execution failed: %s", exc)
        return (
            jsonify(
                {
                    "success": False,
                    "error": str(exc),
                    "operation": operation,
                    "timestamp": now(),
                }
            ),
            500,
        )


def convert_numpy_to_list(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    return obj


@app.route("/api/kuhul/batch", methods=["POST"])
def kuhul_batch():
    if gpu_engine is None:
        return jsonify({"error": "GPU engine not initialized"}), 503

    data = request.get_json(silent=True)
    operations = data.get("operations", []) if data else []
    if not operations:
        return jsonify({"error": "No operations provided"}), 400

    results = []
    total_start = time.time()
    for op_data in operations:
        op_start = time.time()
        operation = op_data.get("operation")
        args = op_data.get("args", [])
        kwargs = op_data.get("kwargs", {})
        try:
            result = gpu_engine.execute_kuhul(operation, *args, **kwargs)
            results.append(
                {
                    "operation": operation,
                    "success": True,
                    "result": convert_numpy_to_list(result),
                    "execution_time": time.time() - op_start,
                }
            )
        except Exception as exc:
            results.append(
                {
                    "operation": operation,
                    "success": False,
                    "error": str(exc),
                    "execution_time": time.time() - op_start,
                }
            )

    return jsonify(
        {
            "success": True,
            "total_operations": len(operations),
            "total_time": time.time() - total_start,
            "operations": results,
            "timestamp": now(),
        }
    )


@app.route("/api/gpu/encrypt", methods=["POST"])
def gpu_encrypt():
    if gpu_engine is None:
        return jsonify({"error": "GPU engine not initialized"}), 503

    data = request.get_json(silent=True) or {}
    plaintext = data.get("data")
    path_key = data.get("path_key", "M0,0 C100,50 200,150 300,0")

    if not plaintext:
        return jsonify({"error": "No data provided"}), 400

    try:
        result = gpu_engine.vector_encrypt(plaintext, path_key)
        return jsonify(
            {
                "success": True,
                "encrypted": convert_numpy_to_list(result),
                "timestamp": now(),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/gpu/compress", methods=["POST"])
def gpu_compress():
    if gpu_engine is None:
        return jsonify({"error": "GPU engine not initialized"}), 503

    data = request.get_json(silent=True) or {}
    geometry = data.get("geometry")
    method = data.get("method", "rotational")
    angle = data.get("angle", 45)

    if geometry is None:
        return jsonify({"error": "No geometry data provided"}), 400

    try:
        if method == "rotational":
            result = gpu_engine.rotational_compression(np.array(geometry), angle)
        else:
            return jsonify({"error": f"Unknown compression method: {method}"}), 400
        return jsonify({"success": True, "method": method, "result": result, "timestamp": now()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/neural/generate", methods=["POST"])
def neural_generate():
    if gpu_engine is None:
        return jsonify({"error": "GPU engine not initialized"}), 503

    data = request.get_json(silent=True) or {}
    input_data = data.get("input")
    params = data.get("params", {})

    if not input_data:
        return jsonify({"error": "No input data provided"}), 400

    try:
        result = gpu_engine.neural_path_generation(input_data, params)
        return jsonify({"success": True, "result": result, "timestamp": now()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/gpu/memory", methods=["GET"])
def gpu_memory():
    if gpu_engine is None:
        return jsonify({"error": "GPU engine not initialized"}), 503

    info = gpu_engine.get_gpu_info()
    return jsonify(
        {
            "device": info.device,
            "memory_allocated_gb": info.memory_allocated_gb or 0,
            "memory_reserved_gb": info.memory_reserved_gb or 0,
            "total_memory_gb": info.total_memory_gb or 0,
            "gpu_memory_used": info.gpu_memory_used,
            "cuda_version": info.cuda_version,
        }
    )


@app.route("/api/gpu/memory/clear", methods=["POST"])
def clear_memory():
    if gpu_engine is None:
        return jsonify({"error": "GPU engine not initialized"}), 503

    try:
        gpu_engine.clear_gpu_memory()
        return jsonify({"success": True, "message": "GPU memory cleared", "timestamp": now()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.errorhandler(404)
def not_found(_error):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    logger.error("Internal server error: %s", error)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    init_thread.join(timeout=10)
    if gpu_engine is None:
        logger.warning("⚠️ GPU engine not initialized, running in limited mode")

    logger.info("🚀 ASXR-GPU Server starting...")
    print("\n" + "=" * 60)
    print("🚀 ASXR-GPU v4.0 - Python GPU Accelerated Runtime")
    print("=" * 60)
    print("📡 REST API Server")
    print("🌐 http://localhost:4750")
    print("\n📊 Health:   GET    /api/gpu/health")
    print("🎯 Execute:  POST   /api/kuhul/execute")
    print("🔐 Encrypt:  POST   /api/gpu/encrypt")
    print("🗜️ Compress: POST   /api/gpu/compress")
    print("🧠 Generate: POST   /api/neural/generate")
    print("🧠 Batch:    POST   /api/kuhul/batch")
    print("💾 Memory:   GET    /api/gpu/memory")
    print("🧹 Clear:    POST   /api/gpu/memory/clear")
    print("=" * 60 + "\n")

    app.run(host="0.0.0.0", port=4750, debug=False)
