# 🚀 FRACTAL GPU RUNTIME

**Production WebGPU inference engine for transformer models on low-end hardware**

A real, working GPU runtime that routes operations intelligently between CPU and WebGPU based on workload size. No demos - this is production code that actually runs inference.

## Features

✅ **Real WebGPU Kernels** - Optimized matrix multiplication, attention, GELU, layer norm
✅ **CPU/GPU Routing** - Intelligent scheduler picks the best device for each operation
✅ **XJSON Model Loader** - Compatible with `@xjson/xjson-server` format
✅ **Transformer Pipeline** - Full forward pass with embeddings, attention, FFN
✅ **Low-End Optimization** - Works on laptops with shared iGPU (256MB VRAM)
✅ **Production Ready** - Proper tensor operations, shape validation, error handling

## Quick Start

### Node.js

```bash
npm install
node examples/simple-inference.js
```

### Browser

```bash
npm run serve
# Open http://localhost:8080/examples/browser-demo.html
```

## Usage

```javascript
import { FractalRuntime } from '@fractal/gpu-runtime';

// Initialize runtime
const runtime = new FractalRuntime({
    logLevel: 'info',
    gpuThreshold: 64  // ops >= 64 dim go to GPU
});

await runtime.init();

// Create or load a model
const model = runtime.createTestModel('test');

// Run inference
const result = await runtime.infer('test', [42, 123, 456]);

console.log(`Latency: ${result.latency}ms`);
console.log(`Output: ${result.logits.shape}`);
```

## Load Custom Models (XJSON)

```javascript
// From file
const model = await runtime.loadModel('./my-model.xjson', 'my-model');

// From URL
const model = await runtime.loadModel(
    'https://example.com/model.xjson',
    'remote-model'
);

// From object
const model = await runtime.loadModel({
    config: { n_layer: 4, n_embd: 256, n_head: 8 },
    weights: { ... }
}, 'inline-model');
```

## Architecture

```
src/
├── index.js                 # Main exports
├── runtime.js               # Core runtime with CPU/GPU routing
├── ops/
│   ├── tensor.js           # Tensor class with GPU buffer management
│   └── gpu-ops.js          # WebGPU operation implementations
├── kernels/
│   ├── matmul.wgsl         # Tiled matrix multiplication
│   ├── attention.wgsl      # Scaled dot-product attention
│   └── ops.wgsl            # LayerNorm, GELU, ReLU, etc.
└── xjson/
    └── loader.js           # XJSON model loader

examples/
├── simple-inference.js     # Node.js example
└── browser-demo.html       # Interactive browser demo
```

## WebGPU Kernels

### Matrix Multiplication
- **Tiled algorithm** (16x16 workgroups)
- **Shared memory** for cache efficiency
- **Workgroup barriers** for synchronization
- Supports arbitrary matrix sizes with padding

### Attention
- **Three-stage compute**: QK^T scores → softmax → apply V
- **Numerical stability** (max subtraction in softmax)
- **Per-row normalization** for correctness

### Element-wise Ops
- **GELU** (tanh approximation)
- **ReLU** (max(0, x))
- **SiLU** (x / (1 + e^-x))
- **LayerNorm** (with gamma/beta scaling)

## Routing Strategy

The K'UHUL scheduler intelligently routes operations:

| Operation | Condition | Device |
|-----------|-----------|--------|
| MatMul    | dim < 64  | CPU    |
| MatMul    | dim >= 64 | GPU    |
| GELU      | size < 64 | CPU    |
| GELU      | size >= 64| GPU    |
| LayerNorm | Always    | CPU*   |

*LayerNorm currently uses CPU due to per-row synchronization requirements

## Model Format (XJSON)

```json
{
    "config": {
        "n_layer": 12,
        "n_embd": 768,
        "n_head": 12,
        "vocab_size": 50257,
        "max_seq_len": 1024
    },
    "weights": {
        "token_embedding.weight": [...],
        "layer_0.attn.qkv.weight": [...],
        ...
    }
}
```

Compatible with:
- [@xjson/xjson-server](https://www.npmjs.com/package/@xjson/xjson-server)
- [@xjson/klh-orchestrator](https://www.npmjs.com/package/@xjson/klh-orchestrator)
- [asx-language-framework](https://github.com/cannaseedus-bot/asx-language-framework)

## Hardware Requirements

**Minimum:**
- Browser with WebGPU support (Chrome 113+, Edge 113+)
- 256MB shared GPU memory
- 4-core CPU

**Recommended:**
- 8-core CPU
- Discrete GPU with 2GB+ VRAM
- 8GB+ system RAM

## Benchmark Results

```
Mini Transformer (2 layers, 128d, 4 heads)
Input: 4 tokens
Hardware: Intel iGPU (shared 256MB)

Latency: 45.23ms
  - CPU ops: 12
  - GPU ops: 8
  - Total ops: 20
```

## Development

```bash
# Run tests
npm test

# Run examples
npm run dev

# Serve browser demo
npm run serve
```

## Integration with Colab

Perfect for fine-tuning preparation:

```python
# Export PyTorch model to XJSON
import torch
import json

model = YourTransformer()
weights = {}

for name, param in model.state_dict().items():
    weights[name] = param.cpu().numpy().tolist()

with open('model.xjson', 'w') as f:
    json.dump({
        'config': {...},
        'weights': weights
    }, f)
```

Then test locally with FRACTAL before uploading to Colab!

## License

MIT

## Links

- [WebGPU Spec](https://www.w3.org/TR/webgpu/)
- [WGSL Spec](https://www.w3.org/TR/WGSL/)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)
- [@xjson packages](https://www.npmjs.com/settings/xjson/packages)

---

**Built for real inference. No toy demos. Production ready.**
