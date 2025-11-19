# 🔄 Colab Integration Guide

**Complete workflow for training in Colab and testing locally with WebGPU runtime**

---

## 🎯 The Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    GOOGLE COLAB (Cloud GPU)                 │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 1. Fine-tune transformer model                        │  │
│  │ 2. Export to XJSON after each epoch                   │  │
│  │ 3. Save to Google Drive                               │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Download checkpoints
                            ↓
┌─────────────────────────────────────────────────────────────┐
│             LOCAL MACHINE (WebGPU Testing)                  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 4. Load XJSON model with FRACTAL runtime             │  │
│  │ 5. Run inference tests (WebGPU + CPU)                │  │
│  │ 6. Generate performance reports                      │  │
│  │ 7. Validate model quality                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Upload test results (optional)
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    BACK TO COLAB                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 8. Review test results                                │  │
│  │ 9. Adjust training if needed                          │  │
│  │ 10. Continue fine-tuning                              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 📚 Quick Start

### 1. Setup Local Environment

```bash
git clone https://github.com/your-username/FRACTAL-GPU-RUNTIME.git
cd FRACTAL-GPU-RUNTIME

# Create directories
mkdir -p models test-reports

# Test the runtime
node examples/simple-inference.js
```

### 2. Open Colab Notebooks

**Option A: Export Existing Model**
- Open: [`colab/01_export_to_xjson.ipynb`](./colab/01_export_to_xjson.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/FRACTAL-GPU-RUNTIME/blob/main/colab/01_export_to_xjson.ipynb)

**Option B: Fine-Tune Model**
- Open: [`colab/02_finetune_with_local_testing.ipynb`](./colab/02_finetune_with_local_testing.ipynb)
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-repo/FRACTAL-GPU-RUNTIME/blob/main/colab/02_finetune_with_local_testing.ipynb)

### 3. Export and Download

In Colab:
```python
# Model automatically exports to XJSON after each epoch
# Download from Google Drive or directly from notebook

from google.colab import files
files.download('fractal_model.xjson')
```

### 4. Test Locally

```bash
# Single model test
node scripts/sync-colab.js --test ./models/checkpoint_epoch_5.xjson

# Watch directory for new models
node scripts/sync-colab.js --watch ./models

# Test all models in directory
node scripts/sync-colab.js --test-all ./models
```

---

## 📖 Detailed Workflow

### Step 1: Training in Colab

#### A. Basic Export (Existing Model)

```python
# In Colab: 01_export_to_xjson.ipynb

from transformers import GPT2LMHeadModel
import json

# Load your model
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Export to XJSON
xjson = export_to_xjson(model, 'my_model.xjson')

# Download
from google.colab import files
files.download('my_model.xjson')
```

#### B. Fine-Tuning with Auto-Export

```python
# In Colab: 02_finetune_with_local_testing.ipynb

# Train model
trained_model, history = train_with_export(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    output_dir='/content/drive/MyDrive/fractal_models',
    epochs=5,
    export_every_epoch=True  # Auto-export after each epoch
)

# Checkpoints saved to Google Drive:
# - checkpoint_epoch_1.xjson
# - checkpoint_epoch_2.xjson
# - checkpoint_epoch_3.xjson
# - checkpoint_epoch_4.xjson
# - checkpoint_epoch_5.xjson
# - final_model.xjson
```

**What gets exported:**
- Model configuration
- All weights (as Float32)
- Layer structure
- Training metadata (epoch, loss, etc.)
- Compatible format for FRACTAL runtime

---

### Step 2: Download Models

#### Option A: Manual Download

1. Download from Colab notebook directly
2. Save to `FRACTAL-GPU-RUNTIME/models/`

#### Option B: Google Drive Sync

1. Mount Google Drive in Colab
2. Models saved automatically
3. Use Google Drive desktop sync
4. Models appear in `models/` directory

#### Option C: Command Line (with rclone)

```bash
# Setup rclone for Google Drive
rclone config

# Sync models
rclone sync gdrive:fractal_models ./models
```

---

### Step 3: Local Testing

#### A. Quick Test

```bash
# Test single model
node examples/load-xjson.js models/checkpoint_epoch_5.xjson
```

**Output:**
```
✅ Model loaded successfully!
  Layers: 2
  Hidden: 128
  Heads: 4
  Vocab: 1000

🚀 Running inference...
✅ Inference complete: 42.15ms
  Output shape: [5, 1000]
  CPU ops: 12
  GPU ops: 8
```

#### B. Automated Testing

```bash
# Watch for new models and auto-test
node scripts/sync-colab.js --watch ./models
```

**What happens:**
1. Monitors `./models` directory
2. Detects new XJSON files
3. Automatically loads and tests each model
4. Generates test reports
5. Saves reports to `./test-reports/`

#### C. Batch Testing

```bash
# Test all models at once
node scripts/sync-colab.js --test-all ./models
```

**Report includes:**
- Model configuration
- Inference latency (multiple sequence lengths)
- Output validation
- CPU vs GPU operation counts
- Performance metrics
- Pass/fail verdict

---

### Step 4: Review Test Results

#### Sample Test Report

```json
{
  "model_name": "checkpoint_epoch_5",
  "test_timestamp": "2025-01-15T10:30:00Z",
  "total_test_time_ms": 1523,
  "model_config": {
    "n_layer": 2,
    "n_embd": 128,
    "n_head": 4,
    "vocab_size": 1000
  },
  "runtime_stats": {
    "hasGPU": true,
    "opsRouted": { "cpu": 12, "gpu": 8 },
    "avgLatency": 42.15
  },
  "test_results": [
    {
      "name": "Short sequence (4 tokens)",
      "input_length": 4,
      "latency_ms": 38.52,
      "output_shape": [4, 1000]
    },
    {
      "name": "Medium sequence (16 tokens)",
      "input_length": 16,
      "latency_ms": 45.23,
      "output_shape": [16, 1000]
    },
    {
      "name": "Long sequence (64 tokens)",
      "input_length": 64,
      "latency_ms": 89.47,
      "output_shape": [64, 1000]
    }
  ],
  "verdict": "PASSED"
}
```

#### Interpret Results

**Good signs:**
- ✅ All tests pass
- ✅ Latency within expected range
- ✅ GPU operations > 0 (if WebGPU available)
- ✅ Output shapes correct

**Warning signs:**
- ⚠️ Very high latency
- ⚠️ All CPU operations (no GPU usage)
- ⚠️ Unexpected output shapes

**Action needed:**
- ❌ Tests fail
- ❌ Runtime errors
- ❌ NaN or Inf values

---

### Step 5: Iterate

Based on test results, go back to Colab and:

#### A. Continue Training

```python
# Resume from checkpoint
model, history = resume_training(
    checkpoint_path='/content/drive/MyDrive/fractal_models/checkpoint_epoch_5.pt',
    additional_epochs=5
)
```

#### B. Adjust Hyperparameters

```python
# Lower learning rate
trained_model, history = train_with_export(
    model=model,
    learning_rate=1e-5,  # Was 1e-4
    epochs=5
)
```

#### C. Change Architecture

```python
# Larger model
model = SimpleTransformer(
    vocab_size=1000,
    n_embd=256,  # Was 128
    n_head=8,    # Was 4
    n_layer=4    # Was 2
)
```

---

## 🛠️ Advanced Features

### Custom Test Scripts

```javascript
// custom-test.js
import { FractalRuntime } from './src/index.js';

async function customTest() {
    const runtime = new FractalRuntime();
    await runtime.init();

    // Load model
    const model = await runtime.loadModel('./models/my_model.xjson', 'test');

    // Your custom test logic
    const tokens = [1, 2, 3, 4, 5];
    const result = await runtime.infer('test', tokens);

    // Custom validation
    assert(result.latency < 100, 'Latency too high');
    assert(result.logits.shape[0] === tokens.length, 'Wrong output length');

    console.log('✅ Custom test passed');
}

customTest();
```

### Benchmark Script

```javascript
// benchmark.js
async function benchmark(modelPath, iterations = 100) {
    const runtime = new FractalRuntime();
    await runtime.init();

    const model = await runtime.loadModel(modelPath, 'bench');
    const latencies = [];

    for (let i = 0; i < iterations; i++) {
        const tokens = Array.from({ length: 32 }, () => Math.floor(Math.random() * 1000));
        const result = await runtime.infer('bench', tokens);
        latencies.push(result.latency);
    }

    const avg = latencies.reduce((a, b) => a + b) / latencies.length;
    const min = Math.min(...latencies);
    const max = Math.max(...latencies);
    const p95 = latencies.sort((a, b) => a - b)[Math.floor(iterations * 0.95)];

    console.log(`Benchmark Results (${iterations} iterations):`);
    console.log(`  Avg: ${avg.toFixed(2)}ms`);
    console.log(`  Min: ${min.toFixed(2)}ms`);
    console.log(`  Max: ${max.toFixed(2)}ms`);
    console.log(`  P95: ${p95.toFixed(2)}ms`);
}

benchmark('./models/final_model.xjson');
```

### Quantization for Faster Downloads

In Colab:

```python
# Export with INT8 quantization
quantized_model = export_quantized(
    model,
    output_path='model_int8.xjson',
    bits=8  # 75% size reduction
)

# File size: ~5MB instead of ~20MB
```

### Continuous Integration

```bash
# .github/workflows/test-models.yml
name: Test Colab Models

on:
  push:
    paths:
      - 'models/*.xjson'

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
      - run: npm install
      - run: node scripts/sync-colab.js --test-all ./models
      - run: |
          if grep -q "FAILED" test-reports/*.json; then
            echo "❌ Some models failed tests"
            exit 1
          fi
```

---

## 📊 Comparison: Colab vs Local

| Aspect | Colab (Cloud) | Local (WebGPU) |
|--------|---------------|----------------|
| **Training** | ✅ Fast (T4/V100 GPU) | ❌ Slow |
| **Inference** | ✅ Fast | ✅ Fast (WebGPU) |
| **Cost** | 💰 Free tier limited | ✅ Free |
| **Latency** | ⚠️ Network overhead | ✅ Instant |
| **Iteration speed** | ⚠️ Upload/download | ✅ Instant |
| **Development** | ⚠️ Notebook-based | ✅ Full IDE |
| **Testing** | ⚠️ Limited to Python | ✅ Real browser/Node.js |

**Best Practice:** Train in Colab, test and iterate locally

---

## 🔗 Integration with XJSON Ecosystem

### Deploy with KLH Orchestrator

After local validation:

```javascript
// Deploy to distributed inference
import { KLHOrchestrator } from '@xjson/klh-orchestrator';

const orchestrator = new KLHOrchestrator();

// Upload validated model
await orchestrator.deployModel({
    name: 'my-finetuned-model',
    xjsonPath: './models/final_model.xjson',
    replicas: 3,
    region: 'us-east-1'
});
```

### Serve with XJSON Server

```javascript
import { XJSONServer } from '@xjson/xjson-server';

const server = new XJSONServer({
    port: 3000,
    runtime: FractalRuntime
});

// Load model
await server.loadModel('./models/final_model.xjson', 'production');

// Start server
server.listen(() => {
    console.log('Server running on http://localhost:3000');
});

// Inference endpoint:
// POST /infer
// { "model": "production", "tokens": [1, 2, 3] }
```

---

## 🎓 Example Workflow Timeline

**Day 1: Initial Training**
- ⏰ 9:00 AM - Start Colab notebook
- ⏰ 9:05 AM - Begin training (5 epochs)
- ⏰ 9:35 AM - Training complete
- ⏰ 9:40 AM - Download epoch 5 checkpoint
- ⏰ 9:45 AM - Test locally → **Latency: 45ms**
- ⏰ 9:50 AM - Validation passed ✅

**Day 2: Refinement**
- ⏰ 2:00 PM - Resume training in Colab
- ⏰ 2:30 PM - Train 5 more epochs
- ⏰ 3:00 PM - Test epoch 10 → **Latency: 42ms** (improved!)
- ⏰ 3:05 PM - Validate quality ✅

**Day 3: Production**
- ⏰ 10:00 AM - Final training run
- ⏰ 11:00 AM - Export final model
- ⏰ 11:15 AM - Local tests pass
- ⏰ 11:30 AM - Deploy to production
- ⏰ 11:45 AM - **Live inference serving** 🎉

---

## 📝 Troubleshooting

### Model won't load

```javascript
// Check XJSON format
const data = JSON.parse(fs.readFileSync('./models/model.xjson'));
console.log(data.config);  // Should have n_layer, n_embd, etc.
console.log(Object.keys(data.weights));  // Should have weight tensors
```

### WebGPU not available

```javascript
// Check browser/environment
if (!navigator.gpu) {
    console.log('WebGPU not supported - using CPU fallback');
}

// Runtime automatically falls back to CPU
const runtime = new FractalRuntime({ gpuThreshold: 99999 });  // Force CPU
```

### High latency

```bash
# Profile operations
node --inspect scripts/sync-colab.js --test models/model.xjson

# Check GPU usage
runtime.getStats()  // Should show gpu ops > 0
```

### Google Drive sync issues

```python
# In Colab - verify mount
!ls /content/drive/MyDrive/fractal_models

# Re-mount if needed
from google.colab import drive
drive.flush_and_unmount()
drive.mount('/content/drive')
```

---

## 📚 Resources

- **Colab Notebooks:** [`/colab`](./colab/)
- **Examples:** [`/examples`](./examples/)
- **Sync Script:** [`/scripts/sync-colab.js`](./scripts/sync-colab.js)
- **Quick Start:** [`QUICKSTART.md`](./QUICKSTART.md)
- **Main README:** [`README.md`](./README.md)

## 🆘 Support

- GitHub Issues: [Report bugs](https://github.com/your-repo/FRACTAL-GPU-RUNTIME/issues)
- Discussions: [Ask questions](https://github.com/your-repo/FRACTAL-GPU-RUNTIME/discussions)

---

**Ready to train in the cloud and test locally!** 🚀
