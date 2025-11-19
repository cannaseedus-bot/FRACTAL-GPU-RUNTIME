# 🚀 Colab Integration - Quick Start

**Get started training in Colab and testing locally in 5 minutes!**

---

## 📋 What You Got

✅ **2 Production Colab Notebooks** for PyTorch model export and fine-tuning
✅ **Sync Utility** for automated testing of exported models
✅ **Web Dashboard** for monitoring training progress
✅ **Complete Documentation** with step-by-step guides
✅ **Real Working Code** - not demos!

---

## 🎯 The 5-Minute Workflow

### Step 1: Open Colab Notebook (1 minute)

Click one of these:

**Option A: Export Existing Model**
```
📓 colab/01_export_to_xjson.ipynb
```
- Load a pre-trained model
- Export to XJSON
- Download and test

**Option B: Fine-Tune New Model** (recommended)
```
📓 colab/02_finetune_with_local_testing.ipynb
```
- Create mini transformer
- Train for 5 epochs
- Auto-export checkpoints
- Download results

### Step 2: Run Colab Notebook (2 minutes)

In Colab:

1. **Connect to GPU**: Runtime → Change runtime type → GPU (T4)
2. **Run all cells**: Runtime → Run all
3. **Wait for training**: ~2 minutes for mini model
4. **Download checkpoint**: File downloads automatically

You'll get: `checkpoint_epoch_5.xjson` (or similar)

### Step 3: Test Locally (2 minutes)

On your machine:

```bash
# Save downloaded file to models directory
mv ~/Downloads/checkpoint_epoch_5.xjson ./models/

# Test the model
node scripts/sync-colab.js --test ./models/checkpoint_epoch_5.xjson
```

**Output:**
```
🧪 Testing: checkpoint_epoch_5.xjson
📦 Loading model...
✅ Model loaded
   Layers: 2
   Hidden: 128
   Heads: 4

🧪 Running test inferences...
   Testing: Short sequence (4 tokens)
   ✓ Latency: 42.15ms
   ✓ Output shape: [4, 1000]

📊 Test Summary:
   Total time: 1523ms
   GPU available: Yes
   CPU ops: 12
   GPU ops: 8
   Avg latency: 42.15ms

✅ Test report saved: test-reports/checkpoint_epoch_5_report.json
```

**Done!** You've successfully:
- ✅ Trained a model in Colab
- ✅ Exported to XJSON
- ✅ Tested locally with WebGPU
- ✅ Got performance metrics

---

## 🔄 Complete Workflow Example

### Day 1: Initial Training

```bash
# 1. Open Colab notebook
# 2. Train model (5 epochs, ~10 minutes)
# 3. Download all checkpoints

# 4. Test all checkpoints locally
npm run sync:test-all

# Output:
# ✅ checkpoint_epoch_1.xjson: 52.3ms latency
# ✅ checkpoint_epoch_2.xjson: 48.1ms latency
# ✅ checkpoint_epoch_3.xjson: 45.7ms latency
# ✅ checkpoint_epoch_4.xjson: 43.2ms latency
# ✅ checkpoint_epoch_5.xjson: 42.1ms latency ⭐ Best!
```

### Day 2: Refinement

```python
# Back in Colab - resume training
model, history = resume_training(
    checkpoint_path='checkpoint_epoch_5.pt',
    additional_epochs=5
)

# Train 5 more epochs, download checkpoint_epoch_10.xjson
```

```bash
# Test new checkpoint
node scripts/sync-colab.js --test ./models/checkpoint_epoch_10.xjson

# Output: 38.5ms latency - even better! ⭐⭐
```

### Day 3: Production

```bash
# All tests pass, deploy!
# Use final model with XJSON server, KLH orchestrator, etc.
```

---

## 📊 Monitoring Dashboard

Launch the web dashboard:

```bash
npm run dashboard
# Open http://localhost:8080
```

**Features:**
- 📈 Performance trends across epochs
- ✅ Test pass/fail status
- 🖥️ WebGPU availability check
- 📊 Latency comparisons
- 🔄 Auto-refresh every 30 seconds

---

## 🛠️ Available Commands

### NPM Scripts

```bash
# Test single model
npm run sync -- --test ./models/checkpoint_epoch_5.xjson

# Test all models in directory
npm run sync:test-all

# Watch for new models (auto-test on download)
npm run sync:watch

# Launch monitoring dashboard
npm run dashboard

# Run local inference example
npm run dev

# Run unit tests
npm test
```

### Direct Commands

```bash
# Sync utility help
node scripts/sync-colab.js --help

# Test with custom directories
node scripts/sync-colab.js --test ./models/my_model.xjson --report-dir ./reports

# Watch custom directory
node scripts/sync-colab.js --watch /path/to/models
```

---

## 📂 Directory Structure

```
FRACTAL-GPU-RUNTIME/
├── colab/
│   ├── 01_export_to_xjson.ipynb      ← Open in Colab
│   ├── 02_finetune_with_local_testing.ipynb  ← Open in Colab
│   └── README.md
├── models/
│   └── ← Put downloaded .xjson files here
├── test-reports/
│   └── ← Generated JSON reports appear here
├── dashboard/
│   └── index.html                     ← Web monitoring UI
├── scripts/
│   └── sync-colab.js                  ← Sync utility
├── examples/
│   ├── simple-inference.js            ← Basic example
│   ├── load-xjson.js                  ← XJSON loading
│   └── browser-demo.html              ← Interactive demo
└── COLAB_INTEGRATION.md               ← Full documentation
```

---

## 🎓 Learning Path

### Beginner
1. Run `npm run dev` to test the runtime
2. Open Colab notebook, train mini model
3. Download and test with sync script
4. View results in dashboard

### Intermediate
1. Modify Colab notebook with your dataset
2. Adjust model architecture (more layers, bigger hidden size)
3. Export quantized models (INT8) for faster downloads
4. Create custom test scripts

### Advanced
1. Integrate with @xjson/klh-orchestrator
2. Deploy to production with XJSON server
3. Set up CI/CD for automatic testing
4. Build distributed inference pipelines

---

## 📖 Documentation

| Document | Description |
|----------|-------------|
| `README.md` | Main project documentation |
| `QUICKSTART.md` | Local runtime quick start |
| `COLAB_INTEGRATION.md` | **Complete Colab workflow guide** ⭐ |
| `colab/README.md` | Colab notebooks overview |
| This file | Quick start for Colab integration |

---

## 🔥 Hot Tips

### Faster Iterations

```bash
# Use watch mode - auto-tests new models
npm run sync:watch &

# In another terminal, download from Colab as you train
# Tests run automatically!
```

### Google Drive Sync

```python
# In Colab - mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Models auto-save to Drive
output_dir = '/content/drive/MyDrive/fractal_models'

# Use Google Drive desktop app on your machine
# → Models sync automatically to local machine
# → Sync utility detects and tests them
```

### Batch Testing

```bash
# Download multiple checkpoints from Colab
# Test them all at once
npm run sync:test-all

# Compare results in dashboard
npm run dashboard
```

### Resume Training

```python
# In Colab - resume from any checkpoint
model, history = resume_training(
    checkpoint_path='/content/drive/MyDrive/fractal_models/checkpoint_epoch_5.pt',
    additional_epochs=10
)
```

---

## ❓ Common Questions

**Q: Do I need WebGPU to test locally?**
A: No! The runtime automatically falls back to CPU if WebGPU isn't available. But WebGPU is faster.

**Q: Can I use my own datasets?**
A: Yes! Replace `DummyTextDataset` in the Colab notebook with your actual data.

**Q: How do I deploy to production?**
A: After local validation, use `@xjson/xjson-server` or `@xjson/klh-orchestrator` to deploy.

**Q: Can I use HuggingFace models?**
A: Yes! The export notebook works with any PyTorch model, including HuggingFace transformers.

**Q: What if my model is too large?**
A: Use INT8 quantization in the export function to reduce file size by ~75%.

---

## 🎯 Next Steps

1. **Try it now**: Open a Colab notebook and run it
2. **Read the full guide**: See `COLAB_INTEGRATION.md`
3. **Experiment**: Modify architectures, datasets, hyperparameters
4. **Deploy**: When tests pass, deploy to production

---

## 🆘 Troubleshooting

### Model won't load locally
```bash
# Verify XJSON format
node -e "console.log(JSON.parse(require('fs').readFileSync('./models/model.xjson')).config)"
```

### WebGPU not available
```bash
# Check browser support
# Chrome/Edge 113+ required
# Runtime falls back to CPU automatically
```

### Tests fail
```bash
# Check test report for details
cat test-reports/checkpoint_epoch_5_report.json

# Look for:
# - "error" field for error messages
# - "verdict": "FAILED" for failed tests
```

### Colab runtime disconnects
```python
# In Colab - reconnect and resume
# Your models are saved to Google Drive
# Just reload from checkpoint
```

---

**Ready to train in the cloud and test locally!** 🚀

See [`COLAB_INTEGRATION.md`](./COLAB_INTEGRATION.md) for the complete guide.
