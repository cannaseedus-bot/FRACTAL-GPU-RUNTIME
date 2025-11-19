# 📓 Colab Notebooks

Google Colab notebooks for training and exporting models.

## Notebooks

### 01_export_to_xjson.ipynb
**Export existing PyTorch models to XJSON format**

- Load pre-trained models (HuggingFace, custom, etc.)
- Export to XJSON format
- Download for local testing

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

### 02_finetune_with_local_testing.ipynb
**Fine-tune models with automatic export and testing**

- Full training loop with GPU acceleration
- Auto-export checkpoints after each epoch
- Save to Google Drive for easy sync
- Resume training from checkpoints

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)

## Quick Start

1. **Open notebook in Colab**
2. **Connect to GPU runtime** (Runtime → Change runtime type → GPU)
3. **Run all cells** or step through
4. **Download exported XJSON** files
5. **Test locally** with WebGPU runtime

## Workflow

```
Colab (Train) → Download → Local (Test) → Back to Colab (Iterate)
```

See [`COLAB_INTEGRATION.md`](../COLAB_INTEGRATION.md) for complete guide.

## Output Format

All notebooks export to **XJSON format**:

```json
{
  "config": { "n_layer": 2, "n_embd": 128, ... },
  "layers": [...],
  "weights": {...},
  "metadata": {...}
}
```

Compatible with:
- `@fractal/gpu-runtime`
- `@xjson/xjson-server`
- `@xjson/klh-orchestrator`

## Tips

- **Use Google Drive** for persistent storage
- **Export every epoch** to track progress
- **Test locally** before continuing training
- **Resume from checkpoints** if needed

## Support

See main [documentation](../README.md) for more information.
