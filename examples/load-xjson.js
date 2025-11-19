/**
 * Example: Loading and running inference with XJSON models
 * Compatible with @xjson/xjson-server format
 */

import { FractalRuntime } from '../src/index.js';
import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

async function main() {
    console.log('='.repeat(60));
    console.log('FRACTAL GPU RUNTIME - XJSON Model Loading Example');
    console.log('='.repeat(60));

    // Initialize runtime
    const runtime = new FractalRuntime({
        logLevel: 'info',
        gpuThreshold: 64
    });

    await runtime.init();

    console.log('\n📦 Loading XJSON model...');

    // Load XJSON model file
    const modelPath = join(__dirname, 'xjson-model.json');
    const model = await runtime.loadModel(modelPath, 'xjson-test');

    console.log('\n✅ Model loaded successfully!');
    console.log('\nModel Details:');
    console.log(`  Name: ${model.metadata.name || 'unnamed'}`);
    console.log(`  Version: ${model.metadata.version || 'unknown'}`);
    console.log(`  Architecture: ${model.architecture}`);
    console.log(`  Layers: ${model.config.n_layer}`);
    console.log(`  Hidden Size: ${model.config.n_embd}`);
    console.log(`  Attention Heads: ${model.config.n_head}`);
    console.log(`  Vocab Size: ${model.config.vocab_size}`);
    console.log(`  Max Sequence Length: ${model.config.max_seq_len}`);

    console.log('\n📊 Layer Structure:');
    model.layers.forEach((layer, i) => {
        console.log(`  ${i}. ${layer.type} (${layer.name})`);
    });

    // Initialize weights (in production, these would be loaded from the file)
    console.log('\n⚙️  Initializing model weights...');
    runtime.initializeWeights(model);

    console.log(`  Total weight tensors: ${model.weights.size}`);

    // Run inference
    console.log('\n🚀 Running inference...');
    console.log('-'.repeat(60));

    const inputTokens = [42, 123, 456, 789, 101];
    console.log(`Input tokens: [${inputTokens.join(', ')}]`);

    const result = await runtime.infer('xjson-test', inputTokens);

    console.log('\n✅ Inference Results:');
    console.log(`  Latency: ${result.latency.toFixed(2)}ms`);
    console.log(`  Output shape: [${result.logits.shape.join(', ')}]`);
    console.log(`  Output size: ${result.logits.size} values`);

    // Show logits for first token
    const firstTokenLogits = result.logits.data.slice(0, model.config.vocab_size);
    const topK = 5;
    const topIndices = firstTokenLogits
        .map((val, idx) => ({ val, idx }))
        .sort((a, b) => b.val - a.val)
        .slice(0, topK);

    console.log(`\n  Top ${topK} predicted tokens (first position):`);
    topIndices.forEach((item, rank) => {
        console.log(`    ${rank + 1}. Token ${item.idx}: ${item.val.toFixed(4)}`);
    });

    // Show runtime stats
    const stats = runtime.getStats();
    console.log('\n📈 Runtime Statistics:');
    console.log(`  GPU Available: ${stats.hasGPU ? 'Yes' : 'No'}`);
    console.log(`  CPU Operations: ${stats.opsRouted.cpu}`);
    console.log(`  GPU Operations: ${stats.opsRouted.gpu}`);
    console.log(`  Total Inferences: ${stats.totalInferences}`);
    console.log(`  Average Latency: ${stats.avgLatency.toFixed(2)}ms`);

    // Demonstrate compatibility info
    console.log('\n🔗 Compatibility:');
    if (model.metadata.compatible_with) {
        model.metadata.compatible_with.forEach(pkg => {
            console.log(`  ✓ ${pkg}`);
        });
    }

    console.log('\n' + '='.repeat(60));
    console.log('Example complete!');
    console.log('='.repeat(60));

    console.log('\n💡 Next Steps:');
    console.log('  1. Export your PyTorch/TF model to XJSON format');
    console.log('  2. Test inference locally with this runtime');
    console.log('  3. Deploy to Colab for fine-tuning');
    console.log('  4. Use @xjson/klh-orchestrator for distributed inference');
}

main().catch(console.error);
