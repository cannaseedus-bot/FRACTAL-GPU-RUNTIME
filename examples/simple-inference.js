/**
 * Simple inference example - runs a mini transformer model
 * Usage: node examples/simple-inference.js
 */

import { FractalRuntime } from '../src/index.js';

async function main() {
    console.log('='.repeat(60));
    console.log('FRACTAL GPU RUNTIME - Simple Inference Example');
    console.log('='.repeat(60));

    // Initialize runtime
    const runtime = new FractalRuntime({
        logLevel: 'info',
        gpuThreshold: 64
    });

    await runtime.init();

    // Create a test model
    const model = runtime.createTestModel('test');
    console.log('\nModel Config:');
    console.log(`  - Layers: ${model.config.n_layer}`);
    console.log(`  - Hidden Size: ${model.config.n_embd}`);
    console.log(`  - Attention Heads: ${model.config.n_head}`);
    console.log(`  - Vocab Size: ${model.config.vocab_size}`);

    // Run inference
    console.log('\n' + '-'.repeat(60));
    console.log('Running inference...');
    console.log('-'.repeat(60));

    const inputTokens = [42, 123, 456, 789]; // Example token IDs

    const result = await runtime.infer('test', inputTokens);

    console.log('\nInference Results:');
    console.log(`  - Latency: ${result.latency.toFixed(2)}ms`);
    console.log(`  - Output shape: [${result.logits.shape.join(', ')}]`);
    console.log(`  - First 5 logits: [${result.logits.data.slice(0, 5).map(v => v.toFixed(4)).join(', ')}...]`);

    // Show stats
    const stats = runtime.getStats();
    console.log('\nRuntime Statistics:');
    console.log(`  - GPU Available: ${stats.hasGPU ? 'Yes' : 'No'}`);
    console.log(`  - Ops routed to CPU: ${stats.opsRouted.cpu}`);
    console.log(`  - Ops routed to GPU: ${stats.opsRouted.gpu}`);
    console.log(`  - Total inferences: ${stats.totalInferences}`);
    console.log(`  - Avg latency: ${stats.avgLatency.toFixed(2)}ms`);

    console.log('\n' + '='.repeat(60));
    console.log('Inference complete!');
    console.log('='.repeat(60));
}

main().catch(console.error);
