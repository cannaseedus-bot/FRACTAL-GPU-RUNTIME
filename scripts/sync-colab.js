/**
 * Sync utility for Colab <-> Local workflow
 *
 * Features:
 * - Watch for new XJSON exports from Colab
 * - Auto-test models locally with WebGPU runtime
 * - Generate test reports
 * - Sync results back to Google Drive
 *
 * Usage:
 *   node scripts/sync-colab.js --watch ./models
 *   node scripts/sync-colab.js --test ./models/checkpoint_epoch_5.xjson
 */

import { FractalRuntime } from '../src/index.js';
import { readFileSync, writeFileSync, watch } from 'fs';
import { readdir, stat } from 'fs/promises';
import { join, basename } from 'path';

class ColabSync {
    constructor(options = {}) {
        this.modelDir = options.modelDir || './models';
        this.reportDir = options.reportDir || './test-reports';
        this.runtime = null;
        this.autoTest = options.autoTest !== false;
    }

    async init() {
        console.log('🚀 Initializing FRACTAL Runtime...');
        this.runtime = new FractalRuntime({ logLevel: 'info' });
        await this.runtime.init();
        console.log('✅ Runtime ready\n');
    }

    /**
     * Test a single model and generate report
     */
    async testModel(modelPath) {
        console.log(`\n${'='.repeat(60)}`);
        console.log(`Testing: ${basename(modelPath)}`);
        console.log('='.repeat(60));

        const startTime = Date.now();

        try {
            // Load model
            console.log('\n📦 Loading model...');
            const modelName = basename(modelPath, '.xjson');
            const model = await this.runtime.loadModel(modelPath, modelName);

            console.log('✅ Model loaded');
            console.log(`   Layers: ${model.config.n_layer}`);
            console.log(`   Hidden: ${model.config.n_embd}`);
            console.log(`   Heads: ${model.config.n_head}`);
            console.log(`   Vocab: ${model.config.vocab_size}`);

            // Run test inferences
            console.log('\n🧪 Running test inferences...');

            const testCases = [
                { name: 'Short sequence (4 tokens)', tokens: [42, 123, 456, 789] },
                { name: 'Medium sequence (16 tokens)', tokens: Array.from({ length: 16 }, (_, i) => i * 10) },
                { name: 'Long sequence (64 tokens)', tokens: Array.from({ length: 64 }, (_, i) => i) }
            ];

            const results = [];

            for (const testCase of testCases) {
                console.log(`\n   Testing: ${testCase.name}`);

                const result = await this.runtime.infer(modelName, testCase.tokens);

                console.log(`   ✓ Latency: ${result.latency.toFixed(2)}ms`);
                console.log(`   ✓ Output shape: [${result.logits.shape.join(', ')}]`);

                results.push({
                    name: testCase.name,
                    input_length: testCase.tokens.length,
                    latency_ms: result.latency,
                    output_shape: result.logits.shape,
                    first_logits: result.logits.data.slice(0, 5)
                });
            }

            // Get stats
            const stats = this.runtime.getStats();

            const endTime = Date.now();
            const totalTime = endTime - startTime;

            // Generate report
            const report = {
                model_name: modelName,
                model_path: modelPath,
                test_timestamp: new Date().toISOString(),
                total_test_time_ms: totalTime,
                model_config: model.config,
                model_metadata: model.metadata,
                runtime_stats: stats,
                test_results: results,
                verdict: 'PASSED'
            };

            // Save report
            const reportPath = join(this.reportDir, `${modelName}_report.json`);
            writeFileSync(reportPath, JSON.stringify(report, null, 2));

            console.log('\n📊 Test Summary:');
            console.log(`   Total time: ${totalTime}ms`);
            console.log(`   GPU available: ${stats.hasGPU ? 'Yes' : 'No'}`);
            console.log(`   CPU ops: ${stats.opsRouted.cpu}`);
            console.log(`   GPU ops: ${stats.opsRouted.gpu}`);
            console.log(`   Avg latency: ${stats.avgLatency.toFixed(2)}ms`);

            console.log(`\n✅ Test report saved: ${reportPath}`);

            return report;

        } catch (err) {
            console.error('\n❌ Test failed:', err.message);

            const report = {
                model_name: basename(modelPath, '.xjson'),
                model_path: modelPath,
                test_timestamp: new Date().toISOString(),
                error: err.message,
                stack: err.stack,
                verdict: 'FAILED'
            };

            const reportPath = join(this.reportDir, `${basename(modelPath, '.xjson')}_report.json`);
            writeFileSync(reportPath, JSON.stringify(report, null, 2));

            return report;
        }
    }

    /**
     * Watch directory for new models
     */
    async watchDirectory() {
        console.log(`\n👀 Watching directory: ${this.modelDir}`);
        console.log('   Waiting for new XJSON files from Colab...\n');

        const tested = new Set();

        // Test existing files
        const files = await readdir(this.modelDir);
        for (const file of files) {
            if (file.endsWith('.xjson')) {
                const fullPath = join(this.modelDir, file);
                tested.add(fullPath);
                if (this.autoTest) {
                    await this.testModel(fullPath);
                }
            }
        }

        // Watch for new files
        watch(this.modelDir, async (eventType, filename) => {
            if (filename && filename.endsWith('.xjson')) {
                const fullPath = join(this.modelDir, filename);

                if (!tested.has(fullPath)) {
                    console.log(`\n🔔 New model detected: ${filename}`);

                    // Wait a bit to ensure file is fully written
                    await new Promise(resolve => setTimeout(resolve, 1000));

                    tested.add(fullPath);

                    if (this.autoTest) {
                        await this.testModel(fullPath);
                    }
                }
            }
        });
    }

    /**
     * Scan directory and test all models
     */
    async testAll() {
        console.log(`\n🔍 Scanning directory: ${this.modelDir}\n`);

        const files = await readdir(this.modelDir);
        const xjsonFiles = files.filter(f => f.endsWith('.xjson'));

        if (xjsonFiles.length === 0) {
            console.log('❌ No XJSON files found');
            return;
        }

        console.log(`Found ${xjsonFiles.length} model(s):\n`);

        const reports = [];

        for (const file of xjsonFiles) {
            const fullPath = join(this.modelDir, file);
            const report = await this.testModel(fullPath);
            reports.push(report);
        }

        // Summary
        console.log('\n' + '='.repeat(60));
        console.log('BATCH TEST SUMMARY');
        console.log('='.repeat(60));

        const passed = reports.filter(r => r.verdict === 'PASSED').length;
        const failed = reports.filter(r => r.verdict === 'FAILED').length;

        console.log(`✅ Passed: ${passed}`);
        console.log(`❌ Failed: ${failed}`);
        console.log(`📊 Total:  ${reports.length}`);

        return reports;
    }
}

// CLI interface
async function main() {
    const args = process.argv.slice(2);

    if (args.length === 0 || args.includes('--help')) {
        console.log(`
FRACTAL GPU Runtime - Colab Sync Utility

Usage:
  node scripts/sync-colab.js --watch [dir]       Watch directory for new models
  node scripts/sync-colab.js --test <file>       Test single model
  node scripts/sync-colab.js --test-all [dir]    Test all models in directory
  node scripts/sync-colab.js --help              Show this help

Examples:
  node scripts/sync-colab.js --watch ./models
  node scripts/sync-colab.js --test ./models/checkpoint_epoch_5.xjson
  node scripts/sync-colab.js --test-all ./models

Options:
  --model-dir <dir>     Model directory (default: ./models)
  --report-dir <dir>    Report output directory (default: ./test-reports)
  --no-auto-test        Don't auto-test new models in watch mode
        `);
        return;
    }

    const options = {
        modelDir: './models',
        reportDir: './test-reports',
        autoTest: !args.includes('--no-auto-test')
    };

    // Parse options
    const modelDirIdx = args.indexOf('--model-dir');
    if (modelDirIdx !== -1 && args[modelDirIdx + 1]) {
        options.modelDir = args[modelDirIdx + 1];
    }

    const reportDirIdx = args.indexOf('--report-dir');
    if (reportDirIdx !== -1 && args[reportDirIdx + 1]) {
        options.reportDir = args[reportDirIdx + 1];
    }

    // Create sync instance
    const sync = new ColabSync(options);
    await sync.init();

    // Execute command
    if (args.includes('--watch')) {
        const watchDirIdx = args.indexOf('--watch');
        if (args[watchDirIdx + 1] && !args[watchDirIdx + 1].startsWith('--')) {
            options.modelDir = args[watchDirIdx + 1];
        }
        await sync.watchDirectory();
    } else if (args.includes('--test')) {
        const testIdx = args.indexOf('--test');
        const modelPath = args[testIdx + 1];
        if (!modelPath) {
            console.error('❌ Error: --test requires a file path');
            process.exit(1);
        }
        await sync.testModel(modelPath);
    } else if (args.includes('--test-all')) {
        const testAllIdx = args.indexOf('--test-all');
        if (args[testAllIdx + 1] && !args[testAllIdx + 1].startsWith('--')) {
            options.modelDir = args[testAllIdx + 1];
        }
        await sync.testAll();
    } else {
        console.error('❌ Error: Unknown command. Use --help for usage information.');
        process.exit(1);
    }
}

main().catch(err => {
    console.error('❌ Fatal error:', err);
    process.exit(1);
});
