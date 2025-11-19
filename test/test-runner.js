/**
 * Test runner for FRACTAL GPU Runtime
 */

import { FractalRuntime, Tensor } from '../src/index.js';

class TestRunner {
    constructor() {
        this.passed = 0;
        this.failed = 0;
        this.tests = [];
    }

    async test(name, fn) {
        try {
            console.log(`\n🧪 Testing: ${name}`);
            await fn();
            console.log(`✅ PASS: ${name}`);
            this.passed++;
        } catch (err) {
            console.log(`❌ FAIL: ${name}`);
            console.log(`   Error: ${err.message}`);
            console.error(err);
            this.failed++;
        }
    }

    assert(condition, message) {
        if (!condition) {
            throw new Error(message || 'Assertion failed');
        }
    }

    assertClose(a, b, tolerance = 0.01, message = '') {
        const diff = Math.abs(a - b);
        if (diff > tolerance) {
            throw new Error(`${message} Expected ${b}, got ${a} (diff: ${diff})`);
        }
    }

    summary() {
        console.log('\n' + '='.repeat(60));
        console.log('TEST SUMMARY');
        console.log('='.repeat(60));
        console.log(`✅ Passed: ${this.passed}`);
        console.log(`❌ Failed: ${this.failed}`);
        console.log(`📊 Total:  ${this.passed + this.failed}`);
        console.log('='.repeat(60));

        return this.failed === 0;
    }
}

async function runTests() {
    const runner = new TestRunner();

    console.log('='.repeat(60));
    console.log('FRACTAL GPU RUNTIME - TEST SUITE');
    console.log('='.repeat(60));

    // Test 1: Tensor creation
    await runner.test('Tensor creation and basic ops', async () => {
        const t1 = Tensor.zeros([2, 3]);
        runner.assert(t1.shape[0] === 2, 'Wrong shape[0]');
        runner.assert(t1.shape[1] === 3, 'Wrong shape[1]');
        runner.assert(t1.size === 6, 'Wrong size');
        runner.assert(t1.data.length === 6, 'Wrong data length');

        const t2 = Tensor.ones([3, 2]);
        runner.assert(t2.data[0] === 1, 'Wrong ones value');

        const t3 = Tensor.fromArray([[1, 2], [3, 4]]);
        runner.assert(t3.at(0, 0) === 1, 'Wrong indexed value');
        runner.assert(t3.at(1, 1) === 4, 'Wrong indexed value');
    });

    // Test 2: Tensor reshape
    await runner.test('Tensor reshape', async () => {
        const t = Tensor.zeros([2, 3]);
        const r = t.reshape([3, 2]);
        runner.assert(r.shape[0] === 3, 'Wrong reshaped shape[0]');
        runner.assert(r.shape[1] === 2, 'Wrong reshaped shape[1]');
    });

    // Test 3: Runtime initialization
    await runner.test('Runtime initialization', async () => {
        const runtime = new FractalRuntime({ logLevel: 'error' });
        await runtime.init();

        runner.assert(runtime.ready === true, 'Runtime not ready');

        const stats = runtime.getStats();
        runner.assert(stats.ready === true, 'Stats show not ready');
    });

    // Test 4: CPU matmul
    await runner.test('CPU matrix multiplication', async () => {
        const runtime = new FractalRuntime({ logLevel: 'error', gpuThreshold: 9999 }); // Force CPU
        await runtime.init();

        const a = Tensor.fromArray([[1, 2], [3, 4]], [2, 2]);
        const b = Tensor.fromArray([[5, 6], [7, 8]], [2, 2]);

        const c = await runtime.matmul(a, b);

        // Expected: [[19, 22], [43, 50]]
        runner.assertClose(c.at(0, 0), 19, 0.01, 'Wrong matmul result [0,0]');
        runner.assertClose(c.at(0, 1), 22, 0.01, 'Wrong matmul result [0,1]');
        runner.assertClose(c.at(1, 0), 43, 0.01, 'Wrong matmul result [1,0]');
        runner.assertClose(c.at(1, 1), 50, 0.01, 'Wrong matmul result [1,1]');
    });

    // Test 5: GELU activation
    await runner.test('GELU activation', async () => {
        const runtime = new FractalRuntime({ logLevel: 'error', gpuThreshold: 9999 });
        await runtime.init();

        const x = Tensor.fromArray([0, 1, -1], [3]);
        const y = await runtime.gelu(x);

        // GELU(0) ≈ 0
        runner.assertClose(y.data[0], 0, 0.01, 'Wrong GELU(0)');

        // GELU(1) ≈ 0.841
        runner.assertClose(y.data[1], 0.841, 0.1, 'Wrong GELU(1)');

        // GELU(-1) ≈ -0.159
        runner.assertClose(y.data[2], -0.159, 0.1, 'Wrong GELU(-1)');
    });

    // Test 6: Model creation
    await runner.test('Model creation', async () => {
        const runtime = new FractalRuntime({ logLevel: 'error' });
        await runtime.init();

        const model = runtime.createTestModel('test');

        runner.assert(model.config.n_layer > 0, 'No layers');
        runner.assert(model.config.n_embd > 0, 'No embedding size');
        runner.assert(model.weights.size > 0, 'No weights');
    });

    // Test 7: Full inference pipeline
    await runner.test('Full inference pipeline', async () => {
        const runtime = new FractalRuntime({ logLevel: 'error' });
        await runtime.init();

        const model = runtime.createTestModel('test');
        const result = await runtime.infer('test', [42, 123]);

        runner.assert(result.logits, 'No logits returned');
        runner.assert(result.logits.shape[0] === 2, 'Wrong output seq length');
        runner.assert(result.logits.shape[1] === model.config.vocab_size, 'Wrong output vocab size');
        runner.assert(result.latency > 0, 'No latency recorded');
    });

    // Test 8: Stats tracking
    await runner.test('Statistics tracking', async () => {
        const runtime = new FractalRuntime({ logLevel: 'error' });
        await runtime.init();

        runtime.createTestModel('test');
        await runtime.infer('test', [1, 2, 3]);

        const stats = runtime.getStats();
        runner.assert(stats.totalInferences === 1, 'Wrong inference count');
        runner.assert(stats.opsRouted.cpu + stats.opsRouted.gpu > 0, 'No ops routed');
        runner.assert(stats.avgLatency > 0, 'No average latency');
    });

    // Test 9: Multiple inferences
    await runner.test('Multiple inferences', async () => {
        const runtime = new FractalRuntime({ logLevel: 'error' });
        await runtime.init();

        runtime.createTestModel('test');

        await runtime.infer('test', [1, 2]);
        await runtime.infer('test', [3, 4]);
        await runtime.infer('test', [5, 6]);

        const stats = runtime.getStats();
        runner.assert(stats.totalInferences === 3, 'Wrong inference count');
    });

    // Test 10: XJSON loader
    await runner.test('XJSON model loader', async () => {
        const { XJSONLoader } = await import('../src/xjson/loader.js');
        const loader = new XJSONLoader();

        const xjsonModel = {
            config: {
                n_layer: 4,
                n_embd: 256,
                n_head: 8,
                vocab_size: 5000
            },
            layers: [
                { type: 'embedding', name: 'emb' },
                { type: 'transformer_block', name: 'layer_0' }
            ],
            weights: {}
        };

        const model = loader.parse(xjsonModel);

        runner.assert(model.config.n_layer === 4, 'Wrong layer count');
        runner.assert(model.config.n_embd === 256, 'Wrong embedding size');
        runner.assert(model.layers.length === 2, 'Wrong layers array length');
    });

    return runner.summary();
}

// Run tests
runTests()
    .then(success => {
        process.exit(success ? 0 : 1);
    })
    .catch(err => {
        console.error('Test runner crashed:', err);
        process.exit(1);
    });
