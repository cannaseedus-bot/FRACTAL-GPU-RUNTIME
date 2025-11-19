/**
 * FRACTAL-K'UHUL Runtime - Production WebGPU inference engine
 */

import { Tensor } from './ops/tensor.js';
import { GPUOps } from './ops/gpu-ops.js';
import { XJSONLoader } from './xjson/loader.js';

export class FractalRuntime {
    constructor(options = {}) {
        this.options = {
            logLevel: options.logLevel || 'info',
            preferGPU: options.preferGPU !== false,
            gpuThreshold: options.gpuThreshold || 64,
            ...options
        };

        this.device = null;
        this.adapter = null;
        this.gpuOps = null;
        this.ready = false;

        this.stats = {
            opsRouted: { cpu: 0, gpu: 0 },
            totalInferences: 0,
            avgLatency: 0
        };

        this.models = new Map();
        this.loader = new XJSONLoader();
    }

    log(level, ...args) {
        const levels = { error: 0, warn: 1, info: 2, debug: 3 };
        const currentLevel = levels[this.options.logLevel] || 2;
        if (levels[level] <= currentLevel) {
            console[level]('[FRACTAL]', ...args);
        }
    }

    /**
     * Initialize WebGPU device
     */
    async init() {
        this.log('info', 'Initializing runtime...');

        if (!navigator.gpu) {
            this.log('warn', 'WebGPU not available - CPU-only mode');
            this.ready = true;
            return this;
        }

        try {
            this.adapter = await navigator.gpu.requestAdapter({
                powerPreference: 'high-performance'
            });

            if (!this.adapter) {
                this.log('warn', 'No WebGPU adapter - CPU-only mode');
                this.ready = true;
                return this;
            }

            this.device = await this.adapter.requestDevice();
            this.gpuOps = new GPUOps(this.device);
            await this.gpuOps.init();

            const info = this.adapter.info || {};
            this.log('info', `WebGPU initialized: ${info.device || 'unknown device'}`);
            this.log('info', `Features: ${[...this.adapter.features].join(', ') || 'none'}`);

            this.ready = true;
            return this;

        } catch (err) {
            this.log('error', 'WebGPU init failed:', err);
            this.log('warn', 'Falling back to CPU-only mode');
            this.ready = true;
            return this;
        }
    }

    /**
     * Load a model from XJSON
     */
    async loadModel(source, name = 'default') {
        this.log('info', `Loading model: ${name}`);
        const model = await this.loader.load(source);
        this.models.set(name, model);
        this.log('info', `Model loaded: ${model.config.n_layer} layers, ${model.config.n_embd}d`);
        return model;
    }

    /**
     * Create a minimal test model
     */
    createTestModel(name = 'test') {
        this.log('info', 'Creating test model...');
        const model = XJSONLoader.createMiniTransformer({
            vocab_size: 1000,
            n_embd: 128,
            n_head: 4,
            n_layer: 2,
            max_seq_len: 256
        });

        // Initialize random weights
        this.initializeWeights(model);
        this.models.set(name, model);

        this.log('info', `Test model created: ${model.config.n_layer} layers`);
        return model;
    }

    /**
     * Initialize model weights with random values
     */
    initializeWeights(model) {
        const { n_embd, n_head, vocab_size, n_layer } = model.config;

        // Token embeddings
        model.weights.set(
            'token_embedding.weight',
            Tensor.randn([vocab_size, n_embd], this.device)
        );

        // Position embeddings
        model.weights.set(
            'position_embedding.weight',
            Tensor.randn([model.config.max_seq_len, n_embd], this.device)
        );

        // Transformer layers
        for (let i = 0; i < n_layer; i++) {
            const prefix = `layer_${i}`;

            // Attention
            model.weights.set(`${prefix}.attn.qkv.weight`, Tensor.randn([n_embd, 3 * n_embd], this.device));
            model.weights.set(`${prefix}.attn.proj.weight`, Tensor.randn([n_embd, n_embd], this.device));

            // FFN
            model.weights.set(`${prefix}.ffn.fc1.weight`, Tensor.randn([n_embd, 4 * n_embd], this.device));
            model.weights.set(`${prefix}.ffn.fc2.weight`, Tensor.randn([4 * n_embd, n_embd], this.device));

            // Layer norms
            model.weights.set(`${prefix}.ln1.weight`, Tensor.ones([n_embd], this.device));
            model.weights.set(`${prefix}.ln1.bias`, Tensor.zeros([n_embd], this.device));
            model.weights.set(`${prefix}.ln2.weight`, Tensor.ones([n_embd], this.device));
            model.weights.set(`${prefix}.ln2.bias`, Tensor.zeros([n_embd], this.device));
        }

        // Final layer norm
        model.weights.set('final_norm.weight', Tensor.ones([n_embd], this.device));
        model.weights.set('final_norm.bias', Tensor.zeros([n_embd], this.device));

        // LM head
        model.weights.set('lm_head.weight', Tensor.randn([n_embd, vocab_size], this.device));
    }

    /**
     * Run inference on input tokens
     */
    async infer(modelName, inputTokens) {
        if (!this.ready) {
            throw new Error('Runtime not initialized - call init() first');
        }

        const model = this.models.get(modelName);
        if (!model) {
            throw new Error(`Model '${modelName}' not found`);
        }

        this.log('debug', `Running inference on ${inputTokens.length} tokens`);
        const t0 = performance.now();

        // Convert tokens to tensor
        let x = this.embed(model, inputTokens);

        // Forward pass through transformer blocks
        for (let i = 0; i < model.config.n_layer; i++) {
            x = await this.transformerBlock(model, x, i);
        }

        // Final layer norm and projection
        x = this.layerNorm(model, x, 'final_norm');
        const logits = await this.linear(model, x, 'lm_head');

        const t1 = performance.now();
        const latency = t1 - t0;

        this.stats.totalInferences++;
        this.stats.avgLatency = (this.stats.avgLatency * (this.stats.totalInferences - 1) + latency) / this.stats.totalInferences;

        this.log('info', `Inference complete: ${latency.toFixed(2)}ms`);

        return {
            logits: await logits.toCPU(),
            latency,
            stats: { ...this.stats }
        };
    }

    /**
     * Embedding layer
     */
    embed(model, tokens) {
        const embedWeight = model.weights.get('token_embedding.weight');
        const posWeight = model.weights.get('position_embedding.weight');

        // Simple embedding lookup (CPU for now)
        const { n_embd } = model.config;
        const seqLen = tokens.length;
        const embedded = new Float32Array(seqLen * n_embd);

        for (let i = 0; i < seqLen; i++) {
            const tokenId = tokens[i];
            for (let j = 0; j < n_embd; j++) {
                embedded[i * n_embd + j] =
                    embedWeight.data[tokenId * n_embd + j] +
                    posWeight.data[i * n_embd + j];
            }
        }

        return new Tensor(embedded, [seqLen, n_embd], this.device);
    }

    /**
     * Transformer block
     */
    async transformerBlock(model, x, layerIdx) {
        const prefix = `layer_${layerIdx}`;

        // Self-attention with residual
        const attnOut = await this.selfAttention(model, x, prefix);
        x = await this.add(x, attnOut);
        x = this.layerNorm(model, x, `${prefix}.ln1`);

        // FFN with residual
        const ffnOut = await this.ffn(model, x, prefix);
        x = await this.add(x, ffnOut);
        x = this.layerNorm(model, x, `${prefix}.ln2`);

        return x;
    }

    /**
     * Self-attention (simplified)
     */
    async selfAttention(model, x, prefix) {
        const qkvWeight = model.weights.get(`${prefix}.attn.qkv.weight`);
        const projWeight = model.weights.get(`${prefix}.attn.proj.weight`);

        // QKV projection
        let qkv = await this.matmul(x, qkvWeight);

        // For now, simplified attention - just use the projection
        const out = await this.matmul(qkv.reshape([x.shape[0], qkvWeight.shape[1]]), projWeight);

        return out;
    }

    /**
     * Feed-forward network
     */
    async ffn(model, x, prefix) {
        const fc1Weight = model.weights.get(`${prefix}.ffn.fc1.weight`);
        const fc2Weight = model.weights.get(`${prefix}.ffn.fc2.weight`);

        // FC1 + GELU
        let h = await this.matmul(x, fc1Weight);
        h = await this.gelu(h);

        // FC2
        h = await this.matmul(h, fc2Weight);

        return h;
    }

    /**
     * Matrix multiplication with routing
     */
    async matmul(a, b) {
        const dim = Math.max(...a.shape, ...b.shape);

        if (this.gpuOps && dim >= this.options.gpuThreshold) {
            this.stats.opsRouted.gpu++;
            this.log('debug', `matmul on GPU: ${a.shape} @ ${b.shape}`);
            return await this.gpuOps.matmul(a, b);
        } else {
            this.stats.opsRouted.cpu++;
            this.log('debug', `matmul on CPU: ${a.shape} @ ${b.shape}`);
            return this.cpuMatmul(a, b);
        }
    }

    /**
     * CPU fallback matmul
     */
    cpuMatmul(a, b) {
        const [m, k] = a.shape;
        const [k2, n] = b.shape;

        if (k !== k2) {
            throw new Error(`Shape mismatch: ${a.shape} @ ${b.shape}`);
        }

        const result = new Float32Array(m * n);

        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                let sum = 0;
                for (let p = 0; p < k; p++) {
                    sum += a.data[i * k + p] * b.data[p * n + j];
                }
                result[i * n + j] = sum;
            }
        }

        return new Tensor(result, [m, n], this.device);
    }

    /**
     * GELU activation
     */
    async gelu(x) {
        if (this.gpuOps && x.size >= this.options.gpuThreshold) {
            this.stats.opsRouted.gpu++;
            return await this.gpuOps.gelu(x);
        }

        this.stats.opsRouted.cpu++;

        const result = new Float32Array(x.size);
        for (let i = 0; i < x.size; i++) {
            const val = x.data[i];
            const x3 = val * val * val;
            const inner = 0.7978845608 * (val + 0.044715 * x3);
            result[i] = 0.5 * val * (1 + Math.tanh(inner));
        }

        return new Tensor(result, x.shape, this.device);
    }

    /**
     * Layer normalization (CPU for now)
     */
    layerNorm(model, x, name) {
        const gamma = model.weights.get(`${name}.weight`);
        const beta = model.weights.get(`${name}.bias`);

        const epsilon = 1e-5;
        const [seqLen, hiddenSize] = x.shape;
        const result = new Float32Array(x.size);

        for (let i = 0; i < seqLen; i++) {
            // Compute mean
            let sum = 0;
            for (let j = 0; j < hiddenSize; j++) {
                sum += x.data[i * hiddenSize + j];
            }
            const mean = sum / hiddenSize;

            // Compute variance
            let varSum = 0;
            for (let j = 0; j < hiddenSize; j++) {
                const diff = x.data[i * hiddenSize + j] - mean;
                varSum += diff * diff;
            }
            const variance = varSum / hiddenSize;
            const std = Math.sqrt(variance + epsilon);

            // Normalize and scale
            for (let j = 0; j < hiddenSize; j++) {
                const normalized = (x.data[i * hiddenSize + j] - mean) / std;
                result[i * hiddenSize + j] = normalized * gamma.data[j] + beta.data[j];
            }
        }

        return new Tensor(result, x.shape, this.device);
    }

    /**
     * Linear layer
     */
    async linear(model, x, name) {
        const weight = model.weights.get(`${name}.weight`);
        return await this.matmul(x, weight);
    }

    /**
     * Tensor addition
     */
    async add(a, b) {
        if (this.gpuOps && a.size >= this.options.gpuThreshold) {
            this.stats.opsRouted.gpu++;
            return await this.gpuOps.add(a, b);
        }

        this.stats.opsRouted.cpu++;

        const result = new Float32Array(a.size);
        for (let i = 0; i < a.size; i++) {
            result[i] = a.data[i] + b.data[i];
        }

        return new Tensor(result, a.shape, this.device);
    }

    /**
     * Get runtime statistics
     */
    getStats() {
        return {
            ready: this.ready,
            hasGPU: !!this.device,
            models: Array.from(this.models.keys()),
            ...this.stats
        };
    }
}
