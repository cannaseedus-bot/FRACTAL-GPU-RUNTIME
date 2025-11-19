/**
 * GPU-accelerated tensor operations
 */

import { readFileSync } from 'fs';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

export class GPUOps {
    constructor(device) {
        this.device = device;
        this.pipelines = {};
        this.shaderCache = {};
    }

    async init() {
        // Load WGSL shaders
        await this.loadShader('matmul', join(__dirname, '../kernels/matmul.wgsl'));
        await this.loadShader('attention', join(__dirname, '../kernels/attention.wgsl'));
        await this.loadShader('ops', join(__dirname, '../kernels/ops.wgsl'));
        return this;
    }

    async loadShader(name, path) {
        try {
            this.shaderCache[name] = readFileSync(path, 'utf-8');
        } catch (err) {
            console.warn(`Could not load shader ${name} from ${path}, using inline`);
            // Fallback to inline shaders if files don't exist
            this.shaderCache[name] = this.getInlineShader(name);
        }
    }

    getInlineShader(name) {
        // Fallback inline shaders for browser environment
        const shaders = {
            simple_matmul: `
                struct Dims { M: u32, N: u32, K: u32, pad: u32 }
                @group(0) @binding(0) var<uniform> dims: Dims;
                @group(0) @binding(1) var<storage, read> A: array<f32>;
                @group(0) @binding(2) var<storage, read> B: array<f32>;
                @group(0) @binding(3) var<storage, read_write> C: array<f32>;

                @compute @workgroup_size(16, 16)
                fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                    let row = id.y;
                    let col = id.x;
                    if (row >= dims.M || col >= dims.N) { return; }
                    var sum = 0.0;
                    for (var k = 0u; k < dims.K; k++) {
                        sum += A[row * dims.K + k] * B[k * dims.N + col];
                    }
                    C[row * dims.N + col] = sum;
                }
            `,
            elementwise: `
                @group(0) @binding(0) var<storage, read> input: array<f32>;
                @group(0) @binding(1) var<storage, read_write> output: array<f32>;

                @compute @workgroup_size(256)
                fn gelu(@builtin(global_invocation_id) id: vec3<u32>) {
                    let idx = id.x;
                    let x = input[idx];
                    let x3 = x * x * x;
                    let inner = 0.7978845608 * (x + 0.044715 * x3);
                    output[idx] = 0.5 * x * (1.0 + tanh(inner));
                }

                @compute @workgroup_size(256)
                fn relu(@builtin(global_invocation_id) id: vec3<u32>) {
                    output[id.x] = max(0.0, input[id.x]);
                }

                @compute @workgroup_size(256)
                fn add(@builtin(global_invocation_id) id: vec3<u32>) {
                    output[id.x] = input[id.x] + output[id.x];
                }
            `
        };
        return shaders[name] || '';
    }

    createPipeline(shaderCode, entryPoint) {
        const key = `${entryPoint}`;
        if (this.pipelines[key]) {
            return this.pipelines[key];
        }

        const shaderModule = this.device.createShaderModule({ code: shaderCode });
        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: shaderModule,
                entryPoint: entryPoint
            }
        });

        this.pipelines[key] = pipeline;
        return pipeline;
    }

    async matmul(A, B) {
        // A: [M, K], B: [K, N] -> C: [M, N]
        if (A.shape.length !== 2 || B.shape.length !== 2) {
            throw new Error('matmul requires 2D tensors');
        }

        const [M, K] = A.shape;
        const [K2, N] = B.shape;

        if (K !== K2) {
            throw new Error(`Shape mismatch: A[${M},${K}] @ B[${K2},${N}]`);
        }

        // Ensure tensors are on GPU
        if (!A.gpuBuffer) await A.toGPU(this.device);
        if (!B.gpuBuffer) await B.toGPU(this.device);

        // Create output buffer
        const outputSize = M * N;
        const outputBuffer = this.device.createBuffer({
            size: outputSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        // Create uniform buffer for dimensions
        const dimsBuffer = this.device.createBuffer({
            size: 16, // 4 u32s
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
        });

        this.device.queue.writeBuffer(
            dimsBuffer,
            0,
            new Uint32Array([M, N, K, 0])
        );

        // Get or create pipeline
        const shader = this.shaderCache.matmul || this.getInlineShader('simple_matmul');
        const pipeline = this.createPipeline(shader, 'main');

        // Create bind group
        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: dimsBuffer } },
                { binding: 1, resource: { buffer: A.gpuBuffer } },
                { binding: 2, resource: { buffer: B.gpuBuffer } },
                { binding: 3, resource: { buffer: outputBuffer } }
            ]
        });

        // Dispatch compute
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);

        const workgroupsX = Math.ceil(N / 16);
        const workgroupsY = Math.ceil(M / 16);
        pass.dispatchWorkgroups(workgroupsX, workgroupsY, 1);
        pass.end();

        this.device.queue.submit([encoder.finish()]);

        // Create output tensor
        const { Tensor } = await import('./tensor.js');
        const output = Tensor.zeros([M, N], this.device);
        output.gpuBuffer = outputBuffer;

        return output;
    }

    async gelu(input) {
        if (!input.gpuBuffer) await input.toGPU(this.device);

        const outputBuffer = this.device.createBuffer({
            size: input.data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC
        });

        const shader = this.shaderCache.ops || this.getInlineShader('elementwise');
        const pipeline = this.createPipeline(shader, 'gelu');

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: input.gpuBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } }
            ]
        });

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(input.size / 256), 1, 1);
        pass.end();

        this.device.queue.submit([encoder.finish()]);

        const { Tensor } = await import('./tensor.js');
        const output = Tensor.zeros(input.shape, this.device);
        output.gpuBuffer = outputBuffer;

        return output;
    }

    async add(a, b) {
        // Element-wise addition
        if (a.size !== b.size) {
            throw new Error('Tensors must have same size for addition');
        }

        if (!a.gpuBuffer) await a.toGPU(this.device);
        if (!b.gpuBuffer) await b.toGPU(this.device);

        // For now, simple implementation - copy b to output, then add a
        const outputBuffer = this.device.createBuffer({
            size: a.data.byteLength,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(b.gpuBuffer, 0, outputBuffer, 0, a.data.byteLength);
        this.device.queue.submit([encoder.finish()]);

        const shader = this.getInlineShader('elementwise');
        const pipeline = this.createPipeline(shader, 'add');

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: a.gpuBuffer } },
                { binding: 1, resource: { buffer: outputBuffer } }
            ]
        });

        const encoder2 = this.device.createCommandEncoder();
        const pass = encoder2.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(a.size / 256), 1, 1);
        pass.end();

        this.device.queue.submit([encoder2.finish()]);

        const { Tensor } = await import('./tensor.js');
        const output = Tensor.zeros(a.shape, this.device);
        output.gpuBuffer = outputBuffer;

        return output;
    }
}
