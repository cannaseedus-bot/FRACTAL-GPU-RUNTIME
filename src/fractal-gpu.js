/**********************************************************************
 * FRACTAL GPU  •  WebGPU matmul kernel
 * Ultra-light GPU backend for FRACTAL-K’UHUL
 **********************************************************************/

export class FractalGPU {
    constructor() {
        this.ready = false;
        this.device = null;
        this.adapter = null;
    }

    async init() {
        if (!navigator.gpu) {
            console.warn('[FRACTAL GPU] WebGPU not available.');
            return false;
        }

        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
            console.warn('[FRACTAL GPU] No WebGPU adapter available.');
            return false;
        }

        this.device = await this.adapter.requestDevice();
        this.ready = true;

        console.log('[FRACTAL GPU] WebGPU engine READY.');
        return true;
    }

    async matmul(a, b, dim) {
        if (!this.ready) return null;

        const shader = `
            @group(0) @binding(0) var<storage, read> A : array<f32>;
            @group(0) @binding(1) var<storage, read> B : array<f32>;
            @group(0) @binding(2) var<storage, read_write> C : array<f32>;

            @compute @workgroup_size(8, 8)
            fn main(@builtin(global_invocation_id) id : vec3<u32>) {
                let row = id.x;
                let col = id.y;

                if (row >= ${dim} || col >= ${dim}) { return; }

                var sum = 0.0;
                for (var k = 0u; k < ${dim}u; k = k + 1u) {
                    sum = sum + A[row * ${dim}u + k] * B[k * ${dim}u + col];
                }
                C[row * ${dim}u + col] = sum;
            }
        `;

        const module = this.device.createShaderModule({ code: shader });

        const pipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: { module, entryPoint: 'main' }
        });

        const size = dim * dim * 4; // bytes

        const bufA = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const bufB = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
        });

        const bufC = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE
        });

        this.device.queue.writeBuffer(bufA, 0, a);
        this.device.queue.writeBuffer(bufB, 0, b);

        const bindGroup = this.device.createBindGroup({
            layout: pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: bufA } },
                { binding: 1, resource: { buffer: bufB } },
                { binding: 2, resource: { buffer: bufC } }
            ]
        });

        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);

        const groups = Math.ceil(dim / 8);
        pass.dispatchWorkgroups(groups, groups, 1);
        pass.end();

        this.device.queue.submit([encoder.finish()]);

        const readBuf = this.device.createBuffer({
            size,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const readEncoder = this.device.createCommandEncoder();
        readEncoder.copyBufferToBuffer(bufC, 0, readBuf, 0, size);
        this.device.queue.submit([readEncoder.finish()]);

        await readBuf.mapAsync(GPUMapMode.READ);
        const result = readBuf.getMappedRange();
        const output = new Float32Array(result.slice());
        readBuf.unmap();
        return output;
    }
}
