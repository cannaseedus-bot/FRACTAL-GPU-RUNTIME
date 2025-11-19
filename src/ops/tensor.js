/**
 * Tensor class - actual tensor operations backed by WebGPU
 */

export class Tensor {
    constructor(data, shape, device = null) {
        this.shape = shape;
        this.device = device;
        this.size = shape.reduce((a, b) => a * b, 1);

        if (data instanceof Float32Array) {
            this.data = data;
        } else if (Array.isArray(data)) {
            this.data = new Float32Array(data.flat(Infinity));
        } else {
            this.data = new Float32Array(this.size);
        }

        if (this.data.length !== this.size) {
            throw new Error(`Data length ${this.data.length} doesn't match shape size ${this.size}`);
        }

        this.gpuBuffer = null;
    }

    static zeros(shape, device = null) {
        const size = shape.reduce((a, b) => a * b, 1);
        return new Tensor(new Float32Array(size), shape, device);
    }

    static ones(shape, device = null) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(size).fill(1);
        return new Tensor(data, shape, device);
    }

    static randn(shape, device = null) {
        const size = shape.reduce((a, b) => a * b, 1);
        const data = new Float32Array(size);
        for (let i = 0; i < size; i++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            data[i] = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        }
        return new Tensor(data, shape, device);
    }

    static fromArray(arr, shape = null) {
        if (!shape) {
            // Infer shape from array
            shape = [];
            let current = arr;
            while (Array.isArray(current)) {
                shape.push(current.length);
                current = current[0];
            }
        }
        return new Tensor(arr, shape);
    }

    reshape(newShape) {
        const newSize = newShape.reduce((a, b) => a * b, 1);
        if (newSize !== this.size) {
            throw new Error(`Cannot reshape ${this.shape} to ${newShape}`);
        }
        return new Tensor(this.data, newShape, this.device);
    }

    async toGPU(device) {
        if (!device) throw new Error("No GPU device provided");

        const bytes = this.data.byteLength;
        this.gpuBuffer = device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: false
        });

        device.queue.writeBuffer(this.gpuBuffer, 0, this.data);
        this.device = device;
        return this;
    }

    async toCPU() {
        if (!this.gpuBuffer || !this.device) {
            return this; // Already on CPU
        }

        const bytes = this.data.byteLength;
        const stagingBuffer = this.device.createBuffer({
            size: bytes,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
        });

        const encoder = this.device.createCommandEncoder();
        encoder.copyBufferToBuffer(this.gpuBuffer, 0, stagingBuffer, 0, bytes);
        this.device.queue.submit([encoder.finish()]);

        await stagingBuffer.mapAsync(GPUMapMode.READ);
        const arrayBuffer = stagingBuffer.getMappedRange();
        this.data = new Float32Array(arrayBuffer.slice(0));
        stagingBuffer.unmap();

        return this;
    }

    clone() {
        return new Tensor(new Float32Array(this.data), [...this.shape], this.device);
    }

    toString() {
        return `Tensor(shape=${JSON.stringify(this.shape)}, data=[${this.data.slice(0, 10).join(', ')}...])`;
    }

    // Accessor for debugging
    at(...indices) {
        if (indices.length !== this.shape.length) {
            throw new Error(`Expected ${this.shape.length} indices, got ${indices.length}`);
        }

        let idx = 0;
        let stride = 1;
        for (let i = this.shape.length - 1; i >= 0; i--) {
            idx += indices[i] * stride;
            stride *= this.shape[i];
        }

        return this.data[idx];
    }
}
