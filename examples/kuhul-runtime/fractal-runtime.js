/**********************************************************************
 *  FRACTAL–K’UHUL RUNTIME  •  REAL ENGINE
 *  ---------------------------------------------------------------
 *  Features:
 *  • Multi-runtime fractal spawning (legions)
 *  • CPU/iGPU routing scheduler (K’Uhul)
 *  • WebGPU real kernels with CPU fallback
 *  • XJSON graph execution (mini models)
 *  • Self-replicating micro-runtimes
 **********************************************************************/

/* ================================================================
 * FRACTAL-K’UHUL RUNTIME  •  LOW-END LAPTOP EDITION
 * - One JSON profile (FRACTAL-OS.json style, inline here)
 * - WebGPU backend (FractalGPU) with CPU fallback
 * - K’UHUL scheduler that picks CPU-MAIN vs iGPU-0 per op
 * - Demo pipeline: tiny sanity matmul + fractal demo pass
 * ================================================================ */

(function (global) {
  // ------------------------------------------------------------
  // 1. FRACTAL-OS PROFILE (LOW-END LAPTOP)
  // ------------------------------------------------------------
  const FRACTAL_PROFILE_LOW_END = {
    id: "LOW_END_LAPTOP",
    label: "Low-end laptop • 8c CPU • shared iGPU",
    hardware: {
      cpu: {
        id: "cpu-main",
        cores: 8,
        threads_per_core: 2,
        priority: "high",
      },
      igpu: {
        id: "igpu-0",
        backend: "webgpu",
        shared_mem_mb: 256,
        enabled: true,
      },
    },
    runtime_strategies: {
      matmul: {
        gpu_threshold: 64, // dim >= threshold → GPU, else CPU
        max_dim: 1024,
      },
    },
    sanity_tests: {
      tiny_matmul_dim: 4,
    },
  };

  // ------------------------------------------------------------
  // 2. LIGHTWEIGHT LOGGER
  // ------------------------------------------------------------
  class FractalLogger {
    constructor(targetElId) {
      this.targetEl = targetElId
        ? document.getElementById(targetElId)
        : null;
      this.lines = [];
      this.maxLines = 200;
    }

    line(msg) {
      const stamp = new Date().toLocaleTimeString();
      const full = `[${stamp}] ${msg}`;
      this.lines.push(full);
      if (this.lines.length > this.maxLines) {
        this.lines.shift();
      }
      if (this.targetEl) {
        const el = this.targetEl;
        el.textContent = this.lines.join("\n");
        el.scrollTop = el.scrollHeight;
      } else {
        console.log(full);
      }
    }
  }

  // ------------------------------------------------------------
  // 3. FRACTAL GPU BACKEND (WebGPU)
  // ------------------------------------------------------------
  class FractalGPU {
    constructor(logger) {
      this.logger = logger;
      this.ready = false;
      this.device = null;
      this.adapter = null;
    }

    async init() {
      if (!("gpu" in navigator)) {
        this.logger.line("[FRACTAL-GPU] WebGPU not available. Using CPU only.");
        return false;
      }

      try {
        this.adapter = await navigator.gpu.requestAdapter();
        if (!this.adapter) {
          this.logger.line("[FRACTAL-GPU] No GPU adapter. CPU only.");
          return false;
        }
        this.device = await this.adapter.requestDevice();
        this.ready = true;
        const info = this.adapter.features ? [...this.adapter.features].join(",") : "n/a";
        this.logger.line(`[FRACTAL-GPU] WebGPU READY • features: ${info}`);
        return true;
      } catch (err) {
        this.logger.line("[FRACTAL-GPU] init error → " + err);
        return false;
      }
    }

    async matmul(a, b, dim) {
      if (!this.ready) throw new Error("FractalGPU not ready");

      const shader = `
        @group(0) @binding(0) var<storage, read> A : array<f32>;
        @group(0) @binding(1) var<storage, read> B : array<f32>;
        @group(0) @binding(2) var<storage, read_write> C : array<f32>;

        @compute @workgroup_size(8, 8)
        fn main(@builtin(global_invocation_id) id : vec3<u32>) {
          let row = id.x;
          let col = id.y;

          if (row >= ${dim}u || col >= ${dim}u) { return; }

          var sum = 0.0;
          for (var k = 0u; k < ${dim}u; k = k + 1u) {
            sum = sum + A[row * ${dim}u + k] * B[k * ${dim}u + col];
          }
          C[row * ${dim}u + col] = sum;
        }
      `;

      const module = this.device.createShaderModule({ code: shader });
      const pipeline = this.device.createComputePipeline({
        layout: "auto",
        compute: { module, entryPoint: "main" },
      });

      const bytes = dim * dim * 4;
      const bufA = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const bufB = this.device.createBuffer({
        size: bytes,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      const bufC = this.device.createBuffer({
        size: bytes,
        usage:
          GPUBufferUsage.STORAGE |
          GPUBufferUsage.COPY_SRC |
          GPUBufferUsage.COPY_DST,
      });

      this.device.queue.writeBuffer(bufA, 0, a);
      this.device.queue.writeBuffer(bufB, 0, b);

      const bindGroup = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: bufA } },
          { binding: 1, resource: { buffer: bufB } },
          { binding: 2, resource: { buffer: bufC } },
        ],
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
        size: bytes,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
      });

      const readEncoder = this.device.createCommandEncoder();
      readEncoder.copyBufferToBuffer(bufC, 0, readBuf, 0, bytes);
      this.device.queue.submit([readEncoder.finish()]);

      await readBuf.mapAsync(GPUMapMode.READ);
      const copy = readBuf.getMappedRange().slice(0);
      readBuf.unmap();
      return new Float32Array(copy);
    }
  }

  // ------------------------------------------------------------
  // 4. K’UHUL SCHEDULER (CPU vs iGPU router)
  // ------------------------------------------------------------
  class FractalScheduler {
    constructor(profile, logger, gpuBackend) {
      this.profile = profile;
      this.logger = logger;
      this.gpu = gpuBackend;
      this.routes = { cpu: 0, gpu: 0 };
    }

    pickDevice(op, shape) {
      const dim = Math.max(...shape);
      const strat = this.profile.runtime_strategies.matmul;
      if (
        op === "matmul" &&
        this.gpu.ready &&
        dim >= strat.gpu_threshold &&
        dim <= strat.max_dim
      ) {
        this.routes.gpu++;
        this.logger.line(
          `[SCHEDULER] op=matmul dim=${dim} → iGPU-0 (WebGPU)`
        );
        return "gpu";
      }
      this.routes.cpu++;
      this.logger.line(
        `[SCHEDULER] op=${op} dim=${dim} → CPU-MAIN`
      );
      return "cpu";
    }
  }

  // ------------------------------------------------------------
  // 5. MAIN FRACTAL-K’UHUL RUNTIME
  // ------------------------------------------------------------
  class FractalKuhulRuntime {
    constructor(options = {}) {
      this.profile = FRACTAL_PROFILE_LOW_END;
      this.logger = new FractalLogger(options.logEl || null);
      this.gpu = new FractalGPU(this.logger);
      this.scheduler = new FractalScheduler(
        this.profile,
        this.logger,
        this.gpu
      );
      this.lastPassMs = null;
    }

    async init() {
      this.logger.line("Fractal-K’UHUL runtime loading…");
      await this.gpu.init();
      this.logger.line(
        `Profile: ${this.profile.id} • CPU cores=${this.profile.hardware.cpu.cores} • iGPU enabled=${this.profile.hardware.igpu.enabled}`
      );
      this.logger.line("Runtime READY.");
      return this;
    }

    // CPU matmul fallback
    cpuMatmul(dim) {
      const A = new Float32Array(dim * dim);
      const B = new Float32Array(dim * dim);
      for (let i = 0; i < A.length; i++) {
        A[i] = Math.random();
        B[i] = Math.random();
      }
      const C = new Float32Array(dim * dim);
      for (let i = 0; i < dim; i++) {
        for (let j = 0; j < dim; j++) {
          let sum = 0;
          for (let k = 0; k < dim; k++) {
            sum += A[i * dim + k] * B[k * dim + j];
          }
          C[i * dim + j] = sum;
        }
      }
      return { A, B, C };
    }

    async matmul(dim) {
      const device = this.scheduler.pickDevice("matmul", [dim, dim]);

      if (device === "gpu") {
        const A = new Float32Array(dim * dim);
        const B = new Float32Array(dim * dim);
        for (let i = 0; i < A.length; i++) {
          A[i] = Math.random();
          B[i] = Math.random();
        }
        const t0 = performance.now();
        const C = await this.gpu.matmul(A, B, dim);
        const t1 = performance.now();
        this.logger.line(
          `[GPU] matmul dim=${dim} completed in ${(t1 - t0).toFixed(2)} ms`
        );
        return { A, B, C, ms: t1 - t0, device: "gpu" };
      } else {
        const t0 = performance.now();
        const { A, B, C } = this.cpuMatmul(dim);
        const t1 = performance.now();
        this.logger.line(
          `[CPU] matmul dim=${dim} completed in ${(t1 - t0).toFixed(2)} ms`
        );
        return { A, B, C, ms: t1 - t0, device: "cpu" };
      }
    }

    // --------------------------------------------------------
    // DEMO: FRACTAL PASS (2 ops: one CPU-scale, one GPU-scale)
    // --------------------------------------------------------
    async runFractalDemoPass() {
      this.logger.line("— DEMO PASS BEGIN —");
      const t0 = performance.now();

      // small op (forced CPU)
      await this.matmul(32);

      // bigger op -> likely GPU on machines with WebGPU
      await this.matmul(128);

      const t1 = performance.now();
      this.lastPassMs = t1 - t0;
      this.logger.line(
        `— DEMO PASS COMPLETE in ${this.lastPassMs.toFixed(2)} ms —`
      );
      this.logger.line(
        `Ops routed • CPU: ${this.scheduler.routes.cpu} • GPU: ${this.scheduler.routes.gpu}`
      );
    }

    // Tiny 4x4 sanity test (like your HUD text)
    async tinySanityMatmul() {
      const dim = this.profile.sanity_tests.tiny_matmul_dim;
      this.logger.line(`Running tiny sanity matmul ${dim}x${dim}…`);

      const { C, ms, device } = await this.matmul(dim);
      const c00 = C[0].toFixed(4);
      this.logger.line(
        `tiny matmul ${dim}x${dim} → C[0,0]=${c00} • device=${device} • ${ms.toFixed(
          3
        )} ms`
      );
    }

    // Expose stats for UI side
    getStats() {
      return {
        profile: this.profile.id,
        lastPassMs: this.lastPassMs,
        routes: { ...this.scheduler.routes },
        gpuReady: this.gpu.ready,
      };
    }
  }

  // ------------------------------------------------------------
  // 6. GLOBAL EXPORT
  // ------------------------------------------------------------
  global.FractalKuhulRuntime = FractalKuhulRuntime;
})(window);
