// Common tensor operations: LayerNorm, GELU, ReLU, Add, etc.

struct OpParams {
    size: u32,
    epsilon: f32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> params: OpParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read> gamma: array<f32>;
@group(0) @binding(4) var<storage, read> beta: array<f32>;

// Layer Normalization
@compute @workgroup_size(256)
fn layer_norm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // Compute mean
    var sum = 0.0;
    for (var i = 0u; i < params.size; i = i + 1u) {
        sum = sum + input[i];
    }
    let mean = sum / f32(params.size);

    // Compute variance
    var var_sum = 0.0;
    for (var i = 0u; i < params.size; i = i + 1u) {
        let diff = input[i] - mean;
        var_sum = var_sum + diff * diff;
    }
    let variance = var_sum / f32(params.size);
    let std = sqrt(variance + params.epsilon);

    // Normalize and scale
    let normalized = (input[idx] - mean) / std;
    output[idx] = normalized * gamma[idx] + beta[idx];
}

// GELU activation
@compute @workgroup_size(256)
fn gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let x = input[idx];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    let x3 = x * x * x;
    let inner = 0.7978845608 * (x + 0.044715 * x3);
    output[idx] = 0.5 * x * (1.0 + tanh(inner));
}

// ReLU activation
@compute @workgroup_size(256)
fn relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    output[idx] = max(0.0, input[idx]);
}

// SiLU/Swish activation
@compute @workgroup_size(256)
fn silu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let x = input[idx];
    output[idx] = x / (1.0 + exp(-x));
}

// Element-wise add
@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // input is A, gamma is B (reusing bindings)
    output[idx] = input[idx] + gamma[idx];
}

// Element-wise multiply
@compute @workgroup_size(256)
fn mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    output[idx] = input[idx] * gamma[idx];
}

// Softmax
@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    // Find max
    var max_val = -1e10;
    for (var i = 0u; i < params.size; i = i + 1u) {
        max_val = max(max_val, input[i]);
    }

    // Compute exp and sum
    var sum_exp = 0.0;
    for (var i = 0u; i < params.size; i = i + 1u) {
        sum_exp = sum_exp + exp(input[i] - max_val);
    }

    // Normalize
    output[idx] = exp(input[idx] - max_val) / sum_exp;
}
