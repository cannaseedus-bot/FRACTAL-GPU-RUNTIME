// Scaled dot-product attention kernel
// Optimized for transformer inference

struct AttentionParams {
    seq_len: u32,
    head_dim: u32,
    scale: f32,
    pad: u32,
}

@group(0) @binding(0) var<uniform> params: AttentionParams;
@group(0) @binding(1) var<storage, read> Q: array<f32>;
@group(0) @binding(2) var<storage, read> K: array<f32>;
@group(0) @binding(3) var<storage, read> V: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;
@group(0) @binding(5) var<storage, read_write> scores: array<f32>; // temp buffer

// Compute Q @ K^T with scaling
@compute @workgroup_size(256)
fn compute_scores(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.seq_len * params.seq_len;

    if (idx >= total) {
        return;
    }

    let i = idx / params.seq_len;
    let j = idx % params.seq_len;

    var sum = 0.0;
    for (var k = 0u; k < params.head_dim; k = k + 1u) {
        sum = sum + Q[i * params.head_dim + k] * K[j * params.head_dim + k];
    }

    scores[idx] = sum * params.scale;
}

// Softmax over attention scores (per row)
@compute @workgroup_size(256)
fn softmax_scores(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;

    if (row >= params.seq_len) {
        return;
    }

    // Find max for numerical stability
    var max_val = -1e10;
    for (var j = 0u; j < params.seq_len; j = j + 1u) {
        max_val = max(max_val, scores[row * params.seq_len + j]);
    }

    // Compute exp and sum
    var sum_exp = 0.0;
    for (var j = 0u; j < params.seq_len; j = j + 1u) {
        let idx = row * params.seq_len + j;
        let val = exp(scores[idx] - max_val);
        scores[idx] = val;
        sum_exp = sum_exp + val;
    }

    // Normalize
    for (var j = 0u; j < params.seq_len; j = j + 1u) {
        let idx = row * params.seq_len + j;
        scores[idx] = scores[idx] / sum_exp;
    }
}

// Multiply attention scores by V
@compute @workgroup_size(256)
fn apply_attention(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.seq_len * params.head_dim;

    if (idx >= total) {
        return;
    }

    let i = idx / params.head_dim;
    let j = idx % params.head_dim;

    var sum = 0.0;
    for (var k = 0u; k < params.seq_len; k = k + 1u) {
        sum = sum + scores[i * params.seq_len + k] * V[k * params.head_dim + j];
    }

    output[idx] = sum;
}
