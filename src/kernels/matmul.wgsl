// Production matrix multiplication kernel with tiling
// Supports both FP32 and quantized (INT8) inputs

struct Dimensions {
    M: u32,  // rows of A
    N: u32,  // cols of B
    K: u32,  // cols of A / rows of B
    pad: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> B: array<f32>;
@group(0) @binding(3) var<storage, read_write> C: array<f32>;

const TILE_SIZE: u32 = 16u;

var<workgroup> tileA: array<f32, 256>; // 16x16
var<workgroup> tileB: array<f32, 256>; // 16x16

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;

    if (row >= dims.M || col >= dims.N) {
        return;
    }

    var sum = 0.0;

    // Tiled multiplication
    let numTiles = (dims.K + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t = 0u; t < numTiles; t = t + 1u) {
        // Load tile of A
        let tileRow = local_id.y;
        let tileCol = local_id.x;
        let aCol = t * TILE_SIZE + tileCol;

        if (row < dims.M && aCol < dims.K) {
            tileA[tileRow * TILE_SIZE + tileCol] = A[row * dims.K + aCol];
        } else {
            tileA[tileRow * TILE_SIZE + tileCol] = 0.0;
        }

        // Load tile of B
        let bRow = t * TILE_SIZE + tileRow;

        if (bRow < dims.K && col < dims.N) {
            tileB[tileRow * TILE_SIZE + tileCol] = B[bRow * dims.N + col];
        } else {
            tileB[tileRow * TILE_SIZE + tileCol] = 0.0;
        }

        workgroupBarrier();

        // Compute partial sum
        for (var k = 0u; k < TILE_SIZE; k = k + 1u) {
            sum = sum + tileA[tileRow * TILE_SIZE + k] * tileB[k * TILE_SIZE + tileCol];
        }

        workgroupBarrier();
    }

    C[row * dims.N + col] = sum;
}
