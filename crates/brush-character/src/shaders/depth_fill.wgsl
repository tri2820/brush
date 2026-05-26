// Fullscreen pass: read brush's linear view-space depth tensor and
// write NDC depth (wgpu [0, 1] convention) into a Depth32Float
// attachment. Run before the mesh pass so the character pass can
// depth-test against splat geometry with the normal hardware path.
//
// Linear depth produced by brush is view-space z with +Z forward
// (camera-local). NDC z in wgpu/Metal is in [0, 1] where 0=near, 1=far.
// For a glam-style perspective_rh projection with near n and far f:
//   ndc_z = f * (vz - n) / (vz * (f - n))
// Pixels with no splat coverage are sentinel `1e30` from the rasterizer
// and end up at ndc_z ≈ 1 (far plane) which is what we want.

struct Uniforms {
    near: f32,
    far: f32,
    width: u32,
    height: u32,
};

@group(0) @binding(0) var<storage, read> linear_depth: array<f32>;
@group(0) @binding(1) var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    // Standard fullscreen-triangle trick: three vertices that cover
    // the entire NDC [-1,1]^2 region with the rasterizer clipping the
    // off-screen corner.
    let x = f32((vid << 1u) & 2u) * 2.0 - 1.0;
    let y = f32(vid & 2u) * 2.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

struct FragOut {
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> FragOut {
    let x = u32(pos.x);
    let y = u32(pos.y);
    let idx = y * u.width + x;
    let vz_raw = linear_depth[idx];
    // Clamp behind the near plane so the conversion can't divide by
    // anything <= 0. Pixels with no splat coverage (sentinel 1e30)
    // sail past `far` and the clamp below collapses them to 1.0.
    let vz = max(vz_raw, u.near);
    let raw = u.far * (vz - u.near) / (vz * (u.far - u.near));
    var out: FragOut;
    out.depth = clamp(raw, 0.0, 1.0);
    return out;
}
