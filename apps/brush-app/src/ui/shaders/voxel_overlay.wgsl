// Semi-transparent overlay for the voxel collision mesh — lets the
// user see where the collider thinks the floor / walls are vs. where
// the splat puts them visually.

struct Camera {
    view_proj: mat4x4<f32>,
    color: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;

@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return camera.view_proj * vec4<f32>(pos, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return camera.color;
}
