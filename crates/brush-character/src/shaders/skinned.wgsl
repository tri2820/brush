// Skinned-mesh vertex+fragment for NPC characters.
//
// The skinning happens here on the GPU: per vertex we look up 4 joint
// matrices from a per-instance storage buffer and blend them by weight.
// Identity skin matrices give a T-pose; an animation evaluator supplies
// the matrices that produce the walk cycle.

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    cam_pos: vec3<f32>,
    _pad: f32,
};

struct InstanceUniforms {
    model: mat4x4<f32>,
    base_color: vec3<f32>,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> instance: InstanceUniforms;
@group(0) @binding(2) var<storage, read> skin_matrices: array<mat4x4<f32>>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) joints:   vec4<u32>,
    @location(3) weights:  vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

fn skin_matrix(j: vec4<u32>, w: vec4<f32>) -> mat4x4<f32> {
    var m: mat4x4<f32> =
        skin_matrices[j.x] * w.x +
        skin_matrices[j.y] * w.y +
        skin_matrices[j.z] * w.z +
        skin_matrices[j.w] * w.w;
    return m;
}

@vertex
fn vs_main(input: VsIn) -> VsOut {
    let skin = skin_matrix(input.joints, input.weights);
    let local_pos = (skin * vec4<f32>(input.position, 1.0)).xyz;
    let local_normal = (skin * vec4<f32>(input.normal, 0.0)).xyz;
    let world_pos4 = instance.model * vec4<f32>(local_pos, 1.0);
    let world_normal = (instance.model * vec4<f32>(local_normal, 0.0)).xyz;

    var out: VsOut;
    out.clip_pos = camera.view_proj * world_pos4;
    out.world_pos = world_pos4.xyz;
    out.world_normal = normalize(world_normal);
    out.color = instance.base_color;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Simple two-source diffuse: a key light from above-front, a soft
    // fill from below. Keeps the character readable against a textured
    // splat backdrop without going PBR.
    let n = normalize(in.world_normal);
    let key_dir = normalize(vec3<f32>(0.4, 0.8, 0.3));
    let fill_dir = normalize(vec3<f32>(-0.2, -0.5, -0.4));
    let key = max(dot(n, key_dir), 0.0);
    let fill = max(dot(n, fill_dir), 0.0) * 0.35;
    let ambient = 0.4;
    let lit = in.color * (ambient + key + fill);
    return vec4<f32>(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0);
}
