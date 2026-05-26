// Skinned-mesh vertex+fragment for NPC characters.
//
// GPU linear-blend skinning + glTF baseColor + tangent-space normal
// mapping. Lighting is two-source diffuse (key + fill) in *linear*
// color space; we apply manual sRGB encoding at the end because the
// recorder's IOSurface texture is Bgra8Unorm (non-sRGB), and the
// splat compositing pipeline implicitly treats its output bytes as
// sRGB-encoded. Keeping the gamma encode in the shader is the simplest
// way to make the character match without disturbing the splat path.

struct CameraUniforms {
    view_proj: mat4x4<f32>,
    cam_pos: vec3<f32>,
    _pad: f32,
};

struct InstanceUniforms {
    model: mat4x4<f32>,
    base_color: vec3<f32>,
    _pad: f32,
    // 6-tap ambient cube in world frame: +X, -X, +Y, -Y, +Z, -Z. Each
    // entry is RGB in linear color space (xyz; .w is padding).
    ambient_cube: array<vec4<f32>, 6>,
};

struct MaterialUniforms {
    base_color_factor: vec4<f32>,
    normal_scale: f32,
    has_base_color_tex: f32,
    has_normal_tex: f32,
    _pad: f32,
};

@group(0) @binding(0) var<uniform> camera: CameraUniforms;
@group(0) @binding(1) var<uniform> instance: InstanceUniforms;
@group(0) @binding(2) var<storage, read> skin_matrices: array<mat4x4<f32>>;

@group(1) @binding(0) var base_color_tex: texture_2d<f32>;
@group(1) @binding(1) var normal_tex: texture_2d<f32>;
@group(1) @binding(2) var mat_sampler: sampler;
@group(1) @binding(3) var<uniform> material: MaterialUniforms;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal:   vec3<f32>,
    @location(2) tangent:  vec4<f32>,
    @location(3) uv:       vec2<f32>,
    @location(4) joints:   vec4<u32>,
    @location(5) weights:  vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos: vec4<f32>,
    @location(0) world_pos: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) world_tangent: vec3<f32>,
    @location(3) world_bitangent: vec3<f32>,
    @location(4) uv: vec2<f32>,
};

fn skin_matrix(j: vec4<u32>, w: vec4<f32>) -> mat4x4<f32> {
    return skin_matrices[j.x] * w.x +
           skin_matrices[j.y] * w.y +
           skin_matrices[j.z] * w.z +
           skin_matrices[j.w] * w.w;
}

@vertex
fn vs_main(input: VsIn) -> VsOut {
    let skin = skin_matrix(input.joints, input.weights);
    let local_pos = (skin * vec4<f32>(input.position, 1.0)).xyz;
    let local_normal = (skin * vec4<f32>(input.normal, 0.0)).xyz;
    let local_tangent = (skin * vec4<f32>(input.tangent.xyz, 0.0)).xyz;
    let world_pos4 = instance.model * vec4<f32>(local_pos, 1.0);
    let world_normal = normalize((instance.model * vec4<f32>(local_normal, 0.0)).xyz);
    let world_tangent = normalize((instance.model * vec4<f32>(local_tangent, 0.0)).xyz);
    // glTF: bitangent = cross(normal, tangent) * tangent.w. The .w
    // sign accounts for mirrored UV charts.
    let world_bitangent = cross(world_normal, world_tangent) * input.tangent.w;

    var out: VsOut;
    out.clip_pos = camera.view_proj * world_pos4;
    out.world_pos = world_pos4.xyz;
    out.world_normal = world_normal;
    out.world_tangent = world_tangent;
    out.world_bitangent = world_bitangent;
    out.uv = input.uv;
    return out;
}

fn linear_to_srgb(c: vec3<f32>) -> vec3<f32> {
    // Piecewise sRGB encode (IEC 61966-2-1).
    let cutoff = vec3<f32>(0.0031308);
    let low = 12.92 * c;
    let high = 1.055 * pow(c, vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(high, low, c < cutoff);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // 1. Base color: texture (hardware-decoded sRGB→linear) × factor,
    //    falling back to plain factor if no texture is bound.
    var base_color = material.base_color_factor.rgb;
    if material.has_base_color_tex > 0.5 {
        let sampled = textureSample(base_color_tex, mat_sampler, in.uv);
        base_color *= sampled.rgb;
    } else {
        // No texture → fall back to instance tint for variety
        // (alice/bob can still be visually distinguished).
        base_color *= instance.base_color;
    }

    // 2. Tangent-space normal from normal map, brought to world space
    //    via TBN.
    var world_normal = normalize(in.world_normal);
    if material.has_normal_tex > 0.5 {
        let nm = textureSample(normal_tex, mat_sampler, in.uv).rgb;
        // [0, 1] → [-1, 1].
        var tangent_n = nm * 2.0 - vec3<f32>(1.0);
        tangent_n.x *= material.normal_scale;
        tangent_n.y *= material.normal_scale;
        let t = normalize(in.world_tangent);
        let b = normalize(in.world_bitangent);
        let n = world_normal;
        world_normal = normalize(tangent_n.x * t + tangent_n.y * b + tangent_n.z * n);
    }

    // 3. Linear-space lighting. Ambient now comes from the
    //    splat-derived 6-tap cube: weight each axis by max(dot(n,
    //    axis), 0)² so the character absorbs the scene's color cast
    //    in the direction the surface faces. A small direct key light
    //    gives shape on top of the ambient (otherwise the character
    //    reads as a flat sticker against the scene).
    let cube_weights = vec3<f32>(
        world_normal.x * world_normal.x,
        world_normal.y * world_normal.y,
        world_normal.z * world_normal.z,
    );
    let ambient_color =
        instance.ambient_cube[select(1u, 0u, world_normal.x > 0.0)].rgb * cube_weights.x +
        instance.ambient_cube[select(3u, 2u, world_normal.y > 0.0)].rgb * cube_weights.y +
        instance.ambient_cube[select(5u, 4u, world_normal.z > 0.0)].rgb * cube_weights.z;
    // Soft key light from above-front to give the character form on
    // top of the (often diffuse) ambient.
    let key_dir = normalize(vec3<f32>(0.3, -0.7, 0.4));  // Y-down world: -Y is up
    let key = max(dot(world_normal, key_dir), 0.0) * 0.3;
    let lit = base_color * (ambient_color + vec3<f32>(key));

    // 4. sRGB-encode for the non-sRGB IOSurface target. Matches the
    //    splat path's implicit assumption that bytes-as-stored are
    //    already sRGB-encoded.
    let encoded = linear_to_srgb(clamp(lit, vec3<f32>(0.0), vec3<f32>(1.0)));
    return vec4<f32>(encoded, 1.0);
}
