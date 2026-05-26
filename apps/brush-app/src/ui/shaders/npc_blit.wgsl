// Fullscreen-triangle blit of the NPC offscreen render texture onto
// egui's color target. Alpha-blended on top of whatever was drawn before
// (the splat backbuffer paint callback).

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> @builtin(position) vec4<f32> {
    // Three vertices that cover the viewport via the standard
    // fullscreen-triangle trick.
    var p = vec2<f32>(-1.0, -1.0);
    if (vi == 1u) { p = vec2<f32>(3.0, -1.0); }
    if (vi == 2u) { p = vec2<f32>(-1.0, 3.0); }
    return vec4<f32>(p, 0.0, 1.0);
}

@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    let dims = textureDimensions(src);
    let uv = vec2<f32>(frag_coord.x / f32(dims.x), frag_coord.y / f32(dims.y));
    return textureSample(src, samp, uv);
}
