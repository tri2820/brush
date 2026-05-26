//! Compute pipeline: brush packed-RGBA8 buffer → BGRA8 IOSurface texture.
//!
//! The brush rasterizer in `TextureMode::Packed` writes one little-endian
//! u32 per pixel where byte 0 = R, byte 1 = G, byte 2 = B, byte 3 = A.
//! VideoToolbox needs a BGRA8 texture. The shader does the byte swap and
//! writes to the IOSurface-backed storage texture in one GPU pass.

use wgpu::util::DeviceExt;

const SHADER: &str = r#"
struct Params {
    width: u32,
    height: u32,
};

@group(0) @binding(0) var<storage, read> src: array<u32>;
@group(0) @binding(1) var dst: texture_storage_2d<bgra8unorm, write>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.width || gid.y >= params.height {
        return;
    }
    let idx = gid.y * params.width + gid.x;
    let packed = src[idx];
    let r = (packed >> 0u) & 0xFFu;
    let g = (packed >> 8u) & 0xFFu;
    let b = (packed >> 16u) & 0xFFu;
    let a = (packed >> 24u) & 0xFFu;
    // wgpu storage textures take vec4 in RGBA shader-component order
    // regardless of the underlying memory layout; the driver writes
    // bytes as BGRA for us when the texture is bgra8unorm.
    let v = vec4<f32>(f32(r), f32(g), f32(b), f32(a)) / 255.0;
    textureStore(dst, vec2<i32>(gid.xy), v);
}
"#;

pub struct SwizzlePipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    width: u32,
    height: u32,
}

impl SwizzlePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("brush-record swizzle"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });

        let bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("brush-record swizzle bgl"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: wgpu::TextureFormat::Bgra8Unorm,
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("brush-record swizzle layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("brush-record swizzle pipeline"),
            layout: Some(&layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        let params_data = Params { width, height };
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("brush-record swizzle params"),
            contents: bytemuck::bytes_of(&params_data),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        Self {
            pipeline,
            bind_group_layout,
            params,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        src_buffer: &wgpu::Buffer,
        src_offset: u64,
        dst_texture: &wgpu::Texture,
        width: u32,
        height: u32,
    ) -> wgpu::SubmissionIndex {
        let view = dst_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("brush-record swizzle bg"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: src_buffer,
                        offset: src_offset,
                        size: std::num::NonZeroU64::new(
                            (width as u64) * (height as u64) * 4,
                        ),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.params.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("brush-record swizzle encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("brush-record swizzle pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        queue.submit(Some(encoder.finish()))
    }
}

// Re-derive bytemuck on Params via manual impls since we don't want to
// pull bytemuck into Cargo.toml; instead use #[repr(C)] and a tiny
// transmute helper. Done inline above with `bytemuck::bytes_of` from
// wgpu's util dep. (bytemuck is a transitive dep of wgpu.)
unsafe impl bytemuck::Pod for Params {}
unsafe impl bytemuck::Zeroable for Params {}
