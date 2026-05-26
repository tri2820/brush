//! wgpu pipeline for drawing skinned mesh characters over an existing
//! color attachment (the IOSurface holding the splat backdrop).
//!
//! Per draw we bind:
//! - camera uniform (view-projection + camera position)
//! - per-instance uniform (model matrix + base color)
//! - skin-matrices storage buffer (one mat4 per joint)
//!
//! The pipeline owns its own depth target. Mesh-vs-splats occlusion is
//! the next phase; mesh-vs-mesh and self-occlusion work today.

use anyhow::Result;
use bytemuck::{Pod, Zeroable};
use glam::Mat4;
use wgpu::util::DeviceExt;

use crate::gltf_load::{Material, MeshAsset, TextureImage};

/// GPU buffers + skinning state for one mesh. Reusable across multiple
/// instances of the same character.
pub struct GpuMesh {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub joint_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Vertex {
    position: [f32; 3],
    _pad0: f32,
    normal: [f32; 3],
    _pad1: f32,
    tangent: [f32; 4],
    uv: [f32; 2],
    _pad2: [f32; 2],
    joints: [u32; 4],
    weights: [f32; 4],
}

impl GpuMesh {
    pub fn upload(device: &wgpu::Device, asset: &MeshAsset) -> Self {
        let mut verts = Vec::with_capacity(asset.positions.len());
        for i in 0..asset.positions.len() {
            verts.push(Vertex {
                position: asset.positions[i].into(),
                _pad0: 0.0,
                normal: asset.normals[i].into(),
                _pad1: 0.0,
                tangent: asset.tangents[i],
                uv: asset.texcoords[i],
                _pad2: [0.0; 2],
                joints: [
                    asset.joints[i][0] as u32,
                    asset.joints[i][1] as u32,
                    asset.joints[i][2] as u32,
                    asset.joints[i][3] as u32,
                ],
                weights: asset.weights[i],
            });
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("character vertex buffer"),
            contents: bytemuck::cast_slice(&verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("character index buffer"),
            contents: bytemuck::cast_slice(&asset.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Self {
            vertex_buffer,
            index_buffer,
            index_count: asset.indices.len() as u32,
            joint_count: asset.skeleton.joints.len() as u32,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CameraUniforms {
    view_proj: [[f32; 4]; 4],
    cam_pos: [f32; 3],
    _pad: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct InstanceUniforms {
    model: [[f32; 4]; 4],
    base_color: [f32; 3],
    _pad: f32,
    /// 6-tap ambient cube (+X, -X, +Y, -Y, +Z, -Z), in linear color
    /// space. Sampled from the splat scene near the NPC's anchor
    /// position; gives the character a directional color cast that
    /// matches the surrounding scene. Each vec3 is padded to vec4 for
    /// uniform-layout alignment.
    ambient_cube: [[f32; 4]; 6],
}

/// Per-NPC GPU state: model uniform + skin-matrix storage buffer. One
/// `NpcInstance` per character placed in the scene.
pub struct NpcInstance {
    pub model: Mat4,
    pub base_color: [f32; 3],
    pub ambient_cube: [[f32; 3]; 6],
    instance_buf: wgpu::Buffer,
    skin_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// `joint_count` mat4's worth of bytes; cached here for upload_skin.
    skin_buf_size: u64,
}

fn pad_cube(c: [[f32; 3]; 6]) -> [[f32; 4]; 6] {
    [
        [c[0][0], c[0][1], c[0][2], 0.0],
        [c[1][0], c[1][1], c[1][2], 0.0],
        [c[2][0], c[2][1], c[2][2], 0.0],
        [c[3][0], c[3][1], c[3][2], 0.0],
        [c[4][0], c[4][1], c[4][2], 0.0],
        [c[5][0], c[5][1], c[5][2], 0.0],
    ]
}

/// Per-mesh material textures + sampler, in their own bind group so
/// the renderer can swap materials without re-creating instance
/// bindings. Today: baseColor (sRGB) + normal (linear) + a single
/// MaterialUniforms for factors / scale.
pub struct GpuMaterial {
    bind_group: wgpu::BindGroup,
    _base_color_view: wgpu::TextureView,
    _normal_view: wgpu::TextureView,
    _sampler: wgpu::Sampler,
    _uniform_buf: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MaterialUniforms {
    base_color_factor: [f32; 4],
    normal_scale: f32,
    /// 1.0 if baseColor texture is bound, 0.0 otherwise → shader can
    /// fall back to the factor only.
    has_base_color_tex: f32,
    /// 1.0 if normal texture is bound.
    has_normal_tex: f32,
    _pad: f32,
}

/// Camera-level state: view-projection uniform + bind group layout.
pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,
    instance_bgl: wgpu::BindGroupLayout,
    material_bgl: wgpu::BindGroupLayout,
    camera_buf: wgpu::Buffer,
    depth_view: Option<wgpu::TextureView>,
    depth_size: (u32, u32),
    /// 1×1 sRGB white used as the baseColor when a glTF material has no
    /// texture (we sample it and rely on the factor for the final
    /// color). Shared by every "no baseColor" material.
    fallback_base_color: wgpu::Texture,
    /// 1×1 flat normal map (128, 128, 255) interpreted as tangent-space
    /// (0, 0, 1) — i.e. no perturbation.
    fallback_normal: wgpu::Texture,
}

impl MeshRenderer {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("character skinned shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/skinned.wgsl").into(),
            ),
        });

        // Bind group 0: camera + per-instance state (model uniform + skin matrices).
        let instance_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("character instance bgl"),
            entries: &[
                // camera (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // instance (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // skin matrices (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Bind group 1: per-material textures + sampler + factors.
        let material_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("character material bgl"),
            entries: &[
                // base color texture (sRGB → linear on sample)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // normal map (linear)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // shared sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // material uniforms (factors)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
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
            label: Some("character pl"),
            bind_group_layouts: &[Some(&instance_bgl), Some(&material_bgl)],
            immediate_size: 0,
        });

        // Vertex layout (matches the `Vertex` struct): offsets in bytes.
        // 0: position (vec3 + pad to vec4)
        // 16: normal (vec3 + pad to vec4)
        // 32: tangent (vec4)
        // 48: uv (vec2 + pad to vec4)
        // 64: joints (uvec4)
        // 80: weights (vec4)
        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 16,
                    shader_location: 1,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 32,
                    shader_location: 2,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x2,
                    offset: 48,
                    shader_location: 3,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32x4,
                    offset: 64,
                    shader_location: 4,
                },
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 80,
                    shader_location: 5,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("character pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                // No back-face cull during bring-up — we don't yet know
                // the model's winding order vs. the view-flip we apply.
                cull_mode: None,
                unclipped_depth: false,
                polygon_mode: wgpu::PolygonMode::Fill,
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let camera_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("character camera ub"),
            size: std::mem::size_of::<CameraUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // 1×1 fallback textures for materials missing baseColor / normal.
        let fallback_base_color = make_solid_texture(
            device,
            queue,
            "fallback baseColor (white)",
            wgpu::TextureFormat::Rgba8UnormSrgb,
            [255, 255, 255, 255],
        );
        let fallback_normal = make_solid_texture(
            device,
            queue,
            "fallback normal (flat)",
            wgpu::TextureFormat::Rgba8Unorm,
            // Tangent-space (0, 0, 1) → unbiased (128, 128, 255).
            [128, 128, 255, 255],
        );

        Self {
            pipeline,
            instance_bgl,
            material_bgl,
            camera_buf,
            depth_view: None,
            depth_size: (0, 0),
            fallback_base_color,
            fallback_normal,
        }
    }

    /// Upload a `Material` to the GPU as a bind-group-ready `GpuMaterial`.
    /// Reuse: cache GpuMaterial by glTF source path in the caller.
    pub fn upload_material(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material: &Material,
    ) -> GpuMaterial {
        let base_color_tex = match &material.base_color {
            Some(img) => upload_texture(
                device,
                queue,
                "character baseColor",
                wgpu::TextureFormat::Rgba8UnormSrgb,
                img,
            ),
            None => self.fallback_base_color.clone(),
        };
        let normal_tex = match &material.normal {
            Some(img) => upload_texture(
                device,
                queue,
                "character normal",
                wgpu::TextureFormat::Rgba8Unorm,
                img,
            ),
            None => self.fallback_normal.clone(),
        };

        let base_color_view = base_color_tex.create_view(&wgpu::TextureViewDescriptor::default());
        let normal_view = normal_tex.create_view(&wgpu::TextureViewDescriptor::default());

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("character sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            address_mode_w: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 1,
            ..Default::default()
        });

        let uniforms = MaterialUniforms {
            base_color_factor: material.base_color_factor,
            normal_scale: material.normal_scale,
            has_base_color_tex: if material.base_color.is_some() { 1.0 } else { 0.0 },
            has_normal_tex: if material.normal.is_some() { 1.0 } else { 0.0 },
            _pad: 0.0,
        };
        let uniform_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("character material ub"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("character material bg"),
            layout: &self.material_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&base_color_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        GpuMaterial {
            bind_group,
            _base_color_view: base_color_view,
            _normal_view: normal_view,
            _sampler: sampler,
            _uniform_buf: uniform_buf,
        }
    }

    /// Update the shared per-camera uniform. Call once per camera per frame.
    pub fn set_camera(&self, queue: &wgpu::Queue, view_proj: Mat4, cam_pos: glam::Vec3) {
        let u = CameraUniforms {
            view_proj: view_proj.to_cols_array_2d(),
            cam_pos: cam_pos.into(),
            _pad: 0.0,
        };
        queue.write_buffer(&self.camera_buf, 0, bytemuck::bytes_of(&u));
    }

    pub fn make_instance(
        &self,
        device: &wgpu::Device,
        mesh: &GpuMesh,
        model: Mat4,
        base_color: [f32; 3],
        ambient_cube: [[f32; 3]; 6],
    ) -> NpcInstance {
        let instance_u = InstanceUniforms {
            model: model.to_cols_array_2d(),
            base_color,
            _pad: 0.0,
            ambient_cube: pad_cube(ambient_cube),
        };
        let instance_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("character instance ub"),
            contents: bytemuck::bytes_of(&instance_u),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let skin_buf_size =
            (mesh.joint_count as u64) * std::mem::size_of::<[[f32; 4]; 4]>() as u64;
        // Initialize with identity matrices → T-pose. Real animation
        // evaluator overwrites this per frame.
        let identity = Mat4::IDENTITY.to_cols_array_2d();
        let mut buf_bytes = Vec::with_capacity(skin_buf_size as usize);
        for _ in 0..mesh.joint_count {
            buf_bytes.extend_from_slice(bytemuck::bytes_of(&identity));
        }
        let skin_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("character skin sb"),
            contents: &buf_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("character instance bg"),
            layout: &self.instance_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.camera_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: instance_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: skin_buf.as_entire_binding(),
                },
            ],
        });

        NpcInstance {
            model,
            base_color,
            ambient_cube,
            instance_buf,
            skin_buf,
            bind_group,
            skin_buf_size,
        }
    }

    /// Run one render pass into `color_view` (the IOSurface texture
    /// view) drawing the given instances. Returns the submission index
    /// the caller should fence on before the encoder reads the
    /// IOSurface.
    pub fn render<'a>(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_texture: &wgpu::Texture,
        instances: impl IntoIterator<Item = (&'a GpuMesh, &'a NpcInstance, &'a GpuMaterial)>,
    ) -> wgpu::SubmissionIndex {
        let (w, h) = (color_texture.width(), color_texture.height());
        self.ensure_depth(device, w, h);
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("character mesh pass"),
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("character mesh pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        // Preserve the splat backdrop the swizzle wrote.
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: self.depth_view.as_ref().expect("ensure_depth"),
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (mesh, inst, material) in instances {
                pass.set_bind_group(0, &inst.bind_group, &[]);
                pass.set_bind_group(1, &material.bind_group, &[]);
                pass.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
                pass.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..mesh.index_count, 0, 0..1);
            }
        }
        queue.submit(Some(encoder.finish()))
    }

    fn ensure_depth(&mut self, device: &wgpu::Device, w: u32, h: u32) {
        if self.depth_size != (w, h) {
            let tex = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("character depth"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Depth32Float,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
            self.depth_size = (w, h);
        }
    }
}

impl NpcInstance {
    /// Update the per-instance model matrix uniform.
    pub fn set_model(
        &mut self,
        queue: &wgpu::Queue,
        model: Mat4,
        base_color: [f32; 3],
        ambient_cube: [[f32; 3]; 6],
    ) {
        self.model = model;
        self.base_color = base_color;
        self.ambient_cube = ambient_cube;
        let u = InstanceUniforms {
            model: model.to_cols_array_2d(),
            base_color,
            _pad: 0.0,
            ambient_cube: pad_cube(ambient_cube),
        };
        queue.write_buffer(&self.instance_buf, 0, bytemuck::bytes_of(&u));
    }

    /// Upload per-joint skinning matrices. `mats.len()` must equal the
    /// joint count of the mesh this instance belongs to.
    pub fn upload_skin(&self, queue: &wgpu::Queue, mats: &[Mat4]) -> Result<()> {
        let need = self.skin_buf_size as usize;
        let supplied = mats.len() * std::mem::size_of::<[[f32; 4]; 4]>();
        if supplied != need {
            anyhow::bail!(
                "skin matrix count mismatch: buffer expects {} bytes, got {}",
                need,
                supplied,
            );
        }
        // glam::Mat4 is not `bytemuck::Pod` directly; flatten through
        // its `to_cols_array_2d` representation, which is `[[f32; 4]; 4]`
        // and matches the WGSL `mat4x4<f32>` layout.
        let mut bytes = Vec::with_capacity(need);
        for m in mats {
            let arr = m.to_cols_array_2d();
            bytes.extend_from_slice(bytemuck::bytes_of(&arr));
        }
        queue.write_buffer(&self.skin_buf, 0, &bytes);
        Ok(())
    }
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    image: &TextureImage,
) -> wgpu::Texture {
    let size = wgpu::Extent3d {
        width: image.width,
        height: image.height,
        depth_or_array_layers: 1,
    };
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &image.rgba,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(image.width * 4),
            rows_per_image: Some(image.height),
        },
        size,
    );
    tex
}

/// Build a 1×1 texture filled with the given color. Used for material
/// fallbacks when a glTF material omits a channel.
fn make_solid_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    format: wgpu::TextureFormat,
    color: [u8; 4],
) -> wgpu::Texture {
    let tex = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &tex,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        &color,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4),
            rows_per_image: Some(1),
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    tex
}
