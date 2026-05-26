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

use crate::gltf_load::MeshAsset;

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
}

/// Per-NPC GPU state: model uniform + skin-matrix storage buffer. One
/// `NpcInstance` per character placed in the scene.
pub struct NpcInstance {
    pub model: Mat4,
    pub base_color: [f32; 3],
    instance_buf: wgpu::Buffer,
    skin_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    /// `joint_count` mat4's worth of bytes; cached here for upload_skin.
    skin_buf_size: u64,
}

/// Camera-level state: view-projection uniform + bind group layout.
pub struct MeshRenderer {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    camera_buf: wgpu::Buffer,
    depth_view: Option<wgpu::TextureView>,
    depth_size: (u32, u32),
}

impl MeshRenderer {
    pub fn new(device: &wgpu::Device, color_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("character skinned shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/skinned.wgsl").into(),
            ),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("character bgl"),
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

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("character pl"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 0,
                    shader_location: 0,
                },
                // normal
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x3,
                    offset: 16,
                    shader_location: 1,
                },
                // joints (uint32x4)
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Uint32x4,
                    offset: 32,
                    shader_location: 2,
                },
                // weights (float32x4)
                wgpu::VertexAttribute {
                    format: wgpu::VertexFormat::Float32x4,
                    offset: 48,
                    shader_location: 3,
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

        Self {
            pipeline,
            bind_group_layout,
            camera_buf,
            depth_view: None,
            depth_size: (0, 0),
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
    ) -> NpcInstance {
        let instance_u = InstanceUniforms {
            model: model.to_cols_array_2d(),
            base_color,
            _pad: 0.0,
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
            label: Some("character bg"),
            layout: &self.bind_group_layout,
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
        instances: impl IntoIterator<Item = (&'a GpuMesh, &'a NpcInstance)>,
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
            for (mesh, inst) in instances {
                pass.set_bind_group(0, &inst.bind_group, &[]);
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
    pub fn set_model(&mut self, queue: &wgpu::Queue, model: Mat4, base_color: [f32; 3]) {
        self.model = model;
        self.base_color = base_color;
        let u = InstanceUniforms {
            model: model.to_cols_array_2d(),
            base_color,
            _pad: 0.0,
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
