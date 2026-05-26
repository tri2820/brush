//! Headless wgpu pipeline that renders the voxel collision mesh into
//! the recorder's color texture. Same purpose as the viewer's V-toggled
//! `voxel_overlay` module, but runs inside `run_record` so `just
//! snapshot` produces PNGs that show what the viewer shows — no need
//! to fight macOS screen capture to verify.

#![cfg(target_os = "macos")]

use std::path::Path;

use anyhow::Result;
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// Read positions + (Uint32) indices from a static (non-skinned) glTF
/// file, applying every ancestor node's transform so the vertices are
/// in world space. Same logic as `brush-app::ui::voxel_overlay::
/// load_static_mesh` — inlined here to avoid pulling brush-app into
/// brush-cli.
fn load_static_mesh(path: &Path) -> Result<(Vec<Vec3>, Vec<u32>)> {
    let (doc, buffers, _images) = gltf::import(path)?;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    fn visit(
        node: &gltf::Node<'_>,
        parent: Mat4,
        buffers: &[gltf::buffer::Data],
        positions: &mut Vec<Vec3>,
        indices: &mut Vec<u32>,
    ) {
        let (t, r, s) = node.transform().decomposed();
        let local = Mat4::from_scale_rotation_translation(
            Vec3::from(s),
            glam::Quat::from_array(r),
            Vec3::from(t),
        );
        let world = parent * local;
        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
                let base = positions.len() as u32;
                if let Some(iter) = reader.read_positions() {
                    for p in iter {
                        positions.push(world.transform_point3(Vec3::from(p)));
                    }
                }
                if let Some(idx) = reader.read_indices() {
                    for i in idx.into_u32() {
                        indices.push(base + i);
                    }
                }
            }
        }
        for child in node.children() {
            visit(&child, world, buffers, positions, indices);
        }
    }

    for scene in doc.scenes() {
        for node in scene.nodes() {
            visit(&node, Mat4::IDENTITY, &buffers, &mut positions, &mut indices);
        }
    }

    if positions.is_empty() {
        anyhow::bail!("no mesh data in {}", path.display());
    }

    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for p in &positions {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    log::info!(
        "[voxel-overlay/record] world bounds X[{:.2},{:.2}] Y[{:.2},{:.2}] Z[{:.2},{:.2}]  ({}v, {}i)",
        lo.x, hi.x, lo.y, hi.y, lo.z, hi.z, positions.len(), indices.len()
    );

    Ok((positions, indices))
}

const SHADER: &str = r#"
struct U {
    view_proj: mat4x4<f32>,
    color: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: U;
@vertex
fn vs_main(@location(0) pos: vec3<f32>) -> @builtin(position) vec4<f32> {
    return u.view_proj * vec4<f32>(pos, 1.0);
}
@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return u.color;
}
"#;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    color: [f32; 4],
}

pub struct VoxelOverlay {
    pipeline: wgpu::RenderPipeline,
    uniform_buf: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    vertex_buf: wgpu::Buffer,
    index_buf: wgpu::Buffer,
    index_count: u32,
}

impl VoxelOverlay {
    /// Loads the mesh from `mesh_path` and prepares a pipeline that
    /// writes into a `target_format` color attachment with alpha
    /// blending and no depth test.
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        mesh_path: &Path,
    ) -> Result<Self> {
        let (positions, indices) = load_static_mesh(mesh_path)?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel-overlay/record shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-overlay/record uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-overlay/record bgl"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("voxel-overlay/record bg"),
            layout: &bgl,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut vert_bytes = Vec::with_capacity(positions.len() * 12);
        for p in &positions {
            vert_bytes.extend_from_slice(bytemuck::bytes_of(&[p.x, p.y, p.z]));
        }
        let vertex_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-overlay/record vbo"),
            contents: &vert_bytes,
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-overlay/record ibo"),
            contents: bytemuck::cast_slice(&indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let pl_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-overlay/record pl layout"),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel-overlay/record pipeline"),
            layout: Some(&pl_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[wgpu::VertexBufferLayout {
                    array_stride: 12,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    }],
                }],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                cull_mode: None,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        Ok(Self {
            pipeline,
            uniform_buf,
            bind_group,
            vertex_buf,
            index_buf,
            index_count: indices.len() as u32,
        })
    }

    /// Append the overlay onto `color_texture`. Caller has already
    /// drawn the splat backdrop and NPC mesh — this paints on top with
    /// alpha blending, no depth (the point is to see the voxels through
    /// everything).
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_texture: &wgpu::Texture,
        view_proj: Mat4,
        color: [f32; 4],
    ) -> wgpu::SubmissionIndex {
        let u = Uniforms {
            view_proj: view_proj.to_cols_array_2d(),
            color,
        };
        queue.write_buffer(&self.uniform_buf, 0, bytemuck::bytes_of(&u));

        let view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("voxel-overlay/record"),
        });
        {
            let mut pass = enc.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("voxel-overlay/record"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buf.slice(..));
            pass.set_index_buffer(self.index_buf.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }
        queue.submit(Some(enc.finish()))
    }
}
