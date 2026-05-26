//! Toggleable overlay that renders the voxel collision mesh on top of
//! the splat scene. Useful for debugging "is the NPC actually on the
//! voxel floor?" — drives the question by showing where the collider
//! thinks the floor is, vs. where the splat shows it.
//!
//! Reads `<scene_collision>.collision.glb` (the voxel surface mesh
//! emitted by `splat-transform … -K faces`); see the bake notes in
//! README/CLAUDE.md.

use std::path::Path;

use anyhow::Result;
use eframe::egui_wgpu::{self, CallbackTrait, wgpu};
use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

/// GPU resources owned by the overlay. Vertex/index buffers + the
/// shared pipeline. Lives in the egui-wgpu `CallbackResources` map.
pub struct VoxelOverlayResources {
    pub pipeline: wgpu::RenderPipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [[f32; 4]; 4],
    color: [f32; 4],
}

impl VoxelOverlayResources {
    pub fn new(
        device: &wgpu::Device,
        target_format: wgpu::TextureFormat,
        positions: &[Vec3],
        indices: &[u32],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("voxel-overlay shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/voxel_overlay.wgsl").into()),
        });

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("voxel-overlay uniforms"),
            size: std::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("voxel-overlay bgl"),
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
            label: Some("voxel-overlay bind group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        // Pack Vec3 positions tightly. Each vertex = 12 bytes.
        let mut vert_bytes = Vec::with_capacity(positions.len() * 12);
        for p in positions {
            vert_bytes.extend_from_slice(bytemuck::bytes_of(&[p.x, p.y, p.z]));
        }
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-overlay vbo"),
            contents: &vert_bytes,
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("voxel-overlay ibo"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("voxel-overlay pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("voxel-overlay pipeline"),
            layout: Some(&pipeline_layout),
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
            // No depth attachment in egui's render pass — overlay will
            // draw over everything regardless of distance. That's fine
            // for a debug visualisation: you WANT to see the voxels
            // even behind walls.
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });

        Self {
            pipeline,
            bind_group_layout,
            uniform_buffer,
            bind_group,
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
        }
    }
}

/// Load positions + indices from a static (non-skinned) glTF mesh.
/// Uses the same `gltf` crate brush-character pulls in. Concatenates
/// every primitive of the file's first mesh — collision.glb from
/// splat-transform is a single mesh anyway.
pub fn load_static_mesh(path: &Path) -> Result<(Vec<Vec3>, Vec<u32>)> {
    let (doc, buffers, _images) = gltf::import(path)?;
    let mut positions: Vec<Vec3> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();

    let mesh = doc
        .meshes()
        .next()
        .ok_or_else(|| anyhow::anyhow!("no mesh in {}", path.display()))?;
    for prim in mesh.primitives() {
        let reader = prim.reader(|buf| Some(&buffers[buf.index()]));
        let base = positions.len() as u32;
        if let Some(iter) = reader.read_positions() {
            for p in iter {
                positions.push(Vec3::from(p));
            }
        }
        if let Some(idx) = reader.read_indices() {
            for i in idx.into_u32() {
                indices.push(base + i);
            }
        }
    }
    Ok((positions, indices))
}

pub struct VoxelOverlay {
    // Marker for "we've registered resources"; the actual GPU state
    // lives in CallbackResources keyed by VoxelOverlayResources.
    _registered: (),
}

impl VoxelOverlay {
    pub fn new(state: &eframe::egui_wgpu::RenderState, mesh_path: &Path) -> Result<Self> {
        let (positions, indices) = load_static_mesh(mesh_path)?;
        log::info!(
            "[voxel-overlay] loaded {} verts, {} indices from {}",
            positions.len(),
            indices.len(),
            mesh_path.display()
        );
        let res = VoxelOverlayResources::new(
            &state.device,
            state.target_format,
            &positions,
            &indices,
        );
        state.renderer.write().callback_resources.insert(res);
        Ok(Self { _registered: () })
    }

    pub fn paint(&self, rect: egui::Rect, ui: &egui::Ui, view_proj: Mat4, color: [f32; 4]) {
        ui.painter().add(egui_wgpu::Callback::new_paint_callback(
            rect,
            VoxelOverlayPainter { view_proj, color },
        ));
    }
}

struct VoxelOverlayPainter {
    view_proj: Mat4,
    color: [f32; 4],
}

impl CallbackTrait for VoxelOverlayPainter {
    fn prepare(
        &self,
        _device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(res) = resources.get::<VoxelOverlayResources>() else {
            return Vec::new();
        };
        let u = Uniforms {
            view_proj: self.view_proj.to_cols_array_2d(),
            color: self.color,
        };
        queue.write_buffer(&res.uniform_buffer, 0, bytemuck::bytes_of(&u));
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<VoxelOverlayResources>() else {
            return;
        };
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, &res.bind_group, &[]);
        render_pass.set_vertex_buffer(0, res.vertex_buffer.slice(..));
        render_pass.set_index_buffer(res.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        render_pass.draw_indexed(0..res.index_count, 0, 0..1);
    }
}
