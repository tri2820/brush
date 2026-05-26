//! Viewer-side NPC composite: drives the shared [`NpcSystem`] each frame
//! with a wall-clock `dt`, then blits its mesh-pass output over the splat
//! backbuffer via an egui-wgpu callback. All the asset loading, physics,
//! and animation lives in `brush_cli::npc_system::NpcSystem` so the viewer
//! and the recorder share one implementation.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::Result;
use brush_cli::SceneConfig;
use brush_cli::npc_system::{self, NpcSystem};
use brush_render::burn_glue::resolve_to_cube_float;
use burn::tensor::Tensor;
use eframe::egui_wgpu::{self, CallbackTrait, wgpu};
use glam::{Mat4, UVec2, Vec3};

pub struct NpcWorld {
    pub system: NpcSystem,
    pub last_tick: Option<Instant>,
    /// Offscreen RGBA8 texture the mesh pass writes into; sampled by
    /// the blit callback to composite over the splat backbuffer.
    pub render_texture: Option<wgpu::Texture>,
    pub render_texture_view: Option<wgpu::TextureView>,
    pub render_size: UVec2,
}

impl NpcWorld {
    pub fn new(
        state: &eframe::egui_wgpu::RenderState,
        scene: Arc<SceneConfig>,
    ) -> Result<Self> {
        // RGBA8 here matches the offscreen render target we allocate
        // below; the recorder uses BGRA8 to match its IOSurface texture.
        let system = NpcSystem::new(
            &state.device,
            &state.queue,
            wgpu::TextureFormat::Rgba8Unorm,
            scene,
            // Viewer doesn't have the splat probe on the CPU side;
            // flat 0.5 ambient is acceptable until probe readback gets
            // wired up.
            |_| [[0.5; 3]; 6],
        )?;
        Ok(Self {
            system,
            last_tick: None,
            render_texture: None,
            render_texture_view: None,
            render_size: UVec2::ZERO,
        })
    }

    fn ensure_render_texture(&mut self, device: &wgpu::Device, size: UVec2) {
        if self.render_size == size && self.render_texture.is_some() {
            return;
        }
        let tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("npc offscreen color"),
            size: wgpu::Extent3d {
                width: size.x.max(1),
                height: size.y.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        self.render_texture_view = Some(tex.create_view(&wgpu::TextureViewDescriptor::default()));
        self.render_texture = Some(tex);
        self.render_size = size;
    }

    /// Step physics + animation by wall-clock dt and re-render NPCs into
    /// the offscreen texture. `view_proj` must come from
    /// [`brush_cli::npc_system::view_projection`] so the orientation
    /// stays in lockstep with record mode.
    ///
    /// `splat_depth` is the per-pixel view-space depth tensor from
    /// `render_splats` for THIS frame. When present, the mesh pass
    /// loads its depth attachment from the splat depth so NPCs get
    /// correctly occluded by foreground splat geometry.
    pub fn tick_and_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view_proj: Mat4,
        cam_pos: Vec3,
        size: UVec2,
        splat_depth: Option<&Tensor<2>>,
    ) -> Result<()> {
        self.ensure_render_texture(device, size);

        let now = Instant::now();
        // Clamp dt so a long alt-tab pause can't fling the NPC.
        let dt = match self.last_tick {
            Some(t) => (now - t).as_secs_f32().min(0.1),
            None => 0.0,
        };
        self.last_tick = Some(now);

        self.system.tick(dt, queue)?;
        self.system
            .mesh_renderer
            .set_camera(queue, view_proj, cam_pos);

        let tex = self.render_texture.as_ref().expect("ensured");

        // Splat depth → mesh depth attachment, so NPC fragments behind
        // walls/columns get z-rejected. Skipping this is what makes the
        // viewer feel "painted on" — NPCs always draw over geometry.
        if let Some(depth) = splat_depth {
            let depth_dims = depth.dims();
            // depth tensor is [h, w] floats. Only feed it through when
            // its resolution matches our offscreen texture; otherwise
            // the depth values land at the wrong pixels.
            if depth_dims[0] == size.y as usize && depth_dims[1] == size.x as usize {
                let prim = resolve_to_cube_float(depth.clone());
                if let Ok(res) = prim.client.get_resource(prim.handle.clone()) {
                    let res = res.resource();
                    self.system.mesh_renderer.fill_depth_from_splats(
                        device,
                        queue,
                        tex,
                        &res.buffer,
                        res.offset,
                        0.05,
                        1000.0,
                    );
                }
            }
        }

        // Clear to transparent so the blit alpha-blends cleanly over
        // whatever was painted before us (the splat backbuffer).
        self.system
            .render_npcs(device, queue, tex, Some(wgpu::Color::TRANSPARENT));
        Ok(())
    }
}

/// Pipeline + bind-group resources for compositing the NPC offscreen
/// texture into egui's render pass via a fullscreen-triangle blit.
pub struct NpcBlitResources {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    bind_group: Option<wgpu::BindGroup>,
}

impl NpcBlitResources {
    pub fn new(device: &wgpu::Device, target_format: wgpu::TextureFormat) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("npc-blit shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/npc_blit.wgsl").into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("npc-blit bgl"),
            entries: &[
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
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("npc-blit sampler"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("npc-blit pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("npc-blit pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
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
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        });
        Self {
            pipeline,
            bind_group_layout,
            sampler,
            bind_group: None,
        }
    }
}

/// Per-frame paint callback. Holds a shared handle to the scene's
/// [`NpcWorld`] (so `prepare` can tick + render into the offscreen
/// texture) plus the current camera info.
pub struct NpcBlitCallback {
    pub world: Arc<Mutex<NpcWorld>>,
    pub view_proj: Mat4,
    pub cam_pos: Vec3,
    pub size: UVec2,
    /// Per-pixel splat depth tensor for this frame (view-space Z). Set
    /// from `SplatBackbuffer::latest().depth`; None during the initial
    /// frames before the splat actor has produced output.
    pub splat_depth: Option<Tensor<2>>,
}

impl CallbackTrait for NpcBlitCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen_descriptor: &egui_wgpu::ScreenDescriptor,
        _egui_encoder: &mut wgpu::CommandEncoder,
        resources: &mut egui_wgpu::CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let mut world = self.world.lock().expect("npc world poisoned");
        // Throttled debug: dump where NPCs actually are vs what camera/
        // view-projection the viewer is handing to the mesh pass.
        static LOG_COUNTER: std::sync::atomic::AtomicU32 =
            std::sync::atomic::AtomicU32::new(0);
        let n = LOG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if n % 60 == 0 {
            if let Some(rt) = world.system.runtimes.first() {
                log::info!(
                    "[viewer-npc] frame={} cam_pos={:?} size={:?}",
                    n, self.cam_pos, self.size
                );
                log::info!(
                    "[viewer-npc] alice current_y={:.3} velocity_y={:.3} offset={:.3}",
                    rt.current_y, rt.velocity_y, world.system.scene.floor_offset_y
                );
                // Project alice's render_pos manually and log where it lands
                // in clip space.
                let pos = glam::Vec3::new(6.0, rt.current_y + world.system.scene.floor_offset_y, 0.21);
                let clip = self.view_proj * pos.extend(1.0);
                log::info!(
                    "[viewer-npc] alice render_pos={:?} → clip=({:.3},{:.3},{:.3}) w={:.3}  ndc=({:.3},{:.3})",
                    pos, clip.x, clip.y, clip.z, clip.w,
                    clip.x / clip.w, clip.y / clip.w
                );
            }
        }
        if let Err(e) = world.tick_and_render(
            device,
            queue,
            self.view_proj,
            self.cam_pos,
            self.size,
            self.splat_depth.as_ref(),
        ) {
            log::warn!("npc tick failed: {e}");
            return Vec::new();
        }
        let Some(res) = resources.get_mut::<NpcBlitResources>() else {
            return Vec::new();
        };
        let Some(view) = world.render_texture_view.as_ref() else {
            return Vec::new();
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("npc-blit bind group"),
            layout: &res.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&res.sampler),
                },
            ],
        });
        res.bind_group = Some(bind_group);
        Vec::new()
    }

    fn paint(
        &self,
        _info: egui::PaintCallbackInfo,
        render_pass: &mut wgpu::RenderPass<'static>,
        resources: &egui_wgpu::CallbackResources,
    ) {
        let Some(res) = resources.get::<NpcBlitResources>() else {
            return;
        };
        let Some(bind_group) = res.bind_group.as_ref() else {
            return;
        };
        render_pass.set_pipeline(&res.pipeline);
        render_pass.set_bind_group(0, bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

/// Re-export so callers don't have to depend on brush-cli directly to
/// build the view-projection matrix.
pub use npc_system::view_projection;
