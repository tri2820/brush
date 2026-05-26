//! Shared NPC subsystem used by both `run_record` and the interactive
//! viewer. Owns the asset cache, per-NPC runtime state, voxel collision,
//! and the [`MeshRenderer`] that drives the per-frame mesh pass. The
//! caller supplies the render target and the dt source — everything else
//! (asset loading, physics, animation eval, GPU uploads) lives here so
//! the two code paths can't drift again.
//!
//! Conventions match the record path because the record path is the
//! known-good reference:
//! - World coords are treated as Y-DOWN (the supersplat warehouse scene's
//!   native convention; cameras and authored positions assume +Y is the
//!   downward direction).
//! - The Mixamo character mesh is Y-UP, so the per-NPC model matrix
//!   includes a 180° rotation around X to flip head→+Y_world to
//!   head→-Y_world.
//! - Brush's renderer puts +Z forward; wgpu's `perspective_rh` expects
//!   -Z forward. The shared [`view_projection`] helper applies a 180°
//!   rotation around X on the view side, which flips both Y *and* Z —
//!   the Z flip is the bridge and the Y flip is the consequence (so
//!   downstream code in the mesh shader sees a consistent orientation).

#![cfg(target_os = "macos")]

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use brush_character::{
    GpuMaterial, GpuMesh, LinearPath, MeshAsset, MeshRenderer, NpcInstance, Path, load_mesh,
};
use brush_collision::VoxelCollision;
use glam::{Affine3A, Mat4, Quat, Vec3};

use crate::{PathConfig, SceneConfig};

/// Per-glTF asset state. Multiple NPCs that share an `asset` path share
/// the same `LoadedAsset` — the asset cache is keyed by path.
pub struct LoadedAsset {
    pub mesh: MeshAsset,
    pub gpu_mesh: GpuMesh,
    pub material: GpuMaterial,
}

/// Per-NPC mutable state.
pub struct NpcRuntime {
    pub scene_index: usize,
    pub instance: NpcInstance,
    pub animation_index: Option<usize>,
    pub path: Option<Box<dyn Path>>,
    /// Current world-space Y, owned by physics — the path drives XZ
    /// only when a collider is present. Initialized from the authored
    /// start Y; gravity + capsule pushout settle it onto the floor.
    pub current_y: f32,
    /// Vertical velocity in world units / second. Positive = falling
    /// (Y-DOWN convention). Zeroed when capsule pushout reports a
    /// ground hit.
    pub velocity_y: f32,
}

pub struct NpcSystem {
    pub scene: Arc<SceneConfig>,
    pub mesh_renderer: MeshRenderer,
    pub mesh_cache: HashMap<PathBuf, LoadedAsset>,
    pub runtimes: Vec<NpcRuntime>,
    pub collision: Option<VoxelCollision>,
    /// Accumulated world time in seconds. Advanced by `tick(dt)`.
    pub world_t: f32,
    /// Y-bias added to every NPC's rendered position. Cached from
    /// `scene.floor_offset_y` — see the field docs for why this exists.
    floor_offset_y: f32,
}

/// Capsule + physics tuning. Matches the values that produced the
/// known-good record output (bob stepping off the box).
const GRAVITY: f32 = 9.81;
const CAPSULE_HALF_HEIGHT: f32 = 0.5; // segment half-length (1.0 m body)
const CAPSULE_RADIUS: f32 = 0.3; // 0.6 m wide → 1.6 m tall total
const BODY_HALF_Y: f32 = CAPSULE_HALF_HEIGHT + CAPSULE_RADIUS;

impl NpcSystem {
    /// Build the subsystem: load each unique glTF asset, upload GPU
    /// resources, optionally load the voxel collider, and create one
    /// `NpcRuntime` per scene NPC.
    ///
    /// `ambient_fn` returns the 6-tap ambient cube for an NPC at a given
    /// world position. Record mode samples the splat probe; the viewer
    /// passes a flat 0.5 (no probe data on hand).
    pub fn new<F>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        scene: Arc<SceneConfig>,
        mut ambient_fn: F,
    ) -> Result<Self>
    where
        F: FnMut(Vec3) -> [[f32; 3]; 6],
    {
        let mesh_renderer = MeshRenderer::new(device, queue, color_format);

        let mut mesh_cache: HashMap<PathBuf, LoadedAsset> = HashMap::new();
        for npc in &scene.npcs {
            if !mesh_cache.contains_key(&npc.asset) {
                log::info!("Loading character asset {}", npc.asset.display());
                let asset = load_mesh(&npc.asset)?;
                let gpu_mesh = GpuMesh::upload(device, &asset);
                let material = mesh_renderer.upload_material(device, queue, &asset.material);
                mesh_cache.insert(
                    npc.asset.clone(),
                    LoadedAsset {
                        mesh: asset,
                        gpu_mesh,
                        material,
                    },
                );
            }
        }

        let collision = match &scene.collision {
            Some(path) => {
                log::info!("Loading voxel collision: {}", path.display());
                Some(VoxelCollision::load(path)?)
            }
            None => None,
        };

        let mut runtimes = Vec::with_capacity(scene.npcs.len());
        for (i, npc) in scene.npcs.iter().enumerate() {
            let loaded = mesh_cache.get(&npc.asset).expect("just inserted");
            let asset = &loaded.mesh;

            let anchor = match &npc.path {
                Some(PathConfig::Linear { start, .. }) => Vec3::from(*start),
                None => Vec3::from(npc.pos),
            };
            let ambient = ambient_fn(anchor);

            let model = model_matrix(npc.scale, npc.yaw_deg, Vec3::from(npc.pos));
            let instance =
                mesh_renderer.make_instance(device, &loaded.gpu_mesh, model, npc.color, ambient);

            let animation_index = match &npc.animation {
                Some(name) => {
                    let idx = asset.animations.iter().position(|a| &a.name == name);
                    if idx.is_none() {
                        let available: Vec<_> = asset.animations.iter().map(|a| &a.name).collect();
                        anyhow::bail!(
                            "npc '{}': animation '{}' not found in {} (have: {:?})",
                            npc.name,
                            name,
                            npc.asset.display(),
                            available,
                        );
                    }
                    idx
                }
                None => None,
            };

            let path: Option<Box<dyn Path>> = match &npc.path {
                Some(PathConfig::Linear {
                    start,
                    end,
                    duration_s,
                }) => Some(Box::new(LinearPath {
                    start: Vec3::from(*start),
                    end: Vec3::from(*end),
                    duration_s: *duration_s,
                })),
                None => None,
            };

            // Seed Y from the authored path start. The voxel data's
            // topmost solid layer IS the warehouse floor (verified
            // visually with the V overlay); ray-cast in tick() settles
            // the capsule onto it. Earlier "cavity floor" interpretation
            // (find inner bottom of empty region) moved NPCs to Y≈3.25,
            // which turned out to be *below* the warehouse entirely.
            let initial_y = match &npc.path {
                Some(PathConfig::Linear { start, .. }) => start[1],
                None => npc.pos[1],
            };

            runtimes.push(NpcRuntime {
                scene_index: i,
                instance,
                animation_index,
                path,
                current_y: initial_y,
                velocity_y: 0.0,
            });
        }

        let floor_offset_y = scene.floor_offset_y;
        Ok(Self {
            scene,
            mesh_renderer,
            mesh_cache,
            runtimes,
            collision,
            world_t: 0.0,
            floor_offset_y,
        })
    }

    /// Advance world time by `dt` and step physics + animation for every
    /// NPC. Uploads new model matrices and skin matrices to the GPU; the
    /// caller can then issue the mesh render pass.
    pub fn tick(&mut self, dt: f32, queue: &wgpu::Queue) -> Result<()> {
        self.world_t += dt;
        let world_t = self.world_t;

        for rt in &mut self.runtimes {
            let npc = &self.scene.npcs[rt.scene_index];
            let asset = &self
                .mesh_cache
                .get(&npc.asset)
                .expect("preloaded")
                .mesh;

            if let Some(path) = rt.path.as_deref() {
                let path_pos = path.position(world_t);
                let mut pos = if self.collision.is_some() {
                    Vec3::new(path_pos.x, rt.current_y, path_pos.z)
                } else {
                    path_pos
                };

                if let Some(c) = self.collision.as_ref() {
                    rt.velocity_y += GRAVITY * dt;
                    pos.y += rt.velocity_y * dt;

                    let capsule_center = Vec3::new(pos.x, pos.y - BODY_HALF_Y, pos.z);
                    if let Some(push) =
                        c.query_capsule(capsule_center, CAPSULE_HALF_HEIGHT, CAPSULE_RADIUS)
                    {
                        pos += push;
                        // Push in the world-up direction (-Y in Y-DOWN)
                        // means we hit ground — stop accelerating into it.
                        if push.y < 0.0 {
                            rt.velocity_y = 0.0;
                        }
                    }
                    rt.current_y = pos.y;
                }

                let yaw_deg = path.heading_deg(world_t);
                // Bias the rendered position only; physics state stays
                // in voxel-floor coordinates so collision stays consistent.
                let render_pos = Vec3::new(pos.x, pos.y + self.floor_offset_y, pos.z);
                let model = model_matrix(npc.scale, yaw_deg, render_pos);
                let ambient = rt.instance.ambient_cube;
                rt.instance.set_model(queue, model, npc.color, ambient);
            }

            let anim = rt.animation_index.map(|i| &asset.animations[i]);
            let skin_mats = asset.skeleton.evaluate(anim, world_t);
            rt.instance.upload_skin(queue, &skin_mats)?;
        }
        Ok(())
    }

    /// Issue the mesh pass for all NPCs into `color_texture`. Camera
    /// uniforms must already be set via `mesh_renderer.set_camera` (and
    /// optionally `fill_depth_from_splats` for record mode's depth
    /// integration). Returns the wgpu submission index so the caller
    /// can fence on it.
    ///
    /// Done as a method (vs. exposing the iterator + render call to the
    /// caller) so the disjoint borrow of `&mut self.mesh_renderer` and
    /// the iterator into `self.runtimes`/`self.mesh_cache` is contained
    /// here via explicit field destructuring.
    pub fn render_npcs(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_texture: &wgpu::Texture,
        clear_color: Option<wgpu::Color>,
    ) -> wgpu::SubmissionIndex {
        let Self {
            mesh_renderer,
            runtimes,
            scene,
            mesh_cache,
            ..
        } = self;
        let draws = runtimes.iter().map(|rt| {
            let asset_path = &scene.npcs[rt.scene_index].asset;
            let loaded = mesh_cache.get(asset_path).expect("preloaded");
            (&loaded.gpu_mesh, &rt.instance, &loaded.material)
        });
        mesh_renderer.render(device, queue, color_texture, clear_color, draws)
    }
}

/// Canonical per-NPC model matrix. Mixamo character (Y-UP) is rotated
/// 180° around X so its head points to world -Y (the up direction in
/// this Y-DOWN scene); `yaw_deg` rotates around the world up axis.
/// Applying yaw on the OUTSIDE of the X-flip keeps yaw_deg behaving like
/// a normal heading.
pub fn model_matrix(scale: f32, yaw_deg: f32, pos: Vec3) -> Mat4 {
    let rot = Quat::from_rotation_y(yaw_deg.to_radians())
        * Quat::from_rotation_x(std::f32::consts::PI);
    Mat4::from_scale_rotation_translation(Vec3::splat(scale), rot, pos)
}

/// Canonical view-projection for the NPC mesh pass. Both record and
/// viewer call this so the orientation can't drift.
///
/// `camera_world_to_local` is `Camera::world_to_local()` from
/// `brush-render`. `fov_y_rad` is the camera's vertical FOV in radians.
/// `aspect` is the render-target's width/height.
pub fn view_projection(
    camera_world_to_local: Affine3A,
    fov_y_rad: f32,
    aspect: f32,
) -> Mat4 {
    let proj = Mat4::perspective_rh(fov_y_rad, aspect, 0.05, 1000.0);
    let view = Mat4::from(camera_world_to_local);
    // Brush renderer uses +Z forward; wgpu `perspective_rh` uses -Z
    // forward. The X-rotation 180° flips Z (and Y as a consequence —
    // see the module-level note). Used identically by record and viewer
    // so the mesh shader sees one convention.
    let view_brush_to_wgpu = Mat4::from_quat(Quat::from_rotation_x(std::f32::consts::PI));
    proj * view_brush_to_wgpu * view
}
