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
    GpuMaterial, GpuMesh, MeshAsset, MeshRenderer, NpcInstance, load_mesh,
};
use glam::{Affine3A, Mat4, Quat, Vec3};

use crate::SceneConfig;

/// Per-glTF asset state. Multiple NPCs that share an `asset` path share
/// the same `LoadedAsset` — the asset cache is keyed by path.
pub struct LoadedAsset {
    pub mesh: MeshAsset,
    pub gpu_mesh: GpuMesh,
    pub material: GpuMaterial,
}

/// Brain "role" — the animation/locomotion category the NPC is in for
/// the current timeline step. Ported from `gsa/scripts/brain.gd` but
/// trimmed to what `character.glb` actually ships: a Walk loop and two
/// stationary fall poses. (No idle/punch/sword/jump clips available, so
/// Idle uses the bind pose.)
#[derive(Copy, Clone, Debug)]
pub enum Role {
    Idle,
    Walk,
    Fall,
    FallSide,
}

/// One step on a brain's timeline. Mirrors gsa's `TimelineStep` dict
/// but typed.
#[derive(Clone, Debug)]
pub struct TimelineStep {
    pub role: Role,
    pub duration: f32,
    /// XZ direction (Y=0). Zero for stationary roles.
    pub direction: Vec3,
    /// Animation name to play during this step; `None` → bind pose
    /// (IDLE). Resolved into a per-NPC `animation_index` once we know
    /// the loaded asset.
    pub anim_name: Option<&'static str>,
}

/// Stand-alone PRNG so we don't pull in `rand` for one usage. xorshift64*
/// is fine — deterministic, fast, decent distribution for the tiny number
/// of samples a brain consumes per step. Mirrors what `RandomNumberGenerator`
/// does in gsa (also xorshift-family in Godot 4).
#[derive(Clone, Debug)]
pub struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the all-zeros state which would lock xorshift at zero.
        Self(if seed == 0 { 0xdead_beef_cafe_babe } else { seed })
    }
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform float in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / (1u64 << 24) as f32)
    }
    pub fn range_f32(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// Per-NPC brain state. Owns the PRNG, the current step, time-in-step,
/// and the "last non-zero locomotion direction" used for the 70%/30%
/// drift-vs-fresh azimuth pick.
#[derive(Clone, Debug)]
pub struct Brain {
    pub rng: Rng,
    pub step: TimelineStep,
    pub elapsed: f32,
    /// Last non-zero walk direction. Walk steps drift around this 70%
    /// of the time so the wander looks coherent across step boundaries.
    pub last_direction: Vec3,
    /// Stuck detector: world-position one frame ago, plus seconds spent
    /// wanting-to-move-but-not-moving. If we're walled in for too long,
    /// force the step to expire so the next roll picks a fresh azimuth.
    pub last_pos_xz: Vec3,
    pub stuck_secs: f32,
}

/// Per-NPC mutable state.
pub struct NpcRuntime {
    pub scene_index: usize,
    pub instance: NpcInstance,
    /// Animation index for the CURRENT step (changes on step entry).
    /// `None` means play the bind pose (used for the IDLE role since
    /// `character.glb` doesn't ship an idle clip).
    pub current_anim_index: Option<usize>,
    /// Seconds since the current step started — drives anim sampling
    /// locally so one-shots like Fall play from frame 0 each time
    /// instead of being phase-locked to `world_t`.
    pub step_anim_t: f32,
    /// Current world-space position. XZ updated by brain locomotion;
    /// Y pinned to `floor_y`.
    pub pos: Vec3,
    /// Current yaw in degrees; updated when walking, frozen when idle/
    /// falling so the character keeps facing its last walk heading.
    pub yaw_deg: f32,
    /// Brain (None for static NPCs without a `brain` config).
    pub brain: Option<Brain>,
    /// Cached spawn box from `BrainConfig` for the bounds clamp. Stored
    /// here so the tick loop doesn't have to look it up through scene.
    pub spawn_min_xz: Vec3,
    pub spawn_max_xz: Vec3,
}

pub struct NpcSystem {
    pub scene: Arc<SceneConfig>,
    pub mesh_renderer: MeshRenderer,
    pub mesh_cache: HashMap<PathBuf, LoadedAsset>,
    pub runtimes: Vec<NpcRuntime>,
    /// Accumulated world time in seconds. Advanced by `tick(dt)`.
    pub world_t: f32,
    /// World Y where NPC feet land. Same plane the grid widget draws
    /// at, so toggling the grid in the viewer is a direct visual check
    /// of where NPCs will stand. Defaults to 0; per-scene override via
    /// `floor_y` in scene.json.
    floor_y: f32,
    /// Per-NPC `[Idle, Walk, Fall, FallSide]` → `animation_index`
    /// lookup table. Built once at `new()`. None entries are valid:
    /// Idle is always None (bind pose); a glb missing Fall just won't
    /// ever play that role.
    anim_table: Vec<[Option<usize>; 4]>,
}

// ---------- Brain tuning -------------------------------------------------
// Mirrors gsa/scripts/constants.gd values that survive the port. We drop
// MOVE_SPEED's 2.0 m/s in favour of 1.4 m/s — Quaternius's gsa character
// is ~1.75m tall whereas ours is the same Mixamo rig the supersplat-viewer
// uses at near full scale, and 2.0 m/s looks like jogging from the side
// cameras. 1.4 m/s lands closer to "walking on a warehouse floor".

const WALK_SPEED: f32 = 1.4;
const STEP_MIN: f32 = 1.5;
const STEP_MAX: f32 = 4.0;
const DIRECTION_DRIFT_PROB: f32 = 0.7;
const DIRECTION_DRIFT_RAD: f32 = 0.524; // ±~30°

// Role probabilities. We drop IDLE entirely because character.glb ships
// no idle clip — the bind pose is a T-pose, which is uglier than just
// always walking. Falls are rare comedic moments. Mirrors the spirit of
// gsa's 65/27/8 split but compressed onto the three animations we have.
const P_WALK: f32 = 0.86;
const P_FALL: f32 = 0.07;
// P_FALL_SIDE = 1.0 - the above two. IDLE role kept in the enum for
// future use (e.g., if an Idle clip is added to the glb) but not
// sampled by `next_brain_step` today.

// Pairwise NPC separation. Ported from gsa Perception::separation —
// repulsion in a 2.5m radius, peaking at full strength when overlapping.
const SEPARATION_RADIUS: f32 = 2.5;
const SEPARATION_STRENGTH: f32 = 1.5;

// Stuck detector. If the brain's current step has a non-zero direction
// (walking) but we've moved less than STUCK_MIN_SPEED for STUCK_TIMEOUT
// seconds, force-expire the step so we roll a fresh azimuth.
const STUCK_MIN_SPEED: f32 = 0.5;
const STUCK_TIMEOUT: f32 = 0.8;

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

        let floor_y = scene.floor_y;
        let mut runtimes = Vec::with_capacity(scene.npcs.len());
        for (i, npc) in scene.npcs.iter().enumerate() {
            let loaded = mesh_cache.get(&npc.asset).expect("just inserted");
            let asset = &loaded.mesh;

            // Initial position. For brained NPCs we seed at the centre
            // of the spawn box so they don't all spawn on the same
            // corner; otherwise use the authored static `pos`.
            let initial_pos = match &npc.brain {
                Some(b) => Vec3::new(
                    (b.spawn_min[0] + b.spawn_max[0]) * 0.5,
                    floor_y,
                    (b.spawn_min[1] + b.spawn_max[1]) * 0.5,
                ),
                None => Vec3::from(npc.pos),
            };
            let ambient = ambient_fn(initial_pos);

            let model = model_matrix(npc.scale, npc.yaw_deg, initial_pos);
            let instance =
                mesh_renderer.make_instance(device, &loaded.gpu_mesh, model, npc.color, ambient);

            // Brain animations expect three clip names to be present in
            // the glb — we resolve each lazily so a glb missing one of
            // them still loads (just won't ever play that fall variant).
            let resolve = |name: &str| -> Option<usize> {
                asset.animations.iter().position(|a| a.name == name)
            };
            let walk_idx = npc
                .animation
                .as_deref()
                .and_then(resolve)
                .or_else(|| resolve("Walk"));
            let fall_idx = resolve("Fall");
            let fall_side_idx = resolve("fall_side");

            // Initial step: spend the first 0.5s in IDLE so all NPCs
            // don't lurch off at t=0 with a synchronised gait.
            let initial_step = TimelineStep {
                role: Role::Idle,
                duration: 0.5,
                direction: Vec3::ZERO,
                anim_name: None,
            };

            let brain = npc.brain.as_ref().map(|cfg| Brain {
                rng: Rng::new(cfg.seed),
                step: initial_step.clone(),
                elapsed: 0.0,
                last_direction: Vec3::ZERO,
                last_pos_xz: initial_pos,
                stuck_secs: 0.0,
            });

            let (spawn_min_xz, spawn_max_xz) = match &npc.brain {
                Some(b) => (
                    Vec3::new(b.spawn_min[0], floor_y, b.spawn_min[1]),
                    Vec3::new(b.spawn_max[0], floor_y, b.spawn_max[1]),
                ),
                None => (initial_pos, initial_pos),
            };

            runtimes.push(NpcRuntime {
                scene_index: i,
                instance,
                current_anim_index: walk_idx
                    .map(|_| walk_idx.unwrap())
                    .filter(|_| brain.is_none()), // static NPC: play `animation` continuously
                step_anim_t: 0.0,
                pos: initial_pos,
                yaw_deg: npc.yaw_deg,
                brain,
                spawn_min_xz,
                spawn_max_xz,
            });

            // Stash the resolved indices per-NPC so the tick loop can
            // map a TimelineStep back to an animation_index without
            // re-walking the glb. We piggyback on a per-NPC array keyed
            // by Role.
            // (Indices end up captured in the closure below; we stash
            // them in NpcSystem so tick() can read them.)
            let _ = (walk_idx, fall_idx, fall_side_idx);
        }

        // Build a per-NPC table mapping Role → animation_index so tick
        // doesn't have to re-resolve names on every step transition.
        let mut anim_table: Vec<[Option<usize>; 4]> = Vec::with_capacity(scene.npcs.len());
        for npc in &scene.npcs {
            let asset = &mesh_cache.get(&npc.asset).expect("preloaded").mesh;
            let resolve = |name: &str| asset.animations.iter().position(|a| a.name == name);
            let walk = npc
                .animation
                .as_deref()
                .and_then(resolve)
                .or_else(|| resolve("Walk"));
            anim_table.push([
                None,             // Role::Idle → bind pose
                walk,             // Role::Walk
                resolve("Fall"),  // Role::Fall
                resolve("fall_side"), // Role::FallSide
            ]);
        }

        Ok(Self {
            scene,
            mesh_renderer,
            mesh_cache,
            runtimes,
            world_t: 0.0,
            floor_y,
            anim_table,
        })
    }

    /// Advance world time by `dt` and step physics + animation for every
    /// NPC. Uploads new model matrices and skin matrices to the GPU; the
    /// caller can then issue the mesh render pass.
    pub fn tick(&mut self, dt: f32, queue: &wgpu::Queue) -> Result<()> {
        self.world_t += dt;

        // Two-pass tick so the separation force in pass 2 can read every
        // NPC's pre-separation position from pass 1.
        // Pass 1: brain advance + locomotion + bounds clamp.
        for ri in 0..self.runtimes.len() {
            let scene_index = self.runtimes[ri].scene_index;
            let npc_scale = self.scene.npcs[scene_index].scale;
            let asset = &self
                .mesh_cache
                .get(&self.scene.npcs[scene_index].asset)
                .expect("preloaded")
                .mesh;

            let rt = &mut self.runtimes[ri];
            rt.step_anim_t += dt;

            if let Some(brain) = rt.brain.as_mut() {
                // Brain step advance.
                brain.elapsed += dt;

                // Stuck check: only meaningful for steps that *want* to
                // move (Role::Walk with non-zero direction).
                let wants_to_move = brain.step.direction.length_squared() > 1e-4;
                let displacement = (rt.pos - brain.last_pos_xz).length();
                brain.last_pos_xz = rt.pos;
                if wants_to_move && displacement < STUCK_MIN_SPEED * dt {
                    brain.stuck_secs += dt;
                    if brain.stuck_secs >= STUCK_TIMEOUT {
                        brain.elapsed = brain.step.duration; // force expiry
                        brain.stuck_secs = 0.0;
                    }
                } else {
                    brain.stuck_secs = 0.0;
                }

                if brain.elapsed >= brain.step.duration {
                    // Snapshot last_direction from the just-finished
                    // step before rolling — keeps walks coherent across
                    // step boundaries.
                    if brain.step.direction.length_squared() > 1e-4 {
                        brain.last_direction = brain.step.direction.normalize();
                    }
                    let clip_dur = |idx: Option<usize>| -> f32 {
                        idx.and_then(|i| asset.animations.get(i))
                            .map(|a| a.duration.max(0.3))
                            .unwrap_or(0.5)
                    };
                    let table = &self.anim_table[scene_index];
                    brain.step = next_brain_step(brain, table, clip_dur);
                    brain.elapsed = 0.0;
                    rt.step_anim_t = 0.0;
                    let table = &self.anim_table[scene_index];
                    rt.current_anim_index = match brain.step.role {
                        Role::Idle => table[0],
                        Role::Walk => table[1],
                        Role::Fall => table[2],
                        Role::FallSide => table[3],
                    };
                }

                // Apply locomotion: WALK moves, others hold still.
                if matches!(brain.step.role, Role::Walk) {
                    rt.pos += brain.step.direction * WALK_SPEED * dt;
                    // Face walking direction. The Mixamo model's local forward
                    // is +Z, so after the X-flip in model_matrix the effective
                    // world facing vector for yaw φ is (-sin φ, 0, -cos φ).
                    // To face direction (dx, 0, dz) we need:
                    //   -sin φ = dx  →  sin φ = -dx
                    //   -cos φ = dz  →  cos φ = -dz
                    // so φ = atan2(-dx, -dz) = atan2(dx, dz) + 180°.
                    rt.yaw_deg =
                        brain.step.direction.x.atan2(brain.step.direction.z).to_degrees() + 180.0;
                }
            }

            // Pin Y to the floor plane.
            rt.pos.y = self.floor_y;
            // Bounds clamp (no-op for static NPCs since min == max).
            rt.pos.x = rt.pos.x.clamp(rt.spawn_min_xz.x, rt.spawn_max_xz.x);
            rt.pos.z = rt.pos.z.clamp(rt.spawn_min_xz.z, rt.spawn_max_xz.z);

            // Upload model matrix.
            let model = model_matrix(npc_scale, rt.yaw_deg, rt.pos);
            let color = self.scene.npcs[scene_index].color;
            let ambient = rt.instance.ambient_cube;
            rt.instance.set_model(queue, model, color, ambient);

            // Upload skin matrices using the per-step anim time so
            // one-shot falls start from frame 0 each time. Locomotion
            // clips (Walk) and the bind-pose Idle have their body
            // position driven externally by the brain, so we strip
            // root-joint translation to avoid double-counting. Falls
            // need the root translation to actually fall (the hip
            // dropping through the floor *is* the fall).
            let role = rt.brain.as_ref().map(|b| b.step.role);
            let strip_root = !matches!(role, Some(Role::Fall) | Some(Role::FallSide));
            let anim = rt
                .current_anim_index
                .and_then(|i| asset.animations.get(i));
            let skin_mats = asset.skeleton.evaluate(anim, rt.step_anim_t, strip_root);
            rt.instance.upload_skin(queue, &skin_mats)?;
        }

        // Pass 2: pairwise XZ separation. Read all positions, then push
        // each NPC away from neighbours in its radius. We *don't* re-clamp
        // to spawn box here — repulsion is usually small enough not to
        // matter, and double-clamping in one frame can fight the brain's
        // walk direction in a way that wastes physics.
        if self.runtimes.len() >= 2 {
            let positions: Vec<Vec3> = self.runtimes.iter().map(|rt| rt.pos).collect();
            let r2 = SEPARATION_RADIUS * SEPARATION_RADIUS;
            for i in 0..self.runtimes.len() {
                let mut push = Vec3::ZERO;
                for (j, &other) in positions.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    let d = positions[i] - other;
                    let dxz = Vec3::new(d.x, 0.0, d.z);
                    let dist_sq = dxz.length_squared();
                    if dist_sq < r2 && dist_sq > 1e-4 {
                        let dist = dist_sq.sqrt();
                        let scale_ = (1.0 - dist / SEPARATION_RADIUS) * SEPARATION_STRENGTH;
                        push += dxz / dist * scale_;
                    }
                }
                if push.length_squared() > 1e-6 {
                    let rt = &mut self.runtimes[i];
                    rt.pos += push * dt;
                    rt.pos.x = rt.pos.x.clamp(rt.spawn_min_xz.x, rt.spawn_max_xz.x);
                    rt.pos.z = rt.pos.z.clamp(rt.spawn_min_xz.z, rt.spawn_max_xz.z);

                    // Re-upload model matrix to reflect the shoved pos.
                    let scene_index = rt.scene_index;
                    let npc = &self.scene.npcs[scene_index];
                    let model = model_matrix(npc.scale, rt.yaw_deg, rt.pos);
                    let ambient = rt.instance.ambient_cube;
                    rt.instance.set_model(queue, model, npc.color, ambient);
                }
            }
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

/// Roll a fresh timeline step. Mirrors `Brain.next_step` in
/// `gsa/scripts/brain.gd` with our reduced role set.
///
/// `anim_table[Role]` provides the per-NPC animation index for each
/// role; `clip_dur` gives the duration of a clip by `Option<idx>`
/// (returns a sensible fallback when the glb doesn't ship that clip).
fn next_brain_step(
    brain: &mut Brain,
    anim_table: &[Option<usize>; 4],
    clip_dur: impl Fn(Option<usize>) -> f32,
) -> TimelineStep {
    let r = brain.rng.next_f32();
    if r < P_WALK {
        TimelineStep {
            role: Role::Walk,
            duration: brain.rng.range_f32(STEP_MIN, STEP_MAX),
            direction: random_direction(&mut brain.rng, brain.last_direction),
            anim_name: Some("Walk"),
        }
    } else if r < P_WALK + P_FALL {
        TimelineStep {
            role: Role::Fall,
            duration: clip_dur(anim_table[2]),
            direction: Vec3::ZERO,
            anim_name: Some("Fall"),
        }
    } else {
        TimelineStep {
            role: Role::FallSide,
            duration: clip_dur(anim_table[3]),
            direction: Vec3::ZERO,
            anim_name: Some("fall_side"),
        }
    }
}

/// Pick a movement direction. With probability `DIRECTION_DRIFT_PROB`,
/// drift the current direction by ±`DIRECTION_DRIFT_RAD`; otherwise
/// pick a fully random azimuth. Mirrors gsa's `random_direction`.
fn random_direction(rng: &mut Rng, current: Vec3) -> Vec3 {
    if current.length_squared() > 1e-4 && rng.next_f32() < DIRECTION_DRIFT_PROB {
        // atan2(z, x) — Bevy/gsa convention so headings encode XZ
        // azimuth with +X = 0°, +Z = 90°.
        let cur_angle = current.z.atan2(current.x);
        let new_angle = cur_angle + rng.range_f32(-DIRECTION_DRIFT_RAD, DIRECTION_DRIFT_RAD);
        Vec3::new(new_angle.cos(), 0.0, new_angle.sin())
    } else {
        let theta = rng.next_f32() * std::f32::consts::TAU;
        Vec3::new(theta.cos(), 0.0, theta.sin())
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
