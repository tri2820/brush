#![recursion_limit = "256"]
#![cfg(not(target_family = "wasm"))]

use brush_async::Actor;
use brush_process::DataSource;
use brush_process::RunningProcess;
use brush_process::config::TrainStreamConfig;
use brush_process::message::ProcessMessage;
use brush_process::message::TrainMessage;

use clap::{Args, Error, Parser, builder::ArgPredicate, error::ErrorKind};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc;
use tokio_stream::StreamExt;
use tracing::trace_span;

/// File extension used per output file in `RenderArgs::output_dir`.
const PNG_EXT: &str = "png";
const MP4_EXT: &str = "mp4";

#[derive(Parser)]
#[command(
    author,
    version,
    arg_required_else_help = false,
    about = "Brush - universal splats"
)]
pub struct Cli {
    /// Source to load from (path or URL).
    #[arg(value_name = "PATH_OR_URL")]
    pub source: Option<DataSource>,

    #[arg(
        long,
        default_value = "true",
        default_value_if("source", ArgPredicate::IsPresent, "false"),
        default_value_if("scene", ArgPredicate::IsPresent, "false"),
        help = "Spawn a viewer to visualize the training"
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,

    #[clap(flatten)]
    pub render: RenderArgs,
}

/// Arguments for headless rendering or recording. `--scene PATH` is the
/// single source of truth for camera setup; presence of `--record-frames`
/// switches from one-PNG-per-camera output to one-mp4-per-camera output.
#[derive(Args, Clone, Debug)]
pub struct RenderArgs {
    /// JSON config with one or more named cameras. Without
    /// `--record-frames`, writes one PNG per camera. With it, records
    /// one mp4 per camera, all frames synchronized to a single
    /// world-state advance per frame.
    #[arg(long, value_name = "PATH")]
    pub scene: Option<PathBuf>,

    /// Output directory. Per-camera files are written as
    /// `{output_dir}/{camera.name}.{png|mp4}`.
    #[arg(long, value_name = "DIR", default_value = "./out")]
    pub output_dir: PathBuf,

    /// Number of frames to record. Presence triggers record mode.
    #[arg(long)]
    pub record_frames: Option<u32>,

    /// Frames per second of the recorded video.
    #[arg(long, default_value_t = 30)]
    pub record_fps: u32,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if self.render.scene.is_some() && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --scene is set, --source must be provided",
            ));
        }
        if self.render.record_frames.is_some() && self.render.scene.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "--record-frames requires --scene to define which cameras to capture",
            ));
        }
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, --source must be provided",
            ));
        }
        Ok(self)
    }
}

/// JSON schema for `--scene PATH`. Defines a set of named cameras the
/// renderer should produce. Top-level fields are defaults; each camera
/// can override `resolution` and `fov_y_deg`.
#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SceneConfig {
    #[serde(default = "default_resolution")]
    pub resolution: [u32; 2],
    #[serde(default = "default_fov_y_deg")]
    pub fov_y_deg: f64,
    #[serde(default)]
    pub background: [f32; 3],
    pub cameras: Vec<CameraEntry>,
    /// NPCs placed in the scene. Each renders one glTF character on top
    /// of the splat output (with optional skeletal animation + scripted
    /// path in later phases).
    #[serde(default)]
    pub npcs: Vec<NpcEntry>,
}

#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct CameraEntry {
    /// Output PNG basename (without extension).
    pub name: String,
    /// Camera position in world space.
    pub pos: [f32; 3],
    /// Yaw, pitch, roll in degrees. Applied as glam's `EulerRot::YXZ` —
    /// matches the in-app HUD readout exactly.
    pub ypr_deg: [f32; 3],
    /// Override `SceneConfig.resolution` for this camera.
    #[serde(default)]
    pub resolution: Option<[u32; 2]>,
    /// Override `SceneConfig.fov_y_deg` for this camera.
    #[serde(default)]
    pub fov_y_deg: Option<f64>,
}

#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct NpcEntry {
    pub name: String,
    /// Path to the glTF .glb file.
    pub asset: PathBuf,
    /// Static world-space placement (overridden by `path` in later phases).
    #[serde(default)]
    pub pos: [f32; 3],
    /// Yaw degrees around +Y axis. Pitch/roll fixed to 0.
    #[serde(default)]
    pub yaw_deg: f32,
    /// Uniform scale applied to the mesh.
    #[serde(default = "default_npc_scale")]
    pub scale: f32,
    /// Base diffuse color (linear 0..1). Overrides whatever the glTF
    /// material would say; we don't sample materials yet.
    #[serde(default = "default_npc_color")]
    pub color: [f32; 3],
    /// Optional name of an animation in the glb to play.
    #[serde(default)]
    pub animation: Option<String>,
}

fn default_resolution() -> [u32; 2] {
    [1280, 720]
}
fn default_fov_y_deg() -> f64 {
    45.0
}
fn default_npc_scale() -> f32 {
    1.0
}
fn default_npc_color() -> [f32; 3] {
    [0.7, 0.55, 0.45]
}

/// Run the CLI: pin the trainer stream to a dedicated [`Actor`] thread,
/// drive the indicatif UI on the main task.
pub async fn run_cli_ui(
    mut process: RunningProcess,
    #[allow(unused)] train_stream_config: TrainStreamConfig,
) -> Result<(), anyhow::Error> {
    // Pump the trainer stream from a dedicated Actor thread; the
    // indicatif UI loop below consumes its output on the main task.
    let (tx, mut messages) = mpsc::unbounded_channel();
    let trainer = Actor::new("cli-trainer");
    trainer
        .run(move || async move {
            while let Some(msg) = process.stream.next().await {
                if tx.send(msg).is_err() {
                    break;
                }
            }
        })
        .detach();

    // Hold the actor for the lifetime of the UI loop; dropping it
    // would kill the pump.
    let _trainer = trainer;

    // Initialize the logger with indicatif integration to prevent
    // progress bars from clobbering log output.
    let sp = {
        let mut builder = env_logger::builder();
        builder.target(env_logger::Target::Stdout);
        let logger = builder.build();
        let level = logger.filter();
        let multi = MultiProgress::new();

        LogWrapper::new(multi.clone(), logger)
            .try_init()
            .expect("Failed to initialize logger");
        log::set_max_level(level);

        multi
    };

    let main_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indacitif config")
            .tick_strings(&[
                "🖌️      ",
                "█🖌️     ",
                "▓█🖌️    ",
                "░▓█🖌️   ",
                "•░▓█🖌️  ",
                "·•░▓█🖌️ ",
                " ·•░▓🖌️ ",
                "  ·•░🖌️ ",
                "   ·•🖌️ ",
                "    ·🖌️ ",
                "     🖌️ ",
                "    🖌️ █",
                "   🖌️ █▓",
                "  🖌️ █▓░",
                " 🖌️ █▓░•",
                "🖌️ █▓░•·",
                "🖌️ ▓░•· ",
                "🖌️ ░•·  ",
                "🖌️ •·   ",
                "🖌️ ·    ",
                "🖌️      ",
            ]),
    );

    let stats_spinner = ProgressBar::new_spinner().with_style(
        ProgressStyle::with_template("{spinner:.blue} {msg}")
            .expect("Invalid indicatif config")
            .tick_strings(&["ℹ️", "ℹ️"]),
    );

    let train_progress = {
        let tc = &train_stream_config.train_config;
        let bar = ProgressBar::new(tc.total_iters() as u64)
        .with_style(
            ProgressStyle::with_template(
                "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg} ({per_sec}, {eta} remaining)",
            )
            .expect("Invalid indicatif config").progress_chars("◍○○"),
        )
        .with_message("Steps");
        sp.add(bar)
    };

    let main_spinner = sp.add(main_spinner);
    main_spinner.enable_steady_tick(Duration::from_millis(120));

    let eval_spinner = sp.add(
        ProgressBar::new_spinner().with_style(
            ProgressStyle::with_template("{spinner:.blue} {msg}")
                .expect("Invalid indicatif config")
                .tick_strings(&["✅", "✅"]),
        ),
    );

    eval_spinner.set_message("waiting for dataset...");

    let stats_spinner = sp.add(stats_spinner);
    stats_spinner.set_message("Starting up");
    log::info!("Starting up");

    if cfg!(debug_assertions) {
        let _ =
            sp.println("ℹ️  running in debug mode, compile with --release for best performance");
    }

    #[allow(unused_mut)]
    let mut duration = Duration::from_secs(0);

    while let Some(msg) = messages.recv().await {
        let _span = trace_span!("CLI UI").entered();

        let msg = match msg {
            Ok(msg) => msg,
            Err(error) => {
                // Don't print the error here. It'll bubble up and be printed as output.
                let _ = sp.println("❌ Encountered an error");
                return Err(error);
            }
        };

        match msg {
            ProcessMessage::NewProcess => {
                main_spinner.set_message("Starting process...");
            }
            ProcessMessage::StartLoading { name, training, .. } => {
                if !training {
                    // Display a big warning saying viewing splats from the CLI doesn't make sense.
                    let _ = sp.println("❌ Only training is supported in the CLI (try passing --with-viewer to view a splat)");
                    break;
                }
                main_spinner.set_message(format!("Loading {name}..."));
            }
            ProcessMessage::SplatsUpdated { .. } => {}
            ProcessMessage::TrainMessage(train) => match train {
                TrainMessage::TrainConfig { .. } => {}
                TrainMessage::Dataset { dataset } => {
                    let train_views = dataset.train.views.len();
                    let eval_views = dataset.eval.as_ref().map_or(0, |v| v.views.len());
                    log::info!(
                        "Loaded dataset with {train_views} training, {eval_views} eval views",
                    );
                    main_spinner.set_message(format!(
                        "Loading dataset with {train_views} training, {eval_views} eval views",
                    ));
                    if eval_views > 0 {
                        eval_spinner.set_message(format!(
                            "evaluating {} views every {} steps",
                            eval_views, train_stream_config.process_config.eval_every,
                        ));
                    } else {
                        eval_spinner.finish_and_clear();
                    }
                }
                TrainMessage::TrainStep {
                    iter,
                    total_elapsed,
                    lod_progress,
                    ..
                } => {
                    if let Some((lod, total_lods)) = lod_progress {
                        main_spinner.set_message(format!("LOD {lod}/{total_lods}"));
                    } else {
                        main_spinner.set_message("Training");
                    }
                    train_progress.set_position(iter as u64);
                    duration = total_elapsed;
                }
                TrainMessage::RefineStep {
                    cur_splat_count,
                    iter,
                    ..
                } => {
                    stats_spinner.set_message(format!("Current splat count {cur_splat_count}"));
                    log::info!("Refine iter {iter}, {cur_splat_count} splats.");
                }
                TrainMessage::EvalResult {
                    iter,
                    avg_psnr,
                    avg_ssim,
                } => {
                    log::info!("Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}");

                    eval_spinner.set_message(format!(
                        "Eval iter {iter}: PSNR {avg_psnr}, ssim {avg_ssim}"
                    ));
                }
                TrainMessage::DoneTraining => {}
            },
            ProcessMessage::DoneLoading => {
                log::info!("Completed loading.");
                main_spinner.set_message("Completed loading");
                stats_spinner.set_message("Completed loading");
            }
            ProcessMessage::Warning { error } => {
                log::warn!("{error}");
                sp.println(format!("⚠️: {error}"))?;
            }
            #[allow(unreachable_patterns)]
            _ => {}
        }
    }

    let duration_secs = Duration::from_secs(duration.as_secs());
    let _ = sp.println(format!(
        "Training took {}",
        humantime::format_duration(duration_secs)
    ));

    log::info!(
        "Done training! Took {:?}.",
        humantime::format_duration(duration_secs)
    );

    Ok(())
}

/// Drive a process to `DoneLoading`, then either render a single frame
/// (`--render-output`) or one frame per camera in `--scene` config.
/// Headless multi-camera PNG render. One file per camera in
/// `scene.json` written to `{output_dir}/{camera.name}.png`. All cameras
/// observe the same world state (a static splat scene for now).
pub async fn run_render(
    process: RunningProcess,
    args: RenderArgs,
) -> Result<(), anyhow::Error> {
    let (splats, scene) = load_splats_and_scene(process, &args).await?;
    tokio::fs::create_dir_all(&args.output_dir).await?;
    let background = glam::Vec3::from(scene.background);

    for entry in &scene.cameras {
        let (img_size, fov_y_deg) = resolve_size_fov(&scene, entry);
        let camera = camera_from_ypr(entry.pos.into(), entry.ypr_deg.into(), fov_y_deg, img_size);
        let output = args.output_dir.join(format!("{}.{PNG_EXT}", entry.name));
        log::info!(
            "Rendering camera '{}' at {}x{} → {}",
            entry.name,
            img_size.x,
            img_size.y,
            output.display(),
        );
        render_and_save(splats.clone(), &camera, img_size, background, &output).await?;
    }
    Ok(())
}

/// Parallel multi-camera recording. One outer per-frame loop advances
/// world state once; the inner loop renders every camera against that
/// single bound instant and feeds each camera's own Recorder. Each
/// camera writes to `{output_dir}/{camera.name}.mp4`. macOS-only.
#[cfg(target_os = "macos")]
pub async fn run_record(
    process: RunningProcess,
    args: RenderArgs,
    device: wgpu::Device,
    queue: wgpu::Queue,
) -> Result<(), anyhow::Error> {
    use brush_character::{GpuMesh, MeshRenderer, NpcInstance, load_mesh};
    use brush_record::{Codec, Recorder, RecorderConfig};
    use brush_render::burn_glue::resolve_to_cube_float;
    use brush_render::{TextureMode, gaussian_splats::render_splats};

    let total = args
        .record_frames
        .ok_or_else(|| anyhow::anyhow!("run_record called without --record-frames"))?;
    let total = total.max(1);

    let (splats, scene) = load_splats_and_scene(process, &args).await?;
    tokio::fs::create_dir_all(&args.output_dir).await?;
    let background = glam::Vec3::from(scene.background);

    // Load each unique NPC glb just once. Multiple NPCs may share the
    // same `asset` path; we keep a HashMap so we only upload one GpuMesh
    // and one MeshAsset per file.
    let mut mesh_cache: std::collections::HashMap<PathBuf, (brush_character::MeshAsset, GpuMesh)> =
        std::collections::HashMap::new();
    for npc in &scene.npcs {
        if !mesh_cache.contains_key(&npc.asset) {
            log::info!("Loading character asset {}", npc.asset.display());
            let asset = load_mesh(&npc.asset)?;
            let gpu = GpuMesh::upload(&device, &asset);
            mesh_cache.insert(npc.asset.clone(), (asset, gpu));
        }
    }

    // Shared mesh renderer state. One per cli invocation; bind groups
    // and depth target are owned by MeshRenderer.
    let mut mesh_renderer = MeshRenderer::new(&device, wgpu::TextureFormat::Bgra8Unorm);

    // Per-NPC GPU instance: model matrix uniform + skin matrices
    // storage. T-pose for now; phase 3 swaps the skin matrices each
    // frame from the animation evaluator.
    // Per NPC: GPU instance + index of the active animation in
    // `mesh_cache[asset_path].0.animations` (None = static bind pose).
    struct NpcRuntime {
        scene_index: usize,
        instance: NpcInstance,
        animation_index: Option<usize>,
    }
    let mut npc_runtime: Vec<NpcRuntime> = Vec::with_capacity(scene.npcs.len());
    for (i, npc) in scene.npcs.iter().enumerate() {
        let (asset, gpu) = mesh_cache.get(&npc.asset).expect("just inserted");
        // The supersplat warehouse scene uses Y-DOWN world coords (see
        // earlier camera calibration: yaw/pitch derived assuming +Y is
        // down). The Meshy/Mixamo character is Y-UP. Flip it via a
        // 180° rotation around X so its head points to world -Y.
        // yaw_deg rotates around the world's "up" (= -Y in this scene),
        // which after the X-flip is the character's own +Y axis;
        // applying it on the OUTSIDE keeps yaw_deg behaving like a
        // normal heading.
        let rot = glam::Quat::from_rotation_y(npc.yaw_deg.to_radians())
            * glam::Quat::from_rotation_x(std::f32::consts::PI);
        let model = glam::Mat4::from_scale_rotation_translation(
            glam::Vec3::splat(npc.scale),
            rot,
            glam::Vec3::from(npc.pos),
        );
        let instance = mesh_renderer.make_instance(&device, gpu, model, npc.color);
        let animation_index = match &npc.animation {
            Some(name) => {
                let idx = asset.animations.iter().position(|a| &a.name == name);
                if idx.is_none() {
                    let available: Vec<_> = asset.animations.iter().map(|a| &a.name).collect();
                    anyhow::bail!(
                        "npc '{}': animation '{}' not found in {} (have: {:?})",
                        npc.name, name, npc.asset.display(), available,
                    );
                }
                idx
            }
            None => None,
        };
        npc_runtime.push(NpcRuntime {
            scene_index: i,
            instance,
            animation_index,
        });
    }

    struct Cam {
        entry: CameraEntry,
        img_size: glam::UVec2,
        fov_y_deg: f64,
        recorder: Recorder,
    }
    let mut cams: Vec<Cam> = Vec::with_capacity(scene.cameras.len());
    for entry in &scene.cameras {
        let (img_size, fov_y_deg) = resolve_size_fov(&scene, entry);
        let output = args.output_dir.join(format!("{}.{MP4_EXT}", entry.name));
        log::info!(
            "Opening recorder for camera '{}' at {}x{} → {}",
            entry.name,
            img_size.x,
            img_size.y,
            output.display(),
        );
        let recorder = Recorder::new(
            device.clone(),
            queue.clone(),
            &output,
            RecorderConfig {
                width: img_size.x,
                height: img_size.y,
                fps: args.record_fps,
                codec: Codec::Hevc,
            },
        )?;
        cams.push(Cam {
            entry: entry.clone(),
            img_size,
            fov_y_deg,
            recorder,
        });
    }

    log::info!(
        "Recording {} frames at {} fps across {} cameras + {} NPCs...",
        total,
        args.record_fps,
        cams.len(),
        scene.npcs.len(),
    );

    let t_start = std::time::Instant::now();
    for frame in 0..total {
        // World state for this frame. NPCs see the same `t` across
        // every camera, so all cameras observe identical poses.
        let world_t = frame as f32 / args.record_fps as f32;
        for rt in &npc_runtime {
            let (asset, _gpu) = mesh_cache
                .get(&scene.npcs[rt.scene_index].asset)
                .expect("preloaded");
            let anim = rt.animation_index.map(|i| &asset.animations[i]);
            let skin_mats = asset.skeleton.evaluate(anim, world_t);
            rt.instance.upload_skin(&queue, &skin_mats)?;
        }

        for cam in &mut cams {
            let camera = camera_from_ypr(
                cam.entry.pos.into(),
                cam.entry.ypr_deg.into(),
                cam.fov_y_deg,
                cam.img_size,
            );

            // (1) Brush splat raster → wgpu::Buffer (packed RGBA8 u32)
            let (tensor, _aux) = render_splats(
                splats.clone(),
                &camera,
                cam.img_size,
                background,
                None,
                TextureMode::Packed,
            )
            .await;
            let cube = resolve_to_cube_float(tensor);
            let resource = cube
                .client
                .get_resource(cube.handle.clone())
                .map_err(|e| anyhow::anyhow!("get_resource failed: {e:?}"))?;
            // Fence brush's submission before the swizzle reads.
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let res = resource.resource();

            // (2) Open the frame, swizzle splats into the IOSurface as
            // the backdrop, then run the mesh pass on top.
            let mut frame = cam.recorder.begin_frame()?;
            frame.swizzle_from(&res.buffer, res.offset);

            if !npc_runtime.is_empty() {
                // Recompute view-projection in the convention the mesh
                // shader uses. brush's renderer uses +Z forward; for
                // wgpu clip space we go through `Mat4::perspective_rh`
                // with the same fov_y and aspect.
                let aspect = cam.img_size.x as f32 / cam.img_size.y as f32;
                let proj = glam::Mat4::perspective_rh(
                    cam.fov_y_deg.to_radians() as f32,
                    aspect,
                    0.05,
                    1000.0,
                );
                let view = glam::Mat4::from(camera.world_to_local());
                let view_brush_to_wgpu =
                    glam::Mat4::from_quat(glam::Quat::from_rotation_x(std::f32::consts::PI));
                let view_proj = proj * view_brush_to_wgpu * view;
                let cam_pos = camera.position;

                mesh_renderer.set_camera(&queue, view_proj, cam_pos);
                let draws: Vec<_> = npc_runtime
                    .iter()
                    .map(|rt| {
                        let asset_path = &scene.npcs[rt.scene_index].asset;
                        (
                            &mesh_cache.get(asset_path).expect("preloaded").1,
                            &rt.instance,
                        )
                    })
                    .collect();
                let mesh_submission =
                    mesh_renderer.render(&device, &queue, frame.color_texture(), draws);
                frame.note_submission(mesh_submission);
            }

            frame.finish()?;
        }
    }

    for cam in cams {
        cam.recorder.finish().await?;
    }

    let elapsed = t_start.elapsed();
    let total_frames = total as f64 * scene.cameras.len() as f64;
    log::info!(
        "Wrote {} frames across {} cameras in {:.2?} ({:.1} frames/s aggregate)",
        total,
        scene.cameras.len(),
        elapsed,
        total_frames / elapsed.as_secs_f64(),
    );
    Ok(())
}

/// Drive `process.stream` to `DoneLoading`, parse the scene config,
/// validate that we have at least one camera. Shared by render + record.
async fn load_splats_and_scene(
    mut process: RunningProcess,
    args: &RenderArgs,
) -> Result<(brush_render::gaussian_splats::Splats, SceneConfig), anyhow::Error> {
    let scene_path = args
        .scene
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("--scene is required"))?;
    log::info!("Loading source...");
    while let Some(msg) = process.stream.next().await {
        match msg? {
            ProcessMessage::DoneLoading => break,
            ProcessMessage::StartLoading { training, .. } if training => {
                anyhow::bail!(
                    "render/record mode expects a single .ply / .compressed.ply source, not a training dataset"
                );
            }
            ProcessMessage::Warning { error } => log::warn!("{error}"),
            _ => {}
        }
    }
    let splats = process
        .splat_view
        .latest()
        .ok_or_else(|| anyhow::anyhow!("no splats were loaded from source"))?;
    log::info!(
        "Loaded {} splats (sh degree {}).",
        splats.num_splats(),
        splats.sh_degree(),
    );

    let raw = tokio::fs::read_to_string(scene_path)
        .await
        .map_err(|e| anyhow::anyhow!("failed to read {}: {e}", scene_path.display()))?;
    let scene: SceneConfig = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("invalid scene config {}: {e}", scene_path.display()))?;
    if scene.cameras.is_empty() {
        anyhow::bail!("scene config has no cameras");
    }
    Ok((splats, scene))
}

fn resolve_size_fov(scene: &SceneConfig, entry: &CameraEntry) -> (glam::UVec2, f64) {
    let img_size = entry
        .resolution
        .map(|[w, h]| glam::uvec2(w, h))
        .unwrap_or_else(|| glam::uvec2(scene.resolution[0], scene.resolution[1]));
    let fov_y_deg = entry.fov_y_deg.unwrap_or(scene.fov_y_deg);
    (img_size, fov_y_deg)
}

/// Render `splats` with `camera` at `img_size` against `background` and
/// write the result as a PNG to `output`. Parent dirs are created as
/// needed.
async fn render_and_save(
    splats: brush_render::gaussian_splats::Splats,
    camera: &brush_render::camera::Camera,
    img_size: glam::UVec2,
    background: glam::Vec3,
    output: &std::path::Path,
) -> Result<(), anyhow::Error> {
    use brush_render::{TextureMode, gaussian_splats::render_splats};
    use burn::tensor::s;
    use image::Rgb32FImage;

    let (image, _aux) = render_splats(splats, camera, img_size, background, None, TextureMode::Float).await;

    // Float-mode output is [h, w, 4] (RGBA); drop alpha to match
    // Rgb32FImage's 3-channel expectation. EvalSample::save_to_disk
    // pre-slices, so a direct port without the slice tiles the image.
    let image = image.slice(s![.., .., 0..3]);
    let [h, w, _] = [image.dims()[0], image.dims()[1], image.dims()[2]];
    let data = image
        .into_data_async()
        .await?
        .into_vec::<f32>()
        .map_err(|e| anyhow::anyhow!("failed to decode render tensor: {e:?}"))?;
    let img: image::DynamicImage = Rgb32FImage::from_raw(w as u32, h as u32, data)
        .ok_or_else(|| anyhow::anyhow!("render tensor dims don't match image dims"))?
        .into();
    let img: image::DynamicImage = img.into_rgb8().into();
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        tokio::fs::create_dir_all(parent).await?;
    }
    img.save(output)?;
    log::info!("Saved render to {}", output.display());
    Ok(())
}

fn camera_from_ypr(
    pos: glam::Vec3,
    ypr_deg: glam::Vec3,
    fov_y_deg: f64,
    img_size: glam::UVec2,
) -> brush_render::camera::Camera {
    use brush_render::camera::Camera;
    use brush_render::kernels::camera_model::CameraModel;
    use glam::{EulerRot, Quat};

    let rotation = Quat::from_euler(
        EulerRot::YXZ,
        ypr_deg.x.to_radians(),
        ypr_deg.y.to_radians(),
        ypr_deg.z.to_radians(),
    );
    let aspect = img_size.x as f64 / img_size.y as f64;
    let fov_y = fov_y_deg.to_radians();
    let fov_x = 2.0 * ((fov_y / 2.0).tan() * aspect).atan();
    Camera::new(
        pos,
        rotation,
        fov_x,
        fov_y,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}

