#![recursion_limit = "256"]
#![cfg(not(target_family = "wasm"))]

#[cfg(target_os = "macos")]
pub mod npc_system;

#[cfg(target_os = "macos")]
pub mod voxel_overlay_record;

use brush_async::Actor;
use brush_process::DataSource;
use brush_process::RunningProcess;
use brush_process::config::TrainStreamConfig;
use brush_process::message::ProcessMessage;
use brush_process::message::TrainMessage;

use clap::{Args, Error, Parser, builder::ArgPredicate, error::ErrorKind};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use indicatif_log_bridge::LogWrapper;
use std::path::{Path, PathBuf};
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
        help = "Spawn a viewer to visualize the training"
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,

    #[clap(flatten)]
    pub render: RenderArgs,

    #[clap(flatten)]
    pub collect: CollectArgs,
}

/// Arguments for headless rendering or recording. The scene description
/// lives in the positional argument — pass a `.json` to enable cameras
/// and NPCs, or a bare splat to render just the geometry. Presence of
/// `--record-frames` switches from one-PNG-per-camera output to one-mp4-
/// per-camera output.
#[derive(Args, Clone, Debug)]
pub struct RenderArgs {
    /// Path to the scene JSON. Populated by bin.rs when the positional
    /// argument is a `.json` file — not a user-facing flag (the
    /// positional is the single source of truth for scene config).
    #[arg(skip)]
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

    /// Render one PNG per camera in `scene.json` to `output_dir` and exit.
    /// Without this flag, `brush <scene.json>` opens the interactive viewer.
    #[arg(long)]
    pub screenshot: bool,

    /// Render the voxel collision overlay on top of recorded/screenshot
    /// frames. Off by default so snapshots show the splat unobstructed;
    /// pass this to debug NPC-vs-collision alignment. The interactive
    /// viewer's V toggle is independent of this flag.
    #[arg(long)]
    pub voxel_overlay: bool,
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if !self.with_viewer && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --with-viewer is false, a source path must be provided",
            ));
        }
        // `--record-frames` needs cameras, which only come from a scene.json.
        // We can't tell here whether the positional is `.json` or a bare
        // splat (it's parsed as a generic DataSource), so the actual
        // enforcement happens in bin.rs after the scene preflight.
        Ok(self)
    }
}

/// JSON schema for a scene file. Describes everything needed to load a
/// world: the splat to render, cameras to capture from, NPCs to animate
/// over the top, and optional collision geometry. Asset paths (splat,
/// collision, npc.asset) are resolved relative to the scene file's own
/// directory by [`load_scene_config`], so a scene is self-contained and
/// portable across machines.
#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct SceneConfig {
    /// Path to the splat (`.ply` / `.compressed.ply` / etc.) — relative
    /// paths resolve against the scene file's directory.
    pub splat: PathBuf,
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
    /// World Y at which NPC feet land (the floor plane). Defaults to 0
    /// — same plane the grid widget draws at, so users can verify the
    /// alignment visually by toggling the grid on. Adjust if a scene's
    /// visible floor sits at a different Y.
    #[serde(default)]
    pub floor_y: f32,
    /// Optional voxel-octree collision asset (.voxel.json). Currently
    /// only consumed by the V-key overlay loader for visualization;
    /// physics uses the flat `floor_y` plane.
    #[serde(default)]
    pub collision: Option<PathBuf>,
    /// Optional `.collision.glb` triangle mesh of the voxel collision
    /// surface (emitted by `splat-transform … -K faces`). When set,
    /// the viewer can toggle it on with the `V` key to overlay where
    /// the collider thinks the floor / walls are. Pure debug — no
    /// runtime cost when toggled off.
    #[serde(default)]
    pub voxel_mesh_overlay: Option<PathBuf>,
}

/// Read a scene JSON file and rewrite every relative path inside it to
/// an absolute path resolved against the scene file's own directory.
/// This is what makes a scene self-contained: you can put `scene.json`
/// anywhere and its asset references stay correct.
pub fn load_scene_config(scene_path: &Path) -> anyhow::Result<SceneConfig> {
    let raw = std::fs::read_to_string(scene_path)
        .map_err(|e| anyhow::anyhow!("read {}: {e}", scene_path.display()))?;
    let mut scene: SceneConfig = serde_json::from_str(&raw)
        .map_err(|e| anyhow::anyhow!("invalid scene config {}: {e}", scene_path.display()))?;

    // Asset paths in the JSON are author-friendly (relative to the scene
    // file). Promote them to absolute up-front so every downstream
    // consumer can just read the path without caring about CWD.
    let base = scene_path
        .parent()
        .ok_or_else(|| anyhow::anyhow!("scene path has no parent: {}", scene_path.display()))?;
    let resolve = |p: &PathBuf| -> PathBuf {
        if p.is_absolute() {
            p.clone()
        } else {
            base.join(p)
        }
    };
    scene.splat = resolve(&scene.splat);
    if let Some(c) = scene.collision.as_ref() {
        scene.collision = Some(resolve(c));
    }
    if let Some(c) = scene.voxel_mesh_overlay.as_ref() {
        scene.voxel_mesh_overlay = Some(resolve(c));
    }
    for npc in &mut scene.npcs {
        npc.asset = resolve(&npc.asset);
    }
    Ok(scene)
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
    /// Static world-space placement. Overridden each frame by `path`
    /// if one is set.
    #[serde(default)]
    pub pos: [f32; 3],
    /// Yaw degrees around +Y axis. Pitch/roll fixed to 0. Overridden
    /// each frame by `path.heading_deg(t)` if a path is set.
    #[serde(default)]
    pub yaw_deg: f32,
    /// Uniform scale applied to the mesh.
    #[serde(default = "default_npc_scale")]
    pub scale: f32,
    /// Base diffuse color (linear 0..1). Overrides whatever the glTF
    /// material would say; we don't sample materials yet.
    #[serde(default = "default_npc_color")]
    pub color: [f32; 3],
    /// Optional name of the WALK animation in the glb. Played while
    /// the brain is in a locomotion-walk step; the IDLE pose is the
    /// skeleton's bind pose (no animation set).
    #[serde(default)]
    pub animation: Option<String>,
    /// Optional brain config. When set, the NPC wanders inside its
    /// spawn box driven by a PRNG timeline of idle/walk steps (ported
    /// from `gsa/scripts/brain.gd`). When absent, the NPC stays
    /// stationary at `pos`.
    #[serde(default)]
    pub brain: Option<BrainConfig>,
}

/// Brain config: a deterministic seed plus the AABB the NPC is allowed
/// to wander inside (so two NPCs in the same warehouse don't collide
/// with the wall splats).
#[derive(serde::Deserialize, Debug, Clone)]
#[serde(deny_unknown_fields)]
pub struct BrainConfig {
    /// PRNG seed; same seed → same wander timeline across runs.
    pub seed: u64,
    /// Min XZ corner of the allowed wander region.
    pub spawn_min: [f32; 2],
    /// Max XZ corner of the allowed wander region.
    pub spawn_max: [f32; 2],
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
    // Same logger bootstrap as run_record so `just screenshot` is not silent.
    let mut builder = env_logger::Builder::from_default_env();
    if std::env::var("RUST_LOG").is_err() {
        builder.filter_level(log::LevelFilter::Info);
    }
    let _ = builder.target(env_logger::Target::Stdout).try_init();

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
/// Cached per-splat data used to build per-NPC ambient probes: world
/// position and the direction-average RGB color of each splat. The
/// color comes from the SH degree-0 coefficient (which IS the
/// direction-independent component of the radiance distribution) via
/// the standard gsplat convention `rgb = 0.5 + SH_C0 * f_dc`.
struct SplatProbeData {
    positions: Vec<glam::Vec3>,
    colors: Vec<glam::Vec3>,
}

/// SH degree-0 basis coefficient. The gaussian-splatting community
/// stores `f_dc_0..2` as raw SH coefficients; the rendered base color
/// is `0.5 + SH_C0 * f_dc`. Pre-applied to convert DC tensor values
/// directly to "albedo at average direction" colors.
const SH_C0: f32 = 0.282_094_79;

/// Sample one NPC's 6-tap directional ambient probe (±X/±Y/±Z) by
/// averaging nearby splat colors weighted by `max(dot(splat_dir, axis),
/// 0)`. Probe radius and the minimum weight per axis are tuned for
/// "warehouse-scale interior" — at scale, splats far past the radius
/// contribute negligibly and ignoring them keeps the per-NPC sampling
/// O(splat_count) but with a fast distance reject.
fn compute_ambient_cube(npc_pos: glam::Vec3, probe: &SplatProbeData) -> [[f32; 3]; 6] {
    const PROBE_RADIUS: f32 = 5.0;
    const RADIUS_SQ: f32 = PROBE_RADIUS * PROBE_RADIUS;
    // Use the world's actual up axis (Y-down in the supersplat scene)
    // as the probe basis. The axes below are in the same world frame
    // we render in; the per-NPC X-flip rotation we apply elsewhere
    // doesn't matter here because we look up the cube by world normal
    // in the fragment shader.
    let axes = [
        glam::Vec3::X,
        glam::Vec3::NEG_X,
        glam::Vec3::Y,
        glam::Vec3::NEG_Y,
        glam::Vec3::Z,
        glam::Vec3::NEG_Z,
    ];
    let mut sums = [glam::Vec3::ZERO; 6];
    let mut weights = [0.0_f32; 6];
    for (&pos, &color) in probe.positions.iter().zip(probe.colors.iter()) {
        let delta = pos - npc_pos;
        let dist_sq = delta.length_squared();
        if dist_sq > RADIUS_SQ {
            continue;
        }
        let dir = delta.normalize_or_zero();
        // Falloff with 1/(1 + dist^2) so nearby splats dominate.
        let falloff = 1.0 / (1.0 + dist_sq);
        for i in 0..6 {
            let w = dir.dot(axes[i]).max(0.0) * falloff;
            if w > 0.0 {
                sums[i] += color * w;
                weights[i] += w;
            }
        }
    }
    let mut out = [[0.0_f32; 3]; 6];
    for i in 0..6 {
        if weights[i] > 0.0 {
            let c = sums[i] / weights[i];
            out[i] = c.into();
        } else {
            // No splats biased toward this axis — neutral mid-grey
            // keeps the character readable instead of pitch-black.
            out[i] = [0.5, 0.5, 0.5];
        }
    }
    out
}

pub async fn run_record(
    process: RunningProcess,
    args: RenderArgs,
    device: wgpu::Device,
    queue: wgpu::Queue,
) -> Result<(), anyhow::Error> {
    use crate::npc_system::NpcSystem;
    use crate::voxel_overlay_record::VoxelOverlay;
    use brush_record::{Codec, Recorder, RecorderConfig};
    use brush_render::burn_glue::resolve_to_cube_float;
    use brush_render::{TextureMode, gaussian_splats::render_splats};
    use burn::tensor::s;

    // run_record is invoked directly from bin.rs without going through the
    // indicatif-flavored logger setup in run_cli_ui, so initialize a plain
    // env_logger here. Default to Info level when RUST_LOG isn't set so
    // `just record` actually shows progress; otherwise the user sees only
    // a row of macOS finishWriting warnings and wonders if anything ran.
    let mut builder = env_logger::Builder::from_default_env();
    if std::env::var("RUST_LOG").is_err() {
        builder.filter_level(log::LevelFilter::Info);
    }
    let _ = builder.target(env_logger::Target::Stdout).try_init();

    let total = args
        .record_frames
        .ok_or_else(|| anyhow::anyhow!("run_record called without --record-frames"))?;
    let total = total.max(1);

    let (splats, scene) = load_splats_and_scene(process, &args).await?;
    tokio::fs::create_dir_all(&args.output_dir).await?;
    let background = glam::Vec3::from(scene.background);

    // Pull splat positions + degree-0 SH coefficients off the GPU so
    // we can build per-NPC ambient probes on the CPU. One-shot at
    // setup — the splats themselves are static through the recording.
    let probe = if !scene.npcs.is_empty() {
        let means_data = splats.means().into_data_async().await?;
        let means_vec = means_data
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("means readback: {e:?}"))?;
        let sh = splats.sh_coeffs.val().slice(s![.., 0..1, ..]);
        let sh_data = sh.into_data_async().await?;
        let sh_vec = sh_data
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("sh readback: {e:?}"))?;
        let n = means_vec.len() / 3;
        let mut positions = Vec::with_capacity(n);
        let mut colors = Vec::with_capacity(n);
        for i in 0..n {
            positions.push(glam::vec3(
                means_vec[i * 3],
                means_vec[i * 3 + 1],
                means_vec[i * 3 + 2],
            ));
            // gsplat convention: rgb = 0.5 + SH_C0 * f_dc. Clamp into
            // [0,1] — degenerate splats can have huge SH values.
            let c = glam::vec3(
                (0.5 + SH_C0 * sh_vec[i * 3]).clamp(0.0, 1.0),
                (0.5 + SH_C0 * sh_vec[i * 3 + 1]).clamp(0.0, 1.0),
                (0.5 + SH_C0 * sh_vec[i * 3 + 2]).clamp(0.0, 1.0),
            );
            colors.push(c);
        }
        log::info!("Built splat probe with {} colored points", n);
        Some(SplatProbeData { positions, colors })
    } else {
        None
    };

    // NPC subsystem — owns the mesh renderer, asset cache, per-NPC
    // runtimes, voxel collider, and the physics step. The recorder
    // supplies the ambient-cube sampler (from the splat probe) and the
    // BGRA8Unorm color format matching the IOSurface target.
    let scene = std::sync::Arc::new(scene);
    let mut npc_system = NpcSystem::new(
        &device,
        &queue,
        wgpu::TextureFormat::Bgra8Unorm,
        scene.clone(),
        |anchor| match &probe {
            Some(p) => compute_ambient_cube(anchor, p),
            None => [[0.5, 0.5, 0.5]; 6],
        },
    )?;

    // Voxel collision overlay — same data the viewer's V toggle uses.
    // Renders on top of the mesh pass per frame so `just snapshot`
    // produces frames that match the viewer-with-V-on view.
    let voxel_overlay = match scene.voxel_mesh_overlay.as_ref() {
        Some(path) if args.voxel_overlay => Some(VoxelOverlay::new(
            &device,
            wgpu::TextureFormat::Bgra8Unorm,
            path,
        )?),
        _ => None,
    };

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

    let dt = 1.0 / args.record_fps as f32;
    let t_start = std::time::Instant::now();
    for _frame in 0..total {
        // World state for this frame. NPCs see the same `t` across every
        // camera, so all cameras observe identical poses.
        npc_system.tick(dt, &queue)?;

        for cam in &mut cams {
            let camera = camera_from_ypr(
                cam.entry.pos.into(),
                cam.entry.ypr_deg.into(),
                cam.fov_y_deg,
                cam.img_size,
            );

            // (1) Brush splat raster → wgpu::Buffer (packed RGBA8 u32)
            // plus a per-pixel view-space depth tensor used to
            // depth-test the NPC mesh against splat geometry.
            let (tensor, aux) = render_splats(
                splats.clone(),
                &camera,
                cam.img_size,
                background,
                None,
                TextureMode::Packed,
            )
            .await;
            // Color buffer.
            let cube_color = resolve_to_cube_float(tensor);
            let resource_color = cube_color
                .client
                .get_resource(cube_color.handle.clone())
                .map_err(|e| anyhow::anyhow!("get_resource (color) failed: {e:?}"))?;
            // Depth buffer.
            let cube_depth = resolve_to_cube_float(aux.depth_img);
            let resource_depth = cube_depth
                .client
                .get_resource(cube_depth.handle.clone())
                .map_err(|e| anyhow::anyhow!("get_resource (depth) failed: {e:?}"))?;
            // Fence brush's submission before either is read by our
            // downstream passes.
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let res = resource_color.resource();
            let depth_res = resource_depth.resource();

            // (2) Open the frame, swizzle splats into the IOSurface as
            // the backdrop, then run the mesh pass on top.
            let mut frame = cam.recorder.begin_frame()?;
            frame.swizzle_from(&res.buffer, res.offset);

            if !npc_system.runtimes.is_empty() {
                let aspect = cam.img_size.x as f32 / cam.img_size.y as f32;
                let view_proj = crate::npc_system::view_projection(
                    camera.world_to_local(),
                    cam.fov_y_deg.to_radians() as f32,
                    aspect,
                );

                npc_system
                    .mesh_renderer
                    .set_camera(&queue, view_proj, camera.position);
                // Splat depth → NDC depth into the mesh pass's depth
                // attachment. Same `near`/`far` the view_projection
                // helper uses; mismatch would shift hardware depth
                // tests off the actual splat surface.
                npc_system.mesh_renderer.fill_depth_from_splats(
                    &device,
                    &queue,
                    frame.color_texture(),
                    &depth_res.buffer,
                    depth_res.offset,
                    0.05,
                    1000.0,
                );
                let mesh_submission =
                    npc_system.render_npcs(&device, &queue, frame.color_texture(), None);
                frame.note_submission(mesh_submission);
            }

            // Voxel collision overlay (matches the viewer's V toggle).
            // Drawn after the mesh pass so it sits visibly on top —
            // the whole point is to compare against the splat. Uses the
            // splat-aligned convention (`perspective_lh * y_flip *
            // view`), NOT the NPC mesh's `perspective_rh * X-rotation *
            // view`. The NPC mesh counter-rotates 180° X in its model
            // matrix; the GLB has no such counter-rotation, so the NPC
            // view-projection would flip it relative to the splat.
            if let Some(ov) = voxel_overlay.as_ref() {
                let aspect = cam.img_size.x as f32 / cam.img_size.y as f32;
                let proj = glam::Mat4::perspective_lh(
                    cam.fov_y_deg.to_radians() as f32,
                    aspect,
                    0.05,
                    1000.0,
                );
                let y_flip = glam::Mat4::from_scale(glam::Vec3::new(1.0, -1.0, 1.0));
                let view = glam::Mat4::from(camera.world_to_local());
                let view_proj = proj * y_flip * view;
                let sub = ov.render(
                    &device,
                    &queue,
                    frame.color_texture(),
                    view_proj,
                    [1.0, 0.9, 0.0, 0.35],
                );
                frame.note_submission(sub);
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

/// Arguments for `run_collect` — the synthetic dataset collection mode.
/// Shares `output_dir`, `record_fps`, and `scene` from `RenderArgs`;
/// only the fields that are unique to collection live here.
#[derive(Args, Clone, Debug)]
pub struct CollectArgs {
    /// Enable dataset collection mode. When set, `brush <scene.json>
    /// --collect` renders the full clip pipeline instead of opening the
    /// viewer.
    #[arg(long)]
    pub collect: bool,

    /// Total number of clips to produce.
    #[arg(long, default_value_t = 100)]
    pub total_clips: u32,

    /// How many of the total clips will contain a fall segment.
    #[arg(long, default_value_t = 30)]
    pub fall_clips: u32,

    /// Duration of each clip in seconds.
    #[arg(long, default_value_t = 10.0)]
    pub clip_duration_secs: f32,

    /// Camera names to draw from (comma-separated). Each clip is recorded
    /// from exactly one camera chosen by a frustum-visibility test at
    /// fall time (or randomly for walk-only clips).
    #[arg(long, value_delimiter = ',', default_values_t = vec![
        String::from("cam_warehouse_floor"),
        String::from("cam_warehouse_wide"),
    ])]
    pub collect_cameras: Vec<String>,

    /// Global PRNG seed for clip-type shuffle and per-clip RNG seeds.
    #[arg(long, default_value_t = 42)]
    pub collect_seed: u64,
}

/// Synthetic dataset collection. Renders `total_clips` MP4 clips (each
/// `clip_duration_secs` long), ensuring exactly `fall_clips` of them
/// contain a guaranteed fall animation segment. A `metadata.json` sidecar
/// accompanies each clip with the camera used and timestamped fall events.
///
/// Each clip is recorded from exactly ONE camera chosen by a frustum
/// visibility test: for fall clips the chosen camera must be able to see
/// the falling NPC; for walk clips a camera is picked at random.
///
/// Shared args (`output_dir`, `record_fps`, `scene`) come from `render`.
/// macOS-only (uses the same VideoToolbox recorder as `run_record`).
#[cfg(target_os = "macos")]
pub async fn run_collect(
    process: RunningProcess,
    render: RenderArgs,
    args: CollectArgs,
    device: wgpu::Device,
    queue: wgpu::Queue,
) -> Result<(), anyhow::Error> {
    use crate::npc_system::{NpcSystem, Role, camera_sees_point};
    use brush_record::{Codec, Recorder, RecorderConfig};
    use brush_render::burn_glue::resolve_to_cube_float;
    use brush_render::{TextureMode, gaussian_splats::render_splats};
    use burn::tensor::s;
    use serde_json::json;

    let mut builder = env_logger::Builder::from_default_env();
    if std::env::var("RUST_LOG").is_err() {
        builder.filter_level(log::LevelFilter::Info);
    }
    let _ = builder.target(env_logger::Target::Stdout).try_init();

    // load_splats_and_scene reads render.scene and render.output_dir.
    let (splats, scene) = load_splats_and_scene(process, &render).await?;
    tokio::fs::create_dir_all(&render.output_dir).await?;
    let background = glam::Vec3::from(scene.background);

    // Splat ambient probe — same as run_record.
    let probe = if !scene.npcs.is_empty() {
        let means_data = splats.means().into_data_async().await?;
        let means_vec = means_data
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("means readback: {e:?}"))?;
        let sh = splats.sh_coeffs.val().slice(s![.., 0..1, ..]);
        let sh_data = sh.into_data_async().await?;
        let sh_vec = sh_data
            .into_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("sh readback: {e:?}"))?;
        let n = means_vec.len() / 3;
        let mut positions = Vec::with_capacity(n);
        let mut colors = Vec::with_capacity(n);
        for i in 0..n {
            positions.push(glam::vec3(
                means_vec[i * 3],
                means_vec[i * 3 + 1],
                means_vec[i * 3 + 2],
            ));
            let c = glam::vec3(
                (0.5 + SH_C0 * sh_vec[i * 3]).clamp(0.0, 1.0),
                (0.5 + SH_C0 * sh_vec[i * 3 + 1]).clamp(0.0, 1.0),
                (0.5 + SH_C0 * sh_vec[i * 3 + 2]).clamp(0.0, 1.0),
            );
            colors.push(c);
        }
        Some(SplatProbeData { positions, colors })
    } else {
        None
    };

    let scene = std::sync::Arc::new(scene);
    let mut npc_system = NpcSystem::new(
        &device,
        &queue,
        wgpu::TextureFormat::Bgra8Unorm,
        scene.clone(),
        |anchor| match &probe {
            Some(p) => compute_ambient_cube(anchor, p),
            None => [[0.5, 0.5, 0.5]; 6],
        },
    )?;

    // Filter scene cameras to the requested pool.
    let cam_pool: Vec<&CameraEntry> = scene
        .cameras
        .iter()
        .filter(|c| args.collect_cameras.contains(&c.name))
        .collect();
    if cam_pool.is_empty() {
        anyhow::bail!(
            "none of the requested cameras ({:?}) exist in scene.json",
            args.collect_cameras
        );
    }

    // Build shuffled clip list: `fall_clips` true entries + rest false.
    let total = args.total_clips;
    let fall_count = args.fall_clips.min(total);
    let mut clip_has_fall: Vec<bool> = (0..total).map(|i| i < fall_count).collect();
    // Fisher-Yates shuffle with our own RNG.
    let mut rng = crate::npc_system::Rng::new(args.collect_seed);
    for i in (1..clip_has_fall.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        clip_has_fall.swap(i, j);
    }

    let fps = render.record_fps;
    let dt = 1.0 / fps as f32;
    let total_frames = ((args.clip_duration_secs * fps as f32).ceil() as u32).max(1);

    log::info!(
        "Collecting {} clips ({} with fall, {} walk-only), {} frames each, {} cameras in pool",
        total,
        fall_count,
        total - fall_count,
        total_frames,
        cam_pool.len(),
    );
    let t_overall = std::time::Instant::now();

    for clip_idx in 0..total {
        let has_fall = clip_has_fall[clip_idx as usize];
        let clip_seed = args
            .collect_seed
            .wrapping_add((clip_idx as u64).wrapping_mul(6_364_136_223_846_793_005));

        // Reset NPCs for this clip.
        npc_system.reset_for_clip(clip_seed, !has_fall, &queue)?;

        // For fall clips: pick a random trigger time in [2s, 7s] so the
        // fall doesn't start at the very beginning or end of the clip.
        let fall_trigger_t: Option<f32> = if has_fall {
            Some(rng.range_f32(2.0, (args.clip_duration_secs - 2.0).max(2.0)))
        } else {
            None
        };

        // Camera selection. For fall clips, check which pool cameras can
        // see NPC[0]'s CURRENT position at trigger time. Walk clips pick
        // randomly. We pre-select now so the recorder can be opened at
        // the right path before the frame loop.
        //
        // For fall clips we don't know the exact fall position yet (the
        // NPC walks until trigger_t), so we check NPC[0]'s spawn position
        // as a proxy. It's within the spawn box, which is always inside
        // the warehouse, so the result is usually correct. If no camera
        // passes the test we fall back to random.
        let chosen_cam: &CameraEntry = {
            let cam_idx = if has_fall && !npc_system.runtimes.is_empty() {
                let npc_pos = npc_system.runtimes[0].pos;
                let visible: Vec<usize> = cam_pool
                    .iter()
                    .enumerate()
                    .filter(|(_, cam)| {
                        let (img_size, fov_y_deg) = resolve_size_fov(&scene, cam);
                        let camera = camera_from_ypr(
                            cam.pos.into(),
                            cam.ypr_deg.into(),
                            fov_y_deg,
                            img_size,
                        );
                        let aspect = img_size.x as f32 / img_size.y as f32;
                        camera_sees_point(
                            camera.world_to_local(),
                            fov_y_deg.to_radians() as f32,
                            aspect,
                            npc_pos,
                        )
                    })
                    .map(|(i, _)| i)
                    .collect();
                if !visible.is_empty() {
                    visible[(rng.next_u64() as usize) % visible.len()]
                } else {
                    (rng.next_u64() as usize) % cam_pool.len()
                }
            } else {
                (rng.next_u64() as usize) % cam_pool.len()
            };
            cam_pool[cam_idx]
        };

        let clip_dir = render.output_dir.join(format!("clip_{clip_idx:03}"));
        tokio::fs::create_dir_all(&clip_dir).await?;

        let (img_size, fov_y_deg) = resolve_size_fov(&scene, chosen_cam);
        let mp4_path = clip_dir.join(format!("{}.{MP4_EXT}", chosen_cam.name));
        let mut recorder = Recorder::new(
            device.clone(),
            queue.clone(),
            &mp4_path,
            RecorderConfig {
                width: img_size.x,
                height: img_size.y,
                fps,
                codec: Codec::Hevc,
            },
        )?;

        // Per-clip state tracking.
        let npc_count = npc_system.runtimes.len();
        let mut prev_roles: Vec<Option<Role>> = vec![None; npc_count];
        // Each entry: (npc_name, variant, start_sec, Option<end_sec>)
        let mut fall_events: Vec<(String, String, f32, Option<f32>)> = Vec::new();
        let mut fall_injected = false;

        for frame in 0..total_frames {
            let world_t = frame as f32 * dt;

            // Inject fall: set force_next_role on NPC[0] and expire its
            // current step so the transition fires on this tick.
            if let Some(trigger) = fall_trigger_t {
                if !fall_injected && world_t >= trigger {
                    if let Some(rt) = npc_system.runtimes.first_mut() {
                        if let Some(brain) = rt.brain.as_mut() {
                            brain.force_next_role = Some(if rng.next_f32() < 0.5 {
                                Role::Fall
                            } else {
                                Role::FallSide
                            });
                            // Force step expiry so tick() calls next_brain_step
                            // this frame rather than waiting for natural expiry.
                            brain.elapsed = f32::INFINITY;
                        }
                    }
                    fall_injected = true;
                }
            }

            npc_system.tick(dt, &queue)?;

            // Observe role transitions to timestamp fall segments.
            for i in 0..npc_count {
                let rt = &npc_system.runtimes[i];
                let cur_role = rt.brain.as_ref().map(|b| b.step.role);
                let prev = prev_roles[i];

                let is_falling = matches!(cur_role, Some(Role::Fall) | Some(Role::FallSide));
                let was_falling = matches!(prev, Some(Role::Fall) | Some(Role::FallSide));

                if is_falling && !was_falling {
                    let variant = match cur_role {
                        Some(Role::Fall) => "Fall",
                        Some(Role::FallSide) => "fall_side",
                        _ => "Fall",
                    };
                    let npc_name = scene.npcs[rt.scene_index].name.clone();
                    fall_events.push((npc_name, variant.to_string(), world_t, None));
                } else if was_falling && !is_falling {
                    // Close the most recent open event for this NPC.
                    let npc_name = &scene.npcs[rt.scene_index].name;
                    if let Some(ev) = fall_events
                        .iter_mut()
                        .rev()
                        .find(|(n, _, _, end)| n == npc_name && end.is_none())
                    {
                        ev.3 = Some(world_t);
                    }
                }
                prev_roles[i] = cur_role;
            }

            // Render the single chosen camera.
            let camera = camera_from_ypr(
                chosen_cam.pos.into(),
                chosen_cam.ypr_deg.into(),
                fov_y_deg,
                img_size,
            );

            let (tensor, aux) = render_splats(
                splats.clone(),
                &camera,
                img_size,
                background,
                None,
                TextureMode::Packed,
            )
            .await;
            let cube_color = resolve_to_cube_float(tensor);
            let resource_color = cube_color
                .client
                .get_resource(cube_color.handle.clone())
                .map_err(|e| anyhow::anyhow!("get_resource (color): {e:?}"))?;
            let cube_depth = resolve_to_cube_float(aux.depth_img);
            let resource_depth = cube_depth
                .client
                .get_resource(cube_depth.handle.clone())
                .map_err(|e| anyhow::anyhow!("get_resource (depth): {e:?}"))?;
            let _ = device.poll(wgpu::PollType::wait_indefinitely());
            let res = resource_color.resource();
            let depth_res = resource_depth.resource();

            let mut frame_handle = recorder.begin_frame()?;
            frame_handle.swizzle_from(&res.buffer, res.offset);

            if !npc_system.runtimes.is_empty() {
                let aspect = img_size.x as f32 / img_size.y as f32;
                let view_proj = crate::npc_system::view_projection(
                    camera.world_to_local(),
                    fov_y_deg.to_radians() as f32,
                    aspect,
                );
                npc_system.mesh_renderer.set_camera(&queue, view_proj, camera.position);
                npc_system.mesh_renderer.fill_depth_from_splats(
                    &device,
                    &queue,
                    frame_handle.color_texture(),
                    &depth_res.buffer,
                    depth_res.offset,
                    0.05,
                    1000.0,
                );
                let sub = npc_system.render_npcs(&device, &queue, frame_handle.color_texture(), None);
                frame_handle.note_submission(sub);
            }

            frame_handle.finish()?;
        }

        // Close any fall events still open at clip end.
        for ev in &mut fall_events {
            ev.3.get_or_insert(args.clip_duration_secs);
        }

        recorder.finish().await?;

        // Write metadata.json sidecar.
        let events_json: Vec<serde_json::Value> = fall_events
            .iter()
            .map(|(npc, variant, start, end)| {
                json!({
                    "npc": npc,
                    "variant": variant,
                    "start_sec": (start * 1000.0).round() / 1000.0,
                    "end_sec": end.map(|e| (e * 1000.0).round() / 1000.0),
                })
            })
            .collect();
        let metadata = json!({
            "clip_id": clip_idx,
            "has_fall": has_fall,
            "duration_secs": args.clip_duration_secs,
            "camera": chosen_cam.name,
            "fall_events": events_json,
        });
        let meta_path = clip_dir.join("metadata.json");
        std::fs::write(&meta_path, serde_json::to_string_pretty(&metadata)?)?;

        log::info!(
            "clip {:03}/{} done  has_fall={:<5}  cam={}  fall_events={}",
            clip_idx + 1,
            total,
            has_fall,
            chosen_cam.name,
            fall_events.len(),
        );
    }

    let elapsed = t_overall.elapsed();
    log::info!(
        "Dataset complete: {} clips in {:.1?}  ({:.1} s/clip avg)",
        total,
        elapsed,
        elapsed.as_secs_f64() / total as f64,
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
        .ok_or_else(|| anyhow::anyhow!("scene config not loaded (expected a .json positional)"))?;
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

    // Re-parse the scene here (bin.rs has already parsed it once to
    // derive the splat path) — the cost is a few hundred bytes of JSON,
    // and downstream code stays simpler by owning a fresh SceneConfig.
    let scene = load_scene_config(scene_path)?;
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

