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
        default_value_if("render_output", ArgPredicate::IsPresent, "false"),
        help = "Spawn a viewer to visualize the training"
    )]
    pub with_viewer: bool,

    #[clap(flatten)]
    pub train_stream: TrainStreamConfig,

    #[clap(flatten)]
    pub render: RenderArgs,
}

/// Arguments for the headless render-to-PNG mode. Activated by passing
/// `--render-output`; in that mode the CLI loads the source, renders a
/// single frame from the configured camera, writes a PNG, and exits.
#[derive(Args, Clone, Debug)]
pub struct RenderArgs {
    /// Write a single rendered PNG to this path and exit. Implies
    /// --with-viewer=false.
    #[arg(long, value_name = "PATH")]
    pub render_output: Option<PathBuf>,

    /// Camera position in world space, "x,y,z".
    #[arg(long, value_name = "X,Y,Z", default_value = "0,0,-2.5")]
    pub camera_pos: Vec3Arg,

    /// World-space point the camera looks at, "x,y,z".
    #[arg(long, value_name = "X,Y,Z", default_value = "0,0,0")]
    pub camera_look: Vec3Arg,

    /// World-space up vector hint, "x,y,z".
    #[arg(long, value_name = "X,Y,Z", default_value = "0,1,0")]
    pub camera_up: Vec3Arg,

    /// Optional Euler angles "yaw,pitch,roll" in degrees, applied as
    /// glam's `EulerRot::YXZ` — matches the in-app HUD readout. When
    /// set, --camera-look / --camera-up are ignored.
    #[arg(long, value_name = "YAW,PITCH,ROLL")]
    pub camera_ypr_deg: Option<Vec3Arg>,

    /// Vertical field of view in degrees.
    #[arg(long, default_value_t = 45.0)]
    pub fov_y_deg: f64,

    /// Output resolution as WIDTHxHEIGHT.
    #[arg(long, default_value = "1280x720", value_parser = parse_resolution)]
    pub resolution: glam::UVec2,

    /// Background color, "r,g,b" in linear 0..1.
    #[arg(long, value_name = "R,G,B", default_value = "0,0,0")]
    pub background: Vec3Arg,
}

/// Newtype wrapper so clap can parse `--foo x,y,z` directly into a glam Vec3
/// while keeping the existing serde-derived `DataSource` parsing intact.
#[derive(Clone, Copy, Debug)]
pub struct Vec3Arg(pub glam::Vec3);

impl std::str::FromStr for Vec3Arg {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(',').collect();
        if parts.len() != 3 {
            return Err(format!("expected 'x,y,z', got {s:?}"));
        }
        let parse = |i: usize| {
            parts[i]
                .trim()
                .parse::<f32>()
                .map_err(|e| format!("component {i}: {e}"))
        };
        Ok(Self(glam::vec3(parse(0)?, parse(1)?, parse(2)?)))
    }
}

fn parse_resolution(s: &str) -> Result<glam::UVec2, String> {
    let (w, h) = s
        .split_once('x')
        .ok_or_else(|| format!("expected WIDTHxHEIGHT, got {s:?}"))?;
    let w: u32 = w.trim().parse().map_err(|e| format!("width: {e}"))?;
    let h: u32 = h.trim().parse().map_err(|e| format!("height: {e}"))?;
    if w == 0 || h == 0 {
        return Err("resolution must be > 0 in both dimensions".into());
    }
    Ok(glam::uvec2(w, h))
}

impl Cli {
    pub fn validate(self) -> Result<Self, Error> {
        if self.render.render_output.is_some() && self.source.is_none() {
            return Err(Error::raw(
                ErrorKind::MissingRequiredArgument,
                "When --render-output is set, --source must be provided",
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

/// Drive a process to `DoneLoading`, then render a single frame from
/// the configured camera and write it as a PNG.
pub async fn run_render(
    mut process: RunningProcess,
    args: RenderArgs,
) -> Result<(), anyhow::Error> {
    use brush_render::{TextureMode, gaussian_splats::render_splats};
    use image::Rgb32FImage;

    let output = args
        .render_output
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("render_output unset"))?;

    log::info!("Loading source...");
    while let Some(msg) = process.stream.next().await {
        match msg? {
            ProcessMessage::DoneLoading => break,
            ProcessMessage::StartLoading { training, .. } if training => {
                anyhow::bail!(
                    "--render-output expects a single .ply / .compressed.ply source, not a training dataset"
                );
            }
            ProcessMessage::Warning { error } => {
                log::warn!("{error}");
            }
            _ => {}
        }
    }

    let splats = process
        .splat_view
        .latest()
        .ok_or_else(|| anyhow::anyhow!("no splats were loaded from source"))?;
    log::info!(
        "Loaded {} splats (sh degree {}). Rendering at {}x{}...",
        splats.num_splats(),
        splats.sh_degree(),
        args.resolution.x,
        args.resolution.y,
    );

    let camera = build_camera(&args);
    let img_size = args.resolution;
    let background = args.background.0;

    let (image, _aux) = render_splats(
        splats,
        &camera,
        img_size,
        background,
        None,
        TextureMode::Float,
    )
    .await;

    // Float-mode output is [h, w, 4] (RGBA); drop alpha to match Rgb32FImage's
    // 3-channel expectation. Eval does the same: see EvalSample::save_to_disk.
    use burn::tensor::s;
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

fn build_camera(args: &RenderArgs) -> brush_render::camera::Camera {
    use brush_render::camera::Camera;
    use brush_render::kernels::camera_model::CameraModel;
    use glam::{EulerRot, Mat3, Quat, Vec3};

    let eye = args.camera_pos.0;

    let rotation = if let Some(ypr) = args.camera_ypr_deg {
        let v = ypr.0;
        Quat::from_euler(
            EulerRot::YXZ,
            v.x.to_radians(),
            v.y.to_radians(),
            v.z.to_radians(),
        )
    } else {
        let target = args.camera_look.0;
        let up_hint = args.camera_up.0;
        // Camera-local axes: +X right, +Y up, +Z forward (camera looks
        // down +Z). See brush-render::camera::Camera and
        // camera_controls.rs, where `position + rotation * Vec3::Z *
        // focus_distance` is the look-at pivot.
        let forward = (target - eye).normalize_or_zero();
        let forward = if forward.length_squared() < 1e-12 {
            Vec3::Z
        } else {
            forward
        };
        let mut up_hint = up_hint.normalize_or(Vec3::Y);
        if up_hint.cross(forward).length_squared() < 1e-6 {
            up_hint = if forward.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
        }
        let right = up_hint.cross(forward).normalize();
        let cam_up = forward.cross(right);
        let rot_mat = Mat3::from_cols(right, cam_up, forward);
        Quat::from_mat3(&rot_mat)
    };

    let aspect = args.resolution.x as f64 / args.resolution.y as f64;
    let fov_y = args.fov_y_deg.to_radians();
    let fov_x = 2.0 * ((fov_y / 2.0).tan() * aspect).atan();

    Camera::new(
        eye,
        rotation,
        fov_x,
        fov_y,
        glam::vec2(0.5, 0.5),
        CameraModel::Pinhole,
    )
}
