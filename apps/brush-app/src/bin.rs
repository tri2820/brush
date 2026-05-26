#![recursion_limit = "256"]

// The desktop binary only compiles on native platforms.
// On WASM, brush-app is used as a library (cdylib) via wasm.rs instead.
#[cfg(not(target_family = "wasm"))]
mod ui;

#[cfg(not(target_family = "wasm"))]
#[allow(clippy::unnecessary_wraps)]
fn main() -> Result<(), anyhow::Error> {
    use brush_cli::Cli;
    use brush_process::create_process;
    use clap::Parser;

    let args = Cli::parse().validate()?;

    #[cfg(target_family = "windows")]
    {
        use winapi::um::wincon::GetConsoleProcessList;

        let mut buffer = [0u32; 1];

        // Safety: FFI. Buffer is valid for duration of call
        let is_console = unsafe { GetConsoleProcessList(buffer.as_mut_ptr(), 1) != 1 };

        if args.with_viewer && !is_console {
            // Safety: FFI
            unsafe {
                winapi::um::wincon::FreeConsole();
            };
        }
    }

    #[cfg(feature = "tracy")]
    {
        use tracing_subscriber::layer::SubscriberExt;

        tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(tracing_tracy::TracyLayer::default()),
        )
        .expect("Failed to set tracing subscriber");
    }

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("Failed to initialize tokio runtime")
        .block_on(async move {
            let init_process = args.source.map(|source| {
                create_process(source, {
                    let cli_config = args.train_stream.clone();
                    async move |init| {
                        Some(brush_process::args_file::merge_configs(&init, &cli_config))
                    }
                })
            });

            #[cfg(target_os = "macos")]
            if args.render.record_output.is_some() {
                // Build wgpu Instance/Adapter/Device/Queue manually so we
                // own handles the recorder (IOSurface texture import,
                // swizzle compute) can share with burn.
                let mut idesc = wgpu::InstanceDescriptor::new_without_display_handle();
                idesc.backends = wgpu::Backends::METAL;
                let instance = wgpu::Instance::new(idesc);
                let adapter = instance
                    .request_adapter(&wgpu::RequestAdapterOptions {
                        power_preference: wgpu::PowerPreference::HighPerformance,
                        compatible_surface: None,
                        ..Default::default()
                    })
                    .await?;
                // Brush's rasterizer needs 512-thread workgroups, 9
                // storage buffers per stage, and SUBGROUP/etc. features
                // it enables internally — match whatever the adapter
                // supports. Plus BGRA8UNORM_STORAGE for the recorder's
                // IOSurface storage-texture write.
                let adapter_limits = adapter.limits();
                let adapter_features = adapter.features();
                // SAFETY: enabling experimental features acknowledges
                // that some wgpu features (RAY_QUERY, MESH_SHADER, etc.)
                // may have UB-containing bugs. None of those are used in
                // brush's render path or in brush-record's swizzle, but
                // burn's adapter enumeration surfaces them so we have to
                // allow them through to match the adapter's full feature
                // set that burn expects.
                let experimental_features = unsafe { wgpu::ExperimentalFeatures::enabled() };
                let (device, queue) = adapter
                    .request_device(&wgpu::DeviceDescriptor {
                        label: Some("brush-record device"),
                        required_features: adapter_features | wgpu::Features::BGRA8UNORM_STORAGE,
                        required_limits: adapter_limits,
                        experimental_features,
                        ..Default::default()
                    })
                    .await?;
                // `burn_init_device` consumes adapter; clones of
                // device/queue stay for our recorder.
                brush_process::burn_init_device(adapter, device.clone(), queue.clone());
                let process = init_process.expect("Must provide a source for --record-output");
                brush_cli::run_record(process, args.render, device, queue).await?;
                return anyhow::Result::<(), anyhow::Error>::Ok(());
            }

            if args.render.render_output.is_some() || args.render.scene.is_some() {
                brush_process::burn_init_setup().await;
                let process = init_process.expect("Must provide a source for render mode");
                brush_cli::run_render(process, args.render).await?;
            } else if args.with_viewer {
                use crate::ui::app::App;

                let logger = env_logger::Builder::from_default_env()
                    .target(env_logger::Target::Stdout)
                    .build();
                let max = logger.filter();
                crate::ui::log_panel::install_global_logger(Box::new(logger), max);

                let icon = eframe::icon_data::from_png_bytes(
                    &include_bytes!("../assets/icon-256.png")[..],
                )
                .expect("Failed to load icon");

                let native_options = eframe::NativeOptions {
                    viewport: egui::ViewportBuilder::default()
                        .with_inner_size(egui::Vec2::new(1450.0, 1200.0))
                        .with_active(true)
                        .with_icon(std::sync::Arc::new(icon)),
                    wgpu_options: ui::create_egui_options(),
                    persist_window: true,
                    ..Default::default()
                };

                let title = if cfg!(debug_assertions) {
                    "Brush  -  Debug"
                } else {
                    "Brush"
                };

                eframe::run_native(
                    title,
                    native_options,
                    Box::new(move |cc| Ok(Box::new(App::new(cc, init_process)))),
                )?;
            } else {
                brush_process::burn_init_setup().await;
                let process = init_process.expect("Must provide a source");
                brush_cli::run_cli_ui(process, args.train_stream).await?;
            }

            anyhow::Result::<(), anyhow::Error>::Ok(())
        })?;

    Ok(())
}

// On WASM, just stub a dummy main.
#[cfg(target_family = "wasm")]
fn main() {}
