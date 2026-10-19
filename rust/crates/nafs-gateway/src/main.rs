//! nafs-gateway: the Rust Gateway (IF-001..IF-004, IF-009, NFR-011, NFR-021, NFR-022).
mod api;
mod gates;
mod security;
use clap::Parser;
use std::sync::Arc;

#[derive(Parser)]
#[command(
    version,
    about = "Nafs-Aql gateway: dual-process chat API, rooms, performance console and evaluation dashboard"
)]
struct Args {
    #[arg(long, env = "NAFS_CONFIG", default_value = "config/nafs-aql.toml")]
    config: String,
    #[arg(long, env = "NAFS_WEB_DIR")]
    web_dir: Option<String>,
    /// Validate configuration, secrets, bind policy and sandbox isolation, print a JSON report and exit.
    #[arg(long)]
    check: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| "info,nafs=debug".into());
    if std::env::var("NAFS_LOG_FORMAT").as_deref() == Ok("json") {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(filter).init();
    }
    let cfg = Arc::new(nafs_core::Config::load(&args.config)?);
    if args.check {
        let tokens = security::load_tokens(&cfg.server.auth_tokens_env);
        let bind = tokens.as_ref().map_err(|e| e.clone()).and_then(|t| {
            security::check_bind(
                &cfg.server.bind,
                t,
                std::env::var("NAFS_ALLOW_INSECURE").as_deref() == Ok("1"),
            )
        });
        let isolation = nafs_verify::sandbox::isolation_available();
        let report = serde_json::json!({"config": args.config, "config_hash": cfg.hash, "bind": cfg.server.bind, "backend": cfg.backend.active,
            "auth_tokens": match &tokens { Ok(t) => serde_json::json!(t.len()), Err(e) => serde_json::json!({"error": e}) },
            "bind_policy": match &bind { Ok(_) => serde_json::json!("ok"), Err(e) => serde_json::json!({"error": e}) },
            "sandbox_isolation": isolation, "retention_days": cfg.storage.retention_days});
        println!("{}", serde_json::to_string_pretty(&report)?);
        std::process::exit(if bind.is_ok() && isolation { 0 } else { 1 });
    }
    // Latency metrics as histograms so PromQL can aggregate p95 over time windows (OR-004 alerts).
    let prom = metrics_exporter_prometheus::PrometheusBuilder::new()
        .set_buckets_for_metric(
            metrics_exporter_prometheus::Matcher::Suffix("_seconds".into()),
            &[
                0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 20.0, 30.0, 60.0, 120.0, 240.0, 480.0, 900.0,
            ],
        )?
        .install_recorder()?;
    let store = Arc::new(nafs_memory::Store::open(
        std::path::Path::new(&cfg.storage.sqlite_path),
        std::path::Path::new(&cfg.storage.trace_path),
    )?);
    let orch = Arc::new(nafs_core::Orchestrator::new(cfg.clone(), store.clone()));
    let tokens = security::load_tokens(&cfg.server.auth_tokens_env).map_err(anyhow::Error::msg)?;
    security::check_bind(
        &cfg.server.bind,
        &tokens,
        std::env::var("NAFS_ALLOW_INSECURE").as_deref() == Ok("1"),
    )
    .map_err(anyhow::Error::msg)?;
    if tokens.is_empty() {
        tracing::warn!(
            "{} is not set: authentication disabled (loopback-only development mode)",
            cfg.server.auth_tokens_env
        );
    }
    let state = api::new_state(cfg.clone(), orch, store, prom, tokens, args.web_dir);
    api::spawn_background(state.clone());
    let app = api::router(state);
    let listener = tokio::net::TcpListener::bind(&cfg.server.bind).await?;
    tracing::info!(
        "Nafs-Aql gateway listening on http://{} (config hash {}, backend '{}')",
        cfg.server.bind,
        cfg.hash,
        cfg.backend.active
    );
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<std::net::SocketAddr>(),
    )
    .with_graceful_shutdown(async {
        // Ctrl-C locally, SIGTERM from systemd / launchd / docker (OR-003 service files).
        #[cfg(unix)]
        {
            let mut term =
                tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                    .expect("signal handler");
            tokio::select! { _ = tokio::signal::ctrl_c() => {}, _ = term.recv() => {} }
        }
        #[cfg(not(unix))]
        let _ = tokio::signal::ctrl_c().await;
        tracing::info!("shutting down");
    })
    .await?;
    Ok(())
}
