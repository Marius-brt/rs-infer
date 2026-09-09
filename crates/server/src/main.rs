mod api;
mod dto;
mod error;
mod metrics;
mod state;

use std::{path::PathBuf, sync::Arc};

use anyhow::Context;
use clap::Parser;
use rsinfer_core::Registry;
use tokio::net::TcpListener;
use tower_http::{limit::RequestBodyLimitLayer, timeout::TimeoutLayer};

use crate::state::AppState;

#[derive(Parser)]
#[command(name = "rsinfer", about = "ONNX Runtime inference server: embeddings, rerank, PII, zero-shot")]
struct Args {
	/// Path to the TOML model/server config.
	#[arg(short, long, default_value = "config.yaml")]
	config: PathBuf,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
	init_tracing();
	let args = Args::parse();

	let text = std::fs::read_to_string(&args.config).with_context(|| format!("cannot read config {}", args.config.display()))?;
	if args.config.extension().map(|e| e == "toml").unwrap_or(false) {
		anyhow::bail!("TOML configs are no longer supported; convert {} to YAML (see configs/config.example.yaml)", args.config.display());
	}
	let config: rsinfer_core::Config = serde_norway::from_str(&text).context("invalid YAML config")?;
	validate(&config)?;
	let config = Arc::new(config);

	tracing::info!(
		"rsinfer v{} starting with {} model(s) on {} ({}); EP features: coreml={} cuda={} tensorrt={} nvrtx={}",
		env!("CARGO_PKG_VERSION"),
		config.models.len(),
		std::env::consts::OS,
		std::env::consts::ARCH,
		cfg!(feature = "ep-coreml"),
		cfg!(feature = "ep-cuda"),
		cfg!(feature = "ep-tensorrt"),
		cfg!(feature = "ep-nvrtx"),
	);
	if let Some(rss) = rsinfer_core::memory::rss_mb() {
		tracing::info!(ram_mb = rss, "process memory at startup");
	}
	tracing::info!(bind = %config.server.bind, "resolving and loading configured models (first run downloads from the HF Hub)");
	let started = std::time::Instant::now();
	let registry = Arc::new(Registry::load(config.clone()).await?);
	tracing::info!(
		elapsed_ms = started.elapsed().as_millis(),
		models = registry.infos().len(),
		ram_total_mb = ram_mb(),
		"all models loaded"
	);

	let state = AppState::new(registry, config.clone());
	let app = api::router(state)
		.layer(TimeoutLayer::with_status_code(
			axum::http::StatusCode::REQUEST_TIMEOUT,
			std::time::Duration::from_millis(config.server.request_timeout_ms),
		))
		.layer(RequestBodyLimitLayer::new(config.server.max_body_mb * 1024 * 1024));

	let listener = TcpListener::bind(&config.server.bind).await.with_context(|| format!("cannot bind {}", config.server.bind))?;
	let addr = listener.local_addr()?;
	tracing::info!(
		%addr,
		url = %format!("http://{addr}"),
		total_startup_ms = started.elapsed().as_millis(),
		ram_total_mb = ram_mb(),
		"HTTP server ready; listening for requests"
	);
	axum::serve(listener, app)
		.with_graceful_shutdown(shutdown_signal())
		.await
		.context("server error")?;
	Ok(())
}

fn ram_mb() -> u64 {
	rsinfer_core::memory::rss_mb().unwrap_or(0)
}

fn validate(config: &rsinfer_core::Config) -> anyhow::Result<()> {
	let mut names = std::collections::HashSet::new();
	for m in &config.models {
		if !names.insert(m.name.clone()) {
			anyhow::bail!("duplicate model name '{}'", m.name);
		}
		m.validate().map_err(|e| anyhow::anyhow!("{e}"))?;
	}
	if config.models.is_empty() {
		anyhow::bail!("config has no models");
	}
	Ok(())
}

fn init_tracing() {
	use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};
	let filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,ort=warn,tokenizers=warn,reqwest=warn,hf_hub=warn"));
	// Startup/lifecycle logs from our own crates are always visible, even if RUST_LOG is set stricter.
	let filter = ["rsinfer", "rsinfer_server", "rsinfer_core"].into_iter().fold(filter, |f, t| {
		f.add_directive(format!("{t}=info").parse().expect("static directive parses"))
	});
	tracing_subscriber::registry()
		.with(filter)
		.with(tracing_subscriber::fmt::layer().with_target(false))
		.init();
}

async fn shutdown_signal() {
	let ctrl_c = async {
		tokio::signal::ctrl_c().await.ok();
	};
	#[cfg(unix)]
	let sigterm = async {
		use tokio::signal::unix::{signal, SignalKind};
		if let Ok(mut s) = signal(SignalKind::terminate()) {
			s.recv().await;
		}
	};
	#[cfg(not(unix))]
	let sigterm = async { std::future::pending::<()>().await };
	tokio::select! {
		_ = ctrl_c => {},
		_ = sigterm => {},
	}
	tracing::info!("shutdown signal received");
}
