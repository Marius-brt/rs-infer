use std::{sync::Arc, time::Duration};

use ortinfer_core::{config::Config as AppConfig, Registry};

use crate::metrics::Metrics;

#[derive(Clone)]
pub struct AppState {
	pub registry: Arc<Registry>,
	pub metrics: Metrics,
	pub queue_wait: Duration,
}

impl AppState {
	pub fn new(registry: Arc<Registry>, cfg: Arc<AppConfig>) -> Self {
		let queue_wait = Duration::from_millis(cfg.server.queue_timeout_ms);
		Self { registry, metrics: Metrics::default(), queue_wait }
	}
}
