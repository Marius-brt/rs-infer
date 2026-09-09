use std::{sync::Arc, time::Duration};
use std::sync::RwLock;

use prometheus_client::{
	encoding::text::encode,
	metrics::{
		counter::Counter,
		family::Family,
		histogram::{Histogram, exponential_buckets},
	},
	registry::Registry,
};

type Labels = Vec<(String, String)>;

#[derive(Clone)]
pub struct Metrics {
	inner: Arc<Inner>,
}

struct Inner {
	requests: Family<Labels, Counter>,
	duration: Family<Labels, Histogram>,
	registry: RwLock<Registry>,
}

impl Default for Metrics {
	fn default() -> Self {
		let requests: Family<Labels, Counter> = Family::default();
		let duration: Family<Labels, Histogram> = Family::new_with_constructor(|| Histogram::new(exponential_buckets(0.005, 2.0, 12)));
		let mut registry = Registry::default();
		registry.register("ortinfer_http_requests", "HTTP requests by route and status", requests.clone()); // prometheus-client appends _total for counters
		registry.register("ortinfer_http_request_duration_seconds", "HTTP request latency by route", duration.clone());
		Self {
			inner: Arc::new(Inner {
				requests,
				duration,
				registry: RwLock::new(registry),
			}),
		}
	}
}

impl Metrics {
	pub fn observe(&self, route: &str, status: u16, duration: Duration) {
		self.inner
			.requests
			.get_or_create(&vec![("route".into(), route.into()), ("status".into(), status.to_string())])
			.inc();
		self.inner
			.duration
			.get_or_create(&vec![("route".into(), route.into())])
			.observe(duration.as_secs_f64());
	}

	pub fn encode(&self) -> String {
		let mut out = String::new();
		let reg = self.inner.registry.read().expect("metrics registry poisoned");
		let _ = encode(&mut out, &reg);
		out
	}
}
