#[allow(unused_imports)]
use ort::{
	ep::{self, ExecutionProvider, ExecutionProviderDispatch},
	logging::LogLevel,
	session::{
		builder::{GraphOptimizationLevel, SessionBuilder},
		Session,
	},
};

use crate::{
	config::{EpName, ModelConfig},
	Error, Result,
};

/// All EPs known to the runtime, with compile-time availability.
pub const ALL_EPS: &[EpName] = &[EpName::Cpu, EpName::Coreml, EpName::Cuda, EpName::Tensorrt, EpName::Nvrtx];

pub fn compiled_in(ep: EpName) -> bool {
	match ep {
		EpName::Cpu => true,
		#[cfg(feature = "ep-coreml")]
		EpName::Coreml => true,
		#[cfg(feature = "ep-cuda")]
		EpName::Cuda => true,
		#[cfg(feature = "ep-tensorrt")]
		EpName::Tensorrt => true,
		#[cfg(feature = "ep-nvrtx")]
		EpName::Nvrtx => true,
		_ => false,
	}
}

/// Whether the linked ONNX Runtime build actually contains this EP.
pub fn runtime_available(ep: EpName) -> bool {
	match ep {
		EpName::Cpu => true,
		#[cfg(feature = "ep-coreml")]
		EpName::Coreml => ep::CoreML::default().is_available().unwrap_or(false),
		#[cfg(feature = "ep-cuda")]
		EpName::Cuda => ep::CUDA::default().is_available().unwrap_or(false),
		#[cfg(feature = "ep-tensorrt")]
		EpName::Tensorrt => ep::TensorRT::default().is_available().unwrap_or(false),
		#[cfg(feature = "ep-nvrtx")]
		EpName::Nvrtx => ep::NVRTX::default().is_available().unwrap_or(false),
		_ => false,
	}
}

#[allow(unused_variables)]
fn dispatch(cfg: &ModelConfig, name: EpName) -> Option<ExecutionProviderDispatch> {
	match name {
		EpName::Cpu => Some(ep::CPU::default().build()),
		#[cfg(feature = "ep-coreml")]
		EpName::Coreml => {
			use crate::config::CoreMlComputeUnits;
			let units = match cfg.coreml_compute_units {
				CoreMlComputeUnits::All => ep::coreml::ComputeUnits::All,
				CoreMlComputeUnits::CpuAndGpu => ep::coreml::ComputeUnits::CPUAndGPU,
				CoreMlComputeUnits::CpuAndNe => ep::coreml::ComputeUnits::CPUAndNeuralEngine,
				CoreMlComputeUnits::CpuOnly => ep::coreml::ComputeUnits::CPUOnly,
			};
			Some(ep::CoreML::default().with_compute_units(units).with_model_cache_dir(std::env::temp_dir().join("ortinfer-coreml").display().to_string()).build())
		}
		#[cfg(feature = "ep-cuda")]
		EpName::Cuda => Some(ep::CUDA::default().with_device_id(cfg.device_id).build()),
		#[cfg(feature = "ep-tensorrt")]
		EpName::Tensorrt => {
			let mut b = ep::TensorRT::default().with_device_id(cfg.device_id).with_engine_cache(true);
			if let Some(dir) = &cfg.trt_engine_cache {
				b = b.with_engine_cache_path(dir.display().to_string());
			}
			Some(b.build())
		}
		#[cfg(feature = "ep-nvrtx")]
		EpName::Nvrtx => {
			let mut b = ep::NVRTX::default().with_device_id(cfg.device_id as u32);
			if let Some(dir) = &cfg.trt_engine_cache {
				b = b.with_runtime_cache_path(dir.display().to_string());
			}
			Some(b.build())
		}
		_ => None,
	}
}

/// Resolves the effective EP list (request order, compiled-in + runtime-available, CPU appended)
/// together with the registrations that can actually be attempted.
pub fn resolve_eps(cfg: &ModelConfig) -> (Vec<String>, Vec<ExecutionProviderDispatch>) {
	let requested: Vec<EpName> = if cfg.eps.is_empty() { vec![EpName::Cpu] } else { cfg.eps.clone() };
	let mut used = Vec::new();
	let mut dispatches = Vec::new();
	for name in requested {
		if !compiled_in(name) {
			tracing::warn!(model = cfg.name.as_str(), ep = %name, "execution provider not compiled in; enable the matching cargo feature to use it");
			continue;
		}
		if !runtime_available(name) {
			tracing::warn!(model = cfg.name.as_str(), ep = %name, "ONNX Runtime build lacks this execution provider; skipping");
			continue;
		}
		match dispatch(cfg, name) {
			Some(d) => {
				used.push(name.as_str().to_owned());
				dispatches.push(d);
			}
			None => tracing::warn!(model = cfg.name.as_str(), ep = %name, "execution provider unavailable on this platform; skipping"),
		}
	}
	if !dispatches.is_empty() && !used.iter().any(|u| u == "cpu") {
		used.push("cpu".into());
	}
	(used, dispatches)
}

/// Creates one session with the resolved execution providers + graph optimizations.
pub fn new_session(model_path: &std::path::Path, cfg: &ModelConfig) -> Result<Session> {
	let (used, dispatches) = resolve_eps(cfg);
	tracing::debug!(model = cfg.name.as_str(), eps = ?used, "building session");

	let mut builder: SessionBuilder = Session::builder()?
		.with_optimization_level(GraphOptimizationLevel::Level3)
		.map_err(|e| Error::Ort(e.into()))?;
	let log_level = match cfg.ort_log_level {
		crate::config::OrtLogLevel::Verbose => LogLevel::Verbose,
		crate::config::OrtLogLevel::Info => LogLevel::Info,
		crate::config::OrtLogLevel::Warn => LogLevel::Warning,
		crate::config::OrtLogLevel::Error => LogLevel::Error,
	};
	builder = builder.with_log_level(log_level).map_err(|e| Error::Ort(e.into()))?;
	if cfg.intra_threads > 0 {
		builder = builder.with_intra_threads(cfg.intra_threads).map_err(|e| Error::Ort(e.into()))?;
	}
	if !dispatches.is_empty() {
		builder = builder.with_execution_providers(dispatches).map_err(|e| Error::Ort(e.into()))?;
	}
	Ok(builder.commit_from_file(model_path)?)
}
