use std::path::{Path, PathBuf};

use hf_hub::api::sync::{Api, ApiBuilder, ApiRepo};

use crate::{config::ModelConfig, Error, Result};

/// Materialized files for one model, local or hub-cached.
#[derive(Debug, Clone)]
pub struct Resolved {
	pub model: PathBuf,
	pub tokenizer: PathBuf,
	pub config_json: Option<PathBuf>,
	pub pooling_config: Option<PathBuf>,
	pub source: String,
}

/// Model-file preference: full precision first, then common quantized variants.
const MODEL_CANDIDATES: &[&str] = &[
	"model.onnx",
	"model_fp16.onnx",
	"model_quantized.onnx",
	"model_q4f16.onnx",
	"model_q4.onnx",
	"model_int8.onnx",
	"model_uint8.onnx",
];
const POOLING_CANDIDATES: &[&str] = &["1_Pooling/config.json"];

pub fn resolve(cfg: &ModelConfig, cache_dir: Option<&Path>) -> Result<Resolved> {
	if let Some(dir) = &cfg.path {
		return resolve_local(cfg, dir);
	}
	resolve_hf(cfg, cache_dir)
}

fn candidates_for<'a>(cfg: &'a ModelConfig, defaults: &'a [&'a str]) -> Vec<&'a str> {
	let mut v = Vec::with_capacity(defaults.len() + 1);
	if let Some(f) = cfg.file.as_deref() {
		v.push(f);
	}
	v.extend_from_slice(defaults);
	v
}

/// Picks the first candidate present among repo siblings, checking (in order for
/// each candidate): repo root, configured subfolder, and the conventional `onnx/`.
pub fn select_remote(siblings: &[String], subfolder: Option<&str>, candidates: &[&str]) -> Option<String> {
	for c in candidates {
		let mut names = vec![c.to_string()];
		if let Some(sub) = subfolder {
			names.push(format!("{sub}/{c}"));
		}
		names.push(format!("onnx/{c}"));
		for n in names {
			if siblings.iter().any(|s| s == &n) {
				return Some(n);
			}
		}
	}
	None
}

fn pick_file(dir: &Path, candidates: &[&str], subfolder: Option<&str>) -> Option<PathBuf> {
	for c in candidates {
		let direct = dir.join(c);
		if direct.is_file() {
			return Some(direct);
		}
		if let Some(sub) = subfolder {
			let p = dir.join(sub).join(c);
			if p.is_file() {
				return Some(p);
			}
		}
		let p = dir.join("onnx").join(c);
		if p.is_file() {
			return Some(p);
		}
	}
	None
}

fn resolve_local(cfg: &ModelConfig, dir: &Path) -> Result<Resolved> {
	let sub = cfg.subfolder.as_deref();
	let model = pick_file(dir, &candidates_for(cfg, MODEL_CANDIDATES), sub).ok_or_else(|| {
		Error::MissingFile(cfg.name.clone(), format!("no model.onnx under {} (or its onnx/ subfolder)", dir.display()))
	})?;
	let tokenizer = pick_file(dir, &["tokenizer.json"], sub).ok_or_else(|| Error::MissingFile(cfg.name.clone(), format!("tokenizer.json in {}", dir.display())))?;
	let config_json = pick_file(dir, &["config.json"], sub);
	let pooling_config = pick_file(dir, POOLING_CANDIDATES, sub);
	Ok(Resolved { model, tokenizer, config_json, pooling_config, source: format!("path:{}", dir.display()) })
}

fn hub_api(cfg_hf_cache: Option<&Path>) -> Result<Api> {
	let mut builder = ApiBuilder::new();
	if let Some(dir) = cfg_hf_cache {
		builder = builder.with_cache_dir(dir.join("hf"));
	}
	builder.build().map_err(|e| Error::Hub(e.to_string()))
}

fn resolve_hf(cfg: &ModelConfig, hf_cache: Option<&Path>) -> Result<Resolved> {
	let id = cfg.hf.as_deref().expect("checked by caller");
	tracing::debug!(model = cfg.name.as_str(), repo = id, revision = %cfg.revision, "resolving model from Hugging Face Hub");
	let api = hub_api(hf_cache)?;
	let repo_for = |rid: &str| -> ApiRepo {
		api.repo(hf_hub::Repo::with_revision(rid.to_string(), hf_hub::RepoType::Model, cfg.revision.clone()))
	};
	let list = |rid: &str| -> Result<Vec<String>> {
		match repo_for(rid).info() {
			Ok(info) => Ok(info.siblings.into_iter().map(|s| s.rfilename).collect()),
			Err(e) => {
				let msg = e.to_string();
				if msg.contains("401") || msg.contains("403") || msg.contains("404") {
					return Err(Error::Hub(format!(
						"repository '{rid}' not found, gated, or private (HTTP auth rejected anonymously). Check the repo id, or set HF_TOKEN for gated models. Original error: {msg}"
					)));
				}
				Err(Error::Hub(format!("cannot list {rid}: {msg}")))
			}
		}
	};
	let download = |rid: &str, name: &str| -> Result<PathBuf> {
		repo_for(rid).get(name).map_err(|e| Error::Hub(format!("cannot download {rid}/{name}: {e}")))
	};

	let siblings = list(id)?;
	let sub = cfg.subfolder.as_deref();
	let model_name = select_remote(&siblings, sub, &candidates_for(cfg, MODEL_CANDIDATES))
		.ok_or_else(|| Error::MissingFile(cfg.name.clone(), format!("no model.onnx/fp16/quantized ONNX graph found in {id} (files: {:?})", sample(&siblings))))?;
	let model = download(id, &model_name)?;
	// Large exports store weights in a sibling `.onnx_data` file; ORT resolves it relative to model.onnx.
	if let Some(companion) = model_name.strip_suffix(".onnx").map(|stem| format!("{stem}.onnx_data")) {
		if siblings.contains(&companion) {
			download(id, &companion)?;
		}
	}

	// Auxiliary files (tokenizer, config) may live in a separate repo for model-only ONNX exports.
	let aux_id = cfg.tokenizer_hf.as_deref();
	let aux_siblings = match aux_id {
		Some(t) => Some(list(t)?),
		None => None,
	};
	let find_aux = |candidates: &[&str]| -> Option<(String, String)> {
		select_remote(&siblings, sub, candidates).map(|n| (id.to_string(), n)).or_else(|| {
			let t = aux_id?;
			Some((t.to_string(), select_remote(aux_siblings.as_ref()?, None, candidates)?))
		})
	};

	let require_aux = |what: &str| -> String {
		match aux_id {
			Some(_) => format!("no {what} in {id} or the tokenizer repo"),
			None => format!("no {what} in {id}; add `tokenizer_hf` pointing at a repo that has it (e.g. the original model repo)"),
		}
	};
	let (tok_repo, tok_name) = find_aux(&["tokenizer.json"]).ok_or_else(|| Error::MissingFile(cfg.name.clone(), require_aux("tokenizer.json")))?;
	let tokenizer = download(&tok_repo, &tok_name)?;
	let config_json = find_aux(&["config.json"]).map(|(r, n)| download(&r, &n)).transpose()?;
	let pooling_config = find_aux(POOLING_CANDIDATES).map(|(r, n)| download(&r, &n)).transpose()?;
	tracing::debug!(
		model = cfg.name.as_str(),
		graph = %model_name,
		tokenizer = %format!("{tok_repo}/{tok_name}"),
		"hub files resolved"
	);

	let source = match aux_id {
		Some(t) => format!("hf:{id}@{}+aux:{t}", cfg.revision),
		None => format!("hf:{id}@{}", cfg.revision),
	};
	Ok(Resolved { model, tokenizer, config_json, pooling_config, source })
}

fn sample(siblings: &[String]) -> Vec<&str> {
	siblings.iter().take(12).map(|s| s.as_str()).collect()
}

#[cfg(test)]
#[path = "tests/hub_tests.rs"]
mod tests;
