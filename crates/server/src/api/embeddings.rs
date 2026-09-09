use axum::{extract::State, response::IntoResponse, Json};
use base64::{Engine as _, engine::general_purpose::STANDARD};
use ortinfer_core::{
	Kind,
	pipeline::embedding,
};

use super::usage;
use crate::{
	dto::{EmbeddingItem, EmbeddingsRequest, EmbeddingsResponse},
	error::ApiError,
	state::AppState,
};

/// OpenAI/vLLM-compatible embeddings endpoint.
pub async fn embeddings(State(state): State<AppState>, Json(req): Json<EmbeddingsRequest>) -> Result<Json<EmbeddingsResponse>, ApiError> {
	let model = state.registry.resolve(req.model.as_deref(), Kind::Embedding)?;
	let (texts, token_rows) = req.input.into_parts();
	if texts.is_empty() && token_rows.is_empty() {
		return Err(ApiError(ortinfer_core::Error::BadRequest("`input` is empty".into())));
	}

	let (vectors, tokens) = if texts.is_empty() {
		let out = embedding::embed_tokens(&model, token_rows, state.queue_wait).await?;
		(out.vectors, out.tokens)
	} else {
		let out = embedding::embed(&model, texts, state.queue_wait).await?;
		(out.vectors, out.tokens)
	};

	let base64_mode = req.encoding_format.as_deref() == Some("base64");
	let mut data = Vec::with_capacity(vectors.len());
	for (index, mut vec) in vectors.into_iter().enumerate() {
		if let Some(d) = req.dimensions {
			if d < vec.len() {
				vec.truncate(d);
				l2_normalize(&mut vec);
			}
		}
		let embedding = if base64_mode {
			let mut bytes = Vec::with_capacity(vec.len() * 4);
			for f in &vec {
				bytes.extend_from_slice(&f.to_le_bytes());
			}
			serde_json::Value::String(STANDARD.encode(bytes))
		} else {
			serde_json::to_value(&vec).unwrap_or_default()
		};
		data.push(EmbeddingItem { object: "embedding", index, embedding });
	}
	Ok(Json(EmbeddingsResponse {
		object: "list",
		data,
		model: model.name().to_string(),
		usage: usage(tokens),
	}))
}

fn l2_normalize(v: &mut [f32]) {
	let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
	if norm > 0.0 {
		v.iter_mut().for_each(|x| *x /= norm);
	}
}

/// TEI-compatible: `{"inputs": ["..."]}` -> `[[f32]]`
pub async fn embed_tei(State(state): State<AppState>, Json(req): Json<crate::dto::TeiEmbedRequest>) -> Result<impl IntoResponse, ApiError> {
	let model = state.registry.resolve(None, Kind::Embedding)?;
	if req.inputs.is_empty() {
		return Err(ApiError(ortinfer_core::Error::BadRequest("`inputs` is empty".into())));
	}
	let out = embedding::embed(&model, req.inputs, state.queue_wait).await?;
	Ok(Json(out.vectors))
}
