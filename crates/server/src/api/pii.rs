use axum::{extract::State, Json};
use ortinfer_core::{
	Kind,
	pipeline::pii,
};

use super::usage;
use crate::{
	dto::{EntityOut, PiiDetectResponse, PiiDetectResult, PiiRedactResponse, PiiRedactResult, PiiRequest, PiiRedactRequest},
	error::ApiError,
	state::AppState,
};

pub async fn pii_detect_h(State(state): State<AppState>, Json(req): Json<PiiRequest>) -> Result<Json<PiiDetectResponse>, ApiError> {
	let texts = texts_or_400(&req)?;
	let single = req.text.is_some();
	let model = state.registry.resolve(req.model.as_deref(), Kind::Pii)?;
	let out = pii::detect(&model, texts.clone(), req.threshold, state.queue_wait).await?;
	let results = out
		.entities
		.into_iter()
		.enumerate()
		.map(|(i, ents)| {
			let entities: Vec<EntityOut> = ents
				.into_iter()
				.filter(|e| req.types.as_ref().is_none_or(|t| t.contains(&e.entity_type)))
				.map(EntityOut::from)
				.collect();
			PiiDetectResult {
				entities,
				text: (!single).then(|| texts[i].clone()),
			}
		})
		.collect();
	Ok(Json(PiiDetectResponse {
		object: "pii.detection",
		model: model.name().to_string(),
		results,
		usage: usage(out.tokens),
	}))
}

pub async fn pii_redact_h(State(state): State<AppState>, Json(req): Json<PiiRedactRequest>) -> Result<Json<PiiRedactResponse>, ApiError> {
	let mode = match req.mode.as_deref().unwrap_or("mask") {
		"remove" => pii::RedactMode::Remove,
		"mask" | "" => pii::RedactMode::Mask,
		other => return Err(ApiError(ortinfer_core::Error::BadRequest(format!("unknown redact mode '{other}'")))),
	};
	let mask_char = match req.mask_char.as_deref().unwrap_or("*").chars().collect::<Vec<char>>().as_slice() {
		[c] => *c,
		_ => return Err(ApiError(ortinfer_core::Error::BadRequest("`mask_char` must be a single character".into()))),
	};

	let texts = texts_or_400(&req.detect)?;
	let model = state.registry.resolve(req.detect.model.as_deref(), Kind::Pii)?;
	let out = pii::detect(&model, texts.clone(), req.detect.threshold, state.queue_wait).await?;
	let results = out
		.entities
		.iter()
		.zip(texts.iter())
		.map(|(ents, text)| PiiRedactResult {
			text: pii::redact(text, ents, mode, mask_char),
			entities: ents.iter().cloned().map(EntityOut::from).collect(),
		})
		.collect();
	Ok(Json(PiiRedactResponse {
		object: "pii.redaction",
		model: model.name().to_string(),
		results,
		usage: usage(out.tokens),
	}))
}

fn texts_or_400(req: &PiiRequest) -> Result<Vec<String>, ApiError> {
	match (req.text.clone(), req.texts.clone()) {
		(Some(t), None) => Ok(vec![t]),
		(None, Some(v)) if !v.is_empty() => Ok(v),
		_ => Err(ApiError(ortinfer_core::Error::BadRequest("provide exactly one of `text` or non-empty `texts`".into()))),
	}
}
