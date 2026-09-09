use axum::{extract::State, Json};
use rsinfer_core::{
	Kind,
	pipeline::zeroshot,
};

use super::usage;
use crate::{
	dto::{ClassifyRequest, ClassifyResponseBody, ClassifyResult, TrueFalseRequest, TrueFalseResponseBody, TrueFalseResult},
	error::ApiError,
	state::AppState,
};

/// HF-pipeline-like zero-shot classification over candidate labels.
pub async fn classify_h(State(state): State<AppState>, Json(req): Json<ClassifyRequest>) -> Result<Json<ClassifyResponseBody>, ApiError> {
	let model = state.registry.resolve(req.model.as_deref(), Kind::Zeroshot)?;
	if req.candidate_labels.is_empty() {
		return Err(ApiError(rsinfer_core::Error::BadRequest("`candidate_labels` is empty".into())));
	}
	let single = matches!(req.input, crate::dto::TextList::One(_));
	let texts = req.input.into_vec();
	let out = zeroshot::classify(&model, texts, req.candidate_labels, req.multi_label.unwrap_or(false), state.queue_wait).await?;
	let mk = |o: zeroshot::ClassOutcome| ClassifyResult { labels: o.labels, scores: o.scores };
	Ok(Json(if single {
		let first = mk(out.outcomes.into_iter().next().expect("at least one input"));
		ClassifyResponseBody::Single { object: "classification", model: model.name().to_string(), result: first, usage: usage(out.tokens) }
	} else {
		ClassifyResponseBody::Multiple {
			object: "classification.list",
			model: model.name().to_string(),
			results: out.outcomes.into_iter().map(mk).collect(),
			usage: usage(out.tokens),
		}
	}))
}

/// True/false judgement: `question` is checked against each `input` via NLI
/// entailment of the hypotheses "True."/"False.".
pub async fn true_false_h(State(state): State<AppState>, Json(req): Json<TrueFalseRequest>) -> Result<Json<TrueFalseResponseBody>, ApiError> {
	let model = state.registry.resolve(req.model.as_deref(), Kind::Zeroshot)?;
	let inputs = req.input.into_vec();
	if inputs.is_empty() {
		return Err(ApiError(rsinfer_core::Error::BadRequest("`input` is empty".into())));
	}
	if req.question.as_deref().map(str::trim) == Some("") || (req.question.is_none() && req.assertion.is_none()) {
		return Err(ApiError(rsinfer_core::Error::BadRequest("provide `question` or `assertion`".into())));
	}
	let threshold = req.threshold.unwrap_or(0.5).clamp(0.0, 1.0);
	let out = zeroshot::classify_true_false(&model, inputs, req.question, req.assertion, state.queue_wait).await?;
	let results: Vec<TrueFalseResult> = out
		.probabilities
		.iter()
		.zip(&out.assertions)
		.map(|(p, a)| TrueFalseResult {
			is_true: *p >= threshold,
			label: if *p >= threshold { "true" } else { "false" },
			true_probability: *p,
			assertion: a.clone(),
		})
		.collect();
	Ok(Json(if out.probabilities.len() == 1 {
		TrueFalseResponseBody::Single {
			object: "classification.true_false",
			model: model.name().to_string(),
			result: results.into_iter().next().expect("one"),
			usage: usage(out.tokens),
		}
	} else {
		TrueFalseResponseBody::Multiple {
			object: "classification.true_false.list",
			model: model.name().to_string(),
			results,
			usage: usage(out.tokens),
		}
	}))
}
