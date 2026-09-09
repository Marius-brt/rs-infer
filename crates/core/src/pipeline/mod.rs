//! Pipelines: turn text/tokens into typed predictions through ORT sessions.

pub mod embedding;
pub mod pii;
pub mod rerank;
pub mod zeroshot;

use ort::session::{Session, SessionOutputs};

use crate::model::OutSel;

pub(crate) struct Fwd {
	pub shape: Vec<usize>,
	pub data: Vec<f32>,
}

/// Runs a forward pass on a pooled session and extracts the selected output as f32.
pub(crate) fn run_forward(
	session: &mut Session,
	inputs: ort::session::SessionInputs<'static, 'static>,
	sel: &OutSel,
) -> crate::Result<Fwd> {
	let names: Vec<String> = session.outputs().iter().map(|o| o.name().to_string()).collect();
	let outputs: SessionOutputs = session.run(inputs)?;
	let value = outputs
		.get(&sel.0)
		.ok_or_else(|| crate::Error::Ort(ort::Error::new(format!("output '{}' not found; model has {names:?}", sel.0))))?;
	let view = value.try_extract_array::<f32>()?;
	let owned = view.to_owned();
	Ok(Fwd {
		shape: owned.shape().to_vec(),
		data: owned.into_raw_vec_and_offset().0,
	})
}

pub(crate) fn softmax(logits: &[f32]) -> Vec<f64> {
	let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
	let exps: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
	let sum: f32 = exps.iter().sum();
	exps.iter().map(|e| (e / sum) as f64).collect()
}
