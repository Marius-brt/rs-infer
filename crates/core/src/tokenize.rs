use std::borrow::Cow;

use ort::{
	session::{Session, SessionInputs},
	value::Tensor,
};
use tokenizers::{EncodeInput, Encoding, PaddingParams, Tokenizer, TruncationParams, TruncationDirection};

use crate::{Error, Result};

/// One tokenized batch, padded to the longest row in the batch.
pub struct Encoded {
	pub input_ids: Vec<Vec<i64>>,
	pub attention_mask: Vec<Vec<i64>>,
	pub token_type_ids: Vec<Vec<i64>>,
	/// Character offsets per token ((0,0) for special/pad tokens).
	pub offsets: Vec<Vec<(usize, usize)>>,
	pub tokens: Vec<Vec<String>>,
	pub batch: usize,
	pub seq: usize,
}

impl Encoded {
	pub fn token_count(&self) -> usize {
		self.attention_mask.iter().map(|r| r.iter().map(|&m| m as usize).sum::<usize>()).sum()
	}
}

pub struct Encoder {
	tokenizer: Tokenizer,
	pub vocab_size: usize,
}

impl Encoder {
	pub fn new(path: &std::path::Path, max_len: Option<usize>) -> Result<Self> {
		let mut tokenizer = Tokenizer::from_file(path).map_err(|e| Error::Tokenize(format!("{}: {e}", path.display())))?;
		tokenizer.with_padding(Some(PaddingParams::default()));
		if let Some(max) = max_len {
			tokenizer
				.with_truncation(Some(TruncationParams {
					max_length: max,
					stride: 0,
					direction: TruncationDirection::Right,
					..Default::default()
				}))
				.map_err(|e| Error::Tokenize(e.to_string()))?;
		}
		let vocab = tokenizer.get_vocab(true).len();
		Ok(Self { tokenizer, vocab_size: vocab })
	}

	pub fn single_token_id(&self, word: &str) -> Option<u32> {
		let enc = self.tokenizer.encode(word, false).ok()?;
		enc.get_ids().first().copied()
	}

	fn from_encodings(encodings: Vec<Encoding>) -> Encoded {
		let mut e = Encoded {
			input_ids: Vec::with_capacity(encodings.len()),
			attention_mask: Vec::with_capacity(encodings.len()),
			token_type_ids: Vec::with_capacity(encodings.len()),
			offsets: Vec::with_capacity(encodings.len()),
			tokens: Vec::with_capacity(encodings.len()),
			batch: encodings.len(),
			seq: encodings.first().map(|x| x.len()).unwrap_or(0),
		};
		for enc in &encodings {
			e.input_ids.push(enc.get_ids().iter().map(|&i| i as i64).collect());
			e.attention_mask.push(enc.get_attention_mask().iter().map(|&m| m as i64).collect());
			e.token_type_ids.push(enc.get_type_ids().iter().map(|&t| t as i64).collect());
			e.offsets.push(enc.get_offsets().to_vec());
			e.tokens.push(enc.get_tokens().to_vec());
		}
		e
	}

	pub fn encode_texts(&self, texts: &[String]) -> Result<Encoded> {
		let inputs: Vec<EncodeInput<'_>> = texts.iter().map(|t| EncodeInput::from(Cow::Borrowed(t.as_str()))).collect();
		let encodings = self
			.tokenizer
			.encode_batch_char_offsets(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		Ok(Self::from_encodings(encodings))
	}

	pub fn encode_pairs(&self, pairs: &[(String, String)]) -> Result<Encoded> {
		let inputs: Vec<EncodeInput<'_>> = pairs
			.iter()
			.map(|(a, b)| EncodeInput::Dual(Cow::Borrowed(a.as_str()).into(), Cow::Borrowed(b.as_str()).into()))
			.collect();
		let encodings = self
			.tokenizer
			.encode_batch_char_offsets(inputs, true)
			.map_err(|e| Error::Tokenize(e.to_string()))?;
		Ok(Self::from_encodings(encodings))
	}
}

/// Builds the token id/mask/type tensors for a session, intersecting the tokenizer
/// outputs with the names the model actually declares as inputs.
pub fn make_inputs(session: &Session, enc: &Encoded) -> Result<SessionInputs<'static, 'static>> {
	let mut flat_ids = Vec::with_capacity(enc.batch * enc.seq);
	for row in &enc.input_ids {
		flat_ids.extend_from_slice(row);
	}
	let shape = vec![enc.batch as i64, enc.seq as i64];

	let mut map: Vec<(Cow<'static, str>, ort::session::SessionInputValue<'static>)> = Vec::with_capacity(3);
	let mut pushed = Vec::with_capacity(3);
	for input in session.inputs() {
		let name = input.name();
		pushed.push(name.to_string());
		let tensor = match name {
			"input_ids" => Tensor::from_array((shape.clone(), std::mem::take(&mut flat_ids)))?,
			"attention_mask" => Tensor::from_array((shape.clone(), enc.attention_mask.iter().flatten().copied().collect::<Vec<i64>>()))?,
			"token_type_ids" => Tensor::from_array((shape.clone(), enc.token_type_ids.iter().flatten().copied().collect::<Vec<i64>>()))?,
			other => {
				return Err(Error::Ort(ort::Error::new(format!(
					"model requires unsupported input '{other}'; required inputs: {pushed:?}"
				))));
			}
		};
		map.push((Cow::Owned(name.to_string()), tensor.into()));
	}
	Ok(SessionInputs::ValueMap(map))
}

/// Builds raw token-id inputs for pre-tokenized inputs (OpenAI numeric token arrays).
pub fn make_token_inputs(session: &Session, token_rows: &[Vec<u32>]) -> Result<SessionInputs<'static, 'static>> {
	let seq = token_rows.iter().map(|r| r.len()).max().unwrap_or(0);
	let mask = token_rows.iter().map(|r| {
		let mut v = vec![1i64; r.len()];
		v.resize(seq, 0);
		v
	});
	let mut ids = Vec::new();
	for r in token_rows {
		let mut v: Vec<i64> = r.iter().map(|&i| i as i64).collect();
		v.resize(seq, 0);
		ids.extend_from_slice(&v);
	}
	let masks: Vec<i64> = mask.flatten().collect();
	let types = vec![0i64; ids.len()];
	let shape = vec![token_rows.len() as i64, seq as i64];

	let mut map: Vec<(Cow<'static, str>, ort::session::SessionInputValue<'static>)> = Vec::new();
	let mut pushed = Vec::new();
	for input in session.inputs() {
		let name = input.name();
		pushed.push(name.to_string());
		let (data, fill): (Vec<i64>, i64) = match name {
			"input_ids" => (ids.clone(), 0),
			"attention_mask" => (masks.clone(), 0),
			"token_type_ids" => (types.clone(), 0),
			other => {
				return Err(Error::Ort(ort::Error::new(format!(
					"model requires unsupported input '{other}'; required inputs: {pushed:?}"
				))));
			}
		};
		let _ = fill;
		map.push((Cow::Owned(name.to_string()), Tensor::from_array((shape.clone(), data))?.into()));
	}
	Ok(SessionInputs::ValueMap(map))
}
