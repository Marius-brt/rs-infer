use std::{sync::Arc, time::Duration};

use crate::{
	config::Pooling,
	model::{Meta, OutSel},
	pipeline::{run_forward, Fwd},
	tokenize::{make_inputs, Encoded},
	Error, LoadedModel, Result,
};

#[derive(Debug)]
pub struct EmbedOutput {
	pub vectors: Vec<Vec<f32>>,
	pub tokens: usize,
}

/// Produces one embedding vector per input text, in request order.
pub async fn embed(model: &Arc<LoadedModel>, texts: Vec<String>, queue_wait: Duration) -> Result<EmbedOutput> {
	let Meta::Embedding { pooling, output, normalize, dimensions } = &model.meta else {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "embedding",
			actual: model.kind().as_str(),
		});
	};
	let (pooling, output, normalize, dimensions) = (*pooling, output.clone(), *normalize, *dimensions);

	let enc = model.encoder.encode_texts(&texts)?;
	let token_count = enc.token_count();
	let attn = enc.attention_mask.clone();

	let pooled = model.pool.acquire(queue_wait).await?;
	let vectors = pooled
		.run_blocking(move |session| -> Result<Vec<Vec<f32>>> {
			let fwd = forward(session, &enc, &output)?;
			pool_embeddings(&fwd, pooling, &attn)
		})
		.await?;

	let n = vectors.len();
	let mut out = Vec::with_capacity(n);
	for mut vec in vectors {
		if let Some(d) = dimensions {
			if d < vec.len() {
				vec.truncate(d);
			}
		}
		if normalize {
			l2_normalize(&mut vec);
		}
		out.push(vec);
	}
	tracing::trace!(model = model.name(), rows = n, tokens = token_count, "embedding done");
	Ok(EmbedOutput { vectors: out, tokens: token_count })
}

/// Embeddings from raw token id rows (OpenAI `input` as token arrays).
pub async fn embed_tokens(model: &Arc<LoadedModel>, rows: Vec<Vec<u32>>, queue_wait: Duration) -> Result<EmbedOutput> {
	let Meta::Embedding { pooling, output, normalize, dimensions } = &model.meta else {
		return Err(Error::KindMismatch {
			name: model.name().into(),
			expected: "embedding",
			actual: model.kind().as_str(),
		});
	};
	let (pooling, output, normalize, dimensions) = (*pooling, output.clone(), *normalize, *dimensions);
	if rows.is_empty() {
		return Ok(EmbedOutput { vectors: Vec::new(), tokens: 0 });
	}
	let tokens: usize = rows.iter().map(|r| r.len()).sum();

	let pooled = model.pool.acquire(queue_wait).await?;
	let vectors = pooled
		.run_blocking(move |session| -> Result<Vec<Vec<f32>>> {
			let inputs = crate::tokenize::make_token_inputs(session, &rows)?;
			let fwd = run_forward(session, inputs, &output)?;
			let attn: Vec<Vec<i64>> = rows.iter().map(|r| r.iter().map(|_| 1i64).collect()).collect();
			pool_embeddings(&fwd, pooling, &attn)
		})
		.await?;

	let mut out = Vec::with_capacity(vectors.len());
	for mut vec in vectors {
		if let Some(d) = dimensions {
			if d < vec.len() {
				vec.truncate(d);
			}
		}
		if normalize {
			l2_normalize(&mut vec);
		}
		out.push(vec);
	}
	Ok(EmbedOutput { vectors: out, tokens })
}

pub(crate) fn forward(session: &mut ort::session::Session, enc: &Encoded, output: &OutSel) -> Result<Fwd> {
	let inputs = make_inputs(session, enc)?;
	run_forward(session, inputs, output)
}

/// Convert a rank-3 tensor [B,T,D] (or rank-2 [B,D]) into [B,D] rows.
fn pool_embeddings(fwd: &Fwd, pooling: Pooling, attn: &[Vec<i64>]) -> Result<Vec<Vec<f32>>> {
	match fwd.shape.as_slice() {
		[b, d] => Ok((0..*b).map(|i| fwd.data[i * d..(i + 1) * d].to_vec()).collect()),
		[b, t, d] => {
			let (b, t, d) = (*b, *t, *d);
			let mut out = Vec::with_capacity(b);
			for i in 0..b {
				let base = &fwd.data[i * t * d..(i + 1) * t * d];
				let mask = attn.get(i).map(|m| m.as_slice()).unwrap_or(&[]);
				let row = match pooling {
					Pooling::Cls | Pooling::Auto => base[..d].to_vec(),
					Pooling::Mean => {
						let mut acc = vec![0f32; d];
						let mut n = 0f32;
						for (step, &token) in mask.iter().enumerate().take(t) {
							if token == 0 {
								continue;
							}
							n += 1.0;
							let off = step * d;
							for j in 0..d {
								acc[j] += base[off + j];
							}
						}
						if n == 0.0 {
							n = 1.0;
						}
						acc.iter_mut().for_each(|v| *v /= n);
						acc
					}
					Pooling::Last => {
						let mut last = 0usize;
						for (step, &token) in mask.iter().enumerate().take(t) {
							if token != 0 {
								last = step;
							}
						}
						base[last * d..(last + 1) * d].to_vec()
					}
				};
				out.push(row);
			}
			Ok(out)
		}
		other => Err(Error::BadOutputShape(other.to_vec())),
	}
}

fn l2_normalize(v: &mut [f32]) {
	let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
	if norm > 0.0 {
		v.iter_mut().for_each(|x| *x /= norm);
	}
}

#[cfg(test)]
#[path = "../tests/pipeline/embedding_tests.rs"]
mod tests;
