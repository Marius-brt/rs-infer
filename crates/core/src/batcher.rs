//! Opt-in cross-request dynamic batching for embedding models.
//!
//! Without a batcher each HTTP request tokenizes, pads and forwards its own batch.
//! With `batching:` configured, requests enqueue *rows* (one per text) into a shared
//! queue; a single gatherer task packs rows into batches and hands each batch to one
//! of the per-replica forward workers. Useful when a forward has fixed cost that
//! amortizes across rows (GPU backends, large per-run overhead, heavy padding
//! waste); strictly opt-in because on backends whose cost scales linearly with
//! rows the gain is small and p99 tails grow (a request's rows can span batches).

use std::{
	sync::{
		atomic::{AtomicBool, Ordering},
		Arc, Mutex as StdMutex,
	},
	time::Duration,
};

use tokio::sync::{mpsc, oneshot};

use crate::{
	config::{Batching, Pooling},
	model::OutSel,
	pipeline::{embedding::pool_rows, run_forward},
	pool::SessionPool,
	tokenize::make_token_inputs,
	Error, Result,
};

struct Unit {
	ids: Vec<u32>,
	timeout: Duration,
	reply: oneshot::Sender<Result<Vec<f32>>>,
}

/// Everything a worker needs; derived from `Meta::Embedding` at load time.
pub struct EmbedMeta {
	pub pooling: Pooling,
	pub output: OutSel,
	pub normalize: bool,
	pub dimensions: Option<usize>,
}

pub struct EmbedBatcher {
	queue_tx: mpsc::Sender<Unit>,
	queue_rx: StdMutex<Option<mpsc::Receiver<Unit>>>,
	batch_tx: mpsc::Sender<Vec<Unit>>,
	batch_rx: StdMutex<Option<mpsc::Receiver<Vec<Unit>>>>,
	pool: Arc<SessionPool>,
	meta: Arc<EmbedMeta>,
	settings: Batching,
	started: AtomicBool,
}

impl EmbedBatcher {
	/// Creates channels but spawns nothing; call [`EmbedBatcher::start`] from within
	/// the tokio runtime (models are loaded on a blocking thread otherwise).
	pub(crate) fn new(pool: Arc<SessionPool>, meta: EmbedMeta, settings: Batching) -> Arc<Self> {
		let queue = settings.queue_rows.max(settings.max_rows).max(1);
		let (queue_tx, queue_rx) = mpsc::channel(queue);
		let (batch_tx, batch_rx) = mpsc::channel(pool.replicas().max(1));
		Arc::new(Self {
			queue_tx,
			queue_rx: StdMutex::new(Some(queue_rx)),
			batch_tx,
			batch_rx: StdMutex::new(Some(batch_rx)),
			pool,
			meta: Arc::new(meta),
			settings,
			started: AtomicBool::new(false),
		})
	}

	/// Spawns the gatherer plus one forward worker per session replica. Idempotent.
	pub fn start(self: &Arc<Self>) {
		if self.started.swap(true, Ordering::SeqCst) {
			return;
		}
		let Some(queue_rx) = self.queue_rx.lock().expect("batcher poisoned").take() else {
			return;
		};
		let Some(batch_rx) = self.batch_rx.lock().expect("batcher poisoned").take() else {
			return;
		};
		tokio::spawn(gatherer(Arc::clone(self), queue_rx));
		let batch_rx = Arc::new(tokio::sync::Mutex::new(batch_rx));
		for _ in 0..self.pool.replicas() {
			tokio::spawn(forwarder(Arc::clone(self), batch_rx.clone()));
		}
	}

	/// Submits token rows (one per text, unpadded) and awaits one vector per row,
	/// in order. Fails with Saturated when the queue is full.
	pub(crate) async fn submit(&self, rows: Vec<Vec<u32>>, timeout: Duration) -> Result<Vec<Vec<f32>>> {
		let mut rxs = Vec::with_capacity(rows.len());
		for ids in rows {
			let (reply_tx, reply_rx) = oneshot::channel();
			self.queue_tx.try_send(Unit { ids, timeout, reply: reply_tx }).map_err(|_| Error::Saturated)?;
			rxs.push(reply_rx);
		}
		let mut out = Vec::with_capacity(rxs.len());
		for reply in rxs {
			out.push(await_reply(reply).await?);
		}
		Ok(out)
	}
}

async fn await_reply(reply: oneshot::Receiver<Result<Vec<f32>>>) -> Result<Vec<f32>> {
	match reply.await {
		Ok(inner) => inner,
		Err(_) => Err(Error::Saturated),
	}
}

async fn gatherer(batcher: Arc<EmbedBatcher>, mut queue_rx: mpsc::Receiver<Unit>) {
	let s = batcher.settings;
	while let Some(first) = queue_rx.recv().await {
		let mut units = vec![first];
		let mut tokens = units[0].ids.len();
		// Fair share: spread the current backlog over the sessions that can start
		// right now, so no replica idles while rows wait. A lone arrival becomes
		// share=1 and is dispatched with zero added latency. Waiting on an empty
		// queue is never useful here: rows that do not exist yet cannot be
		// predicted, and the per-forward fixed cost is tiny compared with a
		// parked batch's tail latency.
		let idle = batcher.pool.available().max(1);
		let share = (queue_rx.len() + 1).div_ceil(idle).clamp(1, s.max_rows);
		while units.len() < share && tokens < s.max_tokens {
			match queue_rx.try_recv() {
				Ok(u) => {
					tokens += u.ids.len();
					units.push(u);
				}
				Err(_) => break,
			}
		}
		if batcher.batch_tx.send(units).await.is_err() {
			return;
		}
	}
}

async fn forwarder(batcher: Arc<EmbedBatcher>, batch_rx: Arc<tokio::sync::Mutex<mpsc::Receiver<Vec<Unit>>>>) {
	loop {
		// Lock held only around recv() so waiting for a batch never blocks other
		// workers' forwards.
		let next = {
			let mut rx = batch_rx.lock().await;
			rx.recv().await
		};
		match next {
			Some(units) => run_batch(&batcher, units).await,
			None => return,
		}
	}
}

async fn run_batch(batcher: &Arc<EmbedBatcher>, units: Vec<Unit>) {
	let timeout = units.iter().map(|u| u.timeout).min().unwrap_or(Duration::from_secs(30));
	let tokens = units.iter().map(|u| u.ids.len()).sum::<usize>();
	let (senders, rows): (Vec<_>, Vec<_>) = units.into_iter().map(|u| (u.reply, u.ids)).unzip();
	let outcome = match batcher.pool.acquire(timeout).await {
		Ok(pooled) => {
			let meta = Arc::clone(&batcher.meta);
			pooled
				.run_blocking(move |session| -> Result<Vec<Vec<f32>>> {
					let inputs = make_token_inputs(session, &rows)?;
					let fwd = run_forward(session, inputs, &meta.output)?;
					let seq = rows.iter().map(|r| r.len()).max().unwrap_or(0);
					let attn: Vec<Vec<i64>> = rows
						.iter()
						.map(|r| {
							let mut v = vec![1i64; r.len()];
							v.resize(seq, 0);
							v
						})
						.collect();
					let mut out = pool_rows(&fwd, meta.pooling, &attn)?;
					for vec in &mut out {
						if let Some(d) = meta.dimensions {
							if d < vec.len() {
								vec.truncate(d);
							}
						}
						if meta.normalize {
							crate::pipeline::embedding::l2_normalize(vec);
						}
					}
					Ok(out)
				})
				.await
		}
		Err(e) => Err(e),
	};
	match outcome {
		Ok(vectors) => {
			tracing::trace!(rows = vectors.len(), tokens, "dynamic batch done");
			for (sender, vector) in senders.into_iter().zip(vectors) {
				let _ = sender.send(Ok(vector));
			}
		}
		Err(e) => {
			let msg = e.to_string();
			for sender in senders {
				let _ = sender.send(Err(Error::Ort(ort::Error::new(msg.clone()))));
			}
		}
	}
}
