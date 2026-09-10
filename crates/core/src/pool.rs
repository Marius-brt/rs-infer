use std::{
	sync::{
		atomic::{AtomicUsize, Ordering},
		Mutex as StdMutex,
	},
	time::Duration,
};

use ort::session::Session;
use tokio::sync::{Semaphore, SemaphorePermit};

use crate::{Error, Result};

/// A pool of independent ORT sessions for one model.
///
/// `ort::session::Session` is `Send` but not `Sync` (and `run` takes `&mut self`),
/// so concurrency comes from replicas, matching ONNX Runtime's own guidance.
pub struct SessionPool {
	sessions: StdMutex<Vec<Session>>,
	sem: Semaphore,
	replicas: usize,
	max_queue: usize,
	waiting: AtomicUsize,
}

impl SessionPool {
	pub fn new(sessions: Vec<Session>, max_queue: usize) -> Self {
		let replicas = sessions.len();
		Self {
			sessions: StdMutex::new(sessions),
			sem: Semaphore::new(replicas),
			replicas,
			max_queue,
			waiting: AtomicUsize::new(0),
		}
	}

	pub fn replicas(&self) -> usize {
		self.replicas
	}

	/// Sessions not currently checked out (idle or gathered-but-not-started).
	pub fn available(&self) -> usize {
		self.sem.available_permits()
	}

	pub async fn acquire(&self, timeout: Duration) -> Result<Pooled<'_>> {
		if self.sem.available_permits() == 0 && self.waiting.load(Ordering::Relaxed) >= self.max_queue {
			return Err(Error::Saturated);
		}
		self.waiting.fetch_add(1, Ordering::SeqCst);
		struct Guard<'a>(&'a SessionPool);
		impl Drop for Guard<'_> {
			fn drop(&mut self) {
				self.0.waiting.fetch_sub(1, Ordering::SeqCst);
			}
		}
		let guard = Guard(self);
		let permit = tokio::time::timeout(timeout, self.sem.acquire())
			.await
			.map_err(|_| Error::PoolTimeout)?
			.map_err(|_| Error::Saturated)?;
		let session = self
			.sessions
			.lock()
			.expect("session pool poisoned")
			.pop()
			.ok_or(Error::Saturated)?;
		drop(guard);
		Ok(Pooled {
			session: Some(session),
			pool: self,
			_permit: permit,
		})
	}
}

pub struct Pooled<'a> {
	session: Option<Session>,
	pool: &'a SessionPool,
	_permit: SemaphorePermit<'a>,
}

impl Pooled<'_> {
	pub fn get_mut(&mut self) -> &mut Session {
		self.session.as_mut().expect("session taken")
	}

	/// Runs a blocking inference closure with the session off the async runtime.
	pub async fn run_blocking<T, F>(mut self, f: F) -> Result<T>
	where
		T: Send + 'static,
		F: FnOnce(&mut Session) -> Result<T> + Send + 'static,
	{
		let mut session = self.session.take().expect("session taken");
		let (session, out) = tokio::task::spawn_blocking(move || {
			let out = f(&mut session);
			(session, out)
		})
		.await
		.map_err(|e| Error::Ort(ort::Error::new(format!("inference task panicked: {e}"))))?;
		pool_session(self.pool, session);
		out
	}
}

impl Drop for Pooled<'_> {
	fn drop(&mut self) {
		if let Some(session) = self.session.take() {
			pool_session(self.pool, session);
		}
	}
}

fn pool_session(pool: &SessionPool, session: Session) {
	if let Ok(mut guard) = pool.sessions.lock() {
		guard.push(session);
	}
}
