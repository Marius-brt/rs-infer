use thiserror::Error;

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Error)]
pub enum Error {
	#[error("io error: {0}")]
	Io(#[from] std::io::Error),

	#[error("config error: {0}")]
	Config(String),

	#[error("hub error: {0}")]
	Hub(String),

	#[error("tokenizer error: {0}")]
	Tokenize(String),

	#[error("ort error: {0}")]
	Ort(#[from] ort::Error),

	#[error("model '{0}' not found")]
	ModelNotFound(String),

	#[error("model '{name}' is a {actual} model but was used for a {expected} request")]
	KindMismatch { name: String, expected: &'static str, actual: &'static str },

	#[error("model '{0}' has no required file {1}")]
	MissingFile(String, String),

	#[error("inference pool saturated; retry later")]
	Saturated,

	#[error("timed out waiting for a session slot")]
	PoolTimeout,

	#[error("bad request: {0}")]
	BadRequest(String),

	#[error("model produced unexpected output shape {0:?}")]
	BadOutputShape(Vec<usize>),
}

impl Error {
	pub fn status_code(&self) -> u16 {
		match self {
			Error::BadRequest(_) | Error::KindMismatch { .. } => 400,
			Error::ModelNotFound(_) => 404,
			Error::Saturated => 429,
			Error::PoolTimeout => 503,
			_ => 500,
		}
	}

	pub fn kind_str(&self) -> &'static str {
		use Error::*;
		match self {
			Io(_) | Hub(_) | MissingFile(_, _) => "internal",
			Config(_) => "config",
			Tokenize(_) | BadRequest(_) => "invalid_request_error",
			Ort(_) | BadOutputShape(_) => "server_error",
			ModelNotFound(_) => "model_not_found",
			KindMismatch { .. } => "invalid_request_error",
			Saturated => "rate_limit_error",
			PoolTimeout => "timeout_error",
		}
	}
}
