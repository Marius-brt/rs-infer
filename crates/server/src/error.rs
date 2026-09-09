use axum::{
	http::StatusCode,
	response::{IntoResponse, Response},
	Json,
};

#[derive(Debug)]
pub struct ApiError(pub rsinfer_core::Error);

impl From<rsinfer_core::Error> for ApiError {
	fn from(e: rsinfer_core::Error) -> Self {
		Self(e)
	}
}

impl From<anyhow::Error> for ApiError {
	fn from(e: anyhow::Error) -> Self {
		Self(rsinfer_core::Error::Config(format!("{e:#}")))
	}
}

impl IntoResponse for ApiError {
	fn into_response(self) -> Response {
		let status = StatusCode::from_u16(self.0.status_code()).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
		if status.is_server_error() {
			tracing::error!(error = %self.0, "request failed");
		} else {
			tracing::debug!(error = %self.0, "request rejected");
		}
		let body = serde_json::json!({
			"error": {
				"message": self.0.to_string(),
				"type": self.0.kind_str(),
				"code": status.as_u16(),
			}
		});
		(status, Json(body)).into_response()
	}
}
