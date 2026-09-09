//! Unit tests for [`hub`](super).

use super::*;

fn s(list: &[&str]) -> Vec<String> {
	list.iter().map(|x| x.to_string()).collect()
}

#[test]
fn prefers_root_fp32() {
	let sib = s(&["model.onnx", "onnx/model.onnx", "tokenizer.json", "config.json"]);
	assert_eq!(select_remote(&sib, None, MODEL_CANDIDATES).unwrap(), "model.onnx");
}

#[test]
fn falls_back_to_onnx_subfolder() {
	let sib = s(&["onnx/model.onnx", "onnx/model.onnx_data", "tokenizer.json"]);
	assert_eq!(select_remote(&sib, None, MODEL_CANDIDATES).unwrap(), "onnx/model.onnx");
}

#[test]
fn quantized_only_repo_picks_fp16_then_quantized() {
	// e.g. onnx-community repos that only ship quantized graphs
	let sib = s(&["onnx/model_q4.onnx", "onnx/model_quantized.onnx"]);
	assert_eq!(select_remote(&sib, None, MODEL_CANDIDATES).unwrap(), "onnx/model_quantized.onnx");
}

#[test]
fn explicit_file_wins() {
	let sib = s(&["onnx/model_q4.onnx", "onnx/model_int8.onnx"]);
	let cfg = crate::config::ModelConfig::default_for_test(Some("onnx/model_int8.onnx"));
	assert_eq!(select_remote(&sib, cfg.subfolder.as_deref(), &candidates_for(&cfg, MODEL_CANDIDATES)).unwrap(), "onnx/model_int8.onnx");
}
