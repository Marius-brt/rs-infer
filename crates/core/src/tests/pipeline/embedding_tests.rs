//! Unit tests for [`embedding`](super).

use super::*;
use crate::pipeline::Fwd;

fn rank3() -> Fwd {
	// [B=1, T=3, D=2]: tokens [a,b,pad]
	Fwd { shape: vec![1, 3, 2], data: vec![1.0, 2.0, 3.0, 4.0, 100.0, 100.0] }
}

#[test]
fn mean_pooling_respects_mask() {
	let attn = vec![vec![1i64, 1, 0]];
	let rows = pool_embeddings(&rank3(), Pooling::Mean, &attn).unwrap();
	assert_eq!(rows[0], vec![2.0, 3.0]);
}

#[test]
fn cls_and_last_pooling() {
	let attn = vec![vec![1i64, 1, 0]];
	let cls = pool_embeddings(&rank3(), Pooling::Cls, &attn).unwrap();
	assert_eq!(cls[0], vec![1.0, 2.0]);
	let last = pool_embeddings(&rank3(), Pooling::Last, &attn).unwrap();
	assert_eq!(last[0], vec![3.0, 4.0]); // last non-pad token
}

#[test]
fn rank2_passthrough_and_normalize() {
	let fwd = Fwd { shape: vec![2, 2], data: vec![3.0, 4.0, 0.0, 5.0] };
	let rows = pool_embeddings(&fwd, Pooling::Mean, &[vec![1], vec![1]]).unwrap();
	let mut v0 = rows[0].clone();
	l2_normalize(&mut v0);
	assert!((v0[0] - 0.6).abs() < 1e-6);
	assert!((v0[1] - 0.8).abs() < 1e-6);
}
