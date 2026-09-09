//! Unit tests for [`pii`](super).

use super::*;

fn labels() -> Vec<String> {
	vec![
		"O".into(),
		"B-person".into(),
		"I-person".into(),
		"B-email".into(),
		"I-email".into(),
		"B-location".into(),
		"I-location".into(),
	]
}

#[test]
fn decodes_bio_spans() {
	let text = "John Smith lives in Berlin";
	let offsets = vec![
		(0, 0),    // [CLS]
		(0, 4),    // John
		(5, 10),   // Smith
		(11, 16),  // lives
		(17, 19),  // in
		(20, 26),  // Berlin
		(0, 0),    // [SEP]
	];
	let row = vec![(0usize, 1.0), (1, 0.99), (2, 0.98), (0, 1.0), (0, 1.0), (5, 0.95), (0, 1.0)];
	let ents = decode_entities(text, &row, &offsets, &labels(), 0.5);
	assert_eq!(ents.len(), 2);
	assert_eq!(ents[0].entity_type, "person");
	assert_eq!(ents[0].text, "John Smith");
	assert_eq!(ents[0].start, 0);
	assert_eq!(ents[0].end, 10);
	assert_eq!(ents[1].entity_type, "location");
	assert_eq!(ents[1].text, "Berlin");
}

#[test]
fn decodes_bioes_single_and_end() {
	let text = "ada@x.io and Berlin";
	let labels: Vec<String> = ["O", "B-email", "I-email", "E-email", "S-phone", "B-location", "E-location"].into_iter().map(String::from).collect();
	let offsets = vec![(0, 0), (0, 3), (3, 4), (4, 8), (0, 0), (13, 19), (0, 0)];
	let row = vec![(0, 1.0), (1, 0.9), (2, 0.9), (3, 0.9), (0, 1.0), (5, 0.9), (0, 1.0)];
	let ents = decode_entities(text, &row, &offsets, &labels, 0.5);
	assert_eq!(ents.len(), 2);
	assert_eq!(ents[0].entity_type, "email");
	assert_eq!(ents[0].text, "ada@x.io");
	assert_eq!(ents[1].entity_type, "location");
	assert_eq!(ents[1].text, "Berlin");
}

#[test]
fn threshold_drops_low_confidence() {
	let text = "John";
	let offsets = vec![(0, 0), (0, 4), (0, 0)];
	let row = vec![(0, 1.0), (1, 0.3), (0, 1.0)];
	let ents = decode_entities(text, &row, &offsets, &labels(), 0.5);
	assert!(ents.is_empty());
}

#[test]
fn redacts_mask_and_remove() {
	let text = "call John Smith now";
	let e = Entity { entity_type: "person".into(), text: "John Smith".into(), score: 0.9, start: 5, end: 15 };
	assert_eq!(redact(text, std::slice::from_ref(&e), RedactMode::Mask, '*'), "call ********** now");
	assert_eq!(redact(text, std::slice::from_ref(&e), RedactMode::Remove, '*'), "call  now");
	let unicode = "héllo Wörld";
	let e2 = Entity { entity_type: "person".into(), text: "Wörld".into(), score: 0.9, start: 6, end: 11 };
	assert_eq!(redact(unicode, &[e2], RedactMode::Mask, '#'), "héllo #####");
}

#[test]
fn overlapping_redaction_applies_once() {
	let text = "abc def";
	let a = Entity { entity_type: "x".into(), text: "abc".into(), score: 0.9, start: 0, end: 3 };
	let b = Entity { entity_type: "y".into(), text: "abcd".into(), score: 0.9, start: 0, end: 4 };
	assert_eq!(redact(text, &[a, b], RedactMode::Mask, '*'), "*** def");
}
