use flay::eval::refusal::REFUSAL_MARKERS;
use flay::eval::reasoning::CANARY_QUESTIONS;
use flay::eval::{DomainRefusalRate, EvalResults, OverRefusalRate, RefusalRate, ReasoningResult};

#[test]
fn test_refusal_markers_are_lowercase() {
    for marker in REFUSAL_MARKERS {
        assert_eq!(
            *marker,
            marker.to_lowercase(),
            "Refusal marker should be lowercase for case-insensitive matching: {}",
            marker,
        );
    }
}

#[test]
fn test_refusal_markers_non_empty() {
    assert!(
        !REFUSAL_MARKERS.is_empty(),
        "Should have at least one refusal marker"
    );
    for marker in REFUSAL_MARKERS {
        assert!(!marker.is_empty(), "Refusal markers should not be empty");
    }
}

#[test]
fn test_canary_questions_have_answers() {
    assert!(
        !CANARY_QUESTIONS.is_empty(),
        "Should have at least one canary question"
    );
    for (question, answer) in CANARY_QUESTIONS {
        assert!(!question.is_empty(), "Canary question should not be empty");
        assert!(!answer.is_empty(), "Canary answer should not be empty");
    }
}

#[test]
fn test_canary_answers_are_non_empty() {
    for (question, answer) in CANARY_QUESTIONS {
        assert!(
            !answer.is_empty(),
            "Canary answer for question '{}' should not be empty",
            question,
        );
        // Answers must be regex-safe (used with \b word boundaries)
        assert!(
            regex::Regex::new(&format!(r"\b{}\b", regex::escape(answer))).is_ok(),
            "Canary answer '{}' for question '{}' should be valid in a regex pattern",
            answer,
            question,
        );
    }
}

#[test]
fn test_refusal_rate_calculation() {
    let rate = RefusalRate {
        refused: 3,
        total: 10,
        rate: 3.0 / 10.0,
    };
    assert_eq!(rate.refused, 3);
    assert_eq!(rate.total, 10);
    assert!((rate.rate - 0.3).abs() < f32::EPSILON);
}

#[test]
fn test_domain_refusal_rate_zero_total() {
    let rate = DomainRefusalRate {
        domain: "test".to_string(),
        refused: 0,
        total: 0,
        rate: 0.0,
    };
    assert_eq!(rate.rate, 0.0);
}

#[test]
fn test_eval_results_serialization() {
    let results = EvalResults {
        refusal_rate: RefusalRate {
            refused: 2,
            total: 10,
            rate: 0.2,
        },
        kl_divergence: 0.001,
        reasoning_canary: Some(ReasoningResult {
            passed: 8,
            total: 10,
        }),
        domain_refusal: vec![DomainRefusalRate {
            domain: "cybersecurity".to_string(),
            refused: 1,
            total: 5,
            rate: 0.2,
        }],
        over_refusal: Some(OverRefusalRate {
            refused: 1,
            total: 100,
            rate: 0.01,
        }),
        utility: None,
    };
    let json = serde_json::to_string(&results).expect("Should serialize");
    assert!(json.contains("refusal_rate"));
    assert!(json.contains("reasoning_canary"));
    assert!(json.contains("domain_refusal"));
    assert!(json.contains("cybersecurity"));
}

#[test]
fn test_eval_results_optional_canary() {
    let results = EvalResults {
        refusal_rate: RefusalRate {
            refused: 0,
            total: 5,
            rate: 0.0,
        },
        kl_divergence: 0.0,
        reasoning_canary: None,
        domain_refusal: vec![],
        over_refusal: None,
        utility: None,
    };
    let json = serde_json::to_string(&results).expect("Should serialize");
    assert!(json.contains("\"reasoning_canary\":null"));
}
