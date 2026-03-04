pub mod domain;
pub mod generate;
pub mod reasoning;
pub mod refusal;
pub mod utility;

use serde::Serialize;

/// Combined evaluation results.
#[derive(Debug, Clone, Serialize)]
pub struct EvalResults {
    pub refusal_rate: RefusalRate,
    pub kl_divergence: f32,
    pub reasoning_canary: Option<ReasoningResult>,
    pub domain_refusal: Vec<DomainRefusalRate>,
    pub over_refusal: Option<OverRefusalRate>,
    pub utility: Option<utility::UtilityResults>,
}

#[derive(Debug, Clone, Serialize)]
pub struct OverRefusalRate {
    pub refused: usize,
    pub total: usize,
    pub rate: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct RefusalRate {
    pub refused: usize,
    pub total: usize,
    pub rate: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct ReasoningResult {
    pub passed: usize,
    pub total: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct DomainRefusalRate {
    pub domain: String,
    pub refused: usize,
    pub total: usize,
    pub rate: f32,
}
