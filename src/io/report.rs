// Analysis output: per-expert scores, abliteration decisions, KL divergence
// JSON + markdown report generation + auto model card

use std::path::Path;

use anyhow::{Context, Result};
use serde::Serialize;

use crate::abliterate::orthogonalize::AbliteratedWeight;
use crate::abliterate::scoring::ExpertScore;
use crate::abliterate::weight_key::WeightKey;

/// Complete report of a flay abliteration run.
#[derive(Debug, Serialize)]
pub struct FlayReport {
    /// Model identifier (HuggingFace ID or local path).
    pub model_id: String,
    /// HuggingFace revision (branch, tag, or commit hash) used to fetch the model.
    /// `None` for local models.
    pub revision: Option<String>,
    /// Total number of MoE experts in the model.
    pub total_experts: usize,
    /// Number of MoE experts that were abliterated.
    pub abliterated_experts: usize,
    /// Number of o_proj layers abliterated (residual-stream mode).
    pub abliterated_o_proj_layers: usize,
    /// Score threshold used for expert selection.
    pub threshold: f32,
    /// How the threshold was determined: "manual" or "auto (elbow method)".
    pub threshold_method: String,
    /// Abliteration mode used: "single", "multi", "projected", "multi-projected".
    pub abliteration_mode: String,
    /// KL divergence between original and abliterated model (if computed).
    pub kl_divergence: Option<f32>,
    /// Per-expert refusal scores, sorted descending by combined score.
    pub expert_scores: Vec<ExpertScoreEntry>,
    /// Details for each abliterated expert.
    pub abliteration_details: Vec<AbliterationDetail>,
    /// Evaluation suite results (if --eval was used).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eval: Option<crate::eval::EvalResults>,
    /// Reproducibility metadata for fully reproducible runs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reproducibility: Option<ReproducibilityMetadata>,
}

/// A single expert's refusal score for serialization.
#[derive(Debug, Serialize)]
pub struct ExpertScoreEntry {
    pub layer: usize,
    pub expert: usize,
    pub refusal_projection: f32,
    pub routing_bias: f32,
    pub combined_score: f32,
    pub abliterated: bool,
}

/// Details about a single abliterated weight (expert down_proj or attention o_proj).
#[derive(Debug, Serialize)]
pub struct AbliterationDetail {
    pub layer: usize,
    pub expert: Option<usize>,
    pub strength: f32,
    pub direction_source: String,
}

/// Metadata for fully reproducible evaluation runs.
#[derive(Debug, Serialize)]
pub struct ReproducibilityMetadata {
    /// Short git commit hash of the flay tool (if run from a git repo).
    /// Includes "-dirty" suffix when uncommitted changes are present.
    pub git_commit: Option<String>,
    /// HuggingFace model revision used.
    pub model_revision: Option<String>,
    /// Stable hash of the concatenated prompt content (FNV-1a, cross-platform stable).
    pub prompt_hash: String,
    /// Generation strategy used for eval ("greedy" — argmax, no sampling).
    pub generation_strategy: String,
}

impl FlayReport {
    /// Build a report from scoring and abliteration results.
    pub fn new(
        model_id: String,
        revision: Option<String>,
        total_experts: usize,
        scores: &[ExpertScore],
        abliterated: &[AbliteratedWeight],
        threshold: f32,
        threshold_method: String,
        kl_divergence: Option<f32>,
        abliteration_mode: String,
        reproducibility: Option<ReproducibilityMetadata>,
    ) -> Self {
        // Build a set of abliterated (layer, expert) pairs for cross-referencing
        let abliterated_set: std::collections::HashSet<(usize, usize)> = abliterated
            .iter()
            .filter_map(|aw| match &aw.key {
                WeightKey::MoeDownProj { layer, expert } => Some((*layer, *expert)),
                _ => None,
            })
            .collect();

        let expert_scores: Vec<ExpertScoreEntry> = scores
            .iter()
            .map(|s| ExpertScoreEntry {
                layer: s.layer_idx,
                expert: s.expert_idx,
                refusal_projection: s.refusal_projection,
                routing_bias: s.routing_bias,
                combined_score: s.combined_score,
                abliterated: abliterated_set.contains(&(s.layer_idx, s.expert_idx)),
            })
            .collect();

        let abliteration_details: Vec<AbliterationDetail> = abliterated
            .iter()
            .map(|aw| match &aw.key {
                WeightKey::MoeDownProj { layer, expert } => {
                    let has_per_expert = scores
                        .iter()
                        .find(|s| s.layer_idx == *layer && s.expert_idx == *expert)
                        .map_or(false, |s| s.has_per_expert_direction);
                    AbliterationDetail {
                        layer: *layer,
                        expert: Some(*expert),
                        strength: aw.strength,
                        direction_source: if has_per_expert {
                            "per-expert".to_string()
                        } else {
                            "global".to_string()
                        },
                    }
                }
                WeightKey::AttnOProj { layer } => AbliterationDetail {
                    layer: *layer,
                    expert: None,
                    strength: aw.strength,
                    direction_source: "o_proj residual".to_string(),
                },
                WeightKey::MoeGate { layer } => AbliterationDetail {
                    layer: *layer,
                    expert: None,
                    strength: aw.strength,
                    direction_source: "gate attenuation".to_string(),
                },
            })
            .collect();

        let expert_count = abliterated
            .iter()
            .filter(|aw| matches!(&aw.key, WeightKey::MoeDownProj { .. }))
            .count();
        let o_proj_count = abliterated
            .iter()
            .filter(|aw| matches!(&aw.key, WeightKey::AttnOProj { .. }))
            .count();

        Self {
            model_id,
            revision,
            total_experts,
            abliterated_experts: expert_count,
            abliterated_o_proj_layers: o_proj_count,
            threshold,
            threshold_method,
            abliteration_mode,
            kl_divergence,
            expert_scores,
            abliteration_details,
            eval: None,
            reproducibility,
        }
    }

    /// Save the report as pretty-printed JSON to `flay-report.json`.
    pub fn save_json(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("flay-report.json");
        let json = serde_json::to_string_pretty(self)
            .context("Failed to serialize report to JSON")?;
        std::fs::write(&path, json)
            .with_context(|| format!("Failed to write report: {}", path.display()))?;
        tracing::info!("Saved JSON report: {}", path.display());
        Ok(())
    }

    /// Save the report as a markdown table to `flay-report.md`.
    pub fn save_markdown(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("flay-report.md");

        let mut md = String::new();
        md.push_str(&format!("# Flay Abliteration Report\n\n"));
        md.push_str(&format!("**Model:** {}\n\n", self.model_id));
        if let Some(ref rev) = self.revision {
            md.push_str(&format!("**Revision:** {}\n\n", rev));
        }
        md.push_str(&format!(
            "**Experts abliterated:** {} / {}\n\n",
            self.abliterated_experts, self.total_experts,
        ));
        if self.abliterated_o_proj_layers > 0 {
            md.push_str(&format!(
                "**o_proj layers abliterated:** {}\n\n",
                self.abliterated_o_proj_layers,
            ));
        }
        md.push_str(&format!(
            "**Threshold:** {:.4} ({})\n\n",
            self.threshold, self.threshold_method,
        ));
        if let Some(kl) = self.kl_divergence {
            md.push_str(&format!("**KL Divergence:** {:.6}\n\n", kl));
        }

        // Expert scores table
        md.push_str("## Expert Scores\n\n");
        md.push_str("| Layer | Expert | Refusal Proj | Routing Bias | Combined | Abliterated |\n");
        md.push_str("|------:|-------:|-------------:|-------------:|---------:|:-----------:|\n");
        for entry in &self.expert_scores {
            let flag = if entry.abliterated { "yes" } else { "" };
            md.push_str(&format!(
                "| {:>5} | {:>6} | {:>12.4} | {:>12.2} | {:>8.4} | {:^11} |\n",
                entry.layer,
                entry.expert,
                entry.refusal_projection,
                entry.routing_bias,
                entry.combined_score,
                flag,
            ));
        }

        // Abliteration details table
        if !self.abliteration_details.is_empty() {
            md.push_str("\n## Abliteration Details\n\n");
            md.push_str("| Layer | Expert | Strength | Direction |\n");
            md.push_str("|------:|-------:|---------:|:----------|\n");
            for detail in &self.abliteration_details {
                let expert_str = detail
                    .expert
                    .map_or("-".to_string(), |e| e.to_string());
                md.push_str(&format!(
                    "| {:>5} | {:>6} | {:>8.2} | {} |\n",
                    detail.layer, expert_str, detail.strength, detail.direction_source,
                ));
            }
        }

        // Evaluation results (if present)
        if let Some(ref eval) = self.eval {
            md.push_str("\n## Evaluation Results\n\n");

            md.push_str(&format!(
                "**Refusal Rate:** {} / {} ({:.1}%)\n\n",
                eval.refusal_rate.refused,
                eval.refusal_rate.total,
                eval.refusal_rate.rate * 100.0,
            ));

            md.push_str(&format!("**KL Divergence:** {:.6}\n\n", eval.kl_divergence));

            if let Some(ref canary) = eval.reasoning_canary {
                md.push_str(&format!(
                    "**Reasoning Canary:** {} / {} passed\n\n",
                    canary.passed, canary.total,
                ));
            }

            if let Some(ref over) = eval.over_refusal {
                md.push_str(&format!(
                    "**Over-Refusal (harmless):** {} / {} ({:.1}%)\n\n",
                    over.refused, over.total, over.rate * 100.0,
                ));
            }

            if !eval.domain_refusal.is_empty() {
                md.push_str("### Domain Refusal Rates\n\n");
                md.push_str("| Domain | Refused | Total | Rate |\n");
                md.push_str("|:-------|--------:|------:|-----:|\n");
                for domain in &eval.domain_refusal {
                    md.push_str(&format!(
                        "| {} | {} | {} | {:.1}% |\n",
                        domain.domain,
                        domain.refused,
                        domain.total,
                        domain.rate * 100.0,
                    ));
                }
            }

            if let Some(ref util) = eval.utility {
                md.push_str(&format!(
                    "\n**Utility Benchmark:** {} / {} ({:.1}%)\n\n",
                    util.passed, util.total, util.rate * 100.0,
                ));
                if !util.per_category.is_empty() {
                    md.push_str("| Category | Passed | Total | Rate |\n");
                    md.push_str("|:---------|-------:|------:|-----:|\n");
                    for cat in &util.per_category {
                        md.push_str(&format!(
                            "| {} | {} | {} | {:.1}% |\n",
                            cat.category, cat.passed, cat.total, cat.rate * 100.0,
                        ));
                    }
                }
            }
        }

        // Reproducibility metadata (if present)
        if let Some(ref repro) = self.reproducibility {
            md.push_str("\n## Reproducibility\n\n");
            if let Some(ref commit) = repro.git_commit {
                md.push_str(&format!("**Git Commit:** `{}`\n\n", commit));
            }
            if let Some(ref rev) = repro.model_revision {
                md.push_str(&format!("**Model Revision:** `{}`\n\n", rev));
            }
            md.push_str(&format!("**Prompt Hash:** `{}`\n\n", repro.prompt_hash));
            md.push_str(&format!(
                "**Generation Strategy:** {}\n\n",
                repro.generation_strategy,
            ));
        }

        std::fs::write(&path, md)
            .with_context(|| format!("Failed to write markdown report: {}", path.display()))?;
        tracing::info!("Saved markdown report: {}", path.display());
        Ok(())
    }

    /// Save a HuggingFace-compatible model card as `README.md` with YAML frontmatter.
    pub fn save_model_card(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("README.md");

        let mut card = String::new();

        // YAML frontmatter
        card.push_str("---\n");
        card.push_str("license: apache-2.0\n");
        card.push_str("tags:\n");
        card.push_str("  - abliterated\n");
        card.push_str("  - uncensored\n");
        card.push_str("  - flay\n");
        card.push_str("  - moe\n");
        card.push_str(&format!("base_model: {}\n", self.model_id));
        if let Some(ref rev) = self.revision {
            card.push_str(&format!("base_model_revision: {}\n", rev));
        }
        card.push_str("pipeline_tag: text-generation\n");
        card.push_str("---\n\n");

        // Title
        let short_name = self.model_id.rsplit('/').next().unwrap_or(&self.model_id);
        card.push_str(&format!(
            "# {} (Flay Abliterated)\n\n",
            short_name,
        ));

        // Description
        card.push_str(&format!(
            "This model was created by applying [Flay](https://github.com/nooesc/flay) \
             per-expert abliteration to [{}]({}).\n\n",
            self.model_id,
            format!("https://huggingface.co/{}", self.model_id),
        ));
        card.push_str(
            "Flay identifies individual MoE experts responsible for refusal behavior and \
             surgically removes the refusal direction from their weight matrices, preserving \
             the model's general capabilities while reducing refusal.\n\n",
        );

        // Summary
        card.push_str("## Abliteration Summary\n\n");
        card.push_str("| Metric | Value |\n|:-------|------:|\n");
        card.push_str(&format!("| Base model | `{}` |\n", self.model_id));
        if let Some(ref rev) = self.revision {
            card.push_str(&format!("| Revision | `{}` |\n", rev));
        }
        card.push_str(&format!(
            "| Total experts | {} |\n\
             | Experts abliterated | {} |\n",
            self.total_experts,
            self.abliterated_experts,
        ));
        if self.abliterated_o_proj_layers > 0 {
            card.push_str(&format!(
                "| o_proj layers abliterated | {} |\n",
                self.abliterated_o_proj_layers,
            ));
        }
        card.push_str(&format!(
            "| Threshold | {:.4} ({}) |\n",
            self.threshold,
            self.threshold_method,
        ));
        if let Some(kl) = self.kl_divergence {
            card.push_str(&format!("| KL divergence | {:.6} |\n", kl));
        }
        card.push('\n');

        // Affected experts
        if !self.abliteration_details.is_empty() {
            card.push_str("## Modified Experts\n\n");
            card.push_str("| Layer | Expert | Strength | Direction Source |\n");
            card.push_str("|------:|-------:|---------:|:----------------|\n");
            for detail in &self.abliteration_details {
                let expert_str = detail
                    .expert
                    .map_or("-".to_string(), |e| e.to_string());
                card.push_str(&format!(
                    "| {} | {} | {:.2} | {} |\n",
                    detail.layer, expert_str, detail.strength, detail.direction_source,
                ));
            }
            card.push('\n');
        }

        // Usage
        card.push_str("## Usage\n\n");
        card.push_str("This model is a drop-in replacement for the base model. ");
        card.push_str("Use it with any framework that supports the original architecture ");
        card.push_str("(transformers, vLLM, llama.cpp, etc.).\n\n");

        // Disclaimer
        card.push_str("## Disclaimer\n\n");
        card.push_str(
            "This model has had safety-trained refusal behavior reduced. \
             It may produce outputs that the original model would have refused. \
             Use responsibly and in accordance with applicable laws and regulations.\n",
        );

        std::fs::write(&path, card)
            .with_context(|| format!("Failed to write model card: {}", path.display()))?;
        tracing::info!("Saved model card: {}", path.display());
        Ok(())
    }
}
