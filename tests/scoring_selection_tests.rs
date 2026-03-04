use candle_core::{Device, Tensor};

use flay::abliterate::directions::RefusalDirections;
use flay::abliterate::scoring::{
    decompose_hcdg_hrcg, score_experts, select_experts_stable, select_guilty_experts, ExpertScore,
};
use flay::analysis::activations::ExpertStats;

fn make_score(layer_idx: usize, expert_idx: usize, combined_score: f32) -> ExpertScore {
    ExpertScore {
        layer_idx,
        expert_idx,
        refusal_projection: combined_score,
        routing_bias: 1.0,
        combined_score,
        has_per_expert_direction: false,
    }
}

fn toy_stats_and_directions() -> anyhow::Result<(ExpertStats, RefusalDirections)> {
    let device = Device::Cpu;

    // 3 experts, one MoE layer. Expert 2 is never activated and should be skipped.
    let harmful_means = vec![vec![
        Some(Tensor::new(&[1.0f32, 0.0], &device)?),
        Some(Tensor::new(&[2.0f32, 0.0], &device)?),
        None,
    ]];
    let harmless_means = vec![vec![
        Some(Tensor::new(&[0.0f32, 0.0], &device)?),
        Some(Tensor::new(&[0.0f32, 0.0], &device)?),
        None,
    ]];

    let stats = ExpertStats {
        harmful_means,
        harmless_means,
        harmful_residual_means: vec![Tensor::new(&[0.0f32, 0.0], &device)?],
        harmless_residual_means: vec![Tensor::new(&[0.0f32, 0.0], &device)?],
        harmful_counts: vec![vec![1, 2, 0]],
        harmless_counts: vec![vec![1, 1, 0]],
        moe_layer_indices: vec![3],
        num_experts: 3,
        harmful_residuals_raw: vec![Vec::new()],
        harmless_residuals_raw: vec![Vec::new()],
        harmful_expert_raw: vec![vec![Vec::new(); 3]],
        harmless_expert_raw: vec![vec![Vec::new(); 3]],
    };

    let directions = RefusalDirections {
        global: vec![Tensor::new(&[1.0f32, 0.0], &device)?],
        per_expert: vec![vec![None, Some(Tensor::new(&[1.0f32, 0.0], &device)?), None]],
        moe_layer_indices: vec![3],
    };

    Ok((stats, directions))
}

#[test]
fn test_score_experts_skips_never_active_and_sorts_descending() {
    let (stats, directions) = toy_stats_and_directions().unwrap();
    let scores = score_experts(&stats, &directions).unwrap();

    assert_eq!(scores.len(), 2, "never-activated expert should be skipped");
    assert_eq!(scores[0].expert_idx, 1, "higher score should rank first");
    assert_eq!(scores[1].expert_idx, 0);
    assert!(scores[0].combined_score > scores[1].combined_score);
    assert!(scores[0].has_per_expert_direction);
    assert!(!scores[1].has_per_expert_direction);
}

#[test]
fn test_select_guilty_manual_threshold_is_inclusive() {
    let scores = vec![
        make_score(0, 0, 0.9),
        make_score(0, 1, 0.4),
        make_score(0, 2, 0.39),
    ];

    let selected = select_guilty_experts(&scores, Some(0.4));
    assert_eq!(selected.len(), 2);
    assert!(selected.iter().any(|s| s.expert_idx == 1));
}

#[test]
fn test_select_guilty_auto_single_item_returns_it() {
    let scores = vec![make_score(0, 0, 0.2)];
    let selected = select_guilty_experts(&scores, None);

    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].expert_idx, 0);
}

#[test]
fn test_select_guilty_auto_flat_scores_selects_one() {
    let scores = vec![
        make_score(0, 0, 0.7),
        make_score(0, 1, 0.7),
        make_score(0, 2, 0.7),
    ];

    let selected = select_guilty_experts(&scores, None);
    assert_eq!(selected.len(), 1);
}

#[test]
fn test_select_guilty_auto_no_positive_scores_selects_top_only() {
    let scores = vec![
        make_score(0, 0, 0.0),
        make_score(0, 1, -0.1),
        make_score(0, 2, -0.2),
    ];

    let selected = select_guilty_experts(&scores, None);
    assert_eq!(selected.len(), 1);
    assert_eq!(selected[0].combined_score, 0.0);
}

#[test]
fn test_score_experts_flat_combined_scores_behave_with_auto_threshold() {
    let device = Device::Cpu;
    let zero = Tensor::new(&[0.0f32, 0.0], &device).unwrap();

    let stats = ExpertStats {
        harmful_means: vec![vec![Some(zero.clone()), Some(zero.clone())]],
        harmless_means: vec![vec![Some(zero.clone()), Some(zero.clone())]],
        harmful_residual_means: vec![zero.clone()],
        harmless_residual_means: vec![zero.clone()],
        harmful_counts: vec![vec![1, 1]],
        harmless_counts: vec![vec![1, 1]],
        moe_layer_indices: vec![0],
        num_experts: 2,
        harmful_residuals_raw: vec![Vec::new()],
        harmless_residuals_raw: vec![Vec::new()],
        harmful_expert_raw: vec![vec![Vec::new(); 2]],
        harmless_expert_raw: vec![vec![Vec::new(); 2]],
    };

    let directions = RefusalDirections {
        global: vec![Tensor::new(&[1.0f32, 0.0], &device).unwrap()],
        per_expert: vec![vec![None, None]],
        moe_layer_indices: vec![0],
    };

    let scores = score_experts(&stats, &directions).unwrap();
    assert_eq!(scores.len(), 2);
    assert!(scores.iter().all(|s| s.combined_score == 0.0));

    let selected = select_guilty_experts(&scores, None);
    assert_eq!(selected.len(), 1);
}

/// Near-flat distribution: scores within 2x of each other. Should still select
/// just the top expert since there's no meaningful variance to split on.
#[test]
fn test_select_guilty_auto_narrow_range_selects_one() {
    let scores: Vec<ExpertScore> = (0..50)
        .map(|i| {
            let score = 10.0 - i as f32 * 0.1; // 10.0 down to 5.1 (< 3x range)
            make_score(0, i, score)
        })
        .collect();

    let selected = select_guilty_experts(&scores, None);
    assert_eq!(selected.len(), 1, "narrow range should select top expert only");
}

/// Simulates multi-directional scoring with a smooth, gradually tapering distribution
/// where no single gap exceeds 30%. The fallback should use 15% of max score as
/// the cutoff, selecting experts with meaningful refusal participation.
#[test]
fn test_select_guilty_auto_smooth_distribution_uses_percentage_fallback() {
    // 200 experts with exponential decay from 30.0 to ~1.5 (>3x variance).
    // No consecutive gap exceeds 30% (each step is ~1.5% drop).
    let scores: Vec<ExpertScore> = (0..200)
        .map(|i| {
            let score = 30.0 * 0.985_f32.powi(i as i32);
            make_score(0, i, score)
        })
        .collect();

    let selected = select_guilty_experts(&scores, None);

    // 15% of max = 30.0 * 0.15 = 4.5
    // 30.0 * 0.985^k = 4.5 → k ≈ 126
    // Should select roughly 126 experts (NOT 1, NOT all 200)
    assert!(selected.len() > 1, "should not fall back to top-1-only");
    assert!(selected.len() < 200, "should not select everything");

    let cutoff = 30.0 * 0.15;
    assert!(
        selected.iter().all(|s| s.combined_score >= cutoff),
        "all selected should be above 15% of max ({cutoff})"
    );
}

/// Simulates multi-directional scoring: a small high-score cluster followed by
/// a long tail of small non-zero scores. The elbow method should find the gap
/// between the cluster and the tail, NOT a gap within the tail itself.
#[test]
fn test_select_guilty_auto_long_tail_ignores_noise_floor() {
    let mut scores = vec![
        // High-score cluster (the "guilty" experts)
        make_score(0, 0, 25.0),
        make_score(0, 1, 22.0),
        make_score(0, 2, 20.0),
        make_score(0, 3, 18.0),
        // Gap: 18.0 -> 5.0 = 72% relative gap — this is the real elbow
        make_score(0, 4, 5.0),
        make_score(0, 5, 4.5),
        make_score(0, 6, 4.0),
    ];
    // Long tail of noise scores with large relative gaps between them
    // (e.g., 0.002 -> 0.001 = 50% gap, which previously won the global search)
    for i in 0..50 {
        scores.push(make_score(0, 7 + i, 0.5 - i as f32 * 0.01));
    }

    let selected = select_guilty_experts(&scores, None);
    // Should select the top 4 (before the 72% gap), NOT 50+ experts
    assert_eq!(selected.len(), 4);
    assert_eq!(selected[0].combined_score, 25.0);
    assert_eq!(selected[3].combined_score, 18.0);
}

// ── Stability-Based Expert Selection (SES) tests ────────────────────────────

#[test]
fn test_stable_selection_unanimous_experts_survive() {
    // 5 experts, scores designed so experts 0 and 1 always rank top-2
    let scores = vec![
        make_score(0, 0, 10.0),
        make_score(0, 1, 9.0),
        make_score(0, 2, 2.0),
        make_score(0, 3, 1.5),
        make_score(0, 4, 1.0),
    ];
    // With K=3 and top_n=2, experts 0 and 1 should always be in top-2
    // (they dominate regardless of subsample)
    let result = select_experts_stable(&scores, 3, 2, 42);
    assert!(result.len() >= 2);
    assert!(result.iter().any(|s| s.expert_idx == 0));
    assert!(result.iter().any(|s| s.expert_idx == 1));
}

#[test]
fn test_stable_selection_noisy_expert_filtered() {
    // Expert 2 has a score close to top but noisy — sometimes in top-3, sometimes not
    let scores = vec![
        make_score(0, 0, 10.0),
        make_score(0, 1, 9.0),
        make_score(0, 2, 5.1),  // borderline
        make_score(0, 3, 5.0),  // borderline
        make_score(0, 4, 1.0),
    ];
    let result = select_experts_stable(&scores, 5, 2, 42);
    // Only experts 0 and 1 should survive strict intersection
    assert_eq!(result.len(), 2);
}

#[test]
fn test_stable_selection_fallback_to_frequency() {
    // All experts very close in score — strict intersection may be empty
    let scores = vec![
        make_score(0, 0, 5.0),
        make_score(0, 1, 4.9),
        make_score(0, 2, 4.8),
        make_score(0, 3, 4.7),
    ];
    // With K=5 and top_n=2, strict intersection might be empty due to noise
    // Fallback (frequency >= K-1) should still return something
    let result = select_experts_stable(&scores, 5, 2, 42);
    assert!(!result.is_empty(), "Frequency fallback should prevent empty result");
    assert!(result.len() <= 4);
}

#[test]
fn test_stable_selection_k1_returns_topn() {
    let scores = vec![
        make_score(0, 0, 10.0),
        make_score(0, 1, 5.0),
        make_score(0, 2, 1.0),
    ];
    // K=1 is just top-N of a single sample (all data)
    let result = select_experts_stable(&scores, 1, 2, 42);
    assert_eq!(result.len(), 2);
    assert_eq!(result[0].expert_idx, 0);
    assert_eq!(result[1].expert_idx, 1);
}

// ── HCDG / HRCG decomposition tests ─────────────────────────────────────────

#[test]
fn test_hcdg_hrcg_decomposition() {
    // Expert 0: in both regular and jailbreak stable sets → HCDG (detection)
    // Expert 1: in regular only → HRCG (control) — this is our target
    // Expert 2: in jailbreak only → neither (not in regular set, so irrelevant)
    let regular_stable = vec![
        make_score(0, 0, 10.0),
        make_score(0, 1, 8.0),
    ];
    let jailbreak_stable = vec![
        make_score(0, 0, 10.0),
        make_score(0, 2, 7.0),
    ];

    let regular_refs: Vec<&ExpertScore> = regular_stable.iter().collect();
    let jailbreak_refs: Vec<&ExpertScore> = jailbreak_stable.iter().collect();

    let (hcdg, hrcg) = decompose_hcdg_hrcg(&regular_refs, &jailbreak_refs);

    assert_eq!(hcdg.len(), 1);
    assert_eq!(hcdg[0].expert_idx, 0); // detection: in both

    assert_eq!(hrcg.len(), 1);
    assert_eq!(hrcg[0].expert_idx, 1); // control: in regular only
}

#[test]
fn test_hcdg_hrcg_empty_jailbreak_all_become_hrcg() {
    let regular_stable = vec![
        make_score(0, 0, 10.0),
        make_score(0, 1, 8.0),
    ];

    let regular_refs: Vec<&ExpertScore> = regular_stable.iter().collect();
    let jailbreak_refs: Vec<&ExpertScore> = vec![];

    let (hcdg, hrcg) = decompose_hcdg_hrcg(&regular_refs, &jailbreak_refs);

    assert!(hcdg.is_empty());
    assert_eq!(hrcg.len(), 2); // all regular become HRCG when no jailbreak data
}
