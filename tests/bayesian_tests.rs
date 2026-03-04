use flay::optimize::bayesian::{HyperParams, Observation, SearchSpace, TpeOptimizer};

#[test]
fn test_tpe_startup_is_random() {
    let mut opt = TpeOptimizer::new(SearchSpace::default());
    let p1 = opt.suggest();
    let p2 = opt.suggest();
    // Two independent random samples should almost certainly differ
    // in at least one parameter.
    assert!(
        (p1.threshold - p2.threshold).abs() > 1e-6
            || p1.num_directions != p2.num_directions
            || (p1.direction_energy - p2.direction_energy).abs() > 1e-6,
        "Two random suggestions should differ"
    );
}

#[test]
fn test_tpe_pareto_front() {
    let mut opt = TpeOptimizer::new(SearchSpace::default());
    // Point A: low KL, high refusal
    opt.tell(Observation {
        params: HyperParams {
            threshold: 1.0,
            num_directions: 1,
            direction_energy: 0.1,
            strength_min: 0.5,
            abliteration_mode: 0,
        },
        kl_divergence: 0.05,
        refusal_rate: 0.5,
    });
    // Point B: higher KL, low refusal
    opt.tell(Observation {
        params: HyperParams {
            threshold: 0.5,
            num_directions: 3,
            direction_energy: 0.1,
            strength_min: 0.5,
            abliteration_mode: 3,
        },
        kl_divergence: 0.10,
        refusal_rate: 0.1,
    });
    let front = opt.pareto_front();
    // Neither dominates the other, so both should be on the Pareto front
    assert_eq!(front.len(), 2);
}

#[test]
fn test_tpe_pareto_front_dominated_point() {
    let mut opt = TpeOptimizer::new(SearchSpace::default());
    let base = HyperParams {
        threshold: 1.0,
        num_directions: 1,
        direction_energy: 0.1,
        strength_min: 0.5,
        abliteration_mode: 0,
    };
    // Point A dominates point B on both objectives
    opt.tell(Observation {
        params: base.clone(),
        kl_divergence: 0.05,
        refusal_rate: 0.1,
    });
    opt.tell(Observation {
        params: base,
        kl_divergence: 0.10,
        refusal_rate: 0.3,
    });
    let front = opt.pareto_front();
    assert_eq!(front.len(), 1);
    assert!((front[0].kl_divergence - 0.05).abs() < 1e-6);
}

#[test]
fn test_tpe_best_trial_respects_max_refusal() {
    let mut opt = TpeOptimizer::new(SearchSpace::default());
    // Observation with excellent KL but terrible refusal rate
    opt.tell(Observation {
        params: HyperParams {
            threshold: 1.0,
            num_directions: 1,
            direction_energy: 0.1,
            strength_min: 0.5,
            abliteration_mode: 0,
        },
        kl_divergence: 0.02,
        refusal_rate: 0.8,
    });
    // Observation with acceptable KL and acceptable refusal rate
    opt.tell(Observation {
        params: HyperParams {
            threshold: 0.3,
            num_directions: 3,
            direction_energy: 0.1,
            strength_min: 0.5,
            abliteration_mode: 3,
        },
        kl_divergence: 0.10,
        refusal_rate: 0.15,
    });
    let best = opt.best_trial(0.2).unwrap();
    assert!(best.refusal_rate <= 0.2);
    assert!((best.kl_divergence - 0.10).abs() < 1e-6);
}

#[test]
fn test_tpe_best_trial_none_when_all_exceed_max_refusal() {
    let mut opt = TpeOptimizer::new(SearchSpace::default());
    opt.tell(Observation {
        params: HyperParams {
            threshold: 1.0,
            num_directions: 1,
            direction_energy: 0.1,
            strength_min: 0.5,
            abliteration_mode: 0,
        },
        kl_divergence: 0.02,
        refusal_rate: 0.8,
    });
    assert!(opt.best_trial(0.2).is_none());
}

#[test]
fn test_tpe_suggest_after_observations_uses_tpe() {
    let mut opt = TpeOptimizer::new(SearchSpace::default());
    // Feed more than n_startup_trials (10) observations
    for i in 0..12 {
        let params = opt.suggest();
        opt.tell(Observation {
            params,
            kl_divergence: 0.1 * (i as f32),
            refusal_rate: 1.0 - 0.05 * (i as f32),
        });
    }
    // After startup phase, suggestions should still be within bounds
    let suggestion = opt.suggest();
    assert!(suggestion.threshold >= 0.05);
    assert!(suggestion.threshold <= 5.0);
    assert!(suggestion.num_directions >= 1);
    assert!(suggestion.num_directions <= 10);
    assert!(suggestion.direction_energy >= 0.01);
    assert!(suggestion.direction_energy <= 0.5);
    assert!(suggestion.strength_min >= 0.3);
    assert!(suggestion.strength_min <= 1.0);
    assert!(suggestion.abliteration_mode < 4);
}

#[test]
fn test_tpe_observation_count() {
    let mut opt = TpeOptimizer::new(SearchSpace::default());
    assert_eq!(opt.n_observations(), 0);
    opt.tell(Observation {
        params: HyperParams {
            threshold: 1.0,
            num_directions: 1,
            direction_energy: 0.1,
            strength_min: 0.5,
            abliteration_mode: 0,
        },
        kl_divergence: 0.05,
        refusal_rate: 0.1,
    });
    assert_eq!(opt.n_observations(), 1);
}
