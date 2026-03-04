// Tree-structured Parzen Estimator (TPE) for multi-objective optimization
// over the abliteration hyperparameter space.
//
// Minimizes both KL divergence and refusal rate simultaneously,
// maintaining a Pareto front of non-dominated solutions.

use rand::Rng;

/// A point in the search space with its observed objectives.
#[derive(Debug, Clone)]
pub struct Observation {
    pub params: HyperParams,
    pub kl_divergence: f32,
    pub refusal_rate: f32,
}

/// Abliteration hyperparameters to optimize.
#[derive(Debug, Clone)]
pub struct HyperParams {
    pub threshold: f32,
    pub num_directions: usize,
    pub direction_energy: f32,
    pub strength_min: f32,
    /// 0=single, 1=multi, 2=projected, 3=multi-projected
    pub abliteration_mode: usize,
}

/// Parameter bounds for the search space.
pub struct SearchSpace {
    pub threshold_range: (f32, f32),
    pub num_directions_range: (usize, usize),
    pub direction_energy_range: (f32, f32),
    pub strength_min_range: (f32, f32),
    pub num_modes: usize,
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            threshold_range: (0.05, 5.0),
            num_directions_range: (1, 10),
            direction_energy_range: (0.01, 0.5),
            strength_min_range: (0.3, 1.0),
            num_modes: 4,
        }
    }
}

/// Pareto front entry -- non-dominated solution.
#[derive(Debug, Clone)]
pub struct ParetoEntry {
    pub params: HyperParams,
    pub kl_divergence: f32,
    pub refusal_rate: f32,
}

/// Tree-structured Parzen Estimator for multi-objective optimization.
///
/// During the startup phase (first `n_startup_trials` suggestions), points
/// are sampled uniformly at random. Once enough observations accumulate,
/// the TPE splits them into "good" and "bad" halves (by KL divergence),
/// builds a KDE over each group, and maximises l(x)/g(x) to bias
/// exploration toward the promising region.
pub struct TpeOptimizer {
    space: SearchSpace,
    observations: Vec<Observation>,
    rng: rand::rngs::ThreadRng,
    n_startup_trials: usize,
}

impl TpeOptimizer {
    pub fn new(space: SearchSpace) -> Self {
        Self {
            space,
            observations: Vec::new(),
            rng: rand::thread_rng(),
            n_startup_trials: 10,
        }
    }

    /// Suggest the next set of hyperparameters to evaluate.
    ///
    /// Returns a random sample during the startup phase, then switches
    /// to TPE-guided sampling once enough observations are available.
    pub fn suggest(&mut self) -> HyperParams {
        if self.observations.len() < self.n_startup_trials {
            return self.random_sample();
        }
        self.tpe_sample()
    }

    /// Record an observation (trial result).
    pub fn tell(&mut self, obs: Observation) {
        self.observations.push(obs);
    }

    /// Get the current Pareto front (non-dominated solutions).
    ///
    /// A solution is non-dominated if no other observed point is strictly
    /// better on both KL divergence and refusal rate.
    pub fn pareto_front(&self) -> Vec<ParetoEntry> {
        let mut front = Vec::new();
        for obs in &self.observations {
            let dominated = self.observations.iter().any(|other| {
                other.kl_divergence <= obs.kl_divergence
                    && other.refusal_rate <= obs.refusal_rate
                    && (other.kl_divergence < obs.kl_divergence
                        || other.refusal_rate < obs.refusal_rate)
            });
            if !dominated {
                front.push(ParetoEntry {
                    params: obs.params.clone(),
                    kl_divergence: obs.kl_divergence,
                    refusal_rate: obs.refusal_rate,
                });
            }
        }
        front.sort_by(|a, b| a.kl_divergence.total_cmp(&b.kl_divergence));
        front
    }

    /// Select the best trial: lowest KL among those with `refusal_rate <= max_refusal`.
    pub fn best_trial(&self, max_refusal: f32) -> Option<&Observation> {
        self.observations
            .iter()
            .filter(|o| o.refusal_rate <= max_refusal)
            .min_by(|a, b| a.kl_divergence.total_cmp(&b.kl_divergence))
    }

    /// Number of observations recorded so far.
    pub fn n_observations(&self) -> usize {
        self.observations.len()
    }

    /// Select the best trial by lowest KL divergence, ignoring refusal constraint.
    pub fn best_unconstrained(&self) -> Option<&Observation> {
        self.observations
            .iter()
            .min_by(|a, b| a.kl_divergence.total_cmp(&b.kl_divergence))
    }

    // -----------------------------------------------------------------------
    // Private implementation
    // -----------------------------------------------------------------------

    fn random_sample(&mut self) -> HyperParams {
        let (tlo, thi) = self.space.threshold_range;
        let (dlo, dhi) = self.space.num_directions_range;
        let (elo, ehi) = self.space.direction_energy_range;
        let (slo, shi) = self.space.strength_min_range;

        HyperParams {
            threshold: self.rng.gen_range(tlo..thi),
            num_directions: self.rng.gen_range(dlo..=dhi),
            direction_energy: self.rng.gen_range(elo..ehi),
            strength_min: self.rng.gen_range(slo..shi),
            abliteration_mode: self.rng.gen_range(0..self.space.num_modes),
        }
    }

    /// TPE sampling: split observations into good/bad by KL, then pick
    /// the candidate that maximises l(x)/g(x).
    fn tpe_sample(&mut self) -> HyperParams {
        // Sort observation indices by KL divergence to split into good/bad.
        // We work with indices to avoid holding borrows on self.observations
        // while also mutating self.rng.
        let mut indices: Vec<usize> = (0..self.observations.len()).collect();
        indices.sort_by(|&a, &b| {
            self.observations[a]
                .kl_divergence
                .total_cmp(&self.observations[b].kl_divergence)
        });
        let mid = indices.len() / 2;
        let good_idx: Vec<usize> = indices[..mid].to_vec();
        let bad_idx: Vec<usize> = indices[mid..].to_vec();

        let n_candidates = 24;
        let mut best_candidate = self.random_sample();
        let mut best_score = f32::NEG_INFINITY;

        for _ in 0..n_candidates {
            let pick = self.rng.gen_range(0..good_idx.len());
            let base = self.observations[good_idx[pick]].params.clone();
            let candidate = self.perturb(&base);

            let l = self.kde_score_by_idx(&candidate, &good_idx);
            let g = self.kde_score_by_idx(&candidate, &bad_idx);

            let score = if g > 1e-12 { l / g } else { l };
            if score > best_score {
                best_score = score;
                best_candidate = candidate;
            }
        }

        best_candidate
    }

    /// Perturb a base parameter set with Gaussian-like noise, clamped
    /// to stay within the search space bounds.
    fn perturb(&mut self, base: &HyperParams) -> HyperParams {
        let (tlo, thi) = self.space.threshold_range;
        let (dlo, dhi) = self.space.num_directions_range;
        let (elo, ehi) = self.space.direction_energy_range;
        let (slo, shi) = self.space.strength_min_range;
        let bandwidth = 0.2;

        HyperParams {
            threshold: (base.threshold + self.rng.gen_range(-1.0..1.0) * (thi - tlo) * bandwidth)
                .clamp(tlo, thi),
            num_directions: {
                let delta: i32 = self.rng.gen_range(-2..=2);
                (base.num_directions as i32 + delta).clamp(dlo as i32, dhi as i32) as usize
            },
            direction_energy: (base.direction_energy
                + self.rng.gen_range(-1.0..1.0) * (ehi - elo) * bandwidth)
                .clamp(elo, ehi),
            strength_min: (base.strength_min
                + self.rng.gen_range(-1.0..1.0) * (shi - slo) * bandwidth)
                .clamp(slo, shi),
            abliteration_mode: if self.rng.gen_bool(0.8) {
                base.abliteration_mode
            } else {
                self.rng.gen_range(0..self.space.num_modes)
            },
        }
    }

    /// Kernel Density Estimation score for a candidate point against a set
    /// of observations identified by index. Uses Gaussian kernels for
    /// continuous parameters and a categorical kernel for the abliteration mode.
    fn kde_score_by_idx(&self, candidate: &HyperParams, obs_indices: &[usize]) -> f32 {
        if obs_indices.is_empty() {
            return 1.0;
        }
        let (tlo, thi) = self.space.threshold_range;
        let (elo, ehi) = self.space.direction_energy_range;
        let (slo, shi) = self.space.strength_min_range;

        let bw_t = (thi - tlo) * 0.2;
        let bw_e = (ehi - elo) * 0.2;
        let bw_s = (shi - slo) * 0.2;

        let dir_range = (self.space.num_directions_range.1 - self.space.num_directions_range.0) as f32;

        let score: f32 = obs_indices
            .iter()
            .map(|&i| {
                let obs = &self.observations[i];
                let dt = (candidate.threshold - obs.params.threshold) / bw_t;
                let de = (candidate.direction_energy - obs.params.direction_energy) / bw_e;
                let ds = (candidate.strength_min - obs.params.strength_min) / bw_s;
                let continuous = (-0.5 * (dt * dt + de * de + ds * ds)).exp();

                // Discrete ordinal kernel for num_directions
                let dd = (candidate.num_directions as f32 - obs.params.num_directions as f32).abs();
                let discrete_kernel = (-0.5 * (dd / (dir_range * 0.3)).powi(2)).exp();

                let mode_match = if candidate.abliteration_mode == obs.params.abliteration_mode {
                    0.7
                } else {
                    0.3 / (self.space.num_modes as f32 - 1.0)
                };

                continuous * discrete_kernel * mode_match
            })
            .sum();

        score / obs_indices.len() as f32
    }
}
