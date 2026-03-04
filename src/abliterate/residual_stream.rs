// All-layer o_proj orthogonalization for residual-stream abliteration.

use anyhow::Result;

use crate::abliterate::orthogonalize::{orthogonalize_weight, AbliteratedWeight};
use crate::abliterate::residual_directions::ResidualDirections;
use crate::abliterate::weight_key::WeightKey;
use crate::model::arch::MoeModel;

/// Per-layer strength schedule for o_proj orthogonalization.
///
/// Returns `alpha_l = band_weight * global_strength` for the given decoder layer index.
///
/// Layer band   | band_weight | Rationale
/// -------------|-------------|------------------------------------------
/// l  0-7       | 0.25        | Early: lexical/syntactic, low refusal signal
/// l  8-15      | 0.45        | Transitional
/// l 16-23      | 0.70        | Building behavioral features
/// l 24-39      | 1.00        | Peak refusal/policy layers
/// l 40-47      | 0.85        | Late taper for fluency/stability
pub fn alpha_schedule(layer: usize, global_strength: f32) -> f32 {
    let band_weight = match layer {
        0..=7 => 0.25,
        8..=15 => 0.45,
        16..=23 => 0.70,
        24..=39 => 1.00,
        _ => 0.85, // 40+
    };
    band_weight * global_strength
}

/// Abliterate o_proj weights across all decoder layers.
///
/// For each layer:
/// 1. Get the current o_proj weight from the model
/// 2. Compute alpha using the band schedule
/// 3. Orthogonalize with the per-layer refusal direction
/// 4. Package as AbliteratedWeight with WeightKey::AttnOProj
///
/// Returns the list of modified weights ready for `set_weight` / `save_model`.
pub fn abliterate_residual_stream(
    model: &dyn MoeModel,
    directions: &ResidualDirections,
    global_strength: f32,
) -> Result<Vec<AbliteratedWeight>> {
    let num_layers = model.num_decoder_layers();
    if directions.per_layer.len() != num_layers {
        anyhow::bail!(
            "Direction count ({}) doesn't match decoder layer count ({})",
            directions.per_layer.len(),
            num_layers,
        );
    }

    let mut results = Vec::with_capacity(num_layers);

    for layer in 0..num_layers {
        let alpha = alpha_schedule(layer, global_strength);
        if alpha < 1e-6 {
            continue; // Skip layers with negligible strength
        }

        let key = WeightKey::AttnOProj { layer };
        let weight = model.get_weight(&key)?;
        let direction = &directions.per_layer[layer];

        // Explicit shape check (Codex review #8): direction dim must match weight d_out
        let w_dims = weight.dims();
        let d_dims = direction.dims();
        if w_dims.len() != 2 || d_dims.len() != 1 || w_dims[0] != d_dims[0] {
            anyhow::bail!(
                "Shape mismatch for {key}: weight is {:?}, direction is {:?} (expected d_out={} to match)",
                w_dims, d_dims, w_dims.first().copied().unwrap_or(0),
            );
        }

        let new_weight = orthogonalize_weight(&weight, direction, alpha)?;

        tracing::debug!(
            "o_proj layer {} abliterated (alpha={:.3}, lambda={:.3})",
            layer,
            alpha,
            directions.lambda[layer],
        );

        results.push(AbliteratedWeight {
            key,
            new_weight,
            strength: alpha,
        });
    }

    tracing::info!(
        "Abliterated {} o_proj weights (global_strength={:.2})",
        results.len(),
        global_strength,
    );

    Ok(results)
}
