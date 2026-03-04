//! Router weight modification: attenuate gate rows for safety-critical experts.

use anyhow::Result;
use candle_core::Tensor;

use crate::abliterate::weight_key::WeightKey;
use crate::model::arch::MoeModel;

/// Attenuate gate weight rows for specified experts.
///
/// For each (layer, expert), scales `W_gate[expert, :] *= (1 - strength)`.
/// Gate weight shape is `[num_experts, hidden_dim]`.
/// Strength of 1.0 zeros the row (maximum attenuation).
/// Strength of 0.0 is a no-op.
pub fn attenuate_router_weights(
    model: &mut dyn MoeModel,
    experts: &[(usize, usize)], // (decoder_layer_idx, expert_idx)
    strength: f32,
) -> Result<Vec<(WeightKey, Tensor)>> {
    if strength <= 0.0 || experts.is_empty() {
        return Ok(vec![]);
    }

    let scale = 1.0 - strength.clamp(0.0, 1.0);
    let mut modified = Vec::new();

    // Group experts by layer to modify each gate weight once (BTreeMap for deterministic order)
    let mut by_layer: std::collections::BTreeMap<usize, Vec<usize>> =
        std::collections::BTreeMap::new();
    for &(layer, expert) in experts {
        by_layer.entry(layer).or_default().push(expert);
    }

    for (layer, expert_ids) in &by_layer {
        let key = WeightKey::MoeGate { layer: *layer };
        let gate_weight = model.get_weight(&key)?;
        let (out_features, in_features) = gate_weight.dims2()?;

        // Gate weight shape: [num_experts, hidden_dim] -- each row is an expert's gate vector
        // Cast to F32 for manipulation (gate weights may be BF16/F16 on Metal)
        let gate_f32 = gate_weight.to_dtype(candle_core::DType::F32)?;
        let mut data: Vec<f32> = gate_f32
            .to_vec2::<f32>()?
            .into_iter()
            .flatten()
            .collect();

        for &expert_idx in expert_ids {
            if expert_idx >= out_features {
                anyhow::bail!(
                    "Expert index {} out of bounds for gate with {} experts (layer {})",
                    expert_idx, out_features, layer,
                );
            }
            let start = expert_idx * in_features;
            for i in start..start + in_features {
                data[i] *= scale;
            }
        }

        let new_weight = Tensor::from_vec(data, gate_weight.shape(), gate_weight.device())?
            .to_dtype(gate_weight.dtype())?;
        let old = model.set_weight(&key, &new_weight)?;
        modified.push((key, old));
    }

    tracing::info!(
        "Attenuated router weights for {} experts across {} layers (strength={:.2})",
        experts.len(),
        by_layer.len(),
        strength,
    );

    Ok(modified)
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_scale_calculation() {
        // strength 0.5 → scale 0.5, strength 1.0 → scale 0.0, strength 0.0 → scale 1.0
        assert!((1.0 - 0.5_f32.clamp(0.0, 1.0) - 0.5).abs() < f32::EPSILON);
        assert!((1.0 - 1.0_f32.clamp(0.0, 1.0) - 0.0).abs() < f32::EPSILON);
        assert!((1.0 - 0.0_f32.clamp(0.0, 1.0) - 1.0).abs() < f32::EPSILON);
    }

    /// Test the core row-scaling logic directly on a tensor.
    #[test]
    fn test_row_attenuation_logic() {
        use candle_core::{Device, Tensor};

        // 3 experts, hidden_dim=2 → gate shape [3, 2]
        let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let gate = Tensor::from_vec(data.clone(), (3, 2), &Device::Cpu).unwrap();

        let (out_features, in_features) = gate.dims2().unwrap();
        assert_eq!(out_features, 3);
        assert_eq!(in_features, 2);

        // Attenuate expert 1 with strength 0.5 → scale 0.5
        let scale = 0.5f32;
        let expert_idx = 1usize;
        let mut flat: Vec<f32> = gate.to_vec2::<f32>().unwrap().into_iter().flatten().collect();

        let start = expert_idx * in_features;
        for i in start..start + in_features {
            flat[i] *= scale;
        }

        // Expert 0 unchanged: [1.0, 2.0]
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[1], 2.0);
        // Expert 1 halved: [1.5, 2.0]
        assert_eq!(flat[2], 1.5);
        assert_eq!(flat[3], 2.0);
        // Expert 2 unchanged: [5.0, 6.0]
        assert_eq!(flat[4], 5.0);
        assert_eq!(flat[5], 6.0);
    }
}
