// Safetensors read/write
// Load individual expert weights, write modified weights back with selective replacement

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::abliterate::orthogonalize::AbliteratedWeight;

/// Load tensors from one or more safetensors files into a single HashMap.
///
/// Each file is loaded fully into memory on the given device. If a tensor name
/// appears in multiple files, the last occurrence wins.
pub fn load_tensors(paths: &[PathBuf], device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut all_tensors = HashMap::new();
    for path in paths {
        let tensors = candle_core::safetensors::load(path, device)
            .with_context(|| format!("Failed to load safetensors: {}", path.display()))?;
        all_tensors.extend(tensors);
    }
    tracing::info!("Loaded {} tensors from {} shard(s)", all_tensors.len(), paths.len());
    Ok(all_tensors)
}

/// Selectively load only the tensors matching the given keys from safetensors shards.
///
/// Much more memory-efficient than `load_tensors` for large models when only a
/// small subset of weights is needed (e.g., expert down_proj weights for abliteration).
pub fn load_tensors_selective(
    paths: &[PathBuf],
    keys: &std::collections::HashSet<String>,
    device: &Device,
) -> Result<HashMap<String, Tensor>> {
    let mut result = HashMap::new();
    for path in paths {
        let raw_data = std::fs::read(path)
            .with_context(|| format!("Failed to read shard: {}", path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&raw_data)
            .with_context(|| format!("Failed to parse shard: {}", path.display()))?;

        for (name, view) in st.tensors() {
            if keys.contains(name.as_str()) {
                let tensor = candle_core::safetensors::Load::load(&view, device)
                    .with_context(|| format!("Failed to load tensor: {name}"))?;
                result.insert(name, tensor);
            }
        }
    }
    tracing::info!(
        "Selectively loaded {} / {} requested tensors from {} shard(s)",
        result.len(),
        keys.len(),
        paths.len(),
    );
    Ok(result)
}

/// Save a modified model by reading each original shard, replacing abliterated
/// weights in-place, and writing to the output directory.
///
/// The weight key format for MoE expert down_proj is:
/// `model.layers.{layer}.mlp.experts.{expert}.down_proj.weight`
///
/// For sharded models, output files preserve the original shard naming scheme
/// (`model-00001-of-NNNNN.safetensors`) and an index file is written.
pub fn save_model(
    original_paths: &[PathBuf],
    abliterated: &[AbliteratedWeight],
    output_dir: &Path,
    device: &Device,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output directory: {}", output_dir.display()))?;

    // Build replacement map: tensor name -> new tensor
    let mut replacements: HashMap<String, &Tensor> = HashMap::new();
    for aw in abliterated {
        let key = aw.key.safetensors_key();
        replacements.insert(key, &aw.new_weight);
    }
    tracing::info!(
        "Prepared {} weight replacements for saving",
        replacements.len(),
    );

    let num_shards = original_paths.len();
    let mut replaced_keys: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Collect weight map entries during save to avoid re-reading shards for the index
    let mut weight_map_entries: Vec<(String, String, u64)> = Vec::new();

    for (shard_idx, shard_path) in original_paths.iter().enumerate() {
        // Read the raw bytes and deserialize the safetensors header
        let raw_data = std::fs::read(shard_path)
            .with_context(|| format!("Failed to read shard: {}", shard_path.display()))?;
        let st = safetensors::SafeTensors::deserialize(&raw_data)
            .with_context(|| format!("Failed to parse shard: {}", shard_path.display()))?;

        // Determine output filename
        let output_filename = if num_shards > 1 {
            format!("model-{:05}-of-{:05}.safetensors", shard_idx + 1, num_shards)
        } else {
            "model.safetensors".to_string()
        };

        // Build the output tensor map, replacing abliterated weights
        let mut output_tensors: Vec<(String, Tensor)> = Vec::new();
        for (name, view) in st.tensors() {
            // Collect weight map entry for the shard index
            let elem_count: usize = view.shape().iter().product();
            let bytes = (elem_count * dtype_size(view.dtype())) as u64;
            weight_map_entries.push((name.clone(), output_filename.clone(), bytes));

            let tensor = if let Some(replacement) = replacements.get(name.as_str()) {
                tracing::debug!("Replacing tensor: {name}");
                replaced_keys.insert(name.clone());
                // Cast replacement to match the original tensor's dtype
                let orig_dtype = safetensors_dtype_to_candle(view.dtype())?;
                (*replacement).to_dtype(orig_dtype)?
            } else {
                // Load the original tensor unchanged
                candle_core::safetensors::Load::load(&view, device)
                    .with_context(|| format!("Failed to load tensor: {name}"))?
            };
            output_tensors.push((name, tensor));
        }

        let output_path = output_dir.join(&output_filename);

        // Convert to HashMap for candle's save function
        let tensor_map: HashMap<String, Tensor> = output_tensors.into_iter().collect();
        candle_core::safetensors::save(&tensor_map, &output_path)
            .with_context(|| format!("Failed to save shard: {}", output_path.display()))?;

        tracing::info!(
            "Saved shard {}/{}: {}",
            shard_idx + 1,
            num_shards,
            output_filename,
        );
    }

    // Write the shard index file for sharded models
    if num_shards > 1 {
        write_shard_index_from_entries(&weight_map_entries, output_dir)?;
    }

    // Verify all intended replacements were applied
    if replaced_keys.len() != replacements.len() {
        let missing: Vec<_> = replacements
            .keys()
            .filter(|key| !replaced_keys.contains(*key))
            .collect();
        anyhow::bail!(
            "Weight replacement mismatch: expected {} replacements but only {} were applied. \
             Missing keys: {:?}",
            replacements.len(),
            replaced_keys.len(),
            missing,
        );
    }

    tracing::info!(
        "Model saved to {} ({} weights replaced across {} shard(s))",
        output_dir.display(),
        replaced_keys.len(),
        num_shards,
    );
    Ok(())
}

/// Copy non-weight model files (config, tokenizer, etc.) from the source
/// model directory to the output directory.
///
/// Only copies files that exist in the source. Silently skips missing files.
pub fn copy_model_files(model_dir: &Path, output_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(output_dir)?;

    let files_to_copy = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "generation_config.json",
        "merges.txt",
        "vocab.json",
    ];

    for filename in &files_to_copy {
        let src = model_dir.join(filename);
        if src.exists() {
            let dst = output_dir.join(filename);
            std::fs::copy(&src, &dst).with_context(|| {
                format!("Failed to copy {} -> {}", src.display(), dst.display())
            })?;
            tracing::debug!("Copied {filename}");
        }
    }

    Ok(())
}

/// Convert a safetensors `Dtype` to a candle `DType`.
pub fn safetensors_dtype_to_candle(dtype: safetensors::Dtype) -> Result<DType> {
    let candle_dtype = DType::try_from(dtype)
        .map_err(|e| anyhow::anyhow!("Unsupported safetensors dtype {:?}: {}", dtype, e))?;
    Ok(candle_dtype)
}

/// Write a `model.safetensors.index.json` from pre-collected weight map entries.
///
/// Each entry is `(tensor_name, shard_filename, byte_size)`, collected during
/// the save loop to avoid re-reading multi-GB shard files.
fn write_shard_index_from_entries(
    entries: &[(String, String, u64)],
    output_dir: &Path,
) -> Result<()> {
    let mut weight_map: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
    let mut total_size: u64 = 0;

    for (name, shard_filename, bytes) in entries {
        total_size += bytes;
        weight_map.insert(name.clone(), serde_json::Value::String(shard_filename.clone()));
    }

    let index = serde_json::json!({
        "metadata": {
            "total_size": total_size,
        },
        "weight_map": weight_map,
    });

    let index_path = output_dir.join("model.safetensors.index.json");
    let index_json = serde_json::to_string_pretty(&index)?;
    std::fs::write(&index_path, index_json)
        .with_context(|| format!("Failed to write index: {}", index_path.display()))?;

    tracing::info!("Wrote shard index: {}", index_path.display());
    Ok(())
}

/// Return byte size for a safetensors dtype.
fn dtype_size(dtype: safetensors::Dtype) -> usize {
    match dtype {
        safetensors::Dtype::BOOL | safetensors::Dtype::U8 | safetensors::Dtype::I8 => 1,
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 | safetensors::Dtype::I16
        | safetensors::Dtype::U16 => 2,
        safetensors::Dtype::F32 | safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
        safetensors::Dtype::F8_E4M3 | safetensors::Dtype::F8_E5M2 | safetensors::Dtype::F8_E8M0 => 1,
        _ => 4, // conservative fallback
    }
}
