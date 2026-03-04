use anyhow::{Context, Result};
use hf_hub::api::sync::Api;
use hf_hub::{Repo, RepoType};
use std::path::PathBuf;

pub struct ModelFiles {
    pub config_path: PathBuf,
    pub tokenizer_path: PathBuf,
    pub weight_paths: Vec<PathBuf>,
    pub model_dir: PathBuf,
    /// The revision used to resolve the model ("main", a branch, tag, or commit hash).
    /// `None` for local models.
    pub revision: Option<String>,
}

pub fn resolve_model(model_id: &str, revision: &str) -> Result<ModelFiles> {
    let local_path = PathBuf::from(model_id);
    if local_path.is_dir() {
        resolve_local(&local_path)
    } else {
        download_from_hub(model_id, revision)
    }
}

fn resolve_local(dir: &PathBuf) -> Result<ModelFiles> {
    let config_path = dir.join("config.json");
    anyhow::ensure!(config_path.exists(), "config.json not found in {}", dir.display());
    let tokenizer_path = dir.join("tokenizer.json");
    anyhow::ensure!(tokenizer_path.exists(), "tokenizer.json not found in {}", dir.display());

    let mut weight_paths = Vec::new();
    let index_path = dir.join("model.safetensors.index.json");
    if index_path.exists() {
        let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
        let weight_map = index["weight_map"].as_object().context("Invalid safetensors index")?;
        let mut files: Vec<String> = weight_map.values().filter_map(|v| v.as_str().map(String::from)).collect();
        files.sort();
        files.dedup();
        for f in files {
            weight_paths.push(dir.join(&f));
        }
    } else {
        weight_paths.push(dir.join("model.safetensors"));
    }

    Ok(ModelFiles { config_path, tokenizer_path, weight_paths, model_dir: dir.clone(), revision: None })
}

fn download_from_hub(model_id: &str, revision: &str) -> Result<ModelFiles> {
    tracing::info!("Downloading {model_id} (revision: {revision}) from HuggingFace Hub...");
    let api = Api::new().context("Failed to initialize HF Hub API")?;
    let repo = api.repo(Repo::with_revision(model_id.to_string(), RepoType::Model, revision.to_string()));

    let config_path = repo.get("config.json").context("Failed to download config.json")?;
    let tokenizer_path = repo.get("tokenizer.json").context("Failed to download tokenizer.json")?;

    let weight_paths = match repo.get("model.safetensors.index.json") {
        Ok(index_path) => {
            let index: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let weight_map = index["weight_map"].as_object().context("Invalid safetensors index")?;
            let mut files: Vec<String> = weight_map.values().filter_map(|v| v.as_str().map(String::from)).collect();
            files.sort();
            files.dedup();
            let mut paths = Vec::new();
            for f in &files {
                tracing::info!("Downloading {f}...");
                paths.push(repo.get(f).with_context(|| format!("Failed to download {f}"))?);
            }
            paths
        }
        Err(_) => {
            vec![repo.get("model.safetensors").context("Failed to download model.safetensors")?]
        }
    };

    let model_dir = config_path.parent().unwrap().to_path_buf();
    Ok(ModelFiles { config_path, tokenizer_path, weight_paths, model_dir, revision: Some(revision.to_string()) })
}
