use anyhow::{Context, Result};
use std::path::Path;

pub struct PromptDatasets {
    pub harmful: Vec<String>,
    pub harmless: Vec<String>,
    pub eval: Vec<String>,
}

pub fn load_datasets(harmful_path: Option<&str>, harmless_path: Option<&str>) -> Result<PromptDatasets> {
    let harmful = match harmful_path {
        Some(path) => load_prompts_from_file(Path::new(path)).context("Failed to load harmful prompts")?,
        None => default_harmful_prompts(),
    };
    let harmless = match harmless_path {
        Some(path) => load_prompts_from_file(Path::new(path)).context("Failed to load harmless prompts")?,
        None => default_harmless_prompts(),
    };
    tracing::info!("Loaded {} harmful and {} harmless prompts", harmful.len(), harmless.len());
    Ok(PromptDatasets { harmful, harmless, eval: vec![] })
}

pub fn load_prompts_from_file(path: &Path) -> Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    if path.extension().is_some_and(|e| e == "json") {
        let prompts: Vec<String> = serde_json::from_str(&content)?;
        Ok(prompts)
    } else {
        Ok(content.lines().filter(|l| !l.trim().is_empty()).map(String::from).collect())
    }
}

fn default_harmful_prompts() -> Vec<String> {
    include_str!("../../data/harmful.txt")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect()
}

fn default_harmless_prompts() -> Vec<String> {
    include_str!("../../data/harmless.txt")
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(String::from)
        .collect()
}

/// Load prompts from a domain directory containing harmful.txt, harmless.txt,
/// and optionally eval.txt.
pub fn load_domain(domain_dir: &Path) -> Result<PromptDatasets> {
    let harmful_path = domain_dir.join("harmful.txt");
    let harmless_path = domain_dir.join("harmless.txt");
    let eval_path = domain_dir.join("eval.txt");

    anyhow::ensure!(
        harmful_path.exists(),
        "Domain directory missing harmful.txt: {}",
        domain_dir.display()
    );
    anyhow::ensure!(
        harmless_path.exists(),
        "Domain directory missing harmless.txt: {}",
        domain_dir.display()
    );

    let harmful = load_prompts_from_file(&harmful_path)
        .with_context(|| format!("Failed to load {}", harmful_path.display()))?;
    let harmless = load_prompts_from_file(&harmless_path)
        .with_context(|| format!("Failed to load {}", harmless_path.display()))?;
    let eval = if eval_path.exists() {
        load_prompts_from_file(&eval_path)
            .with_context(|| format!("Failed to load {}", eval_path.display()))?
    } else {
        vec![]
    };

    tracing::info!(
        "Loaded domain '{}': {} harmful, {} harmless, {} eval",
        domain_dir.file_name().unwrap_or_default().to_string_lossy(),
        harmful.len(),
        harmless.len(),
        eval.len(),
    );

    Ok(PromptDatasets { harmful, harmless, eval })
}

/// A named domain with its eval prompts, preserved for per-domain reporting.
pub struct NamedDomain {
    pub name: String,
    pub eval: Vec<String>,
}

/// Load multiple domain directories and return both the merged dataset
/// (for training) and per-domain eval sets (for reporting).
pub fn load_domains_with_names(
    domain_paths: &[String],
) -> Result<(PromptDatasets, Vec<NamedDomain>)> {
    let mut domain_sets = Vec::new();
    let mut named_domains = Vec::new();

    for domain_path in domain_paths {
        let ds = load_domain(std::path::Path::new(domain_path))?;
        let name = std::path::Path::new(domain_path)
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| domain_path.clone());

        if !ds.eval.is_empty() {
            named_domains.push(NamedDomain {
                name,
                eval: ds.eval.clone(),
            });
        }
        domain_sets.push(ds);
    }

    let merged = merge_domains(domain_sets);
    Ok((merged, named_domains))
}

/// Merge multiple domain datasets. All harmful, harmless, and eval prompts
/// are concatenated into a single combined dataset.
pub fn merge_domains(domains: Vec<PromptDatasets>) -> PromptDatasets {
    let mut harmful = Vec::new();
    let mut harmless = Vec::new();
    let mut eval = Vec::new();
    for d in domains {
        harmful.extend(d.harmful);
        harmless.extend(d.harmless);
        eval.extend(d.eval);
    }
    PromptDatasets { harmful, harmless, eval }
}
