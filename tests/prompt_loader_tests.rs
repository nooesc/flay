use std::fs;
use std::path::{Path, PathBuf};

use flay::analysis::prompts::{load_datasets, load_domain, merge_domains};

fn make_temp_dir(test_name: &str) -> PathBuf {
    let base = std::env::temp_dir();
    let unique = format!(
        "flay-{}-{}-{}",
        test_name,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let dir = base.join(unique);
    fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).unwrap();
}

#[test]
fn test_load_datasets_from_txt_files_filters_blank_lines() {
    let dir = make_temp_dir("txt");
    let harmful = dir.join("harmful.txt");
    let harmless = dir.join("harmless.txt");

    write_file(&harmful, "h1\n\n h2\n");
    write_file(&harmless, "ok1\n\nok2\n");

    let ds = load_datasets(
        Some(harmful.to_str().unwrap()),
        Some(harmless.to_str().unwrap()),
    )
    .unwrap();

    assert_eq!(ds.harmful, vec!["h1".to_string(), " h2".to_string()]);
    assert_eq!(ds.harmless, vec!["ok1".to_string(), "ok2".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn test_load_datasets_from_json_files() {
    let dir = make_temp_dir("json");
    let harmful = dir.join("harmful.json");
    let harmless = dir.join("harmless.json");

    write_file(&harmful, "[\"h1\", \"h2\"]");
    write_file(&harmless, "[\"ok1\", \"ok2\"]");

    let ds = load_datasets(
        Some(harmful.to_str().unwrap()),
        Some(harmless.to_str().unwrap()),
    )
    .unwrap();

    assert_eq!(ds.harmful, vec!["h1".to_string(), "h2".to_string()]);
    assert_eq!(ds.harmless, vec!["ok1".to_string(), "ok2".to_string()]);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn test_load_datasets_invalid_json_returns_error() {
    let dir = make_temp_dir("invalid-json");
    let harmful = dir.join("harmful.json");
    let harmless = dir.join("harmless.txt");

    write_file(&harmful, "not-json");
    write_file(&harmless, "ok\n");

    let result = load_datasets(
        Some(harmful.to_str().unwrap()),
        Some(harmless.to_str().unwrap()),
    );
    assert!(result.is_err());
    let err = result.err().unwrap();

    assert!(
        err.to_string().contains("Failed to load harmful prompts"),
        "unexpected error: {err}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn test_load_datasets_empty_files_return_empty_vectors() {
    let dir = make_temp_dir("empty");
    let harmful = dir.join("harmful.txt");
    let harmless = dir.join("harmless.txt");

    write_file(&harmful, "\n\n");
    write_file(&harmless, "");

    let ds = load_datasets(
        Some(harmful.to_str().unwrap()),
        Some(harmless.to_str().unwrap()),
    )
    .unwrap();

    assert!(ds.harmful.is_empty());
    assert!(ds.harmless.is_empty());

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn test_load_domain_directory() {
    let dir = make_temp_dir("domain");
    write_file(&dir.join("harmful.txt"), "h1\nh2\nh3\n");
    write_file(&dir.join("harmless.txt"), "b1\nb2\n");
    write_file(&dir.join("eval.txt"), "e1\ne2\n");

    let ds = load_domain(&dir).unwrap();
    assert_eq!(ds.harmful.len(), 3);
    assert_eq!(ds.harmless.len(), 2);
    assert_eq!(ds.eval.len(), 2);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn test_load_domain_missing_harmful_errors() {
    let dir = make_temp_dir("domain-missing");
    write_file(&dir.join("harmless.txt"), "b1\n");

    let result = load_domain(&dir);
    assert!(result.is_err());
    let err = result.err().unwrap();
    assert!(
        err.to_string().contains("missing harmful.txt"),
        "unexpected error: {err}"
    );

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn test_load_domain_no_eval_is_ok() {
    let dir = make_temp_dir("domain-no-eval");
    write_file(&dir.join("harmful.txt"), "h1\n");
    write_file(&dir.join("harmless.txt"), "b1\n");

    let ds = load_domain(&dir).unwrap();
    assert_eq!(ds.eval.len(), 0);
    assert_eq!(ds.harmful.len(), 1);
    assert_eq!(ds.harmless.len(), 1);

    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn test_merge_domains_combines_all_fields() {
    let d1 = flay::analysis::prompts::PromptDatasets {
        harmful: vec!["h1".into(), "h2".into()],
        harmless: vec!["b1".into()],
        eval: vec!["e1".into()],
    };
    let d2 = flay::analysis::prompts::PromptDatasets {
        harmful: vec!["h3".into()],
        harmless: vec!["b2".into(), "b3".into()],
        eval: vec![],
    };

    let merged = merge_domains(vec![d1, d2]);
    assert_eq!(merged.harmful.len(), 3);
    assert_eq!(merged.harmless.len(), 3);
    assert_eq!(merged.eval.len(), 1);
}
