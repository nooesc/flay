// Utility benchmark: auto-gradable capability tests to measure model quality
// preservation after abliteration.
//
// Supports multiple checker types for different answer formats:
// - exact: case-insensitive exact match
// - contains: expected substring appears in response
// - contains_all: all expected substrings appear in response
// - regex: response matches a regex pattern
// - python_test: extracts Python code, runs assertions in subprocess
// - json_fields: parses JSON from response, checks field values

use anyhow::{Context, Result};
use candle_core::Device;
use regex::Regex;
use serde::{Deserialize, Serialize};
use wait_timeout::ChildExt;

use crate::eval::generate::generate_text;
use crate::model::arch::MoeModel;

/// A single utility benchmark question.
#[derive(Debug, Clone, Deserialize)]
pub struct UtilityQuestion {
    /// The prompt to send to the model.
    pub prompt: String,
    /// Expected answer (interpretation depends on checker).
    pub expected: String,
    /// How to check the answer.
    pub checker: CheckerType,
    /// Category for per-category reporting.
    pub category: String,
}

/// Checker types for auto-grading responses.
#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum CheckerType {
    /// Case-insensitive exact match (after trimming whitespace).
    Exact,
    /// Expected string appears as a substring (case-insensitive).
    Contains,
    /// All pipe-separated expected strings appear (case-insensitive).
    /// Format: "substr1|substr2|substr3"
    ContainsAll,
    /// Response matches a regex pattern.
    Regex,
    /// Extract Python code from response, append test assertions, run in subprocess.
    /// Expected format: "func_name::assert1;;assert2;;assert3"
    PythonTest,
    /// Parse JSON from response, check that all pipe-separated key=value pairs exist.
    /// Expected format: "key1=val1|key2=val2|key3=val3"
    /// Numeric values are compared as numbers; string values are case-insensitive.
    JsonFields,
}

/// Results from running the utility benchmark.
#[derive(Debug, Clone, Serialize)]
pub struct UtilityResults {
    pub passed: usize,
    pub total: usize,
    pub rate: f32,
    pub per_category: Vec<CategoryResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CategoryResult {
    pub category: String,
    pub passed: usize,
    pub total: usize,
    pub rate: f32,
}

/// Number of tokens to generate for utility benchmark responses.
const UTILITY_GEN_TOKENS: usize = 128;

/// Load utility benchmark questions from a JSONL file.
pub fn load_utility_benchmark(path: &str) -> Result<Vec<UtilityQuestion>> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read utility benchmark: {path}"))?;

    let mut questions = Vec::new();
    for (i, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let q: UtilityQuestion = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse line {} of {path}", i + 1))?;
        questions.push(q);
    }

    Ok(questions)
}

/// Extract Python code from a model response (strips markdown fences and preamble).
fn extract_python_code(response: &str) -> String {
    // Try to find code in ```python ... ``` or ``` ... ``` blocks
    let fence_re = Regex::new(r"```(?:python)?\s*\n([\s\S]*?)```").unwrap();
    if let Some(cap) = fence_re.captures(response) {
        return cap[1].to_string();
    }
    // Fallback: find lines that look like Python code (start with def, import, etc.)
    let mut code_lines = Vec::new();
    let mut in_code = false;
    for line in response.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("def ")
            || trimmed.starts_with("import ")
            || trimmed.starts_with("from ")
            || trimmed.starts_with("class ")
        {
            in_code = true;
        }
        if in_code {
            code_lines.push(line);
        }
    }
    if !code_lines.is_empty() {
        return code_lines.join("\n");
    }
    // Last resort: return everything
    response.to_string()
}

/// Extract JSON from a model response (finds first { ... } or [ ... ] block).
fn extract_json(response: &str) -> Option<serde_json::Value> {
    // Try to find JSON in code fences first
    let fence_re = Regex::new(r"```(?:json)?\s*\n([\s\S]*?)```").unwrap();
    if let Some(cap) = fence_re.captures(response) {
        if let Ok(v) = serde_json::from_str(cap[1].trim()) {
            return Some(v);
        }
    }
    // Find first { or [ and try to parse from there
    for start_char in ['{', '['] {
        if let Some(start) = response.find(start_char) {
            let substr = &response[start..];
            if let Ok(v) = serde_json::from_str(substr) {
                return Some(v);
            }
            // Try progressively shorter substrings (handle trailing text)
            let end_char = if start_char == '{' { '}' } else { ']' };
            for (i, c) in substr.char_indices().rev() {
                if c == end_char {
                    if let Ok(v) = serde_json::from_str(&substr[..=i]) {
                        return Some(v);
                    }
                }
            }
        }
    }
    None
}

/// Run Python code with test assertions in a subprocess.
fn check_python_test(response: &str, expected: &str) -> bool {
    let parts: Vec<&str> = expected.splitn(2, "::").collect();
    if parts.len() != 2 {
        return false;
    }
    let assertions: Vec<&str> = parts[1].split(";;").collect();
    let code = extract_python_code(response);

    let mut test_script = code.clone();
    test_script.push('\n');
    for assertion in &assertions {
        test_script.push_str(assertion.trim());
        test_script.push('\n');
    }

    // Write to temp file and execute with 5s timeout
    let tmp = std::env::temp_dir().join(format!("flay_test_{}.py", std::process::id()));
    if std::fs::write(&tmp, &test_script).is_err() {
        return false;
    }

    let child = std::process::Command::new("python3")
        .arg(&tmp)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .spawn();

    let result = match child {
        Ok(mut child) => {
            match child.wait_timeout(std::time::Duration::from_secs(5)) {
                Ok(Some(status)) => status.success(),
                Ok(None) => {
                    // Timed out — kill and fail
                    let _ = child.kill();
                    let _ = child.wait();
                    false
                }
                Err(_) => false,
            }
        }
        Err(_) => false,
    };

    let _ = std::fs::remove_file(&tmp);
    result
}

/// Check JSON response against expected field values.
/// Format: "key1=val1|key2=val2"
fn check_json_fields(response: &str, expected: &str) -> bool {
    let json = match extract_json(response) {
        Some(v) => v,
        None => return false,
    };

    for pair in expected.split('|') {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        let kv: Vec<&str> = pair.splitn(2, '=').collect();
        if kv.len() != 2 {
            return false;
        }
        let key = kv[0].trim();
        let expected_val = kv[1].trim();

        // Navigate nested keys with dots, e.g. "roles[0]"
        let json_val = json_get(&json, key);
        match json_val {
            None => return false,
            Some(v) => {
                if !json_value_matches(&v, expected_val) {
                    return false;
                }
            }
        }
    }
    true
}

/// Navigate a JSON value by key path (supports "key", "key[0]").
fn json_get<'a>(val: &'a serde_json::Value, path: &str) -> Option<&'a serde_json::Value> {
    let idx_re = Regex::new(r"^(.+)\[(\d+)\]$").unwrap();

    if let Some(cap) = idx_re.captures(path) {
        let key = &cap[1];
        let idx: usize = cap[2].parse().ok()?;
        val.get(key)?.get(idx)
    } else {
        val.get(path)
    }
}

/// Check if a JSON value matches an expected string representation.
fn json_value_matches(val: &serde_json::Value, expected: &str) -> bool {
    match val {
        serde_json::Value::String(s) => s.to_lowercase() == expected.to_lowercase(),
        serde_json::Value::Number(n) => {
            // Compare as float
            if let Some(f) = n.as_f64() {
                if let Ok(e) = expected.parse::<f64>() {
                    return (f - e).abs() < 0.01;
                }
            }
            n.to_string() == expected
        }
        serde_json::Value::Bool(b) => {
            let expected_lower = expected.to_lowercase();
            (*b && expected_lower == "true") || (!*b && expected_lower == "false")
        }
        _ => val.to_string().to_lowercase() == expected.to_lowercase(),
    }
}

/// Check if a response passes the given checker.
fn check_response(response: &str, expected: &str, checker: &CheckerType) -> bool {
    let response_lower = response.to_lowercase();
    let expected_lower = expected.to_lowercase();

    match checker {
        CheckerType::Exact => {
            response_lower.trim() == expected_lower.trim()
        }
        CheckerType::Contains => {
            response_lower.contains(&expected_lower)
        }
        CheckerType::ContainsAll => {
            expected_lower
                .split('|')
                .all(|part| response_lower.contains(part.trim()))
        }
        CheckerType::Regex => {
            Regex::new(expected)
                .map(|re| re.is_match(response))
                .unwrap_or(false)
        }
        CheckerType::PythonTest => {
            check_python_test(response, expected)
        }
        CheckerType::JsonFields => {
            check_json_fields(response, expected)
        }
    }
}

/// Run the utility benchmark against a model.
pub fn run_utility_benchmark(
    model: &dyn MoeModel,
    tokenizer: &tokenizers::Tokenizer,
    questions: &[UtilityQuestion],
    device: &Device,
) -> Result<UtilityResults> {
    let mut passed = 0;
    let mut category_stats: std::collections::HashMap<String, (usize, usize)> =
        std::collections::HashMap::new();

    for q in questions {
        let formatted = model.format_chat_prompt(&q.prompt);
        let encoding = tokenizer
            .encode(formatted.as_str(), false)
            .map_err(|e| anyhow::anyhow!("Tokenize failed: {e}"))?;

        let text = generate_text(model, tokenizer, encoding.get_ids(), UTILITY_GEN_TOKENS, device)?;

        let ok = check_response(&text, &q.expected, &q.checker);
        if ok {
            passed += 1;
        } else {
            tracing::debug!(
                "Utility failed [{}]: '{}' -> '{}' (expected '{}', checker={:?})",
                q.category,
                q.prompt.chars().take(60).collect::<String>(),
                text.chars().take(80).collect::<String>(),
                q.expected,
                q.checker,
            );
        }

        let entry = category_stats
            .entry(q.category.clone())
            .or_insert((0, 0));
        if ok {
            entry.0 += 1;
        }
        entry.1 += 1;
    }

    let total = questions.len();
    let rate = if total > 0 {
        passed as f32 / total as f32
    } else {
        0.0
    };

    let mut per_category: Vec<CategoryResult> = category_stats
        .into_iter()
        .map(|(category, (p, t))| CategoryResult {
            category,
            passed: p,
            total: t,
            rate: if t > 0 { p as f32 / t as f32 } else { 0.0 },
        })
        .collect();
    per_category.sort_by(|a, b| a.category.cmp(&b.category));

    tracing::info!(
        "Utility benchmark: {}/{} passed ({:.1}%)",
        passed,
        total,
        rate * 100.0,
    );

    Ok(UtilityResults {
        passed,
        total,
        rate,
        per_category,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_checker() {
        assert!(check_response("Hello World", "hello world", &CheckerType::Exact));
        assert!(check_response("  42  ", "42", &CheckerType::Exact));
        assert!(!check_response("Hello World!", "Hello World", &CheckerType::Exact));
    }

    #[test]
    fn test_contains_checker() {
        assert!(check_response("The answer is 42.", "42", &CheckerType::Contains));
        assert!(check_response("Python is great", "python", &CheckerType::Contains));
        assert!(!check_response("Hello", "world", &CheckerType::Contains));
    }

    #[test]
    fn test_contains_all_checker() {
        assert!(check_response(
            "Name: Alice, Age: 30",
            "alice|30",
            &CheckerType::ContainsAll,
        ));
        assert!(!check_response(
            "Name: Alice",
            "alice|30",
            &CheckerType::ContainsAll,
        ));
    }

    #[test]
    fn test_regex_checker() {
        assert!(check_response("42", r"^\d+$", &CheckerType::Regex));
        assert!(check_response(
            "The answer is 3.14",
            r"\d+\.\d+",
            &CheckerType::Regex,
        ));
        assert!(!check_response("hello", r"^\d+$", &CheckerType::Regex));
    }

    #[test]
    fn test_parse_jsonl() {
        let jsonl = r#"{"prompt":"What is 2+2?","expected":"4","checker":"contains","category":"closed_qa"}"#;
        let q: UtilityQuestion = serde_json::from_str(jsonl).unwrap();
        assert_eq!(q.category, "closed_qa");
        assert_eq!(q.checker, CheckerType::Contains);
    }

    #[test]
    fn test_json_fields_checker() {
        let response = r#"{"name": "Alice", "age": 30, "city": "NYC"}"#;
        assert!(check_json_fields(response, "name=Alice|age=30|city=NYC"));
        assert!(!check_json_fields(response, "name=Bob"));
        assert!(!check_json_fields("not json at all", "name=Alice"));
    }

    #[test]
    fn test_json_fields_nested() {
        let response = r#"{"user": "jlee", "roles": ["editor", "reviewer"]}"#;
        assert!(check_json_fields(response, "user=jlee|roles[0]=editor|roles[1]=reviewer"));
    }

    #[test]
    fn test_json_fields_in_markdown() {
        let response = "Here is the result:\n```json\n{\"order_id\": \"A-19\", \"total\": 84.5}\n```";
        assert!(check_json_fields(response, "order_id=A-19|total=84.5"));
    }

    #[test]
    fn test_extract_python_code() {
        let response = "Here's the code:\n```python\ndef add(a, b):\n    return a + b\n```\nDone!";
        let code = extract_python_code(response);
        assert!(code.contains("def add(a, b):"));
        assert!(code.contains("return a + b"));
    }

    #[test]
    fn test_python_test_checker() {
        let response = "def add(a, b):\n    return a + b";
        assert!(check_python_test(response, "add::assert add(2, 3) == 5;;assert add(0, 0) == 0"));
        assert!(!check_python_test(response, "add::assert add(2, 3) == 999"));
    }

    #[test]
    fn test_parse_python_test_jsonl() {
        let jsonl = r#"{"prompt":"Write add","expected":"add::assert add(1,2)==3","checker":"python_test","category":"code_gen"}"#;
        let q: UtilityQuestion = serde_json::from_str(jsonl).unwrap();
        assert_eq!(q.checker, CheckerType::PythonTest);
    }

    #[test]
    fn test_parse_json_fields_jsonl() {
        let jsonl = r#"{"prompt":"Extract data","expected":"name=Alice|age=30","checker":"json_fields","category":"extraction"}"#;
        let q: UtilityQuestion = serde_json::from_str(jsonl).unwrap();
        assert_eq!(q.checker, CheckerType::JsonFields);
    }
}
