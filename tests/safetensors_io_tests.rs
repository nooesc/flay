use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use candle_core::{Device, Tensor};

use flay::abliterate::orthogonalize::AbliteratedWeight;
use flay::abliterate::weight_key::WeightKey;
use flay::io::safetensors::save_model;

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

fn save_fixture(path: &PathBuf, tensors: Vec<(&str, Tensor)>) {
    let map: HashMap<String, Tensor> = tensors
        .into_iter()
        .map(|(name, t)| (name.to_string(), t))
        .collect();
    candle_core::safetensors::save(&map, path).unwrap();
}

#[test]
fn test_save_model_replaces_target_weight_and_keeps_others() {
    let device = Device::Cpu;
    let root = make_temp_dir("save-model-replace");
    let input = root.join("input");
    let output = root.join("output");
    fs::create_dir_all(&input).unwrap();

    let target_key = "model.layers.1.mlp.experts.2.down_proj.weight";
    let other_key = "model.layers.0.self_attn.q_proj.weight";
    let input_path = input.join("model.safetensors");

    save_fixture(
        &input_path,
        vec![
            (
                target_key,
                Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], &device).unwrap(),
            ),
            (
                other_key,
                Tensor::new(&[[9.0f32, 8.0], [7.0, 6.0]], &device).unwrap(),
            ),
        ],
    );

    let replacement = Tensor::new(&[[10.0f32, 20.0], [30.0, 40.0]], &device).unwrap();
    let abliterated = vec![AbliteratedWeight {
        key: WeightKey::MoeDownProj { layer: 1, expert: 2 },
        new_weight: replacement,
        strength: 1.0,
    }];

    save_model(&[input_path], &abliterated, &output, &device).unwrap();

    let out_tensors = candle_core::safetensors::load(output.join("model.safetensors"), &device).unwrap();
    let replaced: Vec<Vec<f32>> = out_tensors[target_key].to_vec2().unwrap();
    let untouched: Vec<Vec<f32>> = out_tensors[other_key].to_vec2().unwrap();

    assert_eq!(replaced, vec![vec![10.0, 20.0], vec![30.0, 40.0]]);
    assert_eq!(untouched, vec![vec![9.0, 8.0], vec![7.0, 6.0]]);

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_save_model_sharded_outputs_include_index_file() {
    let device = Device::Cpu;
    let root = make_temp_dir("save-model-sharded");
    let input = root.join("input");
    let output = root.join("output");
    fs::create_dir_all(&input).unwrap();

    let shard1 = input.join("orig-shard-1.safetensors");
    let shard2 = input.join("orig-shard-2.safetensors");

    save_fixture(
        &shard1,
        vec![(
            "model.layers.0.mlp.experts.0.down_proj.weight",
            Tensor::new(&[[1.0f32]], &device).unwrap(),
        )],
    );
    save_fixture(
        &shard2,
        vec![(
            "model.layers.1.mlp.experts.0.down_proj.weight",
            Tensor::new(&[[2.0f32]], &device).unwrap(),
        )],
    );

    save_model(&[shard1, shard2], &[], &output, &device).unwrap();

    assert!(output.join("model-00001-of-00002.safetensors").exists());
    assert!(output.join("model-00002-of-00002.safetensors").exists());
    assert!(output.join("model.safetensors.index.json").exists());

    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(output.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    let map = index["weight_map"].as_object().unwrap();

    assert_eq!(
        map["model.layers.0.mlp.experts.0.down_proj.weight"],
        "model-00001-of-00002.safetensors"
    );
    assert_eq!(
        map["model.layers.1.mlp.experts.0.down_proj.weight"],
        "model-00002-of-00002.safetensors"
    );

    fs::remove_dir_all(root).unwrap();
}

#[test]
fn test_save_model_errors_when_requested_replacement_missing() {
    let device = Device::Cpu;
    let root = make_temp_dir("save-model-missing");
    let input = root.join("input");
    let output = root.join("output");
    fs::create_dir_all(&input).unwrap();

    let input_path = input.join("model.safetensors");
    save_fixture(
        &input_path,
        vec![(
            "model.layers.0.mlp.experts.0.down_proj.weight",
            Tensor::new(&[[1.0f32]], &device).unwrap(),
        )],
    );

    // This target key does not exist in the input tensor map.
    let abliterated = vec![AbliteratedWeight {
        key: WeightKey::MoeDownProj { layer: 9, expert: 9 },
        new_weight: Tensor::new(&[[99.0f32]], &device).unwrap(),
        strength: 1.0,
    }];

    let err = save_model(&[input_path], &abliterated, &output, &device).unwrap_err();
    assert!(
        err.to_string().contains("Weight replacement mismatch"),
        "unexpected error: {err}"
    );

    fs::remove_dir_all(root).unwrap();
}
