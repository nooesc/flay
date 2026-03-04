use flay::model::deepseek_config::DeepSeekV3Config;

/// Parse a config matching the real DeepSeek-V3 HuggingFace config.json.
#[test]
fn test_deepseek_config_parses_real_config() {
    let json = r#"{
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "vocab_size": 129280,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 163840,
        "n_routed_experts": 256,
        "num_experts_per_tok": 8,
        "n_shared_experts": 1,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1,
        "moe_intermediate_size": 2048,
        "n_group": 8,
        "topk_group": 4,
        "norm_topk_prob": true,
        "routed_scaling_factor": 2.5,
        "q_lora_rank": 1536,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "model_type": "deepseek_v3",
        "tie_word_embeddings": false,
        "attention_bias": false
    }"#;

    let config: DeepSeekV3Config = serde_json::from_str(json).unwrap();
    assert_eq!(config.n_routed_experts, 256);
    assert_eq!(config.n_shared_experts, 1);
    assert_eq!(config.num_experts_per_tok, 8);
    assert_eq!(config.first_k_dense_replace, 3);
    assert_eq!(config.moe_layer_freq, 1);
    assert_eq!(config.q_lora_rank, 1536);
    assert_eq!(config.kv_lora_rank, 512);
    assert_eq!(config.qk_nope_head_dim, 128);
    assert_eq!(config.qk_rope_head_dim, 64);
    assert_eq!(config.v_head_dim, 128);
    assert_eq!(config.n_group, 8);
    assert_eq!(config.topk_group, 4);
    assert!(config.is_moe_layer(3));
    assert!(!config.is_moe_layer(2));
    config.validate().unwrap();
}

/// Validation catches num_experts_per_tok > n_routed_experts.
#[test]
fn test_deepseek_config_validation_catches_bad_experts_per_tok() {
    let json = r#"{
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "vocab_size": 129280,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 163840,
        "n_routed_experts": 256,
        "num_experts_per_tok": 300
    }"#;

    let config: DeepSeekV3Config = serde_json::from_str(json).unwrap();
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("num_experts_per_tok"),
        "Expected error about num_experts_per_tok, got: {err}"
    );
}

/// Validation catches first_k_dense_replace > num_hidden_layers.
#[test]
fn test_deepseek_config_validation_catches_bad_dense_replace() {
    let json = r#"{
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "vocab_size": 129280,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 163840,
        "first_k_dense_replace": 100
    }"#;

    let config: DeepSeekV3Config = serde_json::from_str(json).unwrap();
    let err = config.validate().unwrap_err();
    assert!(
        err.to_string().contains("first_k_dense_replace"),
        "Expected error about first_k_dense_replace, got: {err}"
    );
}

/// All DeepSeek-specific fields have sensible defaults when omitted.
#[test]
fn test_deepseek_config_defaults() {
    let json = r#"{
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "vocab_size": 129280,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 163840
    }"#;

    let config: DeepSeekV3Config = serde_json::from_str(json).unwrap();
    assert_eq!(config.n_routed_experts, 256);
    assert_eq!(config.num_experts_per_tok, 8);
    assert_eq!(config.n_shared_experts, 1);
    assert_eq!(config.first_k_dense_replace, 3);
    assert_eq!(config.moe_layer_freq, 1);
    config.validate().unwrap();
}

/// MoE layer detection: first 3 layers dense, then every layer is MoE (freq=1).
#[test]
fn test_deepseek_moe_layer_detection() {
    let json = r#"{
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "num_hidden_layers": 61,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "vocab_size": 129280,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 163840,
        "first_k_dense_replace": 3,
        "moe_layer_freq": 1
    }"#;

    let config: DeepSeekV3Config = serde_json::from_str(json).unwrap();

    // Dense layers (0, 1, 2)
    assert!(!config.is_moe_layer(0));
    assert!(!config.is_moe_layer(1));
    assert!(!config.is_moe_layer(2));

    // MoE layers (3..60 inclusive)
    assert!(config.is_moe_layer(3));
    assert!(config.is_moe_layer(4));
    assert!(config.is_moe_layer(30));
    assert!(config.is_moe_layer(60));

    // Total MoE layers: 61 - 3 = 58
    let moe_indices = config.moe_layer_indices();
    assert_eq!(moe_indices.len(), 58);
    assert_eq!(moe_indices[0], 3);
    assert_eq!(*moe_indices.last().unwrap(), 60);
}

/// Test moe_layer_freq > 1 (hypothetical config where only every other post-dense layer is MoE).
#[test]
fn test_deepseek_moe_layer_freq() {
    let json = r#"{
        "hidden_size": 7168,
        "intermediate_size": 18432,
        "num_hidden_layers": 10,
        "num_attention_heads": 128,
        "num_key_value_heads": 128,
        "vocab_size": 129280,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "max_position_embeddings": 163840,
        "first_k_dense_replace": 2,
        "moe_layer_freq": 2
    }"#;

    let config: DeepSeekV3Config = serde_json::from_str(json).unwrap();

    // Dense: 0, 1
    // MoE (freq=2, offset from first_k_dense_replace=2): 2, 4, 6, 8
    // Not MoE: 3, 5, 7, 9
    assert!(!config.is_moe_layer(0));
    assert!(!config.is_moe_layer(1));
    assert!(config.is_moe_layer(2));  // (2-2) % 2 == 0
    assert!(!config.is_moe_layer(3)); // (3-2) % 2 == 1
    assert!(config.is_moe_layer(4));  // (4-2) % 2 == 0
    assert!(!config.is_moe_layer(5));
    assert!(config.is_moe_layer(6));
    assert!(!config.is_moe_layer(7));
    assert!(config.is_moe_layer(8));
    assert!(!config.is_moe_layer(9));

    let moe_indices = config.moe_layer_indices();
    assert_eq!(moe_indices, vec![2, 4, 6, 8]);
}
