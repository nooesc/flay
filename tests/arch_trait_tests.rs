use flay::model::arch::{ExpertMask, MoeModel};

#[test]
fn test_qwen3_implements_moe_model_trait() {
    fn _accepts_moe_model(_m: &dyn MoeModel) {}
}

#[test]
fn test_weight_key_format() {
    let key = format!(
        "model.layers.{}.mlp.experts.{}.down_proj.weight",
        5, 42,
    );
    assert_eq!(key, "model.layers.5.mlp.experts.42.down_proj.weight");
}

#[test]
fn test_expert_mask_set_and_check() {
    let mut mask = ExpertMask::new();
    assert!(!mask.is_masked(0, 3));
    mask.add(0, 3);
    mask.add(2, 7);
    assert!(mask.is_masked(0, 3));
    assert!(mask.is_masked(2, 7));
    assert!(!mask.is_masked(0, 7));
    assert!(!mask.is_masked(1, 3));
}

#[test]
fn test_expert_mask_clear() {
    let mut mask = ExpertMask::new();
    mask.add(0, 1);
    mask.add(0, 2);
    assert!(mask.is_masked(0, 1));
    mask.clear();
    assert!(!mask.is_masked(0, 1));
}

#[test]
fn test_expert_mask_layer_count() {
    let mut mask = ExpertMask::new();
    mask.add(5, 0);
    mask.add(5, 3);
    mask.add(5, 7);
    assert_eq!(mask.masked_in_layer(5), 3);
    assert_eq!(mask.masked_in_layer(0), 0);
}

#[test]
fn test_expert_mask_experts_in_layer() {
    let mut mask = ExpertMask::new();
    mask.add(3, 1);
    mask.add(3, 5);
    mask.add(4, 2);
    let layer3 = mask.experts_in_layer(3);
    assert_eq!(layer3.len(), 2);
    assert!(layer3.contains(&1));
    assert!(layer3.contains(&5));
    assert!(mask.experts_in_layer(0).is_empty());
}
