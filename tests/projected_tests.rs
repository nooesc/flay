use candle_core::{Device, Tensor};
use flay::abliterate::projected::project_refusal_direction;

#[test]
fn test_projection_removes_compliance_component() {
    let device = Device::Cpu;
    // Refusal direction at 45 degrees between dim 0 and dim 1
    let v = std::f32::consts::FRAC_1_SQRT_2;
    let refusal = Tensor::new(&[v, v, 0.0], &device).unwrap();
    // Harmless mean along dim 1 (compliance direction)
    let harmless_mean = Tensor::new(&[0.0f32, 1.0, 0.0], &device).unwrap();

    let suppression = project_refusal_direction(&refusal, &harmless_mean).unwrap();
    let vals: Vec<f32> = suppression.to_vec1().unwrap();

    // Suppression should be mostly in dim 0 (orthogonal to compliance dim 1)
    assert!(vals[0].abs() > 0.9, "dim 0 should dominate: {:?}", vals);
    assert!(
        vals[1].abs() < 0.1,
        "dim 1 (compliance) should be removed: {:?}",
        vals
    );
}

#[test]
fn test_projection_aligned_falls_back() {
    let device = Device::Cpu;
    // Refusal direction perfectly aligned with compliance
    let refusal = Tensor::new(&[0.0f32, 1.0, 0.0], &device).unwrap();
    let harmless_mean = Tensor::new(&[0.0f32, 1.0, 0.0], &device).unwrap();

    let result = project_refusal_direction(&refusal, &harmless_mean).unwrap();
    let vals: Vec<f32> = result.to_vec1().unwrap();
    assert!(
        vals[1].abs() > 0.9,
        "should return original direction: {:?}",
        vals
    );
}

#[test]
fn test_projection_orthogonal_returns_full_direction() {
    let device = Device::Cpu;
    // Refusal direction fully orthogonal to compliance
    let refusal = Tensor::new(&[1.0f32, 0.0, 0.0], &device).unwrap();
    let harmless_mean = Tensor::new(&[0.0f32, 1.0, 0.0], &device).unwrap();

    let result = project_refusal_direction(&refusal, &harmless_mean).unwrap();
    let vals: Vec<f32> = result.to_vec1().unwrap();
    assert!(
        vals[0].abs() > 0.9,
        "should return full refusal direction: {:?}",
        vals
    );
}
