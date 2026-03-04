use candle_core::{Device, Tensor};

#[test]
fn test_l2_normalize() {
    let device = Device::Cpu;
    let v = Tensor::new(&[3.0f32, 4.0], &device).unwrap();
    let normalized = flay::abliterate::directions::l2_normalize(&v).unwrap();
    let vals: Vec<f32> = normalized.to_vec1().unwrap();
    assert!((vals[0] - 0.6).abs() < 1e-5);
    assert!((vals[1] - 0.8).abs() < 1e-5);
}

#[test]
fn test_orthogonalize_removes_direction() {
    let device = Device::Cpu;
    let w = Tensor::new(
        &[[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        &device,
    )
    .unwrap();
    let r = Tensor::new(&[1.0f32, 0.0, 0.0], &device).unwrap();
    let w_ortho =
        flay::abliterate::orthogonalize::orthogonalize_weight(&w, &r, 1.0).unwrap();
    let vals: Vec<Vec<f32>> = w_ortho.to_vec2().unwrap();
    // Row 0 = [1, 2, 3], projected along r=[1,0,0]: component = 1*[1,0,0] = [1,0,0]
    // W' row 0 = [1,2,3] - [1,0,0] = [0,2,3]
    // But orthogonalize_weight does W' = W - strength * r_col @ (r_row @ W)
    // r_row @ W = [[1,0,0]] @ [[1,2,3],[4,5,6],[7,8,9]] = [[1,2,3]]
    // r_col @ (r_row @ W) = [[1],[0],[0]] @ [[1,2,3]] = [[1,2,3],[0,0,0],[0,0,0]]
    // W' = W - [[1,2,3],[0,0,0],[0,0,0]] = [[0,0,0],[4,5,6],[7,8,9]]
    assert!(vals[0][0].abs() < 1e-5);
    assert!(vals[0][1].abs() < 1e-5);
    assert!(vals[0][2].abs() < 1e-5);
    assert!((vals[1][0] - 4.0).abs() < 1e-5);
    assert!((vals[2][0] - 7.0).abs() < 1e-5);
}

#[test]
fn test_orthogonalize_partial_strength() {
    let device = Device::Cpu;
    let w = Tensor::new(&[[2.0f32, 0.0], [0.0, 2.0]], &device).unwrap();
    let r = Tensor::new(&[1.0f32, 0.0], &device).unwrap();
    let w_half =
        flay::abliterate::orthogonalize::orthogonalize_weight(&w, &r, 0.5).unwrap();
    let vals: Vec<Vec<f32>> = w_half.to_vec2().unwrap();
    // r_row @ W = [[1,0]] @ [[2,0],[0,2]] = [[2,0]]
    // r_col @ (r_row @ W) = [[1],[0]] @ [[2,0]] = [[2,0],[0,0]]
    // W' = W - 0.5 * [[2,0],[0,0]] = [[2-1,0],[0,2]] = [[1,0],[0,2]]
    assert!((vals[0][0] - 1.0).abs() < 1e-5);
}

#[test]
fn test_kl_divergence_identical() {
    let device = Device::Cpu;
    let logits = Tensor::new(&[[1.0f32, 2.0, 3.0]], &device).unwrap();
    let kl = flay::analysis::kl_divergence::kl_divergence(&logits, &logits).unwrap();
    assert!(kl.abs() < 1e-5, "KL of identical should be ~0, got {kl}");
}

#[test]
fn test_kl_divergence_different() {
    let device = Device::Cpu;
    let p = Tensor::new(&[[10.0f32, 0.0, 0.0]], &device).unwrap();
    let q = Tensor::new(&[[0.0f32, 0.0, 10.0]], &device).unwrap();
    let kl = flay::analysis::kl_divergence::kl_divergence(&p, &q).unwrap();
    assert!(kl > 0.1, "KL of very different should be >> 0, got {kl}");
}
