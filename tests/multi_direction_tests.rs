use candle_core::{Device, Tensor};
use flay::abliterate::multi_direction::{extract_directions_svd, power_iteration_svd};

#[test]
fn test_extract_single_dominant_direction() {
    let device = Device::Cpu;
    let harmful: Vec<Tensor> = (0..10)
        .map(|i| {
            Tensor::new(&[1.0f32 + i as f32 * 0.1, 0.0, 0.0, 0.0], &device).unwrap()
        })
        .collect();
    let harmless: Vec<Tensor> = (0..10)
        .map(|i| {
            Tensor::new(&[-(1.0f32 + i as f32 * 0.1), 0.0, 0.0, 0.0], &device).unwrap()
        })
        .collect();

    let dirs = extract_directions_svd(&harmful, &harmless, 5, 0.1).unwrap();
    assert!(!dirs.is_empty());
    let top_dir: Vec<f32> = dirs[0].direction.to_vec1().unwrap();
    // Top direction should align strongly with dimension 0
    assert!(
        top_dir[0].abs() > 0.9,
        "top dir should align with dim 0: {:?}",
        top_dir
    );
}

#[test]
fn test_extract_two_directions() {
    let device = Device::Cpu;
    // Create data with two orthogonal refusal signals in the difference matrix.
    // The difference matrix D = H_centered - B_centered needs to have two
    // linearly independent principal components.
    //
    // Strategy: vary both harmful and harmless independently along two axes,
    // so that after centering, the difference matrix has variation in both dims.
    let harmful: Vec<Tensor> = (0..20)
        .map(|i| {
            // Harmful prompts have varying signal along both dim 0 and dim 1
            let x = if i < 10 { 2.0f32 } else { 0.0 };
            let y = if i % 2 == 0 { 1.5f32 } else { 0.0 };
            Tensor::new(&[x, y, 0.0, 0.0], &device).unwrap()
        })
        .collect();
    let harmless: Vec<Tensor> = (0..20)
        .map(|_i| {
            // Harmless prompts are constant -- so centering gives zero for harmless,
            // and the diff matrix equals the centered harmful matrix, which has
            // two independent components.
            Tensor::new(&[0.0f32, 0.0, 0.0, 0.0], &device).unwrap()
        })
        .collect();

    let dirs = extract_directions_svd(&harmful, &harmless, 5, 0.1).unwrap();
    assert!(
        dirs.len() >= 2,
        "expected >= 2 directions, got {}",
        dirs.len()
    );
    // Singular values should be in descending order
    assert!(dirs[1].weight <= dirs[0].weight);
}

#[test]
fn test_extract_directions_insufficient_samples_falls_back() {
    let device = Device::Cpu;
    // Only 1 sample each -- should fall back to simple mean difference
    let harmful = vec![Tensor::new(&[1.0f32, 0.0, 0.0], &device).unwrap()];
    let harmless = vec![Tensor::new(&[-1.0f32, 0.0, 0.0], &device).unwrap()];

    let dirs = extract_directions_svd(&harmful, &harmless, 5, 0.1).unwrap();
    assert_eq!(dirs.len(), 1);
    // The fallback direction should align with dim 0
    let top_dir: Vec<f32> = dirs[0].direction.to_vec1().unwrap();
    assert!(
        top_dir[0].abs() > 0.9,
        "fallback dir should align with dim 0: {:?}",
        top_dir
    );
}

#[test]
fn test_power_iteration_recovers_known_direction() {
    let device = Device::Cpu;
    // Matrix with known singular structure: [[3, 0], [0, 1], [0, 0]]
    // Singular values should be 3 and 1, right singular vectors [1,0] and [0,1]
    let matrix = Tensor::new(&[[3.0f32, 0.0], [0.0, 1.0], [0.0, 0.0]], &device).unwrap();
    let (svs, vecs) = power_iteration_svd(&matrix, 2, 100).unwrap();

    assert!(
        (svs[0] - 3.0).abs() < 0.01,
        "first sv should be ~3.0, got {}",
        svs[0]
    );
    assert!(
        (svs[1] - 1.0).abs() < 0.01,
        "second sv should be ~1.0, got {}",
        svs[1]
    );

    // First right singular vector should be [1, 0] or [-1, 0]
    let v0: Vec<f32> = vecs[0].to_vec1().unwrap();
    assert!(
        v0[0].abs() > 0.99,
        "v0 should align with dim 0: {:?}",
        v0
    );
}

#[test]
fn test_power_iteration_single_direction() {
    let device = Device::Cpu;
    // Matrix where all signal is along dim 0
    let matrix = Tensor::new(&[[5.0f32, 0.0, 0.0], [5.0, 0.0, 0.0]], &device).unwrap();
    let (svs, vecs) = power_iteration_svd(&matrix, 3, 100).unwrap();

    // Should find exactly 1 significant direction
    assert!(!svs.is_empty());
    assert!(
        (svs[0] - 50.0f32.sqrt()).abs() < 0.1,
        "first sv should be sqrt(50) ~7.07, got {}",
        svs[0]
    );

    let v0: Vec<f32> = vecs[0].to_vec1().unwrap();
    assert!(
        v0[0].abs() > 0.99,
        "v0 should align with dim 0: {:?}",
        v0
    );
}

#[test]
fn test_energy_threshold_filters_weak_directions() {
    let device = Device::Cpu;
    // Strong signal along dim 0, very weak along dim 1
    let harmful: Vec<Tensor> = (0..10)
        .map(|i| {
            Tensor::new(
                &[10.0f32 + i as f32 * 0.1, 0.001 * i as f32, 0.0, 0.0],
                &device,
            )
            .unwrap()
        })
        .collect();
    let harmless: Vec<Tensor> = (0..10)
        .map(|i| {
            Tensor::new(
                &[-10.0f32 - i as f32 * 0.1, -0.001 * i as f32, 0.0, 0.0],
                &device,
            )
            .unwrap()
        })
        .collect();

    // High energy threshold should filter out weak directions
    let dirs = extract_directions_svd(&harmful, &harmless, 5, 0.5).unwrap();
    assert_eq!(
        dirs.len(),
        1,
        "high threshold should keep only 1 direction, got {}",
        dirs.len()
    );
}
