//! Tests for the safe tensor wrapper. These link against the shared library that CMake builds, so
//! run `cmake --build build --target flint` first, or point LIBWAIFU_LIB_DIR somewhere else.

use waifu::flint::{functional as F, Bound, DType, Device, Fp8Tensor, Tensor};

#[test]
fn reports_metadata() {
    let x = Tensor::zeros(&[2, 3], DType::Float, Device::Cpu).unwrap();

    assert_eq!(x.dim().unwrap(), 2);
    assert_eq!(x.shape(), vec![2, 3]);
    assert_eq!(x.shape_at(-1).unwrap(), 3);
    assert_eq!(x.numel(), 6);
    assert_eq!(x.dtype(), DType::Float);
    assert_eq!(x.device(), Device::Cpu);
    assert!(x.is_contiguous());
    assert_eq!(x.nbytes().unwrap(), 24);
    assert_eq!(x.to_vec_f32().unwrap(), vec![0.0; 6]);
}

#[test]
fn round_trips_element_data() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let x = Tensor::from_f32(&[2, 3], &data).unwrap();
    assert_eq!(x.to_vec_f32().unwrap(), data);

    let ints = vec![1i64, 2, 3, 4];
    let y = Tensor::from_i64(&[4], &ints).unwrap();
    assert_eq!(y.dtype(), DType::Long);
    assert_eq!(y.to_vec_i64().unwrap(), ints);
}

#[test]
fn rejects_data_that_does_not_fill_the_shape() {
    let error = Tensor::from_f32(&[2, 3], &[1.0, 2.0]).unwrap_err();

    assert!(error.is_invalid_arg(), "unexpected error: {error}");
    assert!(
        error.message().contains("data_size"),
        "unexpected error: {error}"
    );
}

#[test]
fn reshapes_without_copying() {
    let x = Tensor::from_f32(&[6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

    let view = x.view(&[2, 3]).unwrap();
    assert_eq!(view.shape(), vec![2, 3]);
    assert_eq!(
        view.to_vec_f32().unwrap(),
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    );

    let row = view.subtensor(1).unwrap();
    assert_eq!(row.shape(), vec![3]);
    assert_eq!(row.to_vec_f32().unwrap(), vec![3.0, 4.0, 5.0]);

    assert_eq!(x.unsqueeze(0).unwrap().shape(), vec![1, 6]);
    assert_eq!(x.unsqueeze(0).unwrap().squeeze(0).unwrap().shape(), vec![6]);
}

#[test]
fn transposes_and_packs_on_read() {
    let x = Tensor::from_f32(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let t = x.transpose(0, 1).unwrap();

    assert_eq!(t.shape(), vec![3, 2]);
    assert!(!t.is_contiguous(), "a transpose should not be contiguous");

    // Reading packs the elements, so the caller sees them in the transposed order.
    assert_eq!(t.to_vec_f32().unwrap(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert!(t.contiguous().unwrap().is_contiguous());
}

#[test]
fn slices_with_open_and_negative_bounds() {
    let x = Tensor::from_f32(&[6], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).unwrap();

    assert_eq!(
        x.slice(0, 1, 3).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0]
    );
    assert_eq!(
        x.slice(0, 4, Bound::End).unwrap().to_vec_f32().unwrap(),
        vec![4.0, 5.0]
    );
    assert_eq!(
        x.slice(0, Bound::End, 2).unwrap().to_vec_f32().unwrap(),
        vec![0.0, 1.0]
    );
    assert_eq!(
        x.slice(0, -2, Bound::End).unwrap().to_vec_f32().unwrap(),
        vec![4.0, 5.0]
    );
}

#[test]
fn clone_shares_storage_and_outlives_the_original() {
    let data = vec![7.0f32, 8.0];
    let clone = {
        let x = Tensor::from_f32(&[2], &data).unwrap();
        x.clone()
    };

    // The original handle is gone, but the storage is not.
    assert_eq!(clone.to_vec_f32().unwrap(), data);
}

#[test]
fn casts_between_element_types() {
    let x = Tensor::from_f32(&[3], &[1.0, 2.0, 3.0]).unwrap();
    let y = x.cast(DType::Float16).unwrap();

    assert_eq!(y.dtype(), DType::Float16);
    assert_eq!(
        y.cast(DType::Float).unwrap().to_vec_f32().unwrap(),
        vec![1.0, 2.0, 3.0]
    );
}

#[test]
fn reading_the_wrong_element_type_is_refused() {
    let x = Tensor::from_i64(&[2], &[1, 2]).unwrap();
    let error = x.to_vec_f32().unwrap_err();

    assert!(
        error.message().contains("Long"),
        "unexpected error: {error}"
    );
}

#[test]
fn debug_shows_the_shape() {
    let x = Tensor::zeros(&[2, 2], DType::Float, Device::Cpu).unwrap();
    let text = format!("{x:?}");

    assert!(text.contains("[2, 2]"), "unexpected debug output: {text}");
    assert!(text.contains("Float"), "unexpected debug output: {text}");
}

#[test]
fn multiplies_by_an_fp8_weight_on_the_cpu() {
    // No CUDA needed for this one: the CPU widens the weight while it packs it, so there is no
    // instruction for the machine to be missing.
    assert!(Fp8Tensor::is_available(Device::Cpu));

    // Magnitudes E4M3 holds exactly, and each of the four channels a different power of two away
    // from the next -- so the round trip is exact, and the channels come back right only if each
    // carried its own scale.
    const CODES: [f32; 8] = [448.0, -224.0, 112.0, 56.0, 0.0, -1.0, 2.0, -4.0];
    const SCALES: [f32; 4] = [1.0, 2.0, 16.0, 0.125];

    let data: Vec<f32> = SCALES
        .iter()
        .flat_map(|scale| CODES.iter().map(move |code| code * scale))
        .collect();
    let weight = Tensor::from_f32(&[4, 8], &data).unwrap();

    let quantized = Fp8Tensor::quantize(&weight).unwrap();
    assert_eq!(quantized.shape().unwrap(), (4, 8));
    assert_eq!(quantized.dequantize().unwrap().to_vec_f32().unwrap(), data);

    // One row of ones against it is each channel's own row sum, so a scale applied to the wrong
    // axis cannot pass.
    let a = Tensor::from_f32(&[1, 8], &[1.0; 8]).unwrap();
    let out = F::fp8_matmul(&a, &quantized).unwrap();
    assert_eq!(out.shape(), vec![1, 4]);

    let expected: Vec<f32> = SCALES
        .iter()
        .map(|scale| CODES.iter().sum::<f32>() * scale)
        .collect();
    for (got, want) in out.to_vec_f32().unwrap().iter().zip(expected.iter()) {
        assert!(
            (got - want).abs() <= want.abs() * 1e-5,
            "fp8 gave {got}, expected {want}"
        );
    }
}
