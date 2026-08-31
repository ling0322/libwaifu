// The MIT License (MIT)
//
// Copyright (c) 2026 Xiaoyang Chen
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//! That a batch is rows that do not read each other.
//!
//! Classifier free guidance sends the prompt and its absence through the U-Net as one batch of
//! two rather than as two passes. That is only the same answer if every operator on the way treats
//! the batch as independent rows -- and an operator that reduces over the wrong axis would not
//! fail loudly, it would quietly mix the two prompts together and give an image that is a little
//! wrong in a way no shape check can see.
//!
//! So each operator is given two unrelated rows at once, and each row is checked against what that
//! row gets on its own.

use waifu::flint::functional as F;
use waifu::flint::Tensor;
use waifu::{DType, Device};

/// How far apart two rows may be before something is wrong, per device.
///
/// Batching does not change what is computed, but it does change how: a GEMM over twice the rows
/// blocks differently and accumulates in a different order. That moves the last bit or two, and
/// nothing more. CUDA gets the looser number because its float type is half, and half's own
/// rounding is already near a percent -- which is still far below what this is looking for, since
/// an operator reading the wrong row is wrong by tens of percent, not by rounding.
fn tolerance(device: Device) -> f32 {
    match device {
        Device::Cuda => 1e-2,
        _ => 1e-5,
    }
}

fn devices() -> Vec<Device> {
    let mut devices = vec![Device::Cpu];
    if Device::Cuda.is_available() {
        devices.push(Device::Cuda);
    }
    devices
}

/// Values that vary without a pattern any of these operators could accidentally satisfy.
///
/// Made on the host rather than drawn from the tensor library's generator, because that generator
/// is one piece of state for the whole process and these tests run alongside each other: a seed
/// set by one of them lands in the middle of another, and what a test compares would then depend
/// on the order the threads happened to run in.
fn spread(shape: &[i32], device: Device, seed: u32) -> Tensor {
    let count: i32 = shape.iter().product();
    let mut state = seed | 1;
    let mut values = Vec::with_capacity(count as usize);
    for _ in 0..count {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        values.push((state >> 8) as f32 / (1 << 24) as f32 * 2.0 - 1.0);
    }

    Tensor::from_f32(shape, &values)
        .unwrap()
        .to_device(device)
        .unwrap()
        .cast(F::default_float_type(device).unwrap())
        .unwrap()
}

/// The largest gap between `batched` row by row and what each row gets alone.
fn row_gap(batched: &Tensor, alone: &[Tensor]) -> f32 {
    let mut worst: f32 = 0.0;
    for (index, one) in alone.iter().enumerate() {
        let row = batched.slice(0, index as i32, index as i32 + 1).unwrap();
        let got = row
            .contiguous()
            .unwrap()
            .cast(DType::Float)
            .unwrap()
            .to_vec_f32()
            .unwrap();
        let want = one.cast(DType::Float).unwrap().to_vec_f32().unwrap();
        assert_eq!(got.len(), want.len(), "row {index} is not the same size");

        for (a, b) in got.iter().zip(want.iter()) {
            let scale = a.abs().max(b.abs()).max(1e-3);
            worst = worst.max((a - b).abs() / scale);
        }
    }
    worst
}

/// Run `op` on two rows at once and on each row alone, and say how far apart the two are.
fn gap_of(rows: &[Tensor], op: impl Fn(&Tensor) -> Tensor) -> f32 {
    let batched = op(&F::cat(&rows[0], &rows[1], 0).unwrap());
    let alone: Vec<Tensor> = rows.iter().map(&op).collect();
    row_gap(&batched, &alone)
}

#[test]
fn a_convolution_reads_one_row_at_a_time() {
    for device in devices() {
        let rows = [spread(&[1, 8, 6, 6], device, 1), spread(&[1, 8, 6, 6], device, 2)];
        let weight = spread(&[4, 8, 3, 3], device, 3);
        let bias = spread(&[4], device, 4);

        let gap = gap_of(&rows, |x| F::conv2d(x, &weight, Some(&bias), 1, 1, 1, 1).unwrap());
        assert!(
            gap < tolerance(device),
            "conv2d on {device:?} mixes rows: {gap}"
        );
    }
}

#[test]
fn a_group_norm_takes_its_statistics_from_one_row() {
    // The one most likely to be wrong: a group's mean and variance are over the channels and
    // pixels of a single image, and an implementation that reduced over the batch as well would
    // still produce the right shape.
    for device in devices() {
        let rows = [
            spread(&[1, 8, 5, 5], device, 5),
            // Deliberately a different scale from the first, so that reducing over both rows
            // together lands somewhere neither row would land alone.
            F::mul_scalar(&spread(&[1, 8, 5, 5], device, 6), 7.0).unwrap(),
        ];
        let weight = spread(&[8], device, 7);
        let bias = spread(&[8], device, 8);

        let gap = gap_of(&rows, |x| F::group_norm(x, Some(&weight), Some(&bias), 4, 1e-5).unwrap());
        assert!(
            gap < tolerance(device),
            "group_norm on {device:?} mixes rows: {gap}"
        );
    }
}

#[test]
fn a_layer_norm_takes_its_statistics_from_one_row() {
    for device in devices() {
        let rows = [
            spread(&[1, 5, 16], device, 9),
            F::mul_scalar(&spread(&[1, 5, 16], device, 10), 7.0).unwrap(),
        ];
        let weight = spread(&[16], device, 11);
        let bias = spread(&[16], device, 12);

        let gap = gap_of(&rows, |x| F::layer_norm(x, Some(&weight), Some(&bias), 1e-5).unwrap());
        assert!(
            gap < tolerance(device),
            "layer_norm on {device:?} mixes rows: {gap}"
        );
    }
}

#[test]
fn a_matmul_keeps_its_batch_dimension_apart() {
    for device in devices() {
        let rows = [spread(&[1, 6, 16], device, 13), spread(&[1, 6, 16], device, 14)];
        let weight = spread(&[16, 24], device, 15);

        let gap = gap_of(&rows, |x| F::matmul(x, &weight).unwrap());
        assert!(
            gap < tolerance(device),
            "matmul on {device:?} mixes rows: {gap}"
        );
    }
}

#[test]
fn attention_does_not_look_across_the_batch() {
    // Every row of a batch attends over its own keys only. A kernel that folded the batch into
    // the key length would let one prompt read the other's, which is exactly what guidance must
    // not do.
    for device in devices() {
        let shape = [1, 4, 7, 16];
        let q = [spread(&shape, device, 16), spread(&shape, device, 17)];
        let k = [spread(&shape, device, 18), spread(&shape, device, 19)];
        let v = [spread(&shape, device, 20), spread(&shape, device, 21)];

        let batched = F::attention(
            &F::cat(&q[0], &q[1], 0).unwrap(),
            &F::cat(&k[0], &k[1], 0).unwrap(),
            &F::cat(&v[0], &v[1], 0).unwrap(),
            false,
        )
        .unwrap();
        let alone: Vec<Tensor> = (0..2)
            .map(|i| F::attention(&q[i], &k[i], &v[i], false).unwrap())
            .collect();

        let gap = row_gap(&batched, &alone);
        assert!(
            gap < tolerance(device),
            "attention on {device:?} mixes rows: {gap}"
        );
    }
}

#[test]
fn an_upsample_repeats_within_a_row() {
    for device in devices() {
        let rows = [spread(&[1, 4, 3, 3], device, 22), spread(&[1, 4, 3, 3], device, 23)];
        let gap = gap_of(&rows, |x| F::upsample_nearest2d(x, 2).unwrap());
        assert!(
            gap < tolerance(device),
            "upsample on {device:?} mixes rows: {gap}"
        );
    }
}

#[test]
fn one_row_and_two_rows_both_accumulate_in_float() {
    // Adding a row changes which kernel a matmul goes through: flint has a vector kernel for a
    // single row and falls to the GEMM for anything wider. Both are meant to accumulate in
    // float32 -- the vector kernel with a float running sum, the GEMM with CUBLAS_COMPUTE_32F --
    // so over a long reduction both should land where float32 lands, and the only thing left
    // between them is which way the last bit of the half they are written back as falls.
    //
    // Worth a test of its own because getting this wrong is quiet. A kernel that accumulated in
    // half would still return the right shape and still look about right, and the way it would
    // show is a picture that is subtly worse -- which is how the CUTLASS backend's accumulator
    // bug behaved before it was found. Over 2816 terms it would also be about fifty times the
    // error below, which is what makes this bar worth asserting rather than eyeballing.
    if !Device::Cuda.is_available() {
        return;
    }

    let (k, n) = (2816, 1280);
    let device = Device::Cuda;
    let x = spread(&[1, k], device, 41);
    let weight = spread(&[n, k], device, 43).transpose(0, 1).unwrap();

    let exact = F::matmul(
        &x.cast(DType::Float).unwrap(),
        &weight.contiguous().unwrap().cast(DType::Float).unwrap(),
    )
    .unwrap();

    let one_row = F::matmul(&x, &weight).unwrap();
    let two_rows = F::matmul(&F::cat(&x, &x, 0).unwrap(), &weight).unwrap();
    let two_rows = two_rows.slice(0, 0, 1).unwrap().contiguous().unwrap();

    // Writing the answer back as half costs about 1.4e-4 on its own, and that is nearly all of
    // what either path is away from float32. Accumulating in half instead would cost percent.
    let gap = |got: &Tensor| {
        let a = got.cast(DType::Float).unwrap().to_vec_f32().unwrap();
        let b = exact.to_vec_f32().unwrap();
        let error: f64 = a.iter().zip(&b).map(|(p, q)| ((p - q) as f64).powi(2)).sum();
        let scale: f64 = b.iter().map(|q| (*q as f64).powi(2)).sum();
        (error / scale).sqrt() as f32
    };

    let one = gap(&one_row);
    let two = gap(&two_rows);
    assert!(one < 1e-3, "the one row path is {one} from float32, which is not float accumulation");
    assert!(two < 1e-3, "the two row path is {two} from float32, which is not float accumulation");
}
