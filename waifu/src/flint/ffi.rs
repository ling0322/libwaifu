//! Raw declarations for the `flint/capi.h` C interface.
//!
//! Nothing here is safe to call directly: handles are raw pointers with manual lifetimes and every
//! function reports failure through a thread-local error rather than the type system. The wrappers
//! in [`crate`] exist to put those rules back.

use std::os::raw::{c_char, c_int, c_void};

pub const FL_OK: i32 = 0;
pub const FL_ERROR_INVALID_ARG: i32 = 0x0100;
pub const FL_ERROR_ABORTED: i32 = 0x0102;

/// Open end of a slice range, matching `FL_NONE`.
pub const FL_NONE: i32 = i32::MIN;

/// Opaque tensor handle. Only ever held behind a pointer.
#[repr(C)]
pub struct FlTensorImpl {
    _private: [u8; 0],
}

pub type FlTensor = *mut FlTensorImpl;

/// Opaque handle to a copy still on its way. Only ever held behind a pointer.
#[repr(C)]
pub struct FlFutureTensorImpl {
    _private: [u8; 0],
}

pub type FlFutureTensor = *mut FlFutureTensorImpl;

/// Opaque handle on the operators of one device. Only ever held behind a pointer.
#[repr(C)]
pub struct FlOperatorsImpl {
    _private: [u8; 0],
}

pub type FlOperators = *mut FlOperatorsImpl;

pub type FlDType = c_int;
pub type FlDeviceType = c_int;

extern "C" {
    pub fn fl_init();
    pub fn fl_is_device_available(device: FlDeviceType, out: *mut i32) -> i32;
    pub fn fl_get_last_error_code() -> i32;
    pub fn fl_get_last_error_message() -> *const c_char;

    pub fn fl_operators_create(device: FlDeviceType, out: *mut FlOperators) -> i32;
    pub fn fl_operators_destroy(operators: FlOperators);
    pub fn fl_operators_get_device(operators: FlOperators, out: *mut FlDeviceType) -> i32;

    pub fn fl_tensor_zeros(
        operators: FlOperators,
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_empty(
        operators: FlOperators,
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_host_empty(
        operators: FlOperators,
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_from_data(
        operators: FlOperators,
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        data: *const c_void,
        data_size: i64,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_clone(tensor: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_host_data(tensor: FlTensor, out: *mut *mut c_void, nbytes: *mut i64) -> i32;

    pub fn fl_tensor_destroy(tensor: FlTensor);

    pub fn fl_tensor_get_dim(tensor: FlTensor, out: *mut i32) -> i32;
    pub fn fl_tensor_get_shape(tensor: FlTensor, dim: i32, out: *mut i32) -> i32;
    pub fn fl_tensor_get_stride(tensor: FlTensor, dim: i32, out: *mut i32) -> i32;
    pub fn fl_tensor_get_numel(tensor: FlTensor, out: *mut i64) -> i32;
    pub fn fl_tensor_get_dtype(tensor: FlTensor, out: *mut FlDType) -> i32;
    pub fn fl_tensor_get_device(tensor: FlTensor, out: *mut FlDeviceType) -> i32;
    pub fn fl_tensor_is_contiguous(tensor: FlTensor, out: *mut i32) -> i32;

    pub fn fl_tensor_view(
        tensor: FlTensor,
        shape: *const i32,
        ndim: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_transpose(tensor: FlTensor, dim0: i32, dim1: i32, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_slice(
        tensor: FlTensor,
        dim: i32,
        begin: i32,
        end: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_subtensor(tensor: FlTensor, index: i32, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_unsqueeze(tensor: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_squeeze(tensor: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_tensor_contiguous(
        operators: FlOperators,
        tensor: FlTensor,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_to_device(
        operators: FlOperators,
        tensor: FlTensor,
        device: FlDeviceType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_cast(
        operators: FlOperators,
        tensor: FlTensor,
        dtype: FlDType,
        out: *mut FlTensor,
    ) -> i32;

    pub fn fl_tensor_get_nbytes(tensor: FlTensor, out: *mut i64) -> i32;
    pub fn fl_tensor_copy_to_host(
        operators: FlOperators,
        tensor: FlTensor,
        buffer: *mut c_void,
        buffer_size: i64,
    ) -> i32;

    pub fn fl_arange(
        operators: FlOperators,
        begin: i64,
        end: i64,
        step: i64,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_rand(
        operators: FlOperators,
        shape: *const i32,
        ndim: i32,
        dtype: FlDType,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_randn(
        operators: FlOperators,
        shape: *const i32,
        ndim: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_manual_seed(operators: FlOperators, seed: u64) -> i32;

    pub fn fl_lookup(
        operators: FlOperators,
        table: FlTensor,
        indices: FlTensor,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_rotary_embedding(
        operators: FlOperators,
        positions: FlTensor,
        query: FlTensor,
        key: FlTensor,
        rotary_cache: FlTensor,
    ) -> i32;
    pub fn fl_rms_norm(
        operators: FlOperators,
        input: FlTensor,
        weight: FlTensor,
        eps: f32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_matmul(operators: FlOperators, a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_layer_norm(
        operators: FlOperators,
        input: FlTensor,
        weight: FlTensor,
        bias: FlTensor,
        eps: f32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_quick_gelu(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_conv2d(
        operators: FlOperators,
        input: FlTensor,
        weight: FlTensor,
        bias: FlTensor,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_conv1d(
        operators: FlOperators,
        input: FlTensor,
        weight: FlTensor,
        bias: FlTensor,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_conv_transpose1d(
        operators: FlOperators,
        input: FlTensor,
        weight: FlTensor,
        bias: FlTensor,
        stride: i32,
        padding: i32,
        output_padding: i32,
        groups: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_snake(
        operators: FlOperators,
        input: FlTensor,
        alpha: FlTensor,
        beta: FlTensor,
        eps: f32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_stft(
        operators: FlOperators,
        input: FlTensor,
        window: FlTensor,
        n_fft: i32,
        hop: i32,
        centered: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_istft(
        operators: FlOperators,
        spectrum: FlTensor,
        window: FlTensor,
        n_fft: i32,
        hop: i32,
        centered: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_group_norm(
        operators: FlOperators,
        input: FlTensor,
        weight: FlTensor,
        bias: FlTensor,
        groups: i32,
        eps: f32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_upsample_nearest2d(
        operators: FlOperators,
        input: FlTensor,
        scale: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_paged_attention_available(out: *mut i32) -> i32;
    pub fn fl_fp8_available(device: FlDeviceType, out: *mut i32) -> i32;
    pub fn fl_fp8_quantize(x: FlTensor, data: *mut FlTensor, channel_scale: *mut FlTensor) -> i32;
    pub fn fl_fp8_dequantize(data: FlTensor, channel_scale: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_fp8_matmul(
        a: FlTensor,
        data: FlTensor,
        channel_scale: FlTensor,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_fp8_matmul_tensor_scale(
        a: FlTensor,
        data: FlTensor,
        scale: FlTensor,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_tensor_to_device_async(
        tensor: FlTensor,
        device: FlDeviceType,
        out: *mut FlFutureTensor,
    ) -> i32;
    pub fn fl_future_tensor_take(future: FlFutureTensor, out: *mut FlTensor) -> i32;
    pub fn fl_future_tensor_take_sync(future: FlFutureTensor, out: *mut FlTensor) -> i32;
    pub fn fl_future_tensor_destroy(future: FlFutureTensor);
    pub fn fl_mul(operators: FlOperators, a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_div(operators: FlOperators, a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_add(operators: FlOperators, a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sub(operators: FlOperators, a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_eq(operators: FlOperators, a: FlTensor, b: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_mul_scalar(
        operators: FlOperators,
        input: FlTensor,
        other: f32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_div_scalar(
        operators: FlOperators,
        input: FlTensor,
        other: f32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_mod_scalar(
        operators: FlOperators,
        input: FlTensor,
        other: i64,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_square(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_neg(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_abs(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_exp(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sqrt(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_rsqrt(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sigmoid(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_tanh(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_relu(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_gelu(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_silu(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sin(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_cos(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_softmax(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_swiglu(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_geglu(operators: FlOperators, input: FlTensor, out: *mut FlTensor) -> i32;
    pub fn fl_sum(operators: FlOperators, input: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_max(operators: FlOperators, input: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_min(operators: FlOperators, input: FlTensor, dim: i32, out: *mut FlTensor) -> i32;
    pub fn fl_cat(
        operators: FlOperators,
        a: FlTensor,
        b: FlTensor,
        dim: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_causal_mask(operators: FlOperators, max_len: i32, out: *mut FlTensor) -> i32;

    pub fn fl_attention(
        operators: FlOperators,
        q: FlTensor,
        k: FlTensor,
        v: FlTensor,
        causal: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_paged_attention(
        operators: FlOperators,
        q: FlTensor,
        key_cache: FlTensor,
        value_cache: FlTensor,
        block_table: FlTensor,
        cu_seqlens_q: FlTensor,
        seqlens_k: FlTensor,
        max_q_len: i32,
        max_k_len: i32,
        causal: i32,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_store_kv_cache(
        operators: FlOperators,
        k: FlTensor,
        v: FlTensor,
        key_cache: FlTensor,
        value_cache: FlTensor,
        slot_mapping: FlTensor,
    ) -> i32;

    pub fn fl_sample_with_params(
        operators: FlOperators,
        logits: FlTensor,
        temperatures: FlTensor,
        top_ks: FlTensor,
        top_ps: FlTensor,
        out: *mut FlTensor,
    ) -> i32;
    pub fn fl_repetition_penalty(
        operators: FlOperators,
        logits: FlTensor,
        history: FlTensor,
        weight: f32,
    ) -> i32;

    pub fn fl_copy(operators: FlOperators, src: FlTensor, dest: FlTensor) -> i32;
    pub fn fl_fill(operators: FlOperators, tensor: FlTensor, value: f32) -> i32;
    pub fn fl_all_close(
        operators: FlOperators,
        a: FlTensor,
        b: FlTensor,
        rtol: f32,
        atol: f32,
        out: *mut i32,
    ) -> i32;
    pub fn fl_all(operators: FlOperators, tensor: FlTensor, out: *mut i32) -> i32;
    pub fn fl_elem(operators: FlOperators, tensor: FlTensor, out: *mut f32) -> i32;
    pub fn fl_get_default_float_type(operators: FlOperators, out: *mut FlDType) -> i32;
    pub fn fl_print(operators: FlOperators, tensor: FlTensor) -> i32;

    pub fn fl_memory_capture(device: FlDeviceType, out: *mut FlMemorySnapshot) -> i32;
    pub fn fl_memory_reset_peak_stats(device: FlDeviceType) -> i32;
    pub fn fl_memory_release_unused(device: FlDeviceType) -> i32;
    pub fn fl_set_fatal_handler(handler: Option<extern "C" fn()>);
}

/// Mirrors `fl_memory_snapshot_t`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct FlMemorySnapshot {
    pub total: i64,
    pub free: i64,
    pub allocated: i64,
    pub peak_allocated: i64,
}
