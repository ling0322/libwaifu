//! Raw declarations for the `flint/capi.h` C interface.
//!
//! Nothing here is safe to call directly: handles are raw pointers with manual lifetimes and every
//! function reports failure through a thread-local error rather than the type system. The wrappers
//! in [`crate`] exist to put those rules back.

use std::os::raw::{c_char, c_int, c_void};

pub const FL_OK: i32 = 0;
pub const FL_ERROR_INVALID_ARG: i32 = 0x0100;
pub const FL_ERROR_ABORTED: i32 = 0x0102;

/// Opaque storage handle, owned by whoever made it. Only ever held behind a pointer.
#[repr(C)]
pub struct FlTensorDataImpl {
    _private: [u8; 0],
}

pub type FlTensorData = *mut FlTensorDataImpl;

/// Opaque view handle: a shape, strides and an offset over storage it does not own.
#[repr(C)]
pub struct FlTensorViewImpl {
    _private: [u8; 0],
}

pub type FlTensorView = *mut FlTensorViewImpl;

/// Opaque handle to a copy still on its way. Only ever held behind a pointer.
#[repr(C)]
pub struct FlTransferImpl {
    _private: [u8; 0],
}

pub type FlTransfer = *mut FlTransferImpl;

/// Opaque handle on the operators of one device. Only ever held behind a pointer.
#[repr(C)]
pub struct FlOperatorsImpl {
    _private: [u8; 0],
}

pub type FlOperators = *mut FlOperatorsImpl;

pub type FlDType = c_int;
pub type FlDeviceType = c_int;

type V = FlTensorView;
type Ops = FlOperators;

extern "C" {
    pub fn fl_init();
    pub fn fl_is_device_available(device: FlDeviceType, out: *mut i32) -> i32;
    pub fn fl_get_last_error_code() -> i32;
    pub fn fl_get_last_error_message() -> *const c_char;

    pub fn fl_operators_create(device: FlDeviceType, out: *mut FlOperators) -> i32;
    pub fn fl_operators_destroy(operators: FlOperators);
    pub fn fl_operators_get_device(operators: FlOperators, out: *mut FlDeviceType) -> i32;
    pub fn fl_get_default_float_type(operators: Ops, out: *mut FlDType) -> i32;

    pub fn fl_tensor_data_create(
        device: FlDeviceType,
        dtype: FlDType,
        numel: i64,
        out: *mut FlTensorData,
    ) -> i32;
    pub fn fl_tensor_data_borrow(
        data: *const c_void,
        dtype: FlDType,
        numel: i64,
        out: *mut FlTensorData,
    ) -> i32;
    pub fn fl_tensor_data_destroy(data: FlTensorData);
    pub fn fl_tensor_data_get_host_ptr(data: FlTensorData, out: *mut *mut c_void) -> i32;

    pub fn fl_tensor_view_create(
        data: FlTensorData,
        shape: *const i32,
        stride: *const i32,
        ndim: i32,
        offset: i64,
        out: *mut FlTensorView,
    ) -> i32;
    pub fn fl_tensor_view_destroy(view: FlTensorView);

    pub fn fl_copy(operators: Ops, src: V, dest: V) -> i32;
    pub fn fl_cast(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_transfer(operators: Ops, src: V, dest: V) -> i32;
    pub fn fl_transfer_async(
        src: V,
        device: FlDeviceType,
        dest: *mut FlTensorData,
        out: *mut FlTransfer,
    ) -> i32;
    pub fn fl_transfer_wait(transfer: FlTransfer) -> i32;
    pub fn fl_transfer_wait_sync(transfer: FlTransfer) -> i32;
    pub fn fl_transfer_destroy(transfer: FlTransfer);

    pub fn fl_fill(operators: Ops, tensor: V, value: f32) -> i32;
    pub fn fl_rand(operators: Ops, out: V) -> i32;
    pub fn fl_randn(operators: Ops, out: V) -> i32;
    pub fn fl_manual_seed(operators: Ops, seed: u64) -> i32;
    pub fn fl_arange(operators: Ops, begin: i64, step: i64, out: V) -> i32;
    pub fn fl_causal_mask(operators: Ops, out: V) -> i32;

    pub fn fl_lookup(operators: Ops, table: V, indices: V, out: V) -> i32;
    pub fn fl_rotary_embedding(operators: Ops, positions: V, query: V, key: V, cache: V) -> i32;
    pub fn fl_rms_norm(operators: Ops, input: V, weight: V, eps: f32, out: V) -> i32;
    pub fn fl_layer_norm(operators: Ops, input: V, weight: V, bias: V, eps: f32, out: V) -> i32;
    pub fn fl_group_norm(
        operators: Ops,
        input: V,
        weight: V,
        bias: V,
        groups: i32,
        eps: f32,
        out: V,
    ) -> i32;
    pub fn fl_conv2d(
        operators: Ops,
        input: V,
        weight: V,
        bias: V,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        out: V,
    ) -> i32;
    pub fn fl_conv1d(
        operators: Ops,
        input: V,
        weight: V,
        bias: V,
        stride: i32,
        padding: i32,
        dilation: i32,
        groups: i32,
        out: V,
    ) -> i32;
    pub fn fl_conv_transpose1d(
        operators: Ops,
        input: V,
        weight: V,
        bias: V,
        stride: i32,
        padding: i32,
        output_padding: i32,
        groups: i32,
        out: V,
    ) -> i32;
    pub fn fl_snake(operators: Ops, input: V, alpha: V, beta: V, eps: f32, out: V) -> i32;
    pub fn fl_stft(
        operators: Ops,
        input: V,
        window: V,
        n_fft: i32,
        hop: i32,
        centered: i32,
        out: V,
    ) -> i32;
    pub fn fl_istft(
        operators: Ops,
        spectrum: V,
        window: V,
        n_fft: i32,
        hop: i32,
        centered: i32,
        out: V,
    ) -> i32;
    pub fn fl_upsample_nearest2d(operators: Ops, input: V, scale: i32, out: V) -> i32;
    pub fn fl_upsample_nearest1d(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_matmul(operators: Ops, a: V, b: V, out: V) -> i32;

    pub fn fl_fp8_available(device: FlDeviceType, out: *mut i32) -> i32;
    pub fn fl_fp8_quantize(x: V, data: V, channel_scale: V) -> i32;
    pub fn fl_fp8_dequantize(data: V, channel_scale: V, out: V) -> i32;
    pub fn fl_fp8_matmul(a: V, data: V, channel_scale: V, out: V) -> i32;
    pub fn fl_fp8_matmul_tensor_scale(a: V, data: V, scale: V, out: V) -> i32;

    pub fn fl_add(operators: Ops, a: V, b: V, out: V) -> i32;
    pub fn fl_sub(operators: Ops, a: V, b: V, out: V) -> i32;
    pub fn fl_mul(operators: Ops, a: V, b: V, out: V) -> i32;
    pub fn fl_div(operators: Ops, a: V, b: V, out: V) -> i32;
    pub fn fl_eq(operators: Ops, a: V, b: V, out: V) -> i32;
    pub fn fl_mul_scalar(operators: Ops, input: V, other: f32, out: V) -> i32;
    pub fn fl_div_scalar(operators: Ops, input: V, other: f32, out: V) -> i32;
    pub fn fl_mod_scalar(operators: Ops, input: V, other: i64, out: V) -> i32;

    pub fn fl_square(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_neg(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_abs(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_exp(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_log(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_round(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_sqrt(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_rsqrt(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_sigmoid(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_tanh(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_relu(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_gelu(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_silu(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_sin(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_cos(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_quick_gelu(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_softmax(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_swiglu(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_geglu(operators: Ops, input: V, out: V) -> i32;

    pub fn fl_sum(operators: Ops, input: V, dim: i32, out: V) -> i32;
    pub fn fl_cumsum(operators: Ops, input: V, dim: i32, out: V) -> i32;
    pub fn fl_max(operators: Ops, input: V, out: V) -> i32;
    pub fn fl_min(operators: Ops, input: V, out: V) -> i32;

    pub fn fl_attention(operators: Ops, q: V, k: V, v: V, causal: i32, out: V) -> i32;
    pub fn fl_paged_attention_available(out: *mut i32) -> i32;
    pub fn fl_paged_attention(
        operators: Ops,
        q: V,
        key_cache: V,
        value_cache: V,
        block_table: V,
        cu_seqlens_q: V,
        seqlens_k: V,
        max_q_len: i32,
        max_k_len: i32,
        causal: i32,
        out: V,
    ) -> i32;
    pub fn fl_store_kv_cache(
        operators: Ops,
        k: V,
        v: V,
        key_cache: V,
        value_cache: V,
        slot_mapping: V,
    ) -> i32;
    pub fn fl_sample_with_params(
        operators: Ops,
        logits: V,
        temperatures: V,
        top_ks: V,
        top_ps: V,
        out: V,
    ) -> i32;
    pub fn fl_repetition_penalty(operators: Ops, logits: V, history: V, weight: f32) -> i32;

    pub fn fl_all_close(operators: Ops, a: V, b: V, rtol: f32, atol: f32, out: *mut i32) -> i32;
    pub fn fl_all(operators: Ops, tensor: V, out: *mut i32) -> i32;
    pub fn fl_elem(operators: Ops, tensor: V, out: *mut f32) -> i32;
    pub fn fl_print(operators: Ops, tensor: V) -> i32;

    pub fn fl_memory_capture(device: FlDeviceType, out: *mut FlMemorySnapshot) -> i32;
    pub fn fl_memory_reset_peak_stats(device: FlDeviceType) -> i32;
    pub fn fl_memory_release_unused(device: FlDeviceType) -> i32;
    pub fn fl_set_fatal_handler(handler: Option<extern "C" fn()>);
    pub fn fl_set_log_sink(
        sink: Option<extern "C" fn(level: i32, source: *const c_char, message: *const c_char)>,
    );
    pub fn fl_set_log_level(level: i32);
    pub fn fl_release_memory();
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
