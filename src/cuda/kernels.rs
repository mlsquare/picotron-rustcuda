//! CUDA kernels for PicoTron

use rustacuda::prelude::*;
use anyhow::Result;
use log::info;

/// CUDA kernels manager
pub struct CudaKernels {
    device: Device,
    context: Context,
    stream: Stream,
}

impl CudaKernels {
    /// Create new CUDA kernels manager
    pub fn new(device: Device, context: Context, stream: Stream) -> Self {
        Self { device, context, stream }
    }
    
    /// Load CUDA module from PTX
    pub fn load_module(&self, ptx_source: &str) -> Result<Module> {
        Module::load_from_string(&self.context, ptx_source)
    }
    
    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get context
    pub fn context(&self) -> &Context {
        &self.context
    }
    
    /// Get stream
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
}

/// Matrix multiplication kernel
#[rustacuda::kernel]
pub fn matmul_kernel(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: u32,
    n: u32,
    k: u32,
) {
    let idx = rustacuda::thread::index_1d();
    let row = idx / n;
    let col = idx % n;
    
    if row < m && col < n {
        let mut sum = 0.0f32;
        for i in 0..k {
            sum += unsafe { *a.add((row * k + i) as usize) } * 
                   unsafe { *b.add((i * n + col) as usize) };
        }
        unsafe { *c.add((row * n + col) as usize) = sum; }
    }
}

/// Dot product kernel
#[rustacuda::kernel]
pub fn dot_product_kernel(
    x: *const f32,
    y: *const f32,
    result: *mut f32,
    n: u32,
) {
    let idx = rustacuda::thread::index_1d();
    
    if idx < n {
        let product = unsafe { *x.add(idx as usize) } * unsafe { *y.add(idx as usize) };
        unsafe { atomic_add(result, product); }
    }
}

/// Layer normalization kernel
#[rustacuda::kernel]
pub fn layer_norm_kernel(
    input: *const f32,
    output: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    batch_size: u32,
    seq_len: u32,
    hidden_size: u32,
    eps: f32,
) {
    let batch_idx = rustacuda::thread::index_1d() / seq_len;
    let seq_idx = rustacuda::thread::index_1d() % seq_len;
    
    if batch_idx < batch_size && seq_idx < seq_len {
        // Compute mean
        let mut mean = 0.0f32;
        for i in 0..hidden_size {
            mean += unsafe { *input.add((batch_idx * seq_len * hidden_size + seq_idx * hidden_size + i) as usize) };
        }
        mean /= hidden_size as f32;
        
        // Compute variance
        let mut variance = 0.0f32;
        for i in 0..hidden_size {
            let diff = unsafe { *input.add((batch_idx * seq_len * hidden_size + seq_idx * hidden_size + i) as usize) } - mean;
            variance += diff * diff;
        }
        variance /= hidden_size as f32;
        
        // Apply normalization
        let std_dev = (variance + eps).sqrt();
        for i in 0..hidden_size {
            let input_idx = (batch_idx * seq_len * hidden_size + seq_idx * hidden_size + i) as usize;
            let output_idx = input_idx;
            let normalized = (unsafe { *input.add(input_idx) } - mean) / std_dev;
            unsafe { *output.add(output_idx) = gamma[i as usize] * normalized + beta[i as usize]; }
        }
    }
}
