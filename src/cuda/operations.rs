//! CUDA operations for PicoTron

use rustacuda::prelude::*;
use anyhow::Result;
use log::info;

/// CUDA operations manager
pub struct CudaOperations {
    device: Device,
    context: Context,
    stream: Stream,
}

impl CudaOperations {
    /// Create new CUDA operations manager
    pub fn new(device: Device, context: Context, stream: Stream) -> Self {
        Self { device, context, stream }
    }
    
    /// Matrix multiplication operation
    pub fn matmul(&self, a: &DeviceBox<f32>, b: &DeviceBox<f32>, c: &mut DeviceBox<f32>, m: u32, n: u32, k: u32) -> Result<()> {
        let block_size = 16;
        let grid_size = ((m * n + block_size - 1) / block_size) as u32;
        
        unsafe {
            launch!(matmul_kernel<<<grid_size, block_size, 0, self.stream>>>(
                a.as_device_ptr(),
                b.as_device_ptr(),
                c.as_device_ptr(),
                m,
                n,
                k
            ))?;
        }
        
        Ok(())
    }
    
    /// Dot product operation
    pub fn dot_product(&self, x: &DeviceBox<f32>, y: &DeviceBox<f32>, result: &mut DeviceBox<f32>, n: u32) -> Result<()> {
        let block_size = 256;
        let grid_size = ((n + block_size - 1) / block_size) as u32;
        
        unsafe {
            launch!(dot_product_kernel<<<grid_size, block_size, 0, self.stream>>>(
                x.as_device_ptr(),
                y.as_device_ptr(),
                result.as_device_ptr(),
                n
            ))?;
        }
        
        Ok(())
    }
    
    /// Layer normalization operation
    pub fn layer_norm(&self, input: &DeviceBox<f32>, output: &mut DeviceBox<f32>, gamma: &DeviceBox<f32>, beta: &DeviceBox<f32>, batch_size: u32, seq_len: u32, hidden_size: u32, eps: f32) -> Result<()> {
        let block_size = 256;
        let grid_size = ((batch_size * seq_len + block_size - 1) / block_size) as u32;
        
        unsafe {
            launch!(layer_norm_kernel<<<grid_size, block_size, 0, self.stream>>>(
                input.as_device_ptr(),
                output.as_device_ptr(),
                gamma.as_device_ptr(),
                beta.as_device_ptr(),
                batch_size,
                seq_len,
                hidden_size,
                eps
            ))?;
        }
        
        Ok(())
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
