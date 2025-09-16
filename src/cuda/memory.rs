//! CUDA memory management for PicoTron

use rustacuda::prelude::*;
use rustacuda::memory::DeviceBox;
use anyhow::Result;
use log::info;

/// CUDA memory manager
pub struct CudaMemory {
    device: Device,
    context: Context,
    stream: Stream,
}

impl CudaMemory {
    /// Create new CUDA memory manager
    pub fn new(device: Device, context: Context, stream: Stream) -> Self {
        Self { device, context, stream }
    }
    
    /// Allocate device memory
    pub fn allocate<T: DeviceCopy>(&self, size: usize) -> Result<DeviceBox<T>> {
        DeviceBox::new(&vec![T::default(); size])
    }
    
    /// Allocate device memory from slice
    pub fn allocate_from_slice<T: DeviceCopy>(&self, data: &[T]) -> Result<DeviceBox<T>> {
        DeviceBox::new(data)
    }
    
    /// Copy data from host to device
    pub fn copy_to_device<T: DeviceCopy>(&self, host_data: &[T], device_data: &mut DeviceBox<T>) -> Result<()> {
        device_data.copy_from(host_data)?;
        Ok(())
    }
    
    /// Copy data from device to host
    pub fn copy_to_host<T: DeviceCopy>(&self, device_data: &DeviceBox<T>, host_data: &mut [T]) -> Result<()> {
        device_data.copy_to(host_data)?;
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
