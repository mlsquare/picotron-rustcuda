//! CUDA device management for PicoTron

use rustacuda::prelude::*;
use log::{info, warn, error};
use anyhow::Result;

/// CUDA device wrapper for PicoTron
pub struct PicoTronCudaDevice {
    pub device: Device,
    pub context: Context,
    pub stream: Stream,
}

impl PicoTronCudaDevice {
    /// Create a new PicoTron CUDA device
    pub fn new(device_id: usize) -> Result<Self> {
        info!("Initializing PicoTron CUDA device: {}", device_id);
        
        // Initialize CUDA
        rustacuda::init(CudaFlags::empty())?;
        
        // Get device
        let device = Device::get_device(device_id as u32)?;
        
        // Create context
        let context = device.create_context()?;
        
        // Create stream
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;
        
        info!("PicoTron CUDA device initialized successfully");
        info!("Device: {}", device.name()?);
        info!("Compute capability: {:?}", device.compute_capability()?);
        info!("Memory: {} MB", device.total_memory()? / (1024 * 1024));
        
        Ok(Self {
            device,
            context,
            stream,
        })
    }
    
    /// Get device information
    pub fn get_device_info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            name: self.device.name()?,
            compute_capability: self.device.compute_capability()?,
            total_memory: self.device.total_memory()?,
            multiprocessor_count: self.device.multiprocessor_count()?,
            max_threads_per_block: self.device.max_threads_per_block()?,
        })
    }
    
    /// Get context
    pub fn context(&self) -> &Context {
        &self.context
    }
    
    /// Get stream
    pub fn stream(&self) -> &Stream {
        &self.stream
    }
    
    /// Synchronize stream
    pub fn synchronize(&self) -> Result<()> {
        self.stream.synchronize()?;
        Ok(())
    }
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: u64,
    pub multiprocessor_count: u32,
    pub max_threads_per_block: u32,
}

impl Drop for PicoTronCudaDevice {
    fn drop(&mut self) {
        info!("Dropping PicoTron CUDA device");
    }
}
