//! Tensor parallelism implementation using RustCUDA

use rustacuda::prelude::*;
use anyhow::Result;
use log::info;

/// Tensor parallelism manager using RustCUDA
pub struct TensorParallel {
    world_size: usize,
    rank: usize,
    device: Device,
    context: Context,
    stream: Stream,
}

impl TensorParallel {
    /// Create new tensor parallel manager
    pub fn new(world_size: usize, rank: usize, device: Device, context: Context, stream: Stream) -> Self {
        info!("Creating tensor parallel manager: world_size={}, rank={}", world_size, rank);
        Self { world_size, rank, device, context, stream }
    }
    
    /// Get world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }
    
    /// Get rank
    pub fn rank(&self) -> usize {
        self.rank
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
    
    /// Split tensor along last dimension
    pub fn split_tensor(&self, tensor: &[f32], dim: usize) -> Result<Vec<f32>> {
        let size = tensor.len();
        let chunk_size = size / self.world_size;
        let start = self.rank * chunk_size;
        let end = if self.rank == self.world_size - 1 {
            size
        } else {
            (self.rank + 1) * chunk_size
        };
        
        Ok(tensor[start..end].to_vec())
    }
    
    /// Concatenate tensors along last dimension
    pub fn concat_tensors(&self, tensors: &[Vec<f32>]) -> Result<Vec<f32>> {
        let mut result = Vec::new();
        for tensor in tensors {
            result.extend(tensor);
        }
        Ok(result)
    }
}
