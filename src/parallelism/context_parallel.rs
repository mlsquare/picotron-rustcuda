//! Context parallelism implementation using RustCUDA

use rustacuda::prelude::*;
use anyhow::Result;
use log::info;

/// Context parallelism manager using RustCUDA
pub struct ContextParallel {
    world_size: usize,
    rank: usize,
    device: Device,
    context: Context,
    stream: Stream,
}

impl ContextParallel {
    /// Create new context parallel manager
    pub fn new(world_size: usize, rank: usize, device: Device, context: Context, stream: Stream) -> Self {
        info!("Creating context parallel manager: world_size={}, rank={}", world_size, rank);
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
    
    /// Split sequence along sequence dimension
    pub fn split_sequence(&self, sequence: &[u32], seq_len: usize) -> Result<Vec<u32>> {
        let chunk_size = seq_len / self.world_size;
        let start = self.rank * chunk_size;
        let end = if self.rank == self.world_size - 1 {
            seq_len
        } else {
            (self.rank + 1) * chunk_size
        };
        
        Ok(sequence[start..end].to_vec())
    }
    
    /// Concatenate sequences along sequence dimension
    pub fn concat_sequences(&self, sequences: &[Vec<u32>]) -> Result<Vec<u32>> {
        let mut result = Vec::new();
        for sequence in sequences {
            result.extend(sequence);
        }
        Ok(result)
    }
}
