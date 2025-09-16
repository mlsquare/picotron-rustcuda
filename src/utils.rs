//! Utility functions for PicoTron RustCUDA

use anyhow::Result;
use log::info;
use rand::Rng;

/// Utility functions for PicoTron RustCUDA
pub struct Utils;

impl Utils {
    /// Create new utils instance
    pub fn new() -> Self {
        Self
    }
    
    /// Create random input tensor for testing
    pub fn create_random_input(batch_size: usize, seq_len: usize, vocab_size: usize) -> Vec<u32> {
        let mut rng = rand::thread_rng();
        (0..batch_size * seq_len)
            .map(|_| rng.gen_range(0..vocab_size as u32))
            .collect()
    }
    
    /// Create random labels for testing
    pub fn create_random_labels(batch_size: usize, seq_len: usize, vocab_size: usize) -> Vec<u32> {
        let mut rng = rand::thread_rng();
        (0..batch_size * seq_len)
            .map(|_| rng.gen_range(0..vocab_size as u32))
            .collect()
    }
    
    /// Print tensor information
    pub fn print_tensor_info(tensor: &[u32], name: &str) {
        info!("{}: length={}, dtype=u32", name, tensor.len());
    }
    
    /// Calculate model size in MB
    pub fn calculate_model_size_mb(num_parameters: usize) -> f64 {
        // Assuming float32 (4 bytes per parameter)
        (num_parameters * 4) as f64 / (1024.0 * 1024.0)
    }
}
