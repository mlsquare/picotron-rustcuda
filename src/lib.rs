//! PicoTron RustCUDA Implementation
//! 
//! A minimalistic distributed training framework for LLaMA-like models
//! using native CUDA for maximum performance on NVIDIA GPUs.

pub mod config;
pub mod model;
pub mod training;
pub mod parallelism;
pub mod cuda;
pub mod utils;

pub use config::*;
pub use model::*;
pub use training::*;
pub use parallelism::*;
pub use cuda::*;
pub use utils::*;

use log::info;
use anyhow::Result;

/// Initialize PicoTron with logging
pub fn init() -> Result<()> {
    env_logger::init();
    info!("Initializing PicoTron RustCUDA");
    Ok(())
}

/// PicoTron version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Get PicoTron version
pub fn version() -> &'static str {
    VERSION
}
