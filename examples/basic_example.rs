//! Basic PicoTron RustCUDA example

use picotron_rustcuda::*;
use anyhow::Result;

fn main() -> Result<()> {
    // Initialize PicoTron
    init()?;
    
    println!("PicoTron RustCUDA Version: {}", version());
    
    // Create a simple configuration
    let mut config = PicoTronConfig::default();
    config.model.hidden_size = 512;  // Smaller model for demo
    config.model.num_hidden_layers = 4;
    config.model.num_attention_heads = 8;
    config.model.vocab_size = 1000;
    config.cuda.device_id = 0;
    config.validate()?;
    
    println!("Configuration validated successfully");
    println!("Model: {}", config.model.name);
    println!("Hidden Size: {}", config.model.hidden_size);
    println!("Attention Heads: {}", config.model.num_attention_heads);
    println!("Hidden Layers: {}", config.model.num_hidden_layers);
    println!("CUDA Device ID: {}", config.cuda.device_id);
    
    // Create model
    let model = PicoTronModel::new(config.model.clone(), config.cuda.device_id)?;
    let num_params = model.num_parameters();
    let model_size_mb = Utils::calculate_model_size_mb(num_params);
    
    println!("Model created successfully");
    println!("Number of parameters: {}", num_params);
    println!("Model size: {:.2} MB", model_size_mb);
    
    // Get device info
    let device_info = model.device().get_device_info()?;
    println!("CUDA Device: {}", device_info.name);
    println!("Compute Capability: {:?}", device_info.compute_capability);
    println!("Total Memory: {} MB", device_info.total_memory / (1024 * 1024));
    println!("Multiprocessors: {}", device_info.multiprocessor_count);
    
    // Create trainer
    let mut trainer = PicoTronTrainer::new(config.training.clone(), &model)?;
    
    // Create sample data
    let batch_size = 2;
    let seq_len = 10;
    let input_ids = Utils::create_random_input(batch_size, seq_len, config.model.vocab_size);
    let labels = Utils::create_random_labels(batch_size, seq_len, config.model.vocab_size);
    
    Utils::print_tensor_info(&input_ids, "input_ids");
    Utils::print_tensor_info(&labels, "labels");
    
    // Training step
    let loss = trainer.train_step(&model, &input_ids, Some(&labels))?;
    println!("Training loss: {:.4}", loss);
    
    // Evaluation step
    let eval_loss = trainer.eval_step(&model, &input_ids, Some(&labels))?;
    println!("Evaluation loss: {:.4}", eval_loss);
    
    // Test forward pass
    let logits = model.forward(&input_ids, None)?;
    println!("Forward pass completed, output size: {}", logits.len());
    
    // Save configuration
    config.to_json("config.json")?;
    println!("Configuration saved to config.json");
    
    // Test parallelism
    let device = model.device();
    let data_parallel = DataParallel::new(
        1, 0, 
        device.device.clone(), 
        device.context.clone(), 
        device.stream.clone()
    );
    let tensor_parallel = TensorParallel::new(
        1, 0, 
        device.device.clone(), 
        device.context.clone(), 
        device.stream.clone()
    );
    let pipeline_parallel = PipelineParallel::new(
        1, 0, 
        device.device.clone(), 
        device.context.clone(), 
        device.stream.clone()
    );
    let context_parallel = ContextParallel::new(
        1, 0, 
        device.device.clone(), 
        device.context.clone(), 
        device.stream.clone()
    );
    
    println!("Parallelism managers created successfully");
    println!("Data parallel world size: {}", data_parallel.world_size());
    println!("Tensor parallel world size: {}", tensor_parallel.world_size());
    println!("Pipeline parallel world size: {}", pipeline_parallel.world_size());
    println!("Context parallel world size: {}", context_parallel.world_size());
    
    // Test CUDA operations
    let operations = model.operations();
    println!("CUDA operations manager created successfully");
    
    // Synchronize device
    model.device().synchronize()?;
    println!("CUDA device synchronized successfully");
    
    Ok(())
}
