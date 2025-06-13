def benchmark_model_performance():
    """Benchmark model performance across different settings."""
    model = load_model("checkpoints/base_model.pt")
    
    results = {}
    
    for seq_len in [1024, 2048, 4096, 8192, 16384, 32768]:
        # Memory usage
        memory_usage = measure_memory_usage(model, seq_len)
        
        # Throughput
        throughput = measure_throughput(model, seq_len)
        
        # FLOPs
        flops = measure_flops(model, seq_len)
        
        results[seq_len] = {
            'memory_gb': memory_usage,
            'tokens_per_second': throughput,
            'flops': flops
        }
    
    return results