import torch
import torch.nn as nn
import torchvision.models as models
from kernelmind.core import ModelParser, GraphOptimizer
from kernelmind.agent import OptimizationAgent
from kernelmind.benchmarks import BenchmarkRunner, MetricsCollector
from kernelmind.kernels import KernelGenerator, MetalBackend
from kernelmind.utils import get_logger, format_bytes, format_time
from kernelmind.config import config

logger = get_logger(__name__)

def run_resnet_example():
    print("\n" + "="*70)
    print("KERNELMIND: ResNet-18 Optimization Example")
    print("="*70 + "\n")
    
    logger.info("Loading pre-trained ResNet-18...")
    model = models.resnet18(pretrained=False)
    model.eval()
    
    device = config.get_device()
    model = model.to(device)
    
    sample_input = torch.randn(1, 3, 224, 224, device=device)
    
    logger.info("Parsing ResNet-18 into computational graph...")
    parser = ModelParser()
    graph = parser.parse_model(model, sample_input)
    
    print("\nGraph Summary:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Tensors: {len(graph.tensors)}")
    print(f"  Total Size: {format_bytes(graph.total_size_bytes())}")
    print()
    
    logger.info("Optimizing computational graph...")
    optimizer = GraphOptimizer(graph)
    optimized_graph = optimizer.optimize()
    
    opt_summary = optimizer.get_optimization_summary()
    print("Optimization Summary:")
    for opt_type, count in opt_summary.items():
        print(f"  {opt_type}: {count}")
    print()
    
    logger.info("Getting LLM optimization suggestions...")
    agent = OptimizationAgent()
    agent.optimize(graph)
    
    suggestions = agent.get_suggestions()
    print(f"Generated {len(suggestions)} optimization suggestions")
    
    logger.info("Running performance benchmarks...")
    benchmark_runner = BenchmarkRunner()
    
    results = benchmark_runner.run(
        model,
        input_shapes=[(3, 224, 224)],
        num_runs=50,
        batch_sizes=[1, 4, 8, 16, 32]
    )
    
    print("\nBenchmark Results:")
    print("-" * 70)
    
    for batch_key in sorted(results.keys()):
        result = results[batch_key]
        batch_size = batch_key.replace("batch_", "")
        print(f"\nBatch Size {batch_size}:")
        print(f"  Mean Latency: {result['mean_latency_ms']:.4f} ms")
        print(f"  Median Latency: {result['median_latency_ms']:.4f} ms")
        print(f"  P95 Latency: {result['p95_latency_ms']:.4f} ms")
        print(f"  Throughput: {result['throughput_samples_per_sec']:.2f} samples/sec")
    print()
    
    logger.info("Profiling memory usage...")
    memory_profile = benchmark_runner.profile_memory(
        model,
        input_shapes=[(3, 224, 224)],
        batch_size=8
    )
    
    print("Memory Profile (Batch Size 8):")
    for key, value in memory_profile.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    print()
    
    logger.info("Collecting metrics...")
    metrics = MetricsCollector()
    
    for batch_key, result in results.items():
        batch_size = int(batch_key.replace("batch_", ""))
        metrics.record_benchmark(
            "ResNet18",
            batch_size=batch_size,
            latency_ms=result["mean_latency_ms"],
            throughput=result["throughput_samples_per_sec"],
            memory_mb=result.get("peak_memory_mb", 0)
        )
    
    metrics.save_metrics("resnet18_baseline.json")
    
    print("="*70)
    print("ResNet optimization example completed!")
    print("="*70 + "\n")
    
    return {
        "model": model,
        "graph": graph,
        "benchmarks": results,
        "metrics": metrics,
    }

if __name__ == "__main__":
    results = run_resnet_example()
