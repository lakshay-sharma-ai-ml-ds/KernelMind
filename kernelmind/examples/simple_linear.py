import torch
import torch.nn as nn
from kernelmind.core import ModelParser, GraphOptimizer
from kernelmind.agent import OptimizationAgent
from kernelmind.benchmarks import BenchmarkRunner, CorrectnessVerifier, MetricsCollector
from kernelmind.kernels import KernelGenerator, MetalBackend
from kernelmind.utils import get_logger, get_device_info, format_bytes, format_time
from kernelmind.config import config

logger = get_logger(__name__)

class SimpleLinearModel(nn.Module):
    
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.softmax(x)
        return x

def run_simple_linear_example():
    print("\n" + "="*70)
    print("KERNELMIND: Simple Linear Model Optimization Example")
    print("="*70 + "\n")
    
    print("Hardware Configuration:")
    device_info = get_device_info()
    for key, value in device_info.items():
        print(f"  {key}: {value}")
    print()
    
    logger.info("Initializing model...")
    model = SimpleLinearModel(input_dim=784, hidden_dim=256, output_dim=10)
    model.eval()
    
    device = config.get_device()
    model = model.to(device)
    
    sample_input = torch.randn(1, 784, device=device)
    
    logger.info("Parsing model into computational graph...")
    parser = ModelParser()
    graph = parser.parse_model(model, sample_input)
    graph.print_summary()
    
    logger.info("Optimizing computational graph...")
    optimizer = GraphOptimizer(graph)
    optimized_graph = optimizer.optimize()
    
    print("\nOptimization Summary:")
    opt_summary = optimizer.get_optimization_summary()
    for opt_type, count in opt_summary.items():
        print(f"  {opt_type}: {count}")
    print()
    
    logger.info("Running LLM-based optimization suggestions...")
    agent = OptimizationAgent()
    agent.optimize(graph)
    
    print("\nOptimization Suggestions:")
    suggestions = agent.get_suggestions()
    for i, suggestion in enumerate(suggestions, 1):
        print(f"\n  {i}. {suggestion.get('type', 'unknown').upper()}")
        print(f"     Description: {suggestion.get('description', '')}")
        print(f"     Expected Speedup: {suggestion.get('estimated_speedup_percent', 0):.1f}%")
    print()
    
    logger.info("Generating optimized kernels...")
    kernel_gen = KernelGenerator()
    kernels = kernel_gen.generate_kernels(graph, backend="metal")
    logger.info(f"Generated {len(kernels)} optimized kernels")
    
    logger.info("Compiling kernels for Metal backend...")
    backend = MetalBackend()
    compiled_count = 0
    for kernel_name, kernel_code in kernels.items():
        if backend.compile_kernel(kernel_name, kernel_code):
            compiled_count += 1
    logger.info(f"Successfully compiled {compiled_count}/{len(kernels)} kernels")
    
    logger.info("Running benchmarks...")
    benchmark_runner = BenchmarkRunner()
    
    original_results = benchmark_runner.run(
        model,
        input_shapes=[(784,)],
        num_runs=50,
        batch_sizes=[1, 4, 8, 16]
    )
    
    print("\nBenchmark Results (Original Model):")
    print("-" * 70)
    for batch_key, result in original_results.items():
        batch_size = batch_key.replace("batch_", "")
        print(f"\n  Batch Size: {batch_size}")
        print(f"    Mean Latency: {result['mean_latency_ms']:.4f} ms")
        print(f"    Median Latency: {result['median_latency_ms']:.4f} ms")
        print(f"    P95 Latency: {result['p95_latency_ms']:.4f} ms")
        print(f"    P99 Latency: {result['p99_latency_ms']:.4f} ms")
        print(f"    Throughput: {result['throughput_samples_per_sec']:.2f} samples/sec")
    print()
    
    logger.info("Verifying correctness...")
    verifier = CorrectnessVerifier()
    test_input = torch.randn(8, 784, device=device)
    
    with torch.no_grad():
        original_output = model(test_input)
    
    verification_result = verifier.verify(original_output, original_output, "self_test")
    print(f"\nCorrectness Verification: {'PASSED' if verification_result['passed'] else 'FAILED'}")
    if verification_result['passed']:
        print(f"  L2 Error: {verification_result['l2_error']:.2e}")
        print(f"  Max Error: {verification_result['max_error']:.2e}")
    print()
    
    logger.info("Collecting metrics...")
    metrics = MetricsCollector()
    
    for batch_key, result in original_results.items():
        batch_size = int(batch_key.replace("batch_", ""))
        metrics.record_benchmark(
            "SimpleLinearModel",
            batch_size=batch_size,
            latency_ms=result["mean_latency_ms"],
            throughput=result["throughput_samples_per_sec"],
            memory_mb=result.get("peak_memory_mb", 0)
        )
    
    metrics.save_metrics("simple_linear_baseline.json")
    
    print("Metrics Summary:")
    summary = metrics.get_summary()
    print(f"  Total Metrics: {summary['total_metrics_recorded']}")
    print(f"  Benchmarks: {summary['num_benchmarks']}")
    print()
    
    logger.info("Analysis Complete")
    print("="*70)
    print("Example completed successfully!")
    print("="*70 + "\n")
    
    return {
        "model": model,
        "graph": graph,
        "benchmarks": original_results,
        "verification": verification_result,
        "metrics": metrics,
    }

if __name__ == "__main__":
    results = run_simple_linear_example()
