import torch
import torch.nn as nn
from kernelmind.core import ModelParser, GraphOptimizer
from kernelmind.agent import OptimizationAgent, DecisionEngine
from kernelmind.benchmarks import BenchmarkRunner, MetricsCollector
from kernelmind.kernels import KernelGenerator, MetalBackend
from kernelmind.utils import get_logger
from kernelmind.config import config, OptimizationLevel

logger = get_logger(__name__)

class SimpleTransformerBlock(nn.Module):
    
    def __init__(self, d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x, mask=None):
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_output)
        
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x

def run_transformer_example():
    print("\n" + "="*70)
    print("KERNELMIND: Transformer Layer Fusion Example")
    print("="*70 + "\n")
    
    logger.info("Creating Transformer block...")
    transformer = SimpleTransformerBlock(
        d_model=512,
        nhead=8,
        d_ff=2048,
        dropout=0.1
    )
    transformer.eval()
    
    device = config.get_device()
    transformer = transformer.to(device)
    
    seq_length = 128
    batch_size = 4
    sample_input = torch.randn(batch_size, seq_length, 512, device=device)
    
    logger.info("Parsing Transformer block into computational graph...")
    parser = ModelParser()
    graph = parser.parse_model(transformer, sample_input)
    
    print("\nGraph Summary:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Tensors: {len(graph.tensors)}")
    print(f"  Total Size: {graph.total_size_bytes() / 1e6:.2f} MB")
    print()
    
    logger.info("Analyzing for fusion opportunities...")
    optimizer = GraphOptimizer(graph)
    
    pre_opt_nodes = len(graph.nodes)
    optimized_graph = optimizer.optimize()
    post_opt_nodes = len(graph.nodes)
    
    print(f"Graph Optimization:")
    print(f"  Nodes Before: {pre_opt_nodes}")
    print(f"  Nodes After: {post_opt_nodes}")
    print(f"  Nodes Eliminated: {pre_opt_nodes - post_opt_nodes}")
    print()
    
    opt_summary = optimizer.get_optimization_summary()
    print("Optimization Types Applied:")
    for opt_type, count in opt_summary.items():
        print(f"  {opt_type}: {count}")
    print()
    
    logger.info("Running LLM-based analysis...")
    agent = OptimizationAgent()
    agent.optimize(graph)
    
    suggestions = agent.get_suggestions()
    print(f"LLM Suggestions: {len(suggestions)} optimizations proposed")
    
    logger.info("Making optimization decisions...")
    decision_engine = DecisionEngine(OptimizationLevel.HIGH)
    decisions = decision_engine.decide_optimizations(suggestions)
    
    print(f"Decisions: {len(decisions)} optimizations approved")
    for decision in decisions:
        print(f"  - {decision.action} (Expected Impact: {decision.expected_impact:.1%}, Risk: {decision.risk_level})")
    print()
    
    logger.info("Running performance benchmarks...")
    benchmark_runner = BenchmarkRunner()
    
    results = benchmark_runner.run(
        transformer,
        input_shapes=[(512, 512)],
        num_runs=50,
        batch_sizes=[1, 4, 8, 16]
    )
    
    print("Benchmark Results:")
    print("-" * 70)
    
    latencies = []
    for batch_key in sorted(results.keys()):
        result = results[batch_key]
        batch_size = batch_key.replace("batch_", "")
        latencies.append(result['mean_latency_ms'])
        print(f"\nBatch {batch_size}:")
        print(f"  Mean Latency: {result['mean_latency_ms']:.4f} ms")
        print(f"  Throughput: {result['throughput_samples_per_sec']:.2f} tokens/sec")
    
    print("\nPerformance Summary:")
    print(f"  Average Latency: {sum(latencies)/len(latencies):.4f} ms")
    print(f"  Min Latency: {min(latencies):.4f} ms")
    print(f"  Max Latency: {max(latencies):.4f} ms")
    print()
    
    logger.info("Generating optimized kernels...")
    kernel_gen = KernelGenerator()
    kernels = kernel_gen.generate_kernels(graph, backend="metal")
    logger.info(f"Generated {len(kernels)} kernels")
    
    logger.info("Compiling kernels...")
    backend = MetalBackend()
    compiled = sum(1 for kernel_name, kernel_code in kernels.items()
                   if backend.compile_kernel(kernel_name, kernel_code))
    logger.info(f"Compiled {compiled}/{len(kernels)} kernels")
    
    logger.info("Collecting final metrics...")
    metrics = MetricsCollector()
    
    for i, suggestion in enumerate(suggestions):
        metrics.record_optimization(
            opt_type=suggestion.get('type', 'unknown'),
            speedup=1.0 + suggestion.get('estimated_speedup_percent', 0) / 100,
            memory_reduction=0,
            accuracy_loss=0
        )
    
    metrics.save_metrics("transformer_optimization.json")
    
    print("="*70)
    print("Transformer optimization example completed!")
    print("="*70 + "\n")
    
    return {
        "model": transformer,
        "graph": graph,
        "benchmarks": results,
        "decisions": decisions,
        "metrics": metrics,
    }

if __name__ == "__main__":
    results = run_transformer_example()
