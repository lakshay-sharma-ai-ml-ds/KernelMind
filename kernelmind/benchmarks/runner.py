import time
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from ..core.graph import ComputationalGraph
from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class BenchmarkRunner:
    
    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.baseline_results: Optional[Dict] = None
        self.batch_sizes = config.BENCHMARK_BATCH_SIZES
        self.warmup_runs = config.BENCHMARK_WARMUP_RUNS
        self.measure_runs = config.BENCHMARK_MEASURE_RUNS
    
    def run(self, model: torch.nn.Module, input_shapes: List[Tuple],
            num_runs: int = 100, batch_sizes: Optional[List[int]] = None) -> Dict:
        logger.info(f"Running benchmarks with {num_runs} runs")
        
        if batch_sizes is None:
            batch_sizes = self.batch_sizes
        
        results = {}
        
        for batch_size in batch_sizes:
            logger.info(f"Benchmarking batch size {batch_size}")
            result = self._benchmark_batch(model, input_shapes, batch_size, num_runs)
            results[f"batch_{batch_size}"] = result
        
        self.results = results
        return results
    
    def _benchmark_batch(self, model: torch.nn.Module, 
                        input_shapes: List[Tuple], 
                        batch_size: int, num_runs: int) -> Dict:
        
        device = config.get_device()
        model = model.to(device)
        model.eval()
        
        inputs = self._create_dummy_inputs(input_shapes, batch_size, device)
        
        for _ in range(self.warmup_runs):
            with torch.no_grad():
                _ = model(*inputs)
        
        if device == "mps":
            torch.mps.synchronize()
        elif device == "cuda":
            torch.cuda.synchronize()
        
        times = []
        memory_peak = 0
        
        for _ in range(num_runs):
            torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
            
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = model(*inputs)
            
            if device == "mps":
                torch.mps.synchronize()
            elif device == "cuda":
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            elapsed = (end_time - start_time) * 1000
            times.append(elapsed)
            
            if device == "cuda":
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024
                memory_peak = max(memory_peak, peak_mem)
        
        times = np.array(times)
        
        return {
            "mean_latency_ms": float(np.mean(times)),
            "median_latency_ms": float(np.median(times)),
            "min_latency_ms": float(np.min(times)),
            "max_latency_ms": float(np.max(times)),
            "std_latency_ms": float(np.std(times)),
            "p95_latency_ms": float(np.percentile(times, 95)),
            "p99_latency_ms": float(np.percentile(times, 99)),
            "throughput_samples_per_sec": float(1000.0 / np.mean(times) * batch_size),
            "peak_memory_mb": memory_peak if device == "cuda" else 0,
        }
    
    def compare_versions(self, original_results: Dict, 
                        optimized_results: Dict) -> Dict:
        logger.info("Comparing original vs optimized versions")
        
        comparison = {}
        
        for batch_key in original_results:
            if batch_key not in optimized_results:
                continue
            
            orig = original_results[batch_key]
            opt = optimized_results[batch_key]
            
            speedup = orig["mean_latency_ms"] / opt["mean_latency_ms"]
            memory_reduction = 0
            
            if orig.get("peak_memory_mb", 0) > 0 and opt.get("peak_memory_mb", 0) > 0:
                memory_reduction = (1 - opt["peak_memory_mb"] / orig["peak_memory_mb"]) * 100
            
            comparison[batch_key] = {
                "latency_speedup": speedup,
                "throughput_improvement_percent": (speedup - 1) * 100,
                "memory_reduction_percent": memory_reduction,
                "original_latency_ms": orig["mean_latency_ms"],
                "optimized_latency_ms": opt["mean_latency_ms"],
            }
        
        return comparison
    
    def _create_dummy_inputs(self, input_shapes: List[Tuple], 
                            batch_size: int, device: str) -> List[torch.Tensor]:
        inputs = []
        
        for shape in input_shapes:
            if isinstance(shape, tuple):
                actual_shape = (batch_size,) + shape
            else:
                actual_shape = (batch_size, shape)
            
            dummy_input = torch.randn(actual_shape, device=device, dtype=torch.float32)
            inputs.append(dummy_input)
        
        return inputs
    
    def profile_memory(self, model: torch.nn.Module, 
                      input_shapes: List[Tuple], batch_size: int) -> Dict:
        logger.info("Profiling memory usage")
        
        device = config.get_device()
        model = model.to(device)
        model.eval()
        
        inputs = self._create_dummy_inputs(input_shapes, batch_size, device)
        
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            with torch.no_grad():
                _ = model(*inputs)
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            allocated = torch.cuda.memory_allocated() / 1024 / 1024
            
            return {
                "peak_memory_mb": peak_memory,
                "allocated_memory_mb": allocated,
                "batch_size": batch_size,
                "memory_per_sample_mb": peak_memory / batch_size,
            }
        else:
            import psutil
            process = psutil.Process()
            
            with torch.no_grad():
                mem_before = process.memory_info().rss / 1024 / 1024
                _ = model(*inputs)
                mem_after = process.memory_info().rss / 1024 / 1024
            
            return {
                "memory_used_mb": mem_after - mem_before,
                "peak_memory_mb": mem_after,
                "batch_size": batch_size,
            }
    
    def get_summary(self) -> Dict:
        if not self.results:
            return {"status": "No benchmarks run yet"}
        
        summary = {
            "total_batch_sizes": len(self.results),
            "results_by_batch": self.results,
        }
        
        latencies = []
        for batch_result in self.results.values():
            latencies.append(batch_result["mean_latency_ms"])
        
        if latencies:
            summary["average_latency_ms"] = float(np.mean(latencies))
            summary["best_latency_ms"] = float(np.min(latencies))
            summary["worst_latency_ms"] = float(np.max(latencies))
        
        return summary
