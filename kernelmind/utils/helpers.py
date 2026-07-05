from typing import Tuple, Union

def format_bytes(num_bytes: Union[int, float]) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} PB"

def format_time(milliseconds: float) -> str:
    if milliseconds < 1:
        return f"{milliseconds * 1000:.2f} µs"
    elif milliseconds < 1000:
        return f"{milliseconds:.2f} ms"
    else:
        seconds = milliseconds / 1000
        if seconds < 60:
            return f"{seconds:.2f} s"
        else:
            minutes = seconds / 60
            return f"{minutes:.2f} min"

def format_throughput(samples_per_second: float) -> str:
    if samples_per_second < 1e3:
        return f"{samples_per_second:.2f} samples/sec"
    elif samples_per_second < 1e6:
        return f"{samples_per_second / 1e3:.2f} K samples/sec"
    elif samples_per_second < 1e9:
        return f"{samples_per_second / 1e6:.2f} M samples/sec"
    else:
        return f"{samples_per_second / 1e9:.2f} G samples/sec"

def calculate_flops(batch_size: int, seq_length: int, hidden_size: int) -> int:
    return 2 * batch_size * seq_length * hidden_size * hidden_size

def format_flops(flops: Union[int, float]) -> str:
    if flops < 1e3:
        return f"{flops:.2f} FLOPs"
    elif flops < 1e6:
        return f"{flops / 1e3:.2f} KFLOPs"
    elif flops < 1e9:
        return f"{flops / 1e6:.2f} MFLOPs"
    elif flops < 1e12:
        return f"{flops / 1e9:.2f} GFLOPs"
    else:
        return f"{flops / 1e12:.2f} TFLOPs"

def format_percentage(value: float, total: float) -> str:
    if total == 0:
        return "0.00%"
    percentage = (value / total) * 100
    return f"{percentage:.2f}%"

def calculate_efficiency(flops: float, memory_bandwidth_gbs: float, 
                        bytes_accessed: float) -> float:
    memory_bound_flops = memory_bandwidth_gbs * 1e9 * bytes_accessed / 8
    return min(flops / memory_bound_flops, 1.0) * 100

def format_config_dict(config_dict: dict, indent: int = 0) -> str:
    lines = []
    prefix = "  " * indent
    
    for key, value in sorted(config_dict.items()):
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(format_config_dict(value, indent + 1))
        elif isinstance(value, (int, float, str, bool)):
            lines.append(f"{prefix}{key}: {value}")
        else:
            lines.append(f"{prefix}{key}: {str(value)}")
    
    return "\n".join(lines)

def get_peak_bandwidth_gbs(device: str) -> float:
    bandwidth_map = {
        "mps": 120.0,
        "cuda_a100": 2000.0,
        "cuda_v100": 900.0,
        "cuda_t4": 320.0,
        "cpu": 50.0,
    }
    return bandwidth_map.get(device, 50.0)

def calculate_roofline_performance(arithmetic_intensity: float, 
                                   device: str) -> Tuple[float, str]:
    peak_flops_tflops = {
        "mps": 2.0,
        "cuda_a100": 312.0,
        "cuda_v100": 125.0,
        "cuda_t4": 65.0,
        "cpu": 0.2,
    }.get(device, 0.2)
    
    peak_bandwidth = get_peak_bandwidth_gbs(device)
    
    ridge_point = peak_flops_tflops * 1e12 / (peak_bandwidth * 1e9)
    
    if arithmetic_intensity < ridge_point:
        performance_tflops = arithmetic_intensity * peak_bandwidth
        bottleneck = "memory_bound"
    else:
        performance_tflops = peak_flops_tflops
        bottleneck = "compute_bound"
    
    return performance_tflops, bottleneck

def estimate_speedup(original_latency_ms: float, 
                    optimized_latency_ms: float) -> float:
    if optimized_latency_ms == 0:
        return float('inf')
    return original_latency_ms / optimized_latency_ms

def estimate_throughput_improvement(original_throughput: float,
                                   optimized_throughput: float) -> float:
    if original_throughput == 0:
        return 0.0
    return ((optimized_throughput - original_throughput) / original_throughput) * 100
