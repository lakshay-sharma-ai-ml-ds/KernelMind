import torch
import psutil
from typing import Dict, Optional
from contextlib import contextmanager
from .logger import get_logger

logger = get_logger(__name__)

class MemoryProfiler:
    
    def __init__(self):
        self.start_memory = 0
        self.peak_memory = 0
        self.snapshots = []
    
    @contextmanager
    def track_memory(self, label: str = "memory tracking"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024
        
        try:
            yield
        finally:
            if device.type == "cuda":
                torch.cuda.synchronize()
                peak = torch.cuda.max_memory_allocated() / 1024 / 1024
                allocated = torch.cuda.memory_allocated() / 1024 / 1024
                logger.info(f"{label}: Peak GPU memory: {peak:.2f} MB, Allocated: {allocated:.2f} MB")
            
            process = psutil.Process()
            mem_after = process.memory_info().rss / 1024 / 1024
            
            mem_used = mem_after - mem_before
            logger.info(f"{label}: CPU memory used: {mem_used:.2f} MB")
            
            self.snapshots.append({
                "label": label,
                "mem_used_mb": mem_used,
                "peak_gpu_mb": peak if device.type == "cuda" else 0,
            })
    
    def get_summary(self) -> Dict:
        return {
            "snapshots": self.snapshots,
            "total_snapshots": len(self.snapshots),
        }

def profile_memory(func, *args, **kwargs) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    process = psutil.Process()
    mem_before = process.memory_info().rss / 1024 / 1024
    
    result = func(*args, **kwargs)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1024 / 1024
        allocated = torch.cuda.memory_allocated() / 1024 / 1024
    else:
        peak = 0
        allocated = 0
    
    process = psutil.Process()
    mem_after = process.memory_info().rss / 1024 / 1024
    
    return {
        "cpu_memory_used_mb": mem_after - mem_before,
        "peak_gpu_memory_mb": peak,
        "allocated_gpu_memory_mb": allocated,
        "result": result,
    }

def get_memory_usage() -> Dict[str, float]:
    vm = psutil.virtual_memory()
    
    usage = {
        "total_mb": vm.total / 1024 / 1024,
        "available_mb": vm.available / 1024 / 1024,
        "used_mb": vm.used / 1024 / 1024,
        "percent": vm.percent,
    }
    
    if torch.cuda.is_available():
        usage["gpu_total_mb"] = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
        usage["gpu_allocated_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        usage["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
    
    return usage

def print_memory_usage():
    usage = get_memory_usage()
    print("\nMemory Usage:")
    for key, value in usage.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
