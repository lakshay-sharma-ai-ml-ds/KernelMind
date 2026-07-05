import torch
import platform
import psutil
import cpuinfo
from typing import Dict
from .logger import get_logger

logger = get_logger(__name__)

def detect_device() -> str:
    if torch.cuda.is_available():
        logger.info("CUDA GPU detected")
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("Apple Metal Performance Shaders detected")
        return "mps"
    else:
        logger.info("Using CPU backend")
        return "cpu"

def get_device_info() -> Dict[str, str]:
    info = {
        "platform": platform.system(),
        "platform_release": platform.release(),
        "processor": platform.processor(),
    }
    
    try:
        cpu_info = cpuinfo.get_cpu_info()
        info["cpu_brand"] = cpu_info.get("brand_raw", "Unknown")
        info["cpu_cores"] = str(psutil.cpu_count(logical=False))
        info["cpu_threads"] = str(psutil.cpu_count(logical=True))
    except:
        pass
    
    try:
        ram_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        info["total_ram_gb"] = f"{ram_gb:.1f}"
    except:
        pass
    
    info["pytorch_version"] = torch.__version__
    
    device = detect_device()
    info["device"] = device
    
    if device == "cuda":
        try:
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_gb"] = str(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024 / 1024)
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
        except:
            pass
    
    elif device == "mps":
        info["gpu_type"] = "Apple Silicon"
        try:
            import subprocess
            result = subprocess.run(["system_profiler", "SPHardwareDataType"], 
                                  capture_output=True, text=True, timeout=5)
            if "Apple" in result.stdout:
                info["silicon_type"] = "Apple Silicon"
        except:
            pass
    
    return info

def get_hardware_summary() -> str:
    info = get_device_info()
    
    lines = [
        "=" * 60,
        "HARDWARE CONFIGURATION",
        "=" * 60,
    ]
    
    for key, value in sorted(info.items()):
        lines.append(f"{key:.<40} {value}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)

def print_hardware_info():
    print(get_hardware_summary())

def get_memory_info() -> Dict[str, float]:
    try:
        vm = psutil.virtual_memory()
        return {
            "total_gb": vm.total / 1024 / 1024 / 1024,
            "available_gb": vm.available / 1024 / 1024 / 1024,
            "used_gb": vm.used / 1024 / 1024 / 1024,
            "percent_used": vm.percent,
        }
    except Exception as e:
        logger.error(f"Failed to get memory info: {e}")
        return {}

def check_device_compatibility(required_device: str) -> bool:
    available_device = detect_device()
    
    if required_device == "cuda" and available_device != "cuda":
        logger.warning(f"Required device {required_device} not available. Using {available_device}")
        return False
    
    if required_device == "mps" and available_device != "mps":
        logger.warning(f"Required device {required_device} not available. Using {available_device}")
        return False
    
    return True
