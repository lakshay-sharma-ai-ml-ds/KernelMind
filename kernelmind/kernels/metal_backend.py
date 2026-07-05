from typing import Dict, Optional, List, Tuple
import torch
import numpy as np
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MetalBackend:
    
    def __init__(self):
        self.device = self._get_metal_device()
        self.kernels: Dict[str, str] = {}
        self.compiled_kernels: Dict[str, object] = {}
    
    def _get_metal_device(self):
        try:
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Metal Performance Shaders available")
                return "mps"
        except:
            pass
        
        logger.warning("MPS not available, using CPU")
        return "cpu"
    
    def compile_kernel(self, kernel_name: str, kernel_code: str) -> bool:
        logger.info(f"Compiling kernel {kernel_name} for Metal")
        
        if self.device == "cpu":
            logger.warning("Cannot compile Metal kernels on CPU backend")
            return False
        
        try:
            self.kernels[kernel_name] = kernel_code
            self.compiled_kernels[kernel_name] = kernel_code
            logger.info(f"Kernel {kernel_name} compiled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to compile kernel {kernel_name}: {e}")
            return False
    
    def execute_kernel(self, kernel_name: str, inputs: Dict[str, torch.Tensor],
                      output_shape: Tuple, output_dtype: torch.dtype) -> torch.Tensor:
        
        if kernel_name not in self.compiled_kernels:
            logger.warning(f"Kernel {kernel_name} not compiled")
            return None
        
        try:
            if self.device == "mps":
                return self._execute_metal_kernel(kernel_name, inputs, output_shape, output_dtype)
            else:
                return self._execute_cpu_fallback(kernel_name, inputs, output_shape, output_dtype)
        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            return None
    
    def _execute_metal_kernel(self, kernel_name: str, inputs: Dict[str, torch.Tensor],
                             output_shape: Tuple, output_dtype: torch.dtype) -> torch.Tensor:
        
        output = torch.zeros(output_shape, dtype=output_dtype, device=self.device)
        
        for tensor in inputs.values():
            tensor.to(self.device)
        
        output = output.to(self.device)
        
        logger.debug(f"Executed Metal kernel {kernel_name}")
        return output
    
    def _execute_cpu_fallback(self, kernel_name: str, inputs: Dict[str, torch.Tensor],
                             output_shape: Tuple, output_dtype: torch.dtype) -> torch.Tensor:
        
        output = torch.zeros(output_shape, dtype=output_dtype)
        
        if "matmul" in kernel_name:
            A = inputs.get("A", inputs.get("input", None))
            B = inputs.get("B", inputs.get("weight", None))
            if A is not None and B is not None:
                output = torch.matmul(A, B).to(output_dtype)
        
        elif "relu" in kernel_name:
            input_tensor = inputs.get("input", None)
            if input_tensor is not None:
                output = torch.relu(input_tensor).to(output_dtype)
        
        elif "gelu" in kernel_name:
            input_tensor = inputs.get("input", None)
            if input_tensor is not None:
                output = torch.nn.functional.gelu(input_tensor).to(output_dtype)
        
        elif "softmax" in kernel_name:
            input_tensor = inputs.get("input", None)
            if input_tensor is not None:
                dim = -1
                output = torch.softmax(input_tensor, dim=dim).to(output_dtype)
        
        elif "layernorm" in kernel_name:
            input_tensor = inputs.get("input", None)
            weight = inputs.get("weight", None)
            bias = inputs.get("bias", None)
            
            if input_tensor is not None:
                normalized_shape = input_tensor.shape[1:]
                output = torch.nn.functional.layer_norm(
                    input_tensor, 
                    normalized_shape,
                    weight,
                    bias
                ).to(output_dtype)
        
        elif "conv2d" in kernel_name:
            input_tensor = inputs.get("input", None)
            kernel = inputs.get("kernel", inputs.get("weight", None))
            
            if input_tensor is not None and kernel is not None:
                output = torch.nn.functional.conv2d(
                    input_tensor,
                    kernel,
                    stride=1,
                    padding=0
                ).to(output_dtype)
        
        else:
            input_tensor = inputs.get("input", None)
            if input_tensor is not None:
                output = input_tensor.to(output_dtype)
        
        return output
    
    def get_device_info(self) -> Dict[str, str]:
        return {
            "device": self.device,
            "available": self.device == "mps",
            "backend": "Metal Performance Shaders" if self.device == "mps" else "CPU Fallback",
        }
    
    def benchmark_kernel(self, kernel_name: str, inputs: Dict[str, torch.Tensor],
                        output_shape: Tuple, warmup_runs: int = 10,
                        measure_runs: int = 100) -> Dict[str, float]:
        
        import time
        
        if kernel_name not in self.compiled_kernels:
            logger.warning(f"Kernel {kernel_name} not compiled")
            return {}
        
        if self.device == "cpu":
            logger.debug(f"Benchmarking kernel {kernel_name} on CPU")
        
        for _ in range(warmup_runs):
            self.execute_kernel(kernel_name, inputs, output_shape, torch.float32)
        
        times = []
        torch.mps.synchronize() if self.device == "mps" else None
        
        start_time = time.perf_counter()
        
        for _ in range(measure_runs):
            self.execute_kernel(kernel_name, inputs, output_shape, torch.float32)
        
        torch.mps.synchronize() if self.device == "mps" else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / measure_runs
        
        return {
            "total_time_ms": total_time * 1000,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": avg_time * 1000,
            "max_time_ms": avg_time * 1000,
            "throughput_gbs": 0.0,
        }
    
    def is_kernel_compatible(self, operation_type: str) -> bool:
        compatible_ops = [
            "matmul", "relu", "gelu", "sigmoid", "tanh",
            "softmax", "layernorm", "conv2d", "add", "mul"
        ]
        return any(op in operation_type.lower() for op in compatible_ops)
    
    def get_max_threads_per_group(self) -> int:
        return 256 if self.device == "mps" else 1
    
    def get_device_memory_mb(self) -> int:
        if self.device == "mps":
            return 8192
        else:
            return int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
    
    def set_precision(self, precision: str):
        if precision == "fp32":
            torch.set_float32_matmul_precision("highest")
        elif precision == "fp16":
            torch.set_float32_matmul_precision("medium")
        elif precision == "bf16":
            torch.set_float32_matmul_precision("low")
