from typing import Dict, Optional, List, Tuple
import torch
from ..utils.logger import get_logger

logger = get_logger(__name__)

class TritonBackend:
    
    def __init__(self):
        self.device = self._get_cuda_device()
        self.kernels: Dict[str, str] = {}
        self.compiled_kernels: Dict[str, object] = {}
        self.triton_available = self._check_triton()
    
    def _get_cuda_device(self):
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            return "cuda"
        else:
            logger.warning("CUDA not available, using CPU fallback")
            return "cpu"
    
    def _check_triton(self) -> bool:
        try:
            import triton
            logger.info("Triton framework available")
            return True
        except ImportError:
            logger.warning("Triton framework not available")
            return False
    
    def compile_kernel(self, kernel_name: str, kernel_code: str) -> bool:
        logger.info(f"Compiling kernel {kernel_name} for Triton")
        
        if not self.triton_available:
            logger.warning("Triton not available, cannot compile kernel")
            return False
        
        if self.device == "cpu":
            logger.warning("Cannot compile Triton kernels on CPU")
            return False
        
        try:
            self.kernels[kernel_name] = kernel_code
            exec_globals = {}
            exec(kernel_code, exec_globals)
            self.compiled_kernels[kernel_name] = exec_globals.get(kernel_name)
            logger.info(f"Kernel {kernel_name} compiled successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to compile kernel {kernel_name}: {e}")
            return False
    
    def execute_kernel(self, kernel_name: str, inputs: Dict[str, torch.Tensor],
                      output_shape: Tuple, output_dtype: torch.dtype) -> torch.Tensor:
        
        if not self.triton_available:
            logger.debug("Using CPU fallback (Triton not available)")
            return self._execute_cpu_fallback(kernel_name, inputs, output_shape, output_dtype)
        
        if kernel_name not in self.compiled_kernels:
            logger.warning(f"Kernel {kernel_name} not compiled")
            return None
        
        try:
            if self.device == "cuda":
                return self._execute_triton_kernel(kernel_name, inputs, output_shape, output_dtype)
            else:
                return self._execute_cpu_fallback(kernel_name, inputs, output_shape, output_dtype)
        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            return None
    
    def _execute_triton_kernel(self, kernel_name: str, inputs: Dict[str, torch.Tensor],
                              output_shape: Tuple, output_dtype: torch.dtype) -> torch.Tensor:
        
        kernel_func = self.compiled_kernels[kernel_name]
        
        output = torch.zeros(output_shape, dtype=output_dtype, device=self.device)
        
        for key, tensor in inputs.items():
            if isinstance(tensor, torch.Tensor):
                inputs[key] = tensor.to(self.device)
        
        try:
            if "matmul" in kernel_name:
                kernel_func[(1024,)](inputs["a"], inputs["b"], output,
                                     output_shape[0], output_shape[1], 
                                     inputs["a"].shape[1],
                                     inputs["a"].stride(0), inputs["a"].stride(1),
                                     inputs["b"].stride(0), inputs["b"].stride(1),
                                     output.stride(0), output.stride(1),
                                     64, 64, 32)
            else:
                grid = (output_shape[0] + 1024 - 1) // 1024,
                kernel_func[grid](inputs.get("x", inputs.get("input")), output, 
                                output_shape[0], 1024)
        except Exception as e:
            logger.debug(f"Triton kernel launch failed, using torch fallback: {e}")
            return self._execute_cpu_fallback(kernel_name, inputs, output_shape, output_dtype)
        
        return output
    
    def _execute_cpu_fallback(self, kernel_name: str, inputs: Dict[str, torch.Tensor],
                             output_shape: Tuple, output_dtype: torch.dtype) -> torch.Tensor:
        
        output = torch.zeros(output_shape, dtype=output_dtype)
        
        if "matmul" in kernel_name:
            A = inputs.get("a", inputs.get("input", None))
            B = inputs.get("b", inputs.get("weight", None))
            if A is not None and B is not None:
                output = torch.matmul(A, B).to(output_dtype)
        
        elif "relu" in kernel_name:
            input_tensor = inputs.get("x", inputs.get("input", None))
            if input_tensor is not None:
                output = torch.relu(input_tensor).to(output_dtype)
        
        elif "gelu" in kernel_name:
            input_tensor = inputs.get("x", inputs.get("input", None))
            if input_tensor is not None:
                output = torch.nn.functional.gelu(input_tensor).to(output_dtype)
        
        else:
            input_tensor = inputs.get("x", inputs.get("input", None))
            if input_tensor is not None:
                output = input_tensor.to(output_dtype)
        
        return output
    
    def get_device_info(self) -> Dict[str, str]:
        device_info = {
            "device": self.device,
            "triton_available": self.triton_available,
            "backend": "Triton" if self.triton_available else "CPU Fallback",
        }
        
        if self.device == "cuda":
            device_info["gpu_name"] = torch.cuda.get_device_name(0)
            device_info["cuda_version"] = torch.version.cuda
        
        return device_info
    
    def benchmark_kernel(self, kernel_name: str, inputs: Dict[str, torch.Tensor],
                        output_shape: Tuple, warmup_runs: int = 10,
                        measure_runs: int = 100) -> Dict[str, float]:
        
        import time
        
        if kernel_name not in self.compiled_kernels and self.triton_available:
            logger.warning(f"Kernel {kernel_name} not compiled")
            return {}
        
        for _ in range(warmup_runs):
            self.execute_kernel(kernel_name, inputs, output_shape, torch.float32)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        
        start_time = time.perf_counter()
        
        for _ in range(measure_runs):
            self.execute_kernel(kernel_name, inputs, output_shape, torch.float32)
        
        torch.cuda.synchronize() if self.device == "cuda" else None
        end_time = time.perf_counter()
        
        total_time = end_time - start_time
        avg_time = total_time / measure_runs
        
        return {
            "total_time_ms": total_time * 1000,
            "avg_time_ms": avg_time * 1000,
            "throughput_gbs": 0.0,
        }
    
    def is_kernel_compatible(self, operation_type: str) -> bool:
        compatible_ops = [
            "matmul", "relu", "gelu", "sigmoid", "tanh",
            "softmax", "layernorm", "conv2d"
        ]
        return any(op in operation_type.lower() for op in compatible_ops)
    
    def get_max_threads_per_block(self) -> int:
        if self.device == "cuda":
            return torch.cuda.get_device_properties(0).max_threads_per_block
        return 1
    
    def get_device_memory_mb(self) -> int:
        if self.device == "cuda":
            return int(torch.cuda.get_device_properties(0).total_memory / 1024 / 1024)
        return 0
