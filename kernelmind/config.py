import os
from enum import Enum
from typing import Optional

class DeviceType(Enum):
    METAL = "metal"
    TRITON = "triton"
    CPU = "cpu"

class OptimizationLevel(Enum):
    NONE = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    AGGRESSIVE = 4

class Config:
    
    # Device Configuration
    DEVICE_TYPE: DeviceType = DeviceType.METAL
    USE_FALLBACK: bool = True
    
    # LLM Configuration
    LLM_MODEL: str = "claude-3-5-sonnet-20241022"
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 2000
    LLM_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    
    # Optimization Configuration
    OPTIMIZATION_LEVEL: OptimizationLevel = OptimizationLevel.HIGH
    ENABLE_OPERATOR_FUSION: bool = True
    ENABLE_QUANTIZATION: bool = True
    ENABLE_MEMORY_OPTIMIZATION: bool = True
    
    # Benchmark Configuration
    BENCHMARK_WARMUP_RUNS: int = 10
    BENCHMARK_MEASURE_RUNS: int = 100
    BENCHMARK_BATCH_SIZES: list = [1, 4, 8, 16, 32]
    BENCHMARK_ENABLE_PROFILING: bool = True
    
    # Graph Optimization
    ENABLE_CONSTANT_FOLDING: bool = True
    ENABLE_DEAD_CODE_ELIMINATION: bool = True
    ENABLE_COMMON_SUBEXPRESSION_ELIMINATION: bool = True
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = "kernelmind.log"
    
    # Cache Configuration
    ENABLE_KERNEL_CACHE: bool = True
    CACHE_DIR: str = ".kernelmind_cache"
    
    # Performance Thresholds
    MIN_SPEEDUP_THRESHOLD: float = 1.05
    MAX_MEMORY_INCREASE_PERCENT: float = 10.0
    NUMERICAL_TOLERANCE: float = 1e-4
    
    # Verification
    ENABLE_CORRECTNESS_CHECK: bool = True
    ENABLE_REGRESSION_TRACKING: bool = True
    
    # Model-specific
    MAX_GRAPH_SIZE: int = 10000
    MAX_TENSOR_BYTES: int = 1024 * 1024 * 1024
    
    @classmethod
    def get_device(cls):
        if cls.DEVICE_TYPE == DeviceType.METAL:
            return "mps" if hasattr(__import__("torch").backends, "mps") else "cpu"
        elif cls.DEVICE_TYPE == DeviceType.TRITON:
            return "cuda"
        return "cpu"
    
    @classmethod
    def validate(cls):
        if cls.LLM_API_KEY is None:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        if cls.OPTIMIZATION_LEVEL not in OptimizationLevel:
            raise ValueError("Invalid optimization level")
        return True

config = Config()
