from enum import Enum, auto
from typing import Dict

class OpType(Enum):
    
    # Arithmetic operations
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    
    # Linear algebra
    MATMUL = auto()
    CONV2D = auto()
    CONV3D = auto()
    
    # Activation functions
    RELU = auto()
    GELU = auto()
    SIGMOID = auto()
    TANH = auto()
    SOFTMAX = auto()
    LAYERNORM = auto()
    BATCHNORM = auto()
    
    # Tensor operations
    TRANSPOSE = auto()
    RESHAPE = auto()
    SQUEEZE = auto()
    UNSQUEEZE = auto()
    GATHER = auto()
    SCATTER = auto()
    
    # Reduction operations
    SUM = auto()
    MEAN = auto()
    MAX = auto()
    MIN = auto()
    
    # Advanced operations
    ATTENTION = auto()
    FFN = auto()
    EMBEDDING = auto()
    
    # Quantization
    QUANTIZE = auto()
    DEQUANTIZE = auto()
    
    # Memory operations
    ALLOC = auto()
    FREE = auto()
    COPY = auto()
    
    # Control flow
    IF = auto()
    LOOP = auto()
    FUNCTION_CALL = auto()

class DataType(Enum):
    
    FLOAT32 = "float32"
    FLOAT16 = "float16"
    BFLOAT16 = "bfloat16"
    INT8 = "int8"
    INT32 = "int32"
    INT64 = "int64"
    BOOL = "bool"
    
    def bytes(self):
        type_sizes = {
            DataType.FLOAT32: 4,
            DataType.FLOAT16: 2,
            DataType.BFLOAT16: 2,
            DataType.INT8: 1,
            DataType.INT32: 4,
            DataType.INT64: 8,
            DataType.BOOL: 1,
        }
        return type_sizes.get(self, 4)

class FusionPattern(Enum):
    
    LINEAR_ACTIVATION = "linear_activation"
    CONV_ACTIVATION = "conv_activation"
    LAYERNORM_ACTIVATION = "layernorm_activation"
    ATTENTION_OUTPUT = "attention_output"
    FFN_BLOCK = "ffn_block"
    RESIDUAL_ADD = "residual_add"

FUSIBLE_PATTERNS: Dict[FusionPattern, list] = {
    FusionPattern.LINEAR_ACTIVATION: [OpType.MATMUL, OpType.RELU],
    FusionPattern.CONV_ACTIVATION: [OpType.CONV2D, OpType.RELU],
    FusionPattern.LAYERNORM_ACTIVATION: [OpType.LAYERNORM, OpType.RELU],
    FusionPattern.ATTENTION_OUTPUT: [OpType.MATMUL, OpType.SOFTMAX],
    FusionPattern.FFN_BLOCK: [OpType.MATMUL, OpType.GELU, OpType.MATMUL],
    FusionPattern.RESIDUAL_ADD: [OpType.MATMUL, OpType.ADD],
}

DEVICE_MEMORY_LIMITS: Dict[str, int] = {
    "mps": 8 * 1024 * 1024 * 1024,
    "cuda": 16 * 1024 * 1024 * 1024,
    "cpu": 32 * 1024 * 1024 * 1024,
}

PEAK_BANDWIDTH: Dict[str, int] = {
    "mps": 120 * 1024 * 1024 * 1024,
    "cuda": 900 * 1024 * 1024 * 1024,
    "cpu": 50 * 1024 * 1024 * 1024,
}
