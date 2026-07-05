from typing import List, Dict, Optional, Tuple
import json
from ..core.graph import ComputationalGraph, Node
from ..core.constants import OpType
from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class KernelGenerator:
    
    def __init__(self):
        self.generated_kernels: Dict[str, str] = {}
        self.kernel_cache: Dict[str, str] = {}
        self.performance_history: Dict[str, List[float]] = {}
    
    def generate_kernels(self, graph: ComputationalGraph, 
                        backend: str = "metal") -> Dict[str, str]:
        logger.info(f"Generating kernels for {len(graph.nodes)} nodes using {backend} backend")
        
        kernels = {}
        
        for node_id, node in graph.nodes.items():
            kernel_name = self._get_kernel_name(node)
            
            if kernel_name in self.kernel_cache:
                kernels[kernel_name] = self.kernel_cache[kernel_name]
                logger.debug(f"Using cached kernel {kernel_name}")
                continue
            
            kernel_code = self._generate_single_kernel(node, backend)
            
            if kernel_code:
                kernels[kernel_name] = kernel_code
                self.kernel_cache[kernel_name] = kernel_code
                self.generated_kernels[kernel_name] = kernel_code
        
        return kernels
    
    def _generate_single_kernel(self, node: Node, backend: str) -> Optional[str]:
        op_type = node.operation.op_type
        
        if backend == "metal":
            return self._generate_metal_kernel(node, op_type)
        elif backend == "triton":
            return self._generate_triton_kernel(node, op_type)
        else:
            logger.warning(f"Unknown backend {backend}")
            return None
    
    def _generate_metal_kernel(self, node: Node, op_type: OpType) -> str:
        if op_type == OpType.MATMUL:
            return self._metal_matmul_kernel(node)
        elif op_type == OpType.RELU:
            return self._metal_relu_kernel(node)
        elif op_type == OpType.GELU:
            return self._metal_gelu_kernel(node)
        elif op_type == OpType.LAYERNORM:
            return self._metal_layernorm_kernel(node)
        elif op_type == OpType.SOFTMAX:
            return self._metal_softmax_kernel(node)
        elif op_type == OpType.CONV2D:
            return self._metal_conv2d_kernel(node)
        else:
            return self._metal_generic_kernel(node, op_type)
    
    def _metal_matmul_kernel(self, node: Node) -> str:
        return """
#include <metal_stdlib>
using namespace metal;

kernel void matmul(
    device float *A [[buffer(0)]],
    device float *B [[buffer(1)]],
    device float *C [[buffer(2)]],
    constant uint *dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 gridsize [[threads_per_grid]]
) {
    uint M = dims[0];
    uint N = dims[1];
    uint K = dims[2];
    
    uint row = gid.y;
    uint col = gid.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint k = 0; k < K; k++) {
        sum += A[row * K + k] * B[k * N + col];
    }
    
    C[row * N + col] = sum;
}
"""
    
    def _metal_relu_kernel(self, node: Node) -> str:
        return """
#include <metal_stdlib>
using namespace metal;

kernel void relu(
    device float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = max(0.0f, input[gid]);
}
"""
    
    def _metal_gelu_kernel(self, node: Node) -> str:
        return """
#include <metal_stdlib>
using namespace metal;

constant float SQRT_2_OVER_PI = 0.7978845608f;

float gelu_approx(float x) {
    float cdf = 0.5f * (1.0f + tanh(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
    return x * cdf;
}

kernel void gelu(
    device float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= size) return;
    output[gid] = gelu_approx(input[gid]);
}
"""
    
    def _metal_layernorm_kernel(self, node: Node) -> str:
        return """
#include <metal_stdlib>
using namespace metal;

kernel void layernorm(
    device float *input [[buffer(0)]],
    device float *weight [[buffer(1)]],
    device float *bias [[buffer(2)]],
    device float *output [[buffer(3)]],
    constant float &eps [[buffer(4)]],
    constant uint *shape [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint normalized_size = shape[0];
    if (gid >= normalized_size) return;
    
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (uint i = 0; i < normalized_size; i++) {
        float val = input[gid * normalized_size + i];
        sum += val;
        sum_sq += val * val;
    }
    
    float mean = sum / normalized_size;
    float var = sum_sq / normalized_size - mean * mean;
    float std = sqrt(var + eps);
    
    for (uint i = 0; i < normalized_size; i++) {
        float normalized = (input[gid * normalized_size + i] - mean) / std;
        float scaled = normalized * weight[i];
        output[gid * normalized_size + i] = scaled + bias[i];
    }
}
"""
    
    def _metal_softmax_kernel(self, node: Node) -> str:
        return """
#include <metal_stdlib>
using namespace metal;

kernel void softmax(
    device float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint *dims [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint vocab_size = dims[0];
    uint batch_size = dims[1];
    
    if (gid >= batch_size) return;
    
    float max_val = input[gid * vocab_size];
    for (uint i = 1; i < vocab_size; i++) {
        max_val = max(max_val, input[gid * vocab_size + i]);
    }
    
    float sum = 0.0f;
    for (uint i = 0; i < vocab_size; i++) {
        float exp_val = exp(input[gid * vocab_size + i] - max_val);
        output[gid * vocab_size + i] = exp_val;
        sum += exp_val;
    }
    
    for (uint i = 0; i < vocab_size; i++) {
        output[gid * vocab_size + i] /= sum;
    }
}
"""
    
    def _metal_conv2d_kernel(self, node: Node) -> str:
        attrs = node.operation.attributes
        kernel_h, kernel_w = attrs.get("kernel_size", (3, 3))
        stride_h, stride_w = attrs.get("stride", (1, 1))
        
        return f"""
#include <metal_stdlib>
using namespace metal;

kernel void conv2d(
    device float *input [[buffer(0)]],
    device float *kernel [[buffer(1)]],
    device float *output [[buffer(2)]],
    constant uint *shape [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]
) {{
    uint out_c = gid.x;
    uint out_h = gid.y;
    uint out_w = gid.z;
    
    uint in_channels = shape[0];
    uint in_h = shape[1];
    uint in_w = shape[2];
    uint out_channels = shape[3];
    
    uint kernel_h = {kernel_h};
    uint kernel_w = {kernel_w};
    uint stride_h = {stride_h};
    uint stride_w = {stride_w};
    
    float sum = 0.0f;
    
    for (uint ic = 0; ic < in_channels; ic++) {{
        for (uint kh = 0; kh < kernel_h; kh++) {{
            for (uint kw = 0; kw < kernel_w; kw++) {{
                uint in_h_idx = out_h * stride_h + kh;
                uint in_w_idx = out_w * stride_w + kw;
                
                if (in_h_idx < in_h && in_w_idx < in_w) {{
                    float in_val = input[ic * in_h * in_w + in_h_idx * in_w + in_w_idx];
                    float k_val = kernel[out_c * in_channels * kernel_h * kernel_w + 
                                         ic * kernel_h * kernel_w + kh * kernel_w + kw];
                    sum += in_val * k_val;
                }}
            }}
        }}
    }}
    
    uint out_idx = out_c * out_h * out_w + out_h * out_w + out_w;
    output[out_idx] = sum;
}}
"""
    
    def _metal_generic_kernel(self, node: Node, op_type: OpType) -> str:
        return f"""
#include <metal_stdlib>
using namespace metal;

kernel void {self._get_kernel_name(node)}(
    device float *input [[buffer(0)]],
    device float *output [[buffer(1)]],
    constant uint &size [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {{
    if (gid >= size) return;
    output[gid] = input[gid];
}}
"""
    
    def _generate_triton_kernel(self, node: Node, op_type: OpType) -> str:
        if op_type == OpType.MATMUL:
            return self._triton_matmul_kernel(node)
        elif op_type == OpType.RELU:
            return self._triton_relu_kernel(node)
        else:
            return self._triton_generic_kernel(node, op_type)
    
    def _triton_matmul_kernel(self, node: Node) -> str:
        return """
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
    
    rm = tl.where(rm < M, rm, M - 1)
    rn = tl.where(rn < N, rn, N - 1)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        rk = k + tl.arange(0, BLOCK_K)
        rk = tl.where(rk < K, rk, K - 1)
        
        a = tl.load(a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak)
        b = tl.load(b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn)
        
        acc += tl.dot(a, b)
    
    c = acc.to(tl.float16)
    
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    
    tl.store(c_ptrs, c, mask=mask)
"""
    
    def _triton_relu_kernel(self, node: Node) -> str:
        return """
import triton
import triton.language as tl

@triton.jit
def relu_kernel(
    x_ptr,
    y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    
    tl.store(y_ptr + offsets, y, mask=mask)
"""
    
    def _triton_generic_kernel(self, node: Node, op_type: OpType) -> str:
        return f"""
import triton
import triton.language as tl

@triton.jit
def {self._get_kernel_name(node)}(
    x_ptr, y_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x, mask=mask)
"""
    
    def _get_kernel_name(self, node: Node) -> str:
        return f"{node.operation.op_type.name.lower()}_{node.node_id}"
    
    def get_kernel_code(self, kernel_name: str) -> Optional[str]:
        return self.generated_kernels.get(kernel_name)
    
    def save_kernels_to_file(self, output_dir: str):
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for kernel_name, code in self.generated_kernels.items():
            filename = os.path.join(output_dir, f"{kernel_name}.metal")
            with open(filename, "w") as f:
                f.write(code)
            logger.info(f"Saved kernel to {filename}")
