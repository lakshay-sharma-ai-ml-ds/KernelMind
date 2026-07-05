import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from .graph import ComputationalGraph, Tensor, Operation
from .constants import OpType, DataType
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ModelParser:
    
    def __init__(self):
        self.graph = None
        self.tensor_shapes: Dict[str, Tuple] = {}
        self.tensor_dtypes: Dict[str, DataType] = {}
        self.module_map: Dict[str, nn.Module] = {}
    
    def parse_model(self, model: nn.Module, 
                   sample_input: Optional[torch.Tensor] = None) -> ComputationalGraph:
        self.graph = ComputationalGraph(name=model.__class__.__name__)
        self._register_modules(model)
        
        if sample_input is not None:
            self._trace_forward(model, sample_input)
        else:
            self._analyze_static(model)
        
        self.graph.build_dependencies()
        logger.info(f"Parsed model with {len(self.graph.nodes)} nodes")
        return self.graph
    
    def _register_modules(self, model: nn.Module, prefix: str = ""):
        for name, module in model.named_modules():
            full_name = f"{prefix}.{name}" if prefix else name
            self.module_map[full_name] = module
    
    def _trace_forward(self, model: nn.Module, sample_input: torch.Tensor):
        try:
            with torch.no_grad():
                traced = torch.jit.trace(model, sample_input)
            self._extract_from_traced(traced, sample_input)
        except Exception as e:
            logger.warning(f"JIT tracing failed: {e}. Falling back to static analysis.")
            self._analyze_static(model)
    
    def _extract_from_traced(self, traced_model, sample_input: torch.Tensor):
        graph = traced_model.graph
        
        for node in graph.nodes():
            op_type = node.kind()
            inputs = node.inputsAt
            outputs = node.outputsAt
            
            self._process_node(op_type, inputs, outputs, node)
    
    def _analyze_static(self, model: nn.Module):
        node_id = 0
        
        for name, module in self.module_map.items():
            if isinstance(module, nn.Linear):
                node_id = self._add_linear_layer(module, name, node_id)
            elif isinstance(module, nn.Conv2d):
                node_id = self._add_conv2d_layer(module, name, node_id)
            elif isinstance(module, nn.LayerNorm):
                node_id = self._add_layernorm_layer(module, name, node_id)
            elif isinstance(module, nn.BatchNorm2d):
                node_id = self._add_batchnorm_layer(module, name, node_id)
            elif isinstance(module, (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh)):
                node_id = self._add_activation_layer(module, name, node_id)
            elif isinstance(module, nn.Softmax):
                node_id = self._add_softmax_layer(module, name, node_id)
    
    def _add_linear_layer(self, module: nn.Linear, name: str, node_id: int) -> int:
        input_tensor = f"{name}_input"
        output_tensor = f"{name}_output"
        
        self._create_tensor(input_tensor, (None, module.in_features), DataType.FLOAT32)
        self._create_tensor(output_tensor, (None, module.out_features), DataType.FLOAT32)
        
        weight_tensor = Tensor(
            name=f"{name}_weight",
            dtype=DataType.FLOAT32,
            shape=(module.out_features, module.in_features),
            is_weight=True
        )
        self.graph.add_tensor(weight_tensor)
        
        if module.bias is not None:
            bias_tensor = Tensor(
                name=f"{name}_bias",
                dtype=DataType.FLOAT32,
                shape=(module.out_features,),
                is_weight=True
            )
            self.graph.add_tensor(bias_tensor)
        
        flops = 2 * module.in_features * module.out_features
        
        operation = Operation(
            op_type=OpType.MATMUL,
            inputs=[input_tensor, f"{name}_weight"],
            outputs=[output_tensor],
            attributes={
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
                "flops": flops
            }
        )
        
        self.graph.add_node(name, operation, 
                           [input_tensor, f"{name}_weight"], 
                           [output_tensor])
        
        return node_id + 1
    
    def _add_conv2d_layer(self, module: nn.Conv2d, name: str, node_id: int) -> int:
        input_tensor = f"{name}_input"
        output_tensor = f"{name}_output"
        
        self._create_tensor(input_tensor, (None, module.in_channels, None, None), DataType.FLOAT32)
        self._create_tensor(output_tensor, (None, module.out_channels, None, None), DataType.FLOAT32)
        
        kernel_tensor = Tensor(
            name=f"{name}_kernel",
            dtype=DataType.FLOAT32,
            shape=(module.out_channels, module.in_channels, *module.kernel_size),
            is_weight=True
        )
        self.graph.add_tensor(kernel_tensor)
        
        operation = Operation(
            op_type=OpType.CONV2D,
            inputs=[input_tensor, f"{name}_kernel"],
            outputs=[output_tensor],
            attributes={
                "kernel_size": module.kernel_size,
                "stride": module.stride,
                "padding": module.padding,
                "groups": module.groups,
            }
        )
        
        self.graph.add_node(name, operation,
                           [input_tensor, f"{name}_kernel"],
                           [output_tensor])
        
        return node_id + 1
    
    def _add_layernorm_layer(self, module: nn.LayerNorm, name: str, node_id: int) -> int:
        input_tensor = f"{name}_input"
        output_tensor = f"{name}_output"
        
        self._create_tensor(input_tensor, tuple(module.normalized_shape), DataType.FLOAT32)
        self._create_tensor(output_tensor, tuple(module.normalized_shape), DataType.FLOAT32)
        
        if module.weight is not None:
            weight_tensor = Tensor(
                name=f"{name}_weight",
                dtype=DataType.FLOAT32,
                shape=module.weight.shape,
                is_weight=True
            )
            self.graph.add_tensor(weight_tensor)
        
        operation = Operation(
            op_type=OpType.LAYERNORM,
            inputs=[input_tensor],
            outputs=[output_tensor],
            attributes={
                "normalized_shape": module.normalized_shape,
                "eps": module.eps,
            }
        )
        
        self.graph.add_node(name, operation, [input_tensor], [output_tensor])
        return node_id + 1
    
    def _add_batchnorm_layer(self, module: nn.BatchNorm2d, name: str, node_id: int) -> int:
        input_tensor = f"{name}_input"
        output_tensor = f"{name}_output"
        
        self._create_tensor(input_tensor, (None, module.num_features, None, None), DataType.FLOAT32)
        self._create_tensor(output_tensor, (None, module.num_features, None, None), DataType.FLOAT32)
        
        operation = Operation(
            op_type=OpType.BATCHNORM,
            inputs=[input_tensor],
            outputs=[output_tensor],
            attributes={
                "num_features": module.num_features,
                "eps": module.eps,
                "momentum": module.momentum,
                "affine": module.affine,
            }
        )
        
        self.graph.add_node(name, operation, [input_tensor], [output_tensor])
        return node_id + 1
    
    def _add_activation_layer(self, module: nn.Module, name: str, node_id: int) -> int:
        input_tensor = f"{name}_input"
        output_tensor = f"{name}_output"
        
        self._create_tensor(input_tensor, (None,), DataType.FLOAT32)
        self._create_tensor(output_tensor, (None,), DataType.FLOAT32)
        
        if isinstance(module, nn.ReLU):
            op_type = OpType.RELU
        elif isinstance(module, nn.GELU):
            op_type = OpType.GELU
        elif isinstance(module, nn.Sigmoid):
            op_type = OpType.SIGMOID
        else:
            op_type = OpType.TANH
        
        operation = Operation(
            op_type=op_type,
            inputs=[input_tensor],
            outputs=[output_tensor],
            attributes={}
        )
        
        self.graph.add_node(name, operation, [input_tensor], [output_tensor])
        return node_id + 1
    
    def _add_softmax_layer(self, module: nn.Softmax, name: str, node_id: int) -> int:
        input_tensor = f"{name}_input"
        output_tensor = f"{name}_output"
        
        self._create_tensor(input_tensor, (None,), DataType.FLOAT32)
        self._create_tensor(output_tensor, (None,), DataType.FLOAT32)
        
        operation = Operation(
            op_type=OpType.SOFTMAX,
            inputs=[input_tensor],
            outputs=[output_tensor],
            attributes={"dim": module.dim}
        )
        
        self.graph.add_node(name, operation, [input_tensor], [output_tensor])
        return node_id + 1
    
    def _create_tensor(self, name: str, shape: Tuple, dtype: DataType):
        tensor = Tensor(name=name, dtype=dtype, shape=shape)
        self.graph.add_tensor(tensor)
    
    def _process_node(self, op_type: str, inputs: List, outputs: List, node):
        pass
    
    def estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        total_flops = 0
        
        def hook_fn(module, input, output):
            nonlocal total_flops
            
            if isinstance(module, nn.Linear):
                batch_size = input[0].shape[0] if len(input[0].shape) > 1 else 1
                total_flops += 2 * batch_size * module.in_features * module.out_features
            elif isinstance(module, nn.Conv2d):
                batch_size = input[0].shape[0]
                h = input[0].shape[2]
                w = input[0].shape[3]
                kernel_ops = module.kernel_size[0] * module.kernel_size[1]
                output_size = batch_size * module.out_channels * h * w
                total_flops += 2 * kernel_ops * module.in_channels * output_size // module.groups
        
        hooks = []
        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hooks.append(module.register_forward_hook(hook_fn))
        
        try:
            with torch.no_grad():
                dummy_input = torch.randn(input_shape)
                model(dummy_input)
        finally:
            for hook in hooks:
                hook.remove()
        
        return total_flops
