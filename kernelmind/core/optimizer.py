from typing import List, Dict, Set, Tuple, Optional
from .graph import ComputationalGraph, Node, Operation, Tensor
from .constants import OpType, FusionPattern, FUSIBLE_PATTERNS
from ..utils.logger import get_logger

logger = get_logger(__name__)

class GraphOptimizer:
    
    def __init__(self, graph: ComputationalGraph):
        self.graph = graph
        self.optimization_log = []
    
    def optimize(self) -> ComputationalGraph:
        logger.info("Starting graph optimization")
        
        initial_nodes = len(self.graph.nodes)
        initial_size = self.graph.total_size_bytes()
        
        self.constant_folding()
        self.dead_code_elimination()
        self.common_subexpression_elimination()
        self.operator_fusion()
        self.layout_optimization()
        
        final_nodes = len(self.graph.nodes)
        final_size = self.graph.total_size_bytes()
        
        logger.info(f"Optimization complete: {initial_nodes} -> {final_nodes} nodes")
        logger.info(f"Memory: {initial_size/1e6:.2f}MB -> {final_size/1e6:.2f}MB")
        
        return self.graph
    
    def constant_folding(self):
        logger.info("Performing constant folding")
        
        nodes_to_remove = []
        
        for node_id, node in self.graph.nodes.items():
            if self._can_fold_constants(node):
                logger.debug(f"Folding constants for node {node.name}")
                result = self._evaluate_constant_node(node)
                
                if result is not None:
                    self.graph.constants[node.outputs[0]] = result
                    nodes_to_remove.append(node_id)
                    self.optimization_log.append({
                        "type": "constant_folding",
                        "node": node.name,
                    })
        
        for node_id in nodes_to_remove:
            del self.graph.nodes[node_id]
    
    def dead_code_elimination(self):
        logger.info("Performing dead code elimination")
        
        live_nodes = self._find_live_nodes()
        nodes_to_remove = set(self.graph.nodes.keys()) - live_nodes
        
        for node_id in nodes_to_remove:
            logger.debug(f"Removing dead node {self.graph.nodes[node_id].name}")
            del self.graph.nodes[node_id]
            self.optimization_log.append({
                "type": "dead_code_elimination",
                "node": node_id,
            })
    
    def common_subexpression_elimination(self):
        logger.info("Performing common subexpression elimination")
        
        expr_map: Dict[str, int] = {}
        nodes_to_remove = []
        
        for node_id, node in list(self.graph.nodes.items()):
            expr_signature = self._get_expression_signature(node)
            
            if expr_signature in expr_map:
                original_node_id = expr_map[expr_signature]
                logger.debug(f"Merging nodes {node.name} and {self.graph.nodes[original_node_id].name}")
                
                self._redirect_outputs(node_id, original_node_id)
                nodes_to_remove.append(node_id)
                self.optimization_log.append({
                    "type": "cse",
                    "source": node.name,
                    "target": self.graph.nodes[original_node_id].name,
                })
            else:
                expr_map[expr_signature] = node_id
        
        for node_id in nodes_to_remove:
            del self.graph.nodes[node_id]
    
    def operator_fusion(self):
        logger.info("Performing operator fusion")
        
        order = self.graph.topological_sort()
        fused_pairs = set()
        
        for i in range(len(order) - 1):
            node_id1 = order[i]
            node_id2 = order[i + 1]
            
            if node_id1 in fused_pairs or node_id2 in fused_pairs:
                continue
            
            node1 = self.graph.nodes.get(node_id1)
            node2 = self.graph.nodes.get(node_id2)
            
            if node1 and node2 and self._can_fuse_nodes(node1, node2):
                logger.debug(f"Fusing {node1.name} -> {node2.name}")
                self._fuse_nodes(node_id1, node_id2)
                fused_pairs.add(node_id1)
                fused_pairs.add(node_id2)
                self.optimization_log.append({
                    "type": "fusion",
                    "nodes": [node1.name, node2.name],
                })
    
    def layout_optimization(self):
        logger.info("Optimizing tensor layouts")
        
        for tensor in self.graph.tensors.values():
            optimal_layout = self._determine_optimal_layout(tensor)
            if optimal_layout != "NCHW":
                logger.debug(f"Optimizing layout for {tensor.name}: NCHW -> {optimal_layout}")
    
    def _can_fold_constants(self, node: Node) -> bool:
        if node.operation.op_type not in [OpType.ADD, OpType.MUL, OpType.DIV, OpType.SUB]:
            return False
        
        for input_name in node.inputs:
            if input_name not in self.graph.constants:
                return False
        
        return True
    
    def _evaluate_constant_node(self, node: Node) -> Optional[object]:
        import numpy as np
        
        try:
            values = [self.graph.constants[name] for name in node.inputs]
            op_type = node.operation.op_type
            
            if op_type == OpType.ADD:
                return values[0] + values[1]
            elif op_type == OpType.SUB:
                return values[0] - values[1]
            elif op_type == OpType.MUL:
                return values[0] * values[1]
            elif op_type == OpType.DIV:
                return values[0] / values[1]
            
            return None
        except Exception as e:
            logger.warning(f"Failed to evaluate node {node.name}: {e}")
            return None
    
    def _find_live_nodes(self) -> Set[int]:
        live_nodes = set(self.graph.output_nodes)
        queue = list(self.graph.output_nodes)
        
        while queue:
            node_id = queue.pop(0)
            if node_id in live_nodes:
                continue
            
            live_nodes.add(node_id)
            node = self.graph.nodes.get(node_id)
            
            if node:
                for dep_id in node.dependencies:
                    if dep_id not in live_nodes:
                        queue.append(dep_id)
        
        return live_nodes
    
    def _get_expression_signature(self, node: Node) -> str:
        op_name = node.operation.op_type.name
        input_types = ",".join(str(type(node.inputs)) for _ in node.inputs)
        return f"{op_name}:{input_types}"
    
    def _redirect_outputs(self, source_id: int, target_id: int):
        source = self.graph.nodes[source_id]
        target = self.graph.nodes[target_id]
        
        for node in self.graph.nodes.values():
            for i, input_name in enumerate(node.inputs):
                for output_name in source.outputs:
                    if input_name == output_name:
                        node.inputs[i] = target.outputs[0]
    
    def _can_fuse_nodes(self, node1: Node, node2: Node) -> bool:
        if not node1.is_fusible(node2):
            return False
        
        pattern = self._detect_fusion_pattern(node1, node2)
        return pattern is not None
    
    def _detect_fusion_pattern(self, node1: Node, node2: Node) -> Optional[FusionPattern]:
        op1 = node1.operation.op_type
        op2 = node2.operation.op_type
        
        for pattern, ops in FUSIBLE_PATTERNS.items():
            if ops == [op1, op2]:
                return pattern
        
        return None
    
    def _fuse_nodes(self, node_id1: int, node_id2: int):
        node1 = self.graph.nodes[node_id1]
        node2 = self.graph.nodes[node_id2]
        
        fused_name = f"{node1.name}_fused_{node2.name}"
        fused_op = Operation(
            op_type=node1.operation.op_type,
            inputs=node1.inputs,
            outputs=node2.outputs,
            attributes={
                "fused_nodes": [node1.name, node2.name],
                "original_ops": [node1.operation.op_type, node2.operation.op_type],
            }
        )
        
        del self.graph.nodes[node_id1]
        del self.graph.nodes[node_id2]
        self.graph.add_node(fused_name, fused_op, node1.inputs, node2.outputs)
    
    def _determine_optimal_layout(self, tensor: Tensor) -> str:
        if len(tensor.shape) == 4:
            return "NCHW"
        elif len(tensor.shape) == 2:
            return "NC"
        else:
            return "DEFAULT"
    
    def get_optimization_summary(self) -> Dict[str, int]:
        summary = {}
        for log_entry in self.optimization_log:
            op_type = log_entry["type"]
            summary[op_type] = summary.get(op_type, 0) + 1
        
        return summary
