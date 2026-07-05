from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
from .constants import OpType, DataType

@dataclass
class Tensor:
    name: str
    dtype: DataType
    shape: Tuple[int, ...]
    device: str = "mps"
    is_weight: bool = False
    requires_grad: bool = False
    
    def size_bytes(self) -> int:
        total_elements = int(np.prod(self.shape))
        return total_elements * self.dtype.bytes()
    
    def __str__(self):
        return f"Tensor({self.name}, shape={self.shape}, dtype={self.dtype.value})"

@dataclass
class Operation:
    op_type: OpType
    inputs: List[str]
    outputs: List[str]
    attributes: Dict[str, Any] = field(default_factory=dict)
    computational_cost: float = 0.0
    memory_cost: float = 0.0
    
    def flops(self) -> int:
        if self.op_type == OpType.MATMUL:
            return self.attributes.get("flops", 0)
        elif self.op_type in [OpType.CONV2D, OpType.CONV3D]:
            return self.attributes.get("flops", 0)
        return 0
    
    def __str__(self):
        return f"Op({self.op_type.name}, inputs={len(self.inputs)}, outputs={len(self.outputs)})"

@dataclass
class Node:
    node_id: int
    name: str
    operation: Operation
    inputs: List[str]
    outputs: List[str]
    dependencies: List[int] = field(default_factory=list)
    dependents: List[int] = field(default_factory=list)
    
    def is_fusible(self, other_node: 'Node') -> bool:
        if not other_node.inputs:
            return False
        return any(out in other_node.inputs for out in self.outputs)

class ComputationalGraph:
    
    def __init__(self, name: str = "graph"):
        self.name = name
        self.nodes: Dict[int, Node] = {}
        self.tensors: Dict[str, Tensor] = {}
        self.node_counter = 0
        self.input_nodes: List[int] = []
        self.output_nodes: List[int] = []
        self.constants: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Any] = {}
    
    def add_node(self, name: str, operation: Operation, 
                 inputs: List[str], outputs: List[str]) -> int:
        node_id = self.node_counter
        node = Node(
            node_id=node_id,
            name=name,
            operation=operation,
            inputs=inputs,
            outputs=outputs
        )
        self.nodes[node_id] = node
        self.node_counter += 1
        return node_id
    
    def add_tensor(self, tensor: Tensor):
        self.tensors[tensor.name] = tensor
    
    def add_constant(self, name: str, value: np.ndarray):
        self.constants[name] = value
    
    def set_input_nodes(self, node_ids: List[int]):
        self.input_nodes = node_ids
    
    def set_output_nodes(self, node_ids: List[int]):
        self.output_nodes = node_ids
    
    def get_node(self, node_id: int) -> Optional[Node]:
        return self.nodes.get(node_id)
    
    def get_tensor(self, tensor_name: str) -> Optional[Tensor]:
        return self.tensors.get(tensor_name)
    
    def build_dependencies(self):
        tensor_to_node: Dict[str, int] = {}
        
        for node_id, node in self.nodes.items():
            for output in node.outputs:
                tensor_to_node[output] = node_id
        
        for node_id, node in self.nodes.items():
            for input_name in node.inputs:
                if input_name in tensor_to_node:
                    producer_id = tensor_to_node[input_name]
                    if producer_id not in node.dependencies:
                        node.dependencies.append(producer_id)
                    if node_id not in self.nodes[producer_id].dependents:
                        self.nodes[producer_id].dependents.append(node_id)
    
    def topological_sort(self) -> List[int]:
        visited = set()
        temp_visited = set()
        order = []
        
        def visit(node_id: int):
            if node_id in visited:
                return
            if node_id in temp_visited:
                raise ValueError("Cycle detected in graph")
            
            temp_visited.add(node_id)
            node = self.nodes[node_id]
            
            for dep_id in node.dependencies:
                visit(dep_id)
            
            temp_visited.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
        
        for node_id in self.nodes:
            visit(node_id)
        
        return order
    
    def get_all_paths(self, start_id: int, end_id: int) -> List[List[int]]:
        paths = []
        
        def dfs(current: int, path: List[int]):
            if current == end_id:
                paths.append(path + [current])
                return
            
            node = self.nodes[current]
            for next_id in node.dependents:
                dfs(next_id, path + [current])
        
        dfs(start_id, [])
        return paths
    
    def get_critical_path(self) -> List[int]:
        order = self.topological_sort()
        costs = {node_id: 0 for node_id in self.nodes}
        
        for node_id in order:
            node = self.nodes[node_id]
            for dep_id in node.dependencies:
                costs[node_id] = max(
                    costs[node_id],
                    costs[dep_id] + node.operation.computational_cost
                )
        
        max_cost = max(costs.values())
        critical_nodes = [nid for nid, cost in costs.items() if cost == max_cost]
        
        return critical_nodes
    
    def total_size_bytes(self) -> int:
        return sum(t.size_bytes() for t in self.tensors.values())
    
    def total_flops(self) -> int:
        return sum(node.operation.flops() for node in self.nodes.values())
    
    def print_summary(self):
        print(f"\nGraph Summary: {self.name}")
        print(f"  Nodes: {len(self.nodes)}")
        print(f"  Tensors: {len(self.tensors)}")
        print(f"  Total size: {self.total_size_bytes() / 1024 / 1024:.2f} MB")
        print(f"  Total FLOPs: {self.total_flops() / 1e9:.2f} GFLOPs")
        print(f"  Input nodes: {self.input_nodes}")
        print(f"  Output nodes: {self.output_nodes}")
