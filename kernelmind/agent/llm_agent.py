from typing import List, Dict, Optional, Tuple
import json
from anthropic import Anthropic
from ..core.graph import ComputationalGraph, Node
from ..core.constants import OpType
from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class OptimizationAgent:
    
    def __init__(self):
        self.client = Anthropic()
        self.conversation_history = []
        self.optimization_suggestions = []
    
    def optimize(self, graph: ComputationalGraph) -> ComputationalGraph:
        logger.info("Starting LLM-based optimization")
        
        graph_analysis = self._analyze_graph(graph)
        logger.debug(f"Graph analysis: {graph_analysis}")
        
        suggestions = self._get_optimization_suggestions(graph_analysis)
        logger.info(f"Generated {len(suggestions)} optimization suggestions")
        
        self.optimization_suggestions = suggestions
        return graph
    
    def _analyze_graph(self, graph: ComputationalGraph) -> Dict:
        total_nodes = len(graph.nodes)
        total_size = graph.total_size_bytes()
        total_flops = graph.total_flops()
        
        op_counts: Dict[OpType, int] = {}
        for node in graph.nodes.values():
            op_type = node.operation.op_type
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
        
        critical_path = graph.get_critical_path()
        
        return {
            "num_nodes": total_nodes,
            "total_size_mb": total_size / 1e6,
            "total_flops_gflops": total_flops / 1e9,
            "operation_distribution": {str(k.name): v for k, v in op_counts.items()},
            "critical_path_length": len(critical_path),
            "input_nodes": len(graph.input_nodes),
            "output_nodes": len(graph.output_nodes),
        }
    
    def _get_optimization_suggestions(self, graph_analysis: Dict) -> List[Dict]:
        system_prompt = """You are an expert ML systems compiler specializing in neural network optimization.
Analyze the computational graph and provide specific, actionable optimization recommendations.
Focus on: operator fusion, kernel optimization, memory efficiency, and latency reduction.
Format responses as JSON with a 'suggestions' array containing optimization strategies."""
        
        analysis_prompt = f"""Analyze this computational graph and provide optimization suggestions:

Graph Analysis:
{json.dumps(graph_analysis, indent=2)}

Provide 3-5 specific optimization recommendations. For each recommendation:
1. Identify the optimization type (fusion, quantization, memory, compute, etc.)
2. Explain the potential benefit
3. Estimate the expected speedup percentage
4. List any constraints or risks

Format as JSON:
{{
  "suggestions": [
    {{
      "type": "optimization_type",
      "description": "what to optimize",
      "benefit": "why this helps",
      "estimated_speedup_percent": 15,
      "constraints": []
    }}
  ]
}}"""
        
        try:
            self.conversation_history.append({
                "role": "user",
                "content": analysis_prompt
            })
            
            response = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                system=system_prompt,
                messages=self.conversation_history
            )
            
            assistant_message = response.content[0].text
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            suggestions = self._parse_suggestions(assistant_message)
            return suggestions
        
        except Exception as e:
            logger.error(f"LLM optimization failed: {e}")
            return self._get_default_suggestions(graph_analysis)
    
    def _parse_suggestions(self, response_text: str) -> List[Dict]:
        try:
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}") + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                parsed = json.loads(json_str)
                return parsed.get("suggestions", [])
        
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM suggestions: {e}")
        
        return []
    
    def _get_default_suggestions(self, graph_analysis: Dict) -> List[Dict]:
        suggestions = []
        
        if graph_analysis.get("total_flops_gflops", 0) > 1:
            suggestions.append({
                "type": "fusion",
                "description": "Fuse consecutive linear and activation operations",
                "benefit": "Reduces memory bandwidth overhead",
                "estimated_speedup_percent": 10,
                "constraints": ["Limited by kernel size limits"]
            })
        
        if graph_analysis.get("total_size_mb", 0) > 100:
            suggestions.append({
                "type": "memory_optimization",
                "description": "Optimize tensor memory layout and caching",
                "benefit": "Improves cache locality",
                "estimated_speedup_percent": 5,
                "constraints": []
            })
        
        suggestions.append({
            "type": "quantization",
            "description": "Apply INT8 quantization to weights",
            "benefit": "Reduces memory footprint and bandwidth",
            "estimated_speedup_percent": 15,
            "constraints": ["Requires accuracy validation"]
        })
        
        return suggestions
    
    def refine_optimization(self, feedback: str) -> List[Dict]:
        logger.info("Refining optimization based on feedback")
        
        refinement_prompt = f"""The user provided this feedback on the optimization suggestions:
        
"{feedback}"

Based on this feedback, provide refined optimization recommendations that address the user's concerns.
Format as JSON with the same structure as before."""
        
        try:
            self.conversation_history.append({
                "role": "user",
                "content": refinement_prompt
            })
            
            response = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=config.LLM_TEMPERATURE,
                messages=self.conversation_history
            )
            
            assistant_message = response.content[0].text
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            refined_suggestions = self._parse_suggestions(assistant_message)
            self.optimization_suggestions = refined_suggestions
            
            return refined_suggestions
        
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return self.optimization_suggestions
    
    def get_kernel_generation_prompt(self, node: Node) -> str:
        prompt = f"""Generate an optimized kernel implementation for this operation:

Operation Type: {node.operation.op_type.name}
Inputs: {node.inputs}
Outputs: {node.outputs}
Attributes: {json.dumps(node.operation.attributes, default=str)}

Requirements:
1. Optimize for Apple Silicon (Metal) or NVIDIA CUDA
2. Use efficient memory access patterns
3. Minimize register usage
4. Include performance-critical optimizations
5. Add detailed comments

Format the response as compilable kernel code."""
        
        return prompt
    
    def generate_kernel_code(self, node: Node, backend: str = "metal") -> str:
        logger.info(f"Generating kernel code for {node.name} using LLM")
        
        prompt = self.get_kernel_generation_prompt(node)
        prompt += f"\nTarget backend: {backend}"
        
        try:
            response = self.client.messages.create(
                model=config.LLM_MODEL,
                max_tokens=config.LLM_MAX_TOKENS,
                temperature=0.5,
                messages=[{"role": "user", "content": prompt}]
            )
            
            kernel_code = response.content[0].text
            return kernel_code
        
        except Exception as e:
            logger.error(f"Kernel generation failed: {e}")
            return ""
    
    def get_suggestions(self) -> List[Dict]:
        return self.optimization_suggestions
    
    def reset_conversation(self):
        self.conversation_history = []
        logger.info("Conversation history cleared")
