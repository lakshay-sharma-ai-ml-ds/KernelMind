import torch
import numpy as np
from typing import Dict, Tuple, Optional
from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class CorrectnessVerifier:
    
    def __init__(self, tolerance: float = config.NUMERICAL_TOLERANCE):
        self.tolerance = tolerance
        self.verification_results: Dict = {}
    
    def verify(self, original_output: torch.Tensor, 
              optimized_output: torch.Tensor, 
              test_name: str = "unknown") -> Dict:
        logger.info(f"Verifying correctness for {test_name}")
        
        if not self._same_shape(original_output, optimized_output):
            logger.error(f"Output shape mismatch: {original_output.shape} vs {optimized_output.shape}")
            return {"passed": False, "reason": "Shape mismatch"}
        
        if not self._same_dtype(original_output, optimized_output):
            original_output = original_output.float()
            optimized_output = optimized_output.float()
        
        results = {
            "passed": True,
            "test_name": test_name,
        }
        
        l2_error = self._compute_l2_error(original_output, optimized_output)
        results["l2_error"] = float(l2_error)
        
        if l2_error > self.tolerance:
            logger.warning(f"L2 error {l2_error} exceeds tolerance {self.tolerance}")
            results["passed"] = False
            results["reason"] = f"L2 error too high: {l2_error}"
        
        max_error = self._compute_max_error(original_output, optimized_output)
        results["max_error"] = float(max_error)
        
        if max_error > self.tolerance * 10:
            logger.warning(f"Max error {max_error} too high")
            results["passed"] = False
            results["reason"] = f"Max error too high: {max_error}"
        
        relative_error = self._compute_relative_error(original_output, optimized_output)
        results["relative_error"] = float(relative_error)
        
        self.verification_results[test_name] = results
        
        if results["passed"]:
            logger.info(f"Verification passed for {test_name} (L2 error: {l2_error:.2e})")
        else:
            logger.error(f"Verification failed for {test_name}")
        
        return results
    
    def verify_model_outputs(self, original_model: torch.nn.Module,
                            optimized_model: torch.nn.Module,
                            test_inputs: list, num_tests: int = 10) -> Dict:
        logger.info(f"Verifying {num_tests} model outputs")
        
        all_passed = True
        errors = []
        
        for i in range(num_tests):
            original_model.eval()
            optimized_model.eval()
            
            with torch.no_grad():
                orig_out = original_model(*test_inputs)
                opt_out = optimized_model(*test_inputs)
            
            result = self.verify(orig_out, opt_out, f"test_{i}")
            
            if not result["passed"]:
                all_passed = False
                errors.append(result)
        
        return {
            "all_passed": all_passed,
            "num_tests": num_tests,
            "num_failed": len(errors),
            "errors": errors,
        }
    
    def _same_shape(self, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        return t1.shape == t2.shape
    
    def _same_dtype(self, t1: torch.Tensor, t2: torch.Tensor) -> bool:
        return t1.dtype == t2.dtype
    
    def _compute_l2_error(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        diff = t1.float() - t2.float()
        l2_error = torch.norm(diff).item() / torch.norm(t1.float()).item()
        return l2_error
    
    def _compute_max_error(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        diff = torch.abs(t1.float() - t2.float())
        return torch.max(diff).item()
    
    def _compute_relative_error(self, t1: torch.Tensor, t2: torch.Tensor) -> float:
        diff = torch.abs(t1.float() - t2.float())
        denom = torch.abs(t1.float()) + 1e-8
        rel_error = torch.mean(diff / denom).item()
        return rel_error
    
    def get_verification_report(self) -> Dict:
        if not self.verification_results:
            return {"status": "No verifications run yet"}
        
        passed_count = sum(1 for r in self.verification_results.values() if r["passed"])
        total_count = len(self.verification_results)
        
        avg_l2_error = np.mean([r.get("l2_error", 0) 
                               for r in self.verification_results.values()])
        
        max_max_error = np.max([r.get("max_error", 0) 
                               for r in self.verification_results.values()])
        
        return {
            "total_tests": total_count,
            "passed": passed_count,
            "failed": total_count - passed_count,
            "pass_rate_percent": (passed_count / total_count * 100) if total_count > 0 else 0,
            "average_l2_error": float(avg_l2_error),
            "max_max_error": float(max_max_error),
            "tolerance": self.tolerance,
        }
