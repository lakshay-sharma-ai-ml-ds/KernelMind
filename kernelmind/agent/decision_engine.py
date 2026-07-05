from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from ..config import OptimizationLevel
from ..utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class OptimizationDecision:
    action: str
    reasoning: str
    expected_impact: float
    risk_level: str

class DecisionEngine:
    
    def __init__(self, optimization_level: OptimizationLevel = OptimizationLevel.HIGH):
        self.optimization_level = optimization_level
        self.decision_history: List[OptimizationDecision] = []
    
    def decide_optimizations(self, suggestions: List[Dict]) -> List[OptimizationDecision]:
        logger.info(f"Making optimization decisions at level {self.optimization_level.name}")
        
        decisions = []
        
        for suggestion in suggestions:
            decision = self._evaluate_suggestion(suggestion)
            if decision:
                decisions.append(decision)
                self.decision_history.append(decision)
        
        decisions = self._prioritize_decisions(decisions)
        return decisions
    
    def _evaluate_suggestion(self, suggestion: Dict) -> Optional[OptimizationDecision]:
        opt_type = suggestion.get("type", "unknown")
        benefit = suggestion.get("estimated_speedup_percent", 0)
        constraints = suggestion.get("constraints", [])
        
        risk_level = self._assess_risk(opt_type, constraints)
        expected_impact = benefit / 100.0
        
        should_apply = self._should_apply_optimization(
            opt_type, expected_impact, risk_level
        )
        
        if should_apply:
            decision = OptimizationDecision(
                action=opt_type,
                reasoning=suggestion.get("description", ""),
                expected_impact=expected_impact,
                risk_level=risk_level
            )
            logger.debug(f"Decision: Apply {opt_type} (impact: {expected_impact:.2%}, risk: {risk_level})")
            return decision
        else:
            logger.debug(f"Decision: Skip {opt_type} (failed safety check)")
            return None
    
    def _assess_risk(self, opt_type: str, constraints: List[str]) -> str:
        if opt_type == "fusion":
            return "low" if not constraints else "medium"
        elif opt_type == "quantization":
            return "high"
        elif opt_type == "memory_optimization":
            return "low"
        elif opt_type == "compute_optimization":
            return "medium"
        else:
            return "medium"
    
    def _should_apply_optimization(self, opt_type: str, 
                                   expected_impact: float, risk_level: str) -> bool:
        
        if self.optimization_level == OptimizationLevel.NONE:
            return False
        
        elif self.optimization_level == OptimizationLevel.LOW:
            return risk_level == "low" and expected_impact > 0.02
        
        elif self.optimization_level == OptimizationLevel.MEDIUM:
            return (risk_level in ["low", "medium"] and expected_impact > 0.01) or \
                   (risk_level == "high" and expected_impact > 0.05)
        
        elif self.optimization_level == OptimizationLevel.HIGH:
            return (risk_level in ["low", "medium"] and expected_impact > 0.005) or \
                   (risk_level == "high" and expected_impact > 0.03)
        
        elif self.optimization_level == OptimizationLevel.AGGRESSIVE:
            return expected_impact > 0.001
        
        return False
    
    def _prioritize_decisions(self, decisions: List[OptimizationDecision]) \
                             -> List[OptimizationDecision]:
        
        def priority_score(decision: OptimizationDecision) -> Tuple[float, float]:
            impact = decision.expected_impact
            
            risk_multiplier = {
                "low": 1.0,
                "medium": 0.7,
                "high": 0.4,
            }.get(decision.risk_level, 0.5)
            
            return (impact * risk_multiplier, impact)
        
        sorted_decisions = sorted(decisions, key=priority_score, reverse=True)
        
        return sorted_decisions
    
    def should_verify(self, decision: OptimizationDecision) -> bool:
        if decision.risk_level == "high":
            return True
        if decision.expected_impact > 0.20:
            return True
        return False
    
    def get_decision_summary(self) -> Dict:
        total_decisions = len(self.decision_history)
        
        avg_impact = sum(d.expected_impact for d in self.decision_history) / max(1, total_decisions)
        
        by_risk = {}
        for decision in self.decision_history:
            risk = decision.risk_level
            by_risk[risk] = by_risk.get(risk, 0) + 1
        
        return {
            "total_decisions": total_decisions,
            "average_impact": avg_impact,
            "by_risk_level": by_risk,
            "optimization_level": self.optimization_level.name,
        }
