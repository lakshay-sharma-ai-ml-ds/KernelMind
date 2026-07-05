from typing import Dict, List, Optional
import json
from datetime import datetime
from pathlib import Path
from ..utils.logger import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    
    def __init__(self, metrics_dir: str = "./metrics"):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(exist_ok=True)
        self.current_metrics: Dict = {}
        self.history: List[Dict] = []
    
    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        timestamp = datetime.now().isoformat()
        
        metric_entry = {
            "timestamp": timestamp,
            "name": name,
            "value": value,
            "tags": tags or {},
        }
        
        self.current_metrics[name] = value
        self.history.append(metric_entry)
        
        logger.debug(f"Recorded metric: {name} = {value}")
    
    def record_optimization(self, opt_type: str, speedup: float, 
                          memory_reduction: float = 0, accuracy_loss: float = 0):
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "optimization",
            "optimization_type": opt_type,
            "speedup": speedup,
            "memory_reduction_percent": memory_reduction,
            "accuracy_loss_percent": accuracy_loss,
        }
        
        self.history.append(metric_entry)
        logger.info(f"Optimization {opt_type}: speedup={speedup:.2f}x, "
                   f"memory_reduction={memory_reduction:.1f}%")
    
    def record_benchmark(self, model_name: str, batch_size: int,
                        latency_ms: float, throughput: float, 
                        memory_mb: float = 0):
        metric_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": "benchmark",
            "model": model_name,
            "batch_size": batch_size,
            "latency_ms": latency_ms,
            "throughput_samples_per_sec": throughput,
            "memory_mb": memory_mb,
        }
        
        self.history.append(metric_entry)
    
    def detect_regressions(self, baseline: Dict, current: Dict, 
                          threshold_percent: float = 10.0) -> List[str]:
        regressions = []
        
        for metric_name in baseline:
            if metric_name not in current:
                continue
            
            baseline_val = baseline[metric_name]
            current_val = current[metric_name]
            
            if baseline_val == 0:
                continue
            
            percent_change = abs((current_val - baseline_val) / baseline_val) * 100
            
            if metric_name.endswith("_latency") or metric_name.endswith("_time"):
                if current_val > baseline_val * (1 + threshold_percent / 100):
                    regressions.append(
                        f"{metric_name}: {baseline_val:.2f} -> {current_val:.2f} "
                        f"({percent_change:.1f}% worse)"
                    )
            
            elif metric_name.startswith("throughput") or metric_name.endswith("_speedup"):
                if current_val < baseline_val * (1 - threshold_percent / 100):
                    regressions.append(
                        f"{metric_name}: {baseline_val:.2f} -> {current_val:.2f} "
                        f"({percent_change:.1f}% worse)"
                    )
        
        if regressions:
            logger.warning(f"Detected {len(regressions)} performance regressions")
            for regression in regressions:
                logger.warning(f"  - {regression}")
        
        return regressions
    
    def save_metrics(self, filename: Optional[str] = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.json"
        
        filepath = self.metrics_dir / filename
        
        output = {
            "timestamp": datetime.now().isoformat(),
            "current_metrics": self.current_metrics,
            "history": self.history,
        }
        
        with open(filepath, "w") as f:
            json.dump(output, f, indent=2)
        
        logger.info(f"Saved metrics to {filepath}")
        return str(filepath)
    
    def load_baseline(self, filename: str) -> Dict:
        filepath = self.metrics_dir / filename
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        logger.info(f"Loaded baseline from {filepath}")
        return data.get("current_metrics", {})
    
    def get_summary(self) -> Dict:
        return {
            "total_metrics_recorded": len(self.history),
            "current_values": self.current_metrics,
            "num_optimizations": len([h for h in self.history if h.get("type") == "optimization"]),
            "num_benchmarks": len([h for h in self.history if h.get("type") == "benchmark"]),
        }
    
    def export_to_csv(self, filename: Optional[str] = None):
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_{timestamp}.csv"
        
        filepath = self.metrics_dir / filename
        
        import csv
        
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            
            if self.history:
                headers = set()
                for entry in self.history:
                    headers.update(entry.keys())
                
                headers = sorted(list(headers))
                writer.writerow(headers)
                
                for entry in self.history:
                    row = [entry.get(h, "") for h in headers]
                    writer.writerow(row)
        
        logger.info(f"Exported metrics to {filepath}")
        return str(filepath)
