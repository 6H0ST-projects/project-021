"""
Performance metrics tracking for Sidewinder.
"""

from typing import Dict, Any, List
import time
import json
from dataclasses import dataclass, asdict


@dataclass
class LLMMetrics:
    """Metrics for LLM operations."""
    time: float = 0.0
    tokens: int = 0
    retries: int = 0
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []


@dataclass
class ExecutionMetrics:
    """Metrics for pipeline execution."""
    total_time: float = 0.0
    analyzer_time: float = 0.0
    transformer_time: float = 0.0
    test_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    success_rate: float = 0.0
    error_count: int = 0


class PerformanceMetrics:
    """Track and aggregate performance metrics."""
    
    def __init__(self):
        self.llm_metrics = {
            "analyzer": LLMMetrics(),
            "transformer": LLMMetrics(),
            "tester": LLMMetrics()
        }
        self.execution_metrics = ExecutionMetrics()
        self._start_time = time.time()
    
    def record_llm_metrics(self, component: str, metrics: Dict[str, Any]):
        """Record metrics for an LLM operation."""
        if component not in self.llm_metrics:
            raise ValueError(f"Invalid component: {component}")
        
        self.llm_metrics[component].time = metrics.get("time", 0)
        self.llm_metrics[component].tokens = metrics.get("tokens", 0)
        self.llm_metrics[component].retries = metrics.get("retries", 0)
        
        if "errors" in metrics:
            self.llm_metrics[component].errors.extend(metrics["errors"])
    
    def record_execution_metrics(self, results: Dict[str, Any]):
        """Record metrics from pipeline execution."""
        self.execution_metrics.total_time = time.time() - self._start_time
        
        # Extract component times from results
        if "timing" in results:
            self.execution_metrics.analyzer_time = results["timing"].get("analyzer", 0)
            self.execution_metrics.transformer_time = results["timing"].get("transformer", 0)
            self.execution_metrics.test_time = results["timing"].get("tests", 0)
        
        # Extract resource usage
        if "resources" in results:
            self.execution_metrics.memory_usage = results["resources"].get("memory_mb", 0)
            self.execution_metrics.cpu_usage = results["resources"].get("cpu_percent", 0)
        
        # Extract success metrics
        self.execution_metrics.success_rate = results.get("success_rate", 0)
        self.execution_metrics.error_count = len(results.get("errors", []))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary format."""
        return {
            "llm": {
                component: asdict(metrics)
                for component, metrics in self.llm_metrics.items()
            },
            "execution": asdict(self.execution_metrics)
        }
    
    def save_to_file(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> "PerformanceMetrics":
        """Load metrics from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        
        metrics = cls()
        
        # Load LLM metrics
        for component, values in data["llm"].items():
            metrics.llm_metrics[component] = LLMMetrics(**values)
        
        # Load execution metrics
        metrics.execution_metrics = ExecutionMetrics(**data["execution"])
        
        return metrics 