from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

@dataclass
class AnalyzerState:
    messages: List[str] = field(default_factory=list)
    error: Optional[str] = None
    completed: bool = False
    schema: Dict[str, Any] = field(default_factory=dict)
    data_quality: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    patterns: Dict[str, Any] = field(default_factory=dict)
    anomalies: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

@dataclass
class TransformerState:
    messages: List[str] = field(default_factory=list)
    error: Optional[str] = None
    completed: bool = False
    source_directory: str = ""
    file_patterns: List[str] = field(default_factory=list)
    discovered_sources: Dict[str, Any] = field(default_factory=dict)
    inferred_relationships: Dict[str, Any] = field(default_factory=dict)
    transformation_steps: List[Dict[str, Any]] = field(default_factory=list)
    bronze_data: Dict[str, Any] = field(default_factory=dict)
    silver_data: Dict[str, Any] = field(default_factory=dict)
    gold_data: Dict[str, Any] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

@dataclass
class TesterState:
    messages: List[str] = field(default_factory=list)
    error: Optional[str] = None
    completed: bool = False
    test_cases: List[Dict[str, Any]] = field(default_factory=list)
    test_results: Dict[str, Any] = field(default_factory=dict)
    test_metrics: Dict[str, Any] = field(default_factory=dict)
    test_report: Dict[str, Any] = field(default_factory=dict)
    execution_time: float = 0.0

@dataclass
class PipelineState:
    analyzer: Optional[AnalyzerState] = None
    transformer: Optional[TransformerState] = None
    tester: Optional[TesterState] = None
    
    def __post_init__(self):
        if self.analyzer is None:
            self.analyzer = AnalyzerState()
        if self.transformer is None:
            self.transformer = TransformerState()
        if self.tester is None:
            self.tester = TesterState() 