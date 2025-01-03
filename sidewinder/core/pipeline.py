"""
Core Pipeline class for Sidewinder.
"""

from typing import Optional, Dict, Any
from langgraph.graph import Graph, StateGraph, END
from langgraph.pregel import Channel
from pydantic import BaseModel
import time

from sidewinder.core.config import Environment, Source, Target
from sidewinder.agents.analyzer import DataAnalyzer, AnalyzerState
from sidewinder.agents.transformer import TransformationDesigner, TransformerState
from sidewinder.agents.tester import TestingAgent, TesterState
from sidewinder.testing.metrics import PerformanceMetrics


class PipelineState(BaseModel):
    """State for the entire pipeline."""
    analyzer: Optional[AnalyzerState] = None
    transformer: Optional[TransformerState] = None
    tester: Optional[TesterState] = None
    error: Optional[str] = None
    metrics: Optional[PerformanceMetrics] = None


class Pipeline:
    """Main pipeline class that orchestrates the ETL process."""
    
    def __init__(
        self,
        source: Source | str,
        target: Target | str,
        environment: Environment | str = "local",
        analyzer_code: Optional[str] = None,
        transformer_code: Optional[str] = None,
        tester_code: Optional[str] = None
    ):
        """
        Initialize a new pipeline.
        
        Args:
            source: Data source configuration or source URI
            target: Data target configuration or target URI
            environment: Environment configuration or environment type
            analyzer_code: LLM-generated analyzer code
            transformer_code: LLM-generated transformer code
            tester_code: LLM-generated tester code
        """
        self.source = source if isinstance(source, Source) else Source(type="file", location=source)
        self.target = target if isinstance(target, Target) else Target(type="file", location=target)
        self.environment = (
            environment if isinstance(environment, Environment)
            else Environment(type=environment)
        )
        self.analyzer_code = analyzer_code
        self.transformer_code = transformer_code
        self.tester_code = tester_code
        self._graph: Optional[Graph] = None
        self._state = PipelineState(metrics=PerformanceMetrics())
        
    def _build_graph(self) -> Graph:
        """
        Build the LangGraph pipeline graph.
        """
        # Initialize agents
        analyzer = DataAnalyzer(config={"code": self.analyzer_code} if self.analyzer_code else None)
        transformer = TransformationDesigner(config={"code": self.transformer_code} if self.transformer_code else None)
        tester = TestingAgent(config={"code": self.tester_code} if self.tester_code else None)
        
        # Create the graph
        workflow = StateGraph(PipelineState)
        
        # Define state update functions
        async def analyze_and_update_state(state: PipelineState) -> Dict[str, Any]:
            if state.analyzer:
                try:
                    result = await analyzer.run(state.analyzer)
                    state.analyzer = result
                    state.analyzer.completed = True
                    state.analyzer.messages.append("Analysis completed successfully")
                except Exception as e:
                    state.analyzer.error = str(e)
                    state.analyzer.messages.append(f"Analysis failed: {str(e)}")
            return {
                "analyzer": state.analyzer,
                "transformer": state.transformer,
                "tester": state.tester,
                "error": state.error,
                "metrics": state.metrics
            }
            
        async def transform_and_update_state(state: PipelineState) -> Dict[str, Any]:
            if state.transformer:
                try:
                    result = await transformer.run(state.transformer)
                    state.transformer = result
                    state.transformer.completed = True
                    state.transformer.messages.append("Transformation completed successfully")
                except Exception as e:
                    state.transformer.error = str(e)
                    state.transformer.messages.append(f"Transformation failed: {str(e)}")
            return {
                "analyzer": state.analyzer,
                "transformer": state.transformer,
                "tester": state.tester,
                "error": state.error,
                "metrics": state.metrics
            }
            
        async def test_and_update_state(state: PipelineState) -> Dict[str, Any]:
            if state.tester:
                try:
                    result = await tester.run(state.tester)
                    state.tester = result
                    state.tester.completed = True
                    state.tester.messages.append("Testing completed successfully")
                except Exception as e:
                    state.tester.error = str(e)
                    state.tester.messages.append(f"Testing failed: {str(e)}")
            return {
                "analyzer": state.analyzer,
                "transformer": state.transformer,
                "tester": state.tester,
                "error": state.error,
                "metrics": state.metrics
            }
        
        # Add nodes
        workflow.add_node("analyze", analyze_and_update_state)
        workflow.add_node("transform", transform_and_update_state)
        workflow.add_node("test", test_and_update_state)
        
        # Add edges
        workflow.add_edge("analyze", "transform")
        workflow.add_edge("transform", "test")
        workflow.add_edge("test", END)
        
        # Set entry point
        workflow.set_entry_point("analyze")
        
        # Compile the graph
        app = workflow.compile()
        
        return app

    async def generate(self) -> None:
        """
        Generate the ETL pipeline using LangGraph.
        This method will:
        1. Analyze the source data
        2. Create appropriate transformations
        3. Generate and validate the pipeline code
        4. Create unit tests
        """
        try:
            # Build the graph if not already built
            if not self._graph:
                self._graph = self._build_graph()
            
            # Initialize all states with required fields
            file_patterns = (
                self.source.options.get("file_patterns", [])
                if self.source.options
                else self.source.file_patterns
                if hasattr(self.source, "file_patterns")
                else ["*.json", "*.csv", "*.parquet"]
            )
            
            analyzer_state = AnalyzerState(
                source_directory=self.source.location,
                file_patterns=file_patterns,
                messages=[],
                error=None,
                completed=False,
                schema={},
                data_quality={},
                statistics={},
                patterns={},
                anomalies={},
                execution_time=0.0
            )
            transformer_state = TransformerState(
                source_directory=self.source.location,
                file_patterns=file_patterns,
                discovered_sources={},
                inferred_relationships=[],
                messages=[],
                error=None,
                completed=False,
                transformation_steps=[],
                bronze_data={},
                silver_data={},
                gold_data={},
                validation_results={},
                execution_time=0.0
            )
            tester_state = TesterState(
                source=self.source,
                target_schema=self.target.target_schema,
                messages=[],
                error=None,
                completed=False,
                test_cases=[],
                test_results={},
                test_metrics={},
                test_report={},
                execution_time=0.0
            )
            
            # Initialize pipeline state
            self._state = PipelineState(
                analyzer=analyzer_state,
                transformer=transformer_state,
                tester=tester_state,
                error=None,
                metrics=PerformanceMetrics()
            )
            
            # Start metrics tracking
            start_time = time.time()
            
            # Execute the graph with initial state
            result = await self._graph.ainvoke(
                {
                    "analyzer": self._state.analyzer,
                    "transformer": self._state.transformer,
                    "tester": self._state.tester,
                    "error": None,
                    "metrics": self._state.metrics
                }
            )
            
            # Update state from result
            if "state" in result:
                self._state = result["state"]
            elif "analyzer" in result:
                self._state.analyzer = result["analyzer"]
                self._state.transformer = result["transformer"]
                self._state.tester = result["tester"]
            
            # Record execution metrics
            if self._state.metrics:
                self._state.metrics.record_execution_metrics({
                    "timing": {
                        "total": time.time() - start_time,
                        "analyzer": self._state.analyzer.execution_time if self._state.analyzer else 0,
                        "transformer": self._state.transformer.execution_time if self._state.transformer else 0,
                        "tests": self._state.tester.execution_time if self._state.tester else 0
                    },
                    "success_rate": 1.0 if not self._state.error else 0.0,
                    "errors": [self._state.error] if self._state.error else []
                })
            
            if self._state.error:
                raise Exception(self._state.error)
            
        except Exception as e:
            self._state.error = str(e)
            raise
    
    def validate(self) -> bool:
        """
        Validate the generated pipeline using unit tests.
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        if not self._state.tester or not self._state.tester.test_results:
            return False
            
        return all(self._state.tester.test_results.values())

    async def execute(self) -> None:
        """
        Execute the generated pipeline in the configured environment.
        """
        if not self._state.transformer or not self._state.transformer.completed:
            raise Exception("Pipeline not generated yet. Call generate() first.")
            
        # TODO: Implement pipeline execution
        raise NotImplementedError("Pipeline execution not yet implemented") 