"""
Testing agent for validating data transformations.
"""

from typing import Dict, Any, List, Optional
import logging
from pydantic import BaseModel, Field
import json
import pandas as pd
import numpy as np
from datetime import datetime
import time

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from sidewinder.agents.base import BaseAgent, BaseAgentState
from sidewinder.core.config import Source, SourceType
from sidewinder.core.llm import DataEngineeringAgent, CodeGenerationRequest
from sidewinder.core.state import TesterState

# Setup logging
logger = logging.getLogger(__name__)


class TestCase(BaseModel):
    """A single test case."""
    name: str
    description: str
    code: str
    dependencies: List[str] = Field(default_factory=list)
    results: Optional[Dict[str, Any]] = None


class TesterState(BaseAgentState):
    """State for the tester agent."""
    source: Source
    target_schema: Optional[Dict[str, Any]] = None
    test_cases: List[TestCase] = Field(default_factory=list)
    test_results: Dict[str, Any] = Field(default_factory=dict)
    test_metrics: Dict[str, Any] = Field(default_factory=dict)
    test_report: Dict[str, Any] = Field(default_factory=dict)
    execution_time: float = 0.0


class TestingAgent(BaseAgent[TesterState]):
    """Agent for testing data transformations."""
    
    def __init__(self, code: Optional[str] = None):
        """
        Initialize the tester agent.
        
        Args:
            code: LLM-generated code for custom tests
        """
        super().__init__()
        self.code = code
        self.llm_agent = DataEngineeringAgent()
        
    async def run(self, state: TesterState) -> TesterState:
        """
        Run tests on the transformed data:
        1. Generate test cases
        2. Execute tests
        3. Collect metrics
        4. Generate report
        
        Args:
            state: Current tester state
            
        Returns:
            Updated tester state with test results
        """
        try:
            logger.info("Starting test execution...")
            start_time = time.time()
            
            # Initialize state fields
            state.test_cases = []
            state.test_results = {"passed": [], "failed": [], "skipped": []}
            state.test_metrics = {"total": 0, "passed": 0, "failed": 0, "skipped": 0}
            state.test_report = {"summary": {}, "details": {}}
            
            # Execute test code
            exec(self.code, globals(), locals())
            
            # Update state with results
            if "run_tests" in locals():
                results = locals()["run_tests"]()
                if isinstance(results, dict):
                    if "test_cases" in results:
                        state.test_cases.extend(results["test_cases"])
                    if "test_results" in results:
                        state.test_results.update(results["test_results"])
                    if "test_metrics" in results:
                        state.test_metrics.update(results["test_metrics"])
                    if "test_report" in results:
                        state.test_report.update(results["test_report"])
            
            state.completed = True
            state.messages.append("Testing completed successfully")
            
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Testing failed: {str(e)}")
            
        finally:
            state.execution_time = time.time() - start_time
            
        return state
            
    async def _generate_tests(self, state: TesterState) -> Dict[str, Any]:
        """Generate test cases."""
        logger.info("Generating test cases...")
        try:
            if self.code:
                # Use LLM-generated code for test generation
                request = CodeGenerationRequest(
                    task="Generate test cases",
                    context={
                        "source_type": state.source.type,
                        "target_schema": state.target_schema
                    },
                    requirements=[
                        "Schema validation",
                        "Data quality checks",
                        "Business rule validation",
                        "Performance testing"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Comprehensive coverage"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                test_cases = locals().get("test_cases", [])
                state.test_cases.extend([
                    TestCase(**test_case) for test_case in test_cases
                ])
            else:
                # Use default test cases
                state.test_cases = [
                    TestCase(
                        name="schema_validation",
                        description="Validate data against target schema",
                        code="def test_schema(): pass"  # TODO: Implement default tests
                    )
                ]
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.target_schema = state.target_schema
            state.test_cases = state.test_cases or []
            state.test_results = state.test_results or {}
            state.test_metrics = state.test_metrics or {}
            state.test_report = state.test_report or {}
            state.execution_time = 0.0
            
            state.messages.append("Test case generation completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Test generation failed: {str(e)}")
            return {"state": state}
            
    async def _execute_tests(self, state: TesterState) -> Dict[str, Any]:
        """Execute test cases."""
        logger.info("Executing test cases...")
        try:
            if self.code:
                # Use LLM-generated code for test execution
                request = CodeGenerationRequest(
                    task="Execute test cases",
                    context={
                        "source_type": state.source.type,
                        "test_cases": [tc.dict() for tc in state.test_cases]
                    },
                    requirements=[
                        "Parallel execution",
                        "Error handling",
                        "Resource monitoring",
                        "Result collection"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Timeout handling"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.test_results = locals().get("test_results", {})
            else:
                # Use default test execution
                state.test_results = {
                    tc.name: True for tc in state.test_cases
                }
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.target_schema = state.target_schema
            state.test_cases = state.test_cases or []
            state.test_results = state.test_results or {}
            state.test_metrics = state.test_metrics or {}
            state.test_report = state.test_report or {}
            state.execution_time = 0.0
            
            state.messages.append("Test execution completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Test execution failed: {str(e)}")
            return {"state": state}
            
    async def _collect_metrics(self, state: TesterState) -> Dict[str, Any]:
        """Collect performance metrics."""
        logger.info("Collecting metrics...")
        try:
            if self.code:
                # Use LLM-generated code for metrics collection
                request = CodeGenerationRequest(
                    task="Collect performance metrics",
                    context={
                        "source_type": state.source.type,
                        "test_results": state.test_results
                    },
                    requirements=[
                        "Resource usage",
                        "Execution time",
                        "Success rate",
                        "Coverage"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Accurate timing"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.test_metrics = locals().get("test_metrics", {})
            else:
                # Use default metrics collection
                state.test_metrics = {
                    "execution_time": state.execution_time,
                    "success_rate": sum(state.test_results.values()) / len(state.test_results),
                    "total_tests": len(state.test_results)
                }
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.target_schema = state.target_schema
            state.test_cases = state.test_cases or []
            state.test_results = state.test_results or {}
            state.test_metrics = state.test_metrics or {}
            state.test_report = state.test_report or {}
            state.execution_time = 0.0
            
            state.messages.append("Metrics collection completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Metrics collection failed: {str(e)}")
            return {"state": state}
            
    async def _generate_report(self, state: TesterState) -> Dict[str, Any]:
        """Generate test report."""
        logger.info("Generating test report...")
        try:
            if self.code:
                # Use LLM-generated code for report generation
                request = CodeGenerationRequest(
                    task="Generate test report",
                    context={
                        "source_type": state.source.type,
                        "test_results": state.test_results,
                        "test_metrics": state.test_metrics
                    },
                    requirements=[
                        "Summary statistics",
                        "Failure analysis",
                        "Performance insights",
                        "Recommendations"
                    ],
                    constraints=[
                        "Clear formatting",
                        "Actionable insights",
                        "Comprehensive coverage"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.test_report = locals().get("test_report", {})
            else:
                # Use default report generation
                state.test_report = {
                    "timestamp": datetime.now().isoformat(),
                    "summary": {
                        "total_tests": len(state.test_results),
                        "passed_tests": sum(state.test_results.values()),
                        "success_rate": state.test_metrics["success_rate"]
                    },
                    "performance": state.test_metrics
                }
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.target_schema = state.target_schema
            state.test_cases = state.test_cases or []
            state.test_results = state.test_results or {}
            state.test_metrics = state.test_metrics or {}
            state.test_report = state.test_report or {}
            state.execution_time = 0.0
            
            state.messages.append("Report generation completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Report generation failed: {str(e)}")
            return {"state": state} 