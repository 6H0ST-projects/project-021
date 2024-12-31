"""
Testing agent for validating data transformations.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import pandas as pd
import logging
from langchain.graphs import StateGraph
from langchain.graphs.state_graph import END

from sidewinder.agents.base import BaseAgent, BaseAgentState
from sidewinder.core.config import Source, SourceType, TestConfig
from sidewinder.core.llm import DataEngineeringAgent, CodeGenerationRequest

# Setup logging
logger = logging.getLogger(__name__)

class TestCase(BaseModel):
    """A single test case in the testing process."""
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
    test_data: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None


class TestingAgent(BaseAgent[TesterState]):
    """Agent responsible for testing data transformations."""
    
    def __init__(self):
        super().__init__()
        self.llm_agent = DataEngineeringAgent()
    
    async def run(self, state: TesterState) -> TesterState:
        """
        Execute the testing process:
        1. Generate test data
        2. Run test cases
        3. Validate results
        4. Measure performance
        
        Args:
            state: Current tester state
            
        Returns:
            Updated tester state with test results
        """
        try:
            # Create testing workflow
            workflow = StateGraph(TesterState)
            
            # Add nodes for each testing phase
            workflow.add_node("generate", self._generate_test_data)
            workflow.add_node("execute", self._execute_test_cases)
            workflow.add_node("validate", self._validate_test_results)
            workflow.add_node("measure", self._measure_performance)
            
            # Add edges
            workflow.add_edge("generate", "execute")
            workflow.add_edge("execute", "validate")
            workflow.add_edge("validate", "measure")
            
            # Set entry point
            workflow.set_entry_point("generate")
            
            # Set end point conditions
            workflow.add_edge("measure", END)
            
            # Execute workflow
            state = await workflow.arun(state)
            
            state.completed = True
            return state
            
        except Exception as e:
            logger.error(f"Testing failed: {str(e)}")
            state.error = str(e)
            return state
    
    async def _generate_test_data(self, state: TesterState) -> TesterState:
        """Generate test data based on source schema and constraints."""
        try:
            # Generate test data code using LLM
            test_data_code = await self.llm_agent.generate_test_data(
                source_type=state.source.type,
                source_data=state.source.sample_data,
                target_schema=state.target_schema,
                context={
                    "source_config": state.source.dict(),
                    "purpose": "generate representative test data"
                }
            )
            
            # Add test data generation step
            state.test_cases.append(
                TestCase(
                    name="generate_test_data",
                    description="Generate representative test data",
                    code=test_data_code
                )
            )
            
            # Execute test data generation code
            exec_globals = {"source_data": state.source.sample_data}
            exec(test_data_code, exec_globals)
            generate_func = exec_globals.get("generate_test_data")
            
            if generate_func:
                state.test_data = generate_func(state.source.sample_data)
            
            return state
            
        except Exception as e:
            logger.error(f"Test data generation failed: {str(e)}")
            raise
    
    async def _execute_test_cases(self, state: TesterState) -> TesterState:
        """Execute test cases on the generated test data."""
        try:
            # Generate test execution code using LLM
            test_exec_code = await self.llm_agent.generate_test_cases(
                source_type=state.source.type,
                source_data=state.test_data,
                target_schema=state.target_schema,
                context={
                    "source_config": state.source.dict(),
                    "purpose": "execute comprehensive test cases"
                }
            )
            
            # Add test execution step
            state.test_cases.append(
                TestCase(
                    name="execute_test_cases",
                    description="Execute comprehensive test cases",
                    code=test_exec_code,
                    dependencies=["generate_test_data"]
                )
            )
            
            # Execute test cases
            exec_globals = {"test_data": state.test_data}
            exec(test_exec_code, exec_globals)
            execute_func = exec_globals.get("execute_test_cases")
            
            if execute_func:
                state.test_results = execute_func(state.test_data)
            
            return state
            
        except Exception as e:
            logger.error(f"Test execution failed: {str(e)}")
            raise
    
    async def _validate_test_results(self, state: TesterState) -> TesterState:
        """Validate test results against expected outcomes."""
        try:
            # Generate validation code using LLM
            validation_code = await self.llm_agent.validate_test_results(
                source_type=state.source.type,
                source_data=state.test_results,
                target_schema=state.target_schema,
                context={
                    "source_config": state.source.dict(),
                    "purpose": "validate test results against expected outcomes"
                }
            )
            
            # Add validation step
            state.test_cases.append(
                TestCase(
                    name="validate_test_results",
                    description="Validate test results against expected outcomes",
                    code=validation_code,
                    dependencies=["execute_test_cases"]
                )
            )
            
            # Execute validation code
            exec_globals = {
                "test_results": state.test_results,
                "target_schema": state.target_schema
            }
            exec(validation_code, exec_globals)
            validate_func = exec_globals.get("validate_test_results")
            
            if validate_func:
                state.validation_results = validate_func(
                    state.test_results,
                    state.target_schema
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Test validation failed: {str(e)}")
            raise
    
    async def _measure_performance(self, state: TesterState) -> TesterState:
        """Measure performance metrics of the test execution."""
        try:
            # Generate performance measurement code using LLM
            performance_code = await self.llm_agent.measure_performance(
                source_type=state.source.type,
                source_data=state.test_results,
                target_schema=state.target_schema,
                context={
                    "source_config": state.source.dict(),
                    "purpose": "measure test execution performance"
                }
            )
            
            # Add performance measurement step
            state.test_cases.append(
                TestCase(
                    name="measure_performance",
                    description="Measure test execution performance",
                    code=performance_code,
                    dependencies=["validate_test_results"]
                )
            )
            
            # Execute performance measurement code
            exec_globals = {
                "test_results": state.test_results,
                "validation_results": state.validation_results
            }
            exec(performance_code, exec_globals)
            measure_func = exec_globals.get("measure_performance")
            
            if measure_func:
                state.performance_metrics = measure_func(
                    state.test_results,
                    state.validation_results
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Performance measurement failed: {str(e)}")
            raise 