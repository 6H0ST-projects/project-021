"""
Data analyzer agent for inspecting and understanding source data.
"""

from typing import Dict, Any, List, Optional
import logging
from pydantic import BaseModel, Field
import json
import pandas as pd
import numpy as np
from enum import Enum
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from sidewinder.agents.base import BaseAgent, BaseAgentState
from sidewinder.core.config import Source, SourceType
from sidewinder.core.llm import DataEngineeringAgent, CodeGenerationRequest

# Setup logging
logger = logging.getLogger(__name__)

class AnalysisStep(BaseModel):
    """A single step in the data analysis process."""
    name: str
    description: str
    code: str
    dependencies: List[str] = Field(default_factory=list)
    results: Optional[Dict[str, Any]] = None


class AnalyzerState(BaseAgentState):
    """State for the analyzer agent."""
    source: Source
    analysis_steps: List[AnalysisStep] = Field(default_factory=list)
    schema: Optional[Dict[str, Any]] = None
    sample_data: Optional[Dict[str, Any]] = None
    data_quality_issues: list[str] = Field(default_factory=list)
    inferred_data_types: Optional[Dict[str, str]] = None
    statistics: Optional[Dict[str, Dict[str, Any]]] = None


class DataAnalyzer(BaseAgent[AnalyzerState]):
    """Agent responsible for analyzing source data and determining its characteristics."""
    
    def __init__(self):
        super().__init__()
        self.llm_agent = DataEngineeringAgent()
    
    async def run(self, state: AnalyzerState) -> AnalyzerState:
        """
        Analyze the source data to determine:
        1. Data schema and types
        2. Data quality issues
        3. Sample data for validation
        
        Args:
            state: Current analyzer state
            
        Returns:
            Updated analyzer state with data insights
        """
        try:
            # Create analysis workflow
            workflow = StateGraph(AnalyzerState)
            
            # Add nodes for each analysis step
            workflow.add_node("connect", self._connect_to_source)
            workflow.add_node("sample", self._extract_sample)
            workflow.add_node("analyze", self._analyze_data)
            workflow.add_node("validate", self._validate_results)
            
            # Add edges
            workflow.add_edge("connect", "sample")
            workflow.add_edge("sample", "analyze")
            workflow.add_edge("analyze", "validate")
            
            # Set entry point
            workflow.set_entry_point("connect")
            
            # Set end point conditions
            workflow.add_edge("validate", END)
            
            # Execute workflow
            state = await workflow.arun(state)
            
            state.completed = True
            return state
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            state.error = str(e)
            return state
    
    async def _connect_to_source(self, state: AnalyzerState) -> AnalyzerState:
        """Connect to data source and validate access."""
        try:
            # Generate connection code using LLM
            connection_code = await self.llm_agent.analyze_data_source(
                source_type=state.source.type,
                sample_data=None,
                context={
                    "source_config": state.source.dict(),
                    "purpose": "establish connection and validate access"
                }
            )
            
            # Add connection step
            state.analysis_steps.append(
                AnalysisStep(
                    name="connect_to_source",
                    description="Establish connection to the data source",
                    code=connection_code
                )
            )
            
            # Execute connection code
            exec_globals = {}
            exec(connection_code, exec_globals)
            connect_func = exec_globals.get("connect_to_source")
            
            if connect_func:
                state.connection = connect_func(state.source)
            
            return state
            
        except Exception as e:
            logger.error(f"Connection failed: {str(e)}")
            raise
    
    async def _extract_sample(self, state: AnalyzerState) -> AnalyzerState:
        """Extract representative data sample."""
        try:
            # Generate sampling code using LLM
            sampling_code = await self.llm_agent.analyze_data_source(
                source_type=state.source.type,
                sample_data=state.connection,
                context={
                    "source_config": state.source.dict(),
                    "purpose": "extract representative data sample"
                }
            )
            
            # Add sampling step
            state.analysis_steps.append(
                AnalysisStep(
                    name="extract_sample",
                    description="Extract a representative data sample",
                    code=sampling_code,
                    dependencies=["connect_to_source"]
                )
            )
            
            # Execute sampling code
            exec_globals = {"source_data": state.connection}
            exec(sampling_code, exec_globals)
            sample_func = exec_globals.get("extract_sample")
            
            if sample_func:
                state.sample_data = sample_func(state.connection)
            
            return state
            
        except Exception as e:
            logger.error(f"Sampling failed: {str(e)}")
            raise
    
    async def _analyze_data(self, state: AnalyzerState) -> AnalyzerState:
        """Perform comprehensive data analysis."""
        try:
            # Generate analysis code using LLM
            analysis_code = await self.llm_agent.analyze_data_source(
                source_type=state.source.type,
                sample_data=state.sample_data,
                context={
                    "source_config": state.source.dict(),
                    "purpose": "comprehensive data analysis"
                }
            )
            
            # Add analysis step
            state.analysis_steps.append(
                AnalysisStep(
                    name="analyze_data",
                    description="Perform comprehensive data analysis",
                    code=analysis_code,
                    dependencies=["extract_sample"]
                )
            )
            
            # Execute analysis code
            exec_globals = {"sample_data": state.sample_data}
            exec(analysis_code, exec_globals)
            analyze_func = exec_globals.get("analyze_data")
            
            if analyze_func:
                analysis_results = analyze_func(state.sample_data)
                state.schema = analysis_results.get("schema")
                state.data_quality_issues = analysis_results.get("quality_issues", [])
                state.inferred_data_types = analysis_results.get("data_types")
                state.statistics = analysis_results.get("statistics")
            
            return state
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    async def _validate_results(self, state: AnalyzerState) -> AnalyzerState:
        """Validate analysis results and generate summary."""
        try:
            # Generate validation code using LLM
            validation_code = await self.llm_agent.analyze_data_source(
                source_type=state.source.type,
                sample_data=state.sample_data,
                context={
                    "source_config": state.source.dict(),
                    "analysis_results": {
                        "schema": state.schema,
                        "quality_issues": state.data_quality_issues,
                        "data_types": state.inferred_data_types,
                        "statistics": state.statistics
                    },
                    "purpose": "validate analysis results"
                }
            )
            
            # Add validation step
            state.analysis_steps.append(
                AnalysisStep(
                    name="validate_results",
                    description="Validate analysis results and generate summary",
                    code=validation_code,
                    dependencies=["analyze_data"]
                )
            )
            
            # Execute validation code
            exec_globals = {
                "analysis_results": {
                    "schema": state.schema,
                    "quality_issues": state.data_quality_issues,
                    "data_types": state.inferred_data_types,
                    "statistics": state.statistics
                }
            }
            exec(validation_code, exec_globals)
            validate_func = exec_globals.get("validate_results")
            
            if validate_func:
                validation_results = validate_func(exec_globals["analysis_results"])
                state.validation = validation_results
            
            return state
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise 