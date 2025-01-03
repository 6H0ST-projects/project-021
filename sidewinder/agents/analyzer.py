"""
Data analyzer agent for analyzing source data.
"""

from typing import Dict, Any, Optional
import logging
from pydantic import BaseModel, Field
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import *
import numpy as np
from datetime import datetime
import time

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from sidewinder.agents.base import BaseAgent, BaseAgentState
from sidewinder.core.config import Source, SourceType
from sidewinder.core.llm import DataEngineeringAgent, CodeGenerationRequest
from sidewinder.core.analyzer import analyze_data
from sidewinder.core.state import AnalyzerState

# Initialize Spark session for local mode
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("SidewinderAnalyzer") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Setup logging
logger = logging.getLogger(__name__)


class AnalyzerState(BaseAgentState):
    """State for the analyzer agent."""
    source: Source
    schema: Optional[Dict[str, Any]] = None
    data_quality: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, Any]] = None
    patterns: Optional[Dict[str, Any]] = None
    anomalies: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class DataAnalyzer(BaseAgent[AnalyzerState]):
    """Agent for analyzing source data."""
    
    def __init__(self, code: Optional[str] = None):
        """
        Initialize the analyzer agent.
        
        Args:
            code: LLM-generated code for custom analysis
        """
        super().__init__()
        self.code = code
        self.llm_agent = DataEngineeringAgent()
        
    async def run(self, state: AnalyzerState) -> AnalyzerState:
        """
        Run data analysis:
        1. Analyze schema
        2. Check data quality
        3. Generate statistics
        4. Detect patterns
        5. Detect anomalies
        
        Args:
            state: Current analyzer state
            
        Returns:
            Updated analyzer state with analysis results
        """
        try:
            logger.info("Starting data analysis...")
            start_time = time.time()
            
            # Initialize state fields
            state.schema = {"fields": [], "metadata": {}}
            state.data_quality = {"metrics": {}, "issues": []}
            state.statistics = {"summary": {}, "distributions": {}}
            state.patterns = {"discovered": [], "analysis": {}}
            state.anomalies = {"detected": [], "details": {}}
            
            # Convert input data to Spark DataFrame if needed
            if isinstance(state.source.sample_data, str):
                sample_data = spark.read.json(state.source.sample_data)
            else:
                sample_data = spark.createDataFrame(state.source.sample_data)
            
            # Execute analysis code
            exec(self.code, globals(), locals())
            
            # Update state with results
            if "analyze_data" in locals():
                results = locals()["analyze_data"](sample_data)
                if isinstance(results, dict):
                    if "schema" in results:
                        state.schema.update(results["schema"])
                    if "data_quality" in results:
                        state.data_quality.update(results["data_quality"])
                    if "statistics" in results:
                        state.statistics.update(results["statistics"])
                    if "patterns" in results:
                        state.patterns.update(results["patterns"])
                    if "anomalies" in results:
                        state.anomalies.update(results["anomalies"])
            
            state.completed = True
            state.messages.append("Analysis completed successfully")
            
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Analysis failed: {str(e)}")
            
        finally:
            state.execution_time = time.time() - start_time
            
        return state

    async def _analyze_schema(self, state: AnalyzerState) -> Dict[str, Any]:
        """Analyze the schema of the source data."""
        try:
            if self.code:
                # Use LLM-generated code for schema analysis
                request = CodeGenerationRequest(
                    task="Analyze data schema using Spark SQL",
                    context={
                        "source_type": state.source.type,
                        "source_location": state.source.location,
                        "source_config": state.source.config
                    },
                    requirements=[
                        "Use Spark SQL to analyze schema",
                        "Detect data types",
                        "Identify primary keys",
                        "Find relationships",
                        "Infer schema structure"
                    ],
                    constraints=[
                        "Handle large datasets efficiently",
                        "Use Spark SQL operations",
                        "Preserve data types"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.schema = locals().get("schema", {})
            else:
                # Use default schema analysis with Spark SQL
                data = spark.createDataFrame(state.source.sample_data)
                state.schema = {
                    "fields": [
                        {
                            "name": field.name,
                            "type": str(field.dataType),
                            "nullable": field.nullable
                        }
                        for field in data.schema.fields
                    ],
                    "metadata": {
                        "total_fields": len(data.schema.fields),
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.schema = state.schema or {}
            state.data_quality = state.data_quality or {}
            state.statistics = state.statistics or {}
            state.patterns = state.patterns or {}
            state.anomalies = state.anomalies or {}
            state.execution_time = 0.0
            
            state.messages.append("Schema analysis completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Schema analysis failed: {str(e)}")
            return {"state": state}

    async def _check_data_quality(self, state: AnalyzerState) -> Dict[str, Any]:
        """Check data quality metrics."""
        try:
            if self.code:
                # Use LLM-generated code for quality checks
                request = CodeGenerationRequest(
                    task="Check data quality",
                    context={
                        "source_type": state.source.type,
                        "schema": state.schema
                    },
                    requirements=[
                        "Check completeness",
                        "Validate formats",
                        "Find duplicates",
                        "Measure consistency"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Comprehensive checks"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.data_quality = locals().get("data_quality", {})
            else:
                # Use default quality checks
                data = analyze_data(state.source)
                state.data_quality = data.get("quality", {})
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.schema = state.schema or {}
            state.data_quality = state.data_quality or {}
            state.statistics = state.statistics or {}
            state.patterns = state.patterns or {}
            state.anomalies = state.anomalies or {}
            state.execution_time = 0.0
            
            state.messages.append("Data quality check completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Data quality check failed: {str(e)}")
            return {"state": state}

    async def _generate_statistics(self, state: AnalyzerState) -> Dict[str, Any]:
        """Generate statistical summaries."""
        try:
            if self.code:
                # Use LLM-generated code for statistics
                request = CodeGenerationRequest(
                    task="Generate statistics",
                    context={
                        "source_type": state.source.type,
                        "schema": state.schema,
                        "quality": state.data_quality
                    },
                    requirements=[
                        "Calculate distributions",
                        "Find correlations",
                        "Identify outliers",
                        "Generate summaries"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Statistical accuracy"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.statistics = locals().get("statistics", {})
            else:
                # Use default statistics generation
                data = analyze_data(state.source)
                state.statistics = data.get("statistics", {})
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.schema = state.schema or {}
            state.data_quality = state.data_quality or {}
            state.statistics = state.statistics or {}
            state.patterns = state.patterns or {}
            state.anomalies = state.anomalies or {}
            state.execution_time = 0.0
            
            state.messages.append("Statistics generation completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Statistics generation failed: {str(e)}")
            return {"state": state}

    async def _detect_patterns(self, state: AnalyzerState) -> Dict[str, Any]:
        """Detect patterns in the data."""
        try:
            if self.code:
                # Use LLM-generated code for pattern detection
                request = CodeGenerationRequest(
                    task="Detect patterns",
                    context={
                        "source_type": state.source.type,
                        "schema": state.schema,
                        "statistics": state.statistics
                    },
                    requirements=[
                        "Find trends",
                        "Detect seasonality",
                        "Identify cycles",
                        "Discover relationships"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Pattern significance"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.patterns = locals().get("patterns", {})
            else:
                # Use default pattern detection
                data = analyze_data(state.source)
                state.patterns = data.get("patterns", {})
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.schema = state.schema or {}
            state.data_quality = state.data_quality or {}
            state.statistics = state.statistics or {}
            state.patterns = state.patterns or {}
            state.anomalies = state.anomalies or {}
            state.execution_time = 0.0
            
            state.messages.append("Pattern detection completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Pattern detection failed: {str(e)}")
            return {"state": state}

    async def _detect_anomalies(self, state: AnalyzerState) -> Dict[str, Any]:
        """Detect anomalies in the data."""
        try:
            if self.code:
                # Use LLM-generated code for anomaly detection
                request = CodeGenerationRequest(
                    task="Detect anomalies",
                    context={
                        "source_type": state.source.type,
                        "schema": state.schema,
                        "statistics": state.statistics,
                        "patterns": state.patterns
                    },
                    requirements=[
                        "Find outliers",
                        "Detect inconsistencies",
                        "Identify errors",
                        "Flag anomalies"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Anomaly significance"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.anomalies = locals().get("anomalies", {})
            else:
                # Use default anomaly detection
                data = analyze_data(state.source)
                state.anomalies = data.get("anomalies", {})
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.schema = state.schema or {}
            state.data_quality = state.data_quality or {}
            state.statistics = state.statistics or {}
            state.patterns = state.patterns or {}
            state.anomalies = state.anomalies or {}
            state.execution_time = 0.0
            
            state.messages.append("Anomaly detection completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Anomaly detection failed: {str(e)}")
            return {"state": state} 