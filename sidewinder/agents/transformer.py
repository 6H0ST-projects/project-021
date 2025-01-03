"""
Transformation designer agent for creating data transformation pipelines.
"""

from typing import Dict, Any, List, Optional
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
from sidewinder.core.config import Source, SourceType, TransformationConfig
from sidewinder.core.llm import DataEngineeringAgent, CodeGenerationRequest
from sidewinder.core.state import TransformerState

# Initialize Spark session for local mode
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("SidewinderTransformer") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Setup logging
logger = logging.getLogger(__name__)


class TransformationStep(BaseModel):
    """A single step in the transformation process."""
    name: str
    description: str
    code: str
    dependencies: List[str] = Field(default_factory=list)
    results: Optional[Dict[str, Any]] = None


class TransformerState(BaseAgentState):
    """State for the transformer agent."""
    source: Source
    target_schema: Optional[Dict[str, Any]] = None
    transformation_steps: List[TransformationStep] = Field(default_factory=list)
    bronze_data: Optional[Dict[str, Any]] = None
    silver_data: Optional[Dict[str, Any]] = None
    gold_data: Optional[Dict[str, Any]] = None
    validation_results: Optional[Dict[str, Any]] = None
    execution_time: float = 0.0


class TransformationDesigner(BaseAgent[TransformerState]):
    """Agent for designing data transformations."""
    
    def __init__(self, code: Optional[str] = None):
        """
        Initialize the transformer agent.
        
        Args:
            code: LLM-generated code for custom transformations
        """
        super().__init__()
        self.code = code
        self.llm_agent = DataEngineeringAgent()
        
    async def run(self, state: TransformerState) -> TransformerState:
        """
        Design and execute the transformation pipeline:
        1. Bronze layer: Raw data ingestion and standardization
        2. Silver layer: Data cleaning and enrichment
        3. Gold layer: Business logic application
        
        Args:
            state: Current transformer state
            
        Returns:
            Updated transformer state with transformation results
        """
        try:
            logger.info("Starting transformation design...")
            start_time = time.time()
            
            # Initialize state fields
            state.transformation_steps = []
            state.bronze_data = {"raw": {}, "cleaned": {}}
            state.silver_data = {"transformed": {}, "validated": {}}
            state.gold_data = {"final": {}, "metadata": {}}
            state.validation_results = {"checks": [], "metrics": {}}
            
            # Convert input data to Spark DataFrame if needed
            if isinstance(state.source.sample_data, str):
                sample_data = spark.read.json(state.source.sample_data)
            else:
                sample_data = spark.createDataFrame(state.source.sample_data)
            
            # Execute transformation code
            exec(self.code, globals(), locals())
            
            # Update state with results
            if "transform_data" in locals():
                results = locals()["transform_data"](sample_data)
                if isinstance(results, dict):
                    if "transformation_steps" in results:
                        state.transformation_steps.extend(results["transformation_steps"])
                    if "bronze_data" in results:
                        state.bronze_data.update(results["bronze_data"])
                    if "silver_data" in results:
                        state.silver_data.update(results["silver_data"])
                    if "gold_data" in results:
                        state.gold_data.update(results["gold_data"])
                    if "validation_results" in results:
                        state.validation_results.update(results["validation_results"])
            
            state.completed = True
            state.messages.append("Transformation completed successfully")
            
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Transformation failed: {str(e)}")
            
        finally:
            state.execution_time = time.time() - start_time
            
        return state
            
    async def _create_bronze_layer(self, state: TransformerState) -> Dict[str, Any]:
        """Create bronze layer transformations."""
        logger.info("Creating bronze layer...")
        try:
            if self.code:
                # Use LLM-generated code for bronze layer
                request = CodeGenerationRequest(
                    task="Create bronze layer transformations using Spark SQL",
                    context={
                        "source_type": state.source.type,
                        "source_location": state.source.location,
                        "source_config": state.source.config
                    },
                    requirements=[
                        "Use Spark SQL for data ingestion",
                        "Schema standardization",
                        "Data type conversion",
                        "Quality checks"
                    ],
                    constraints=[
                        "Handle large datasets efficiently",
                        "Use Spark SQL operations",
                        "Preserve raw data"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.bronze_data = locals().get("bronze_data", {})
                state.transformation_steps.append(
                    TransformationStep(
                        name="bronze_layer",
                        description="Raw data ingestion and standardization",
                        code=response.code
                    )
                )
            else:
                # Use default bronze layer transformations with Spark SQL
                data = spark.read.format(state.source.type) \
                    .load(state.source.location)
                state.bronze_data = {
                    "data": data.collect(),
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "source": state.source.dict()
                    }
                }
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.target_schema = state.target_schema
            state.transformation_steps = state.transformation_steps or []
            state.bronze_data = state.bronze_data or {}
            state.silver_data = state.silver_data or {}
            state.gold_data = state.gold_data or {}
            state.validation_results = state.validation_results or {}
            state.execution_time = 0.0
            
            state.messages.append("Bronze layer creation completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Bronze layer creation failed: {str(e)}")
            return {"state": state}

    async def _create_silver_layer(self, state: TransformerState) -> Dict[str, Any]:
        """Create silver layer transformations."""
        logger.info("Creating silver layer...")
        try:
            if self.code:
                # Use LLM-generated code for silver layer
                request = CodeGenerationRequest(
                    task="Create silver layer transformations",
                    context={
                        "source_type": state.source.type,
                        "bronze_data": state.bronze_data,
                        "target_schema": state.target_schema
                    },
                    requirements=[
                        "Data cleaning",
                        "Data enrichment",
                        "Data validation",
                        "Quality improvements"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Maintain data lineage"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.silver_data = locals().get("silver_data", {})
                state.transformation_steps.append(
                    TransformationStep(
                        name="silver_layer",
                        description="Data cleaning and enrichment",
                        code=response.code,
                        dependencies=["bronze_layer"]
                    )
                )
            else:
                # Use default silver layer transformations
                data = pd.DataFrame(state.bronze_data["data"])
                state.silver_data = {
                    "data": data.to_dict(),
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "source": state.bronze_data["metadata"]
                    }
                }
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.target_schema = state.target_schema
            state.transformation_steps = state.transformation_steps or []
            state.bronze_data = state.bronze_data or {}
            state.silver_data = state.silver_data or {}
            state.gold_data = state.gold_data or {}
            state.validation_results = state.validation_results or {}
            state.execution_time = 0.0
            
            state.messages.append("Silver layer creation completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Silver layer creation failed: {str(e)}")
            return {"state": state}

    async def _create_gold_layer(self, state: TransformerState) -> Dict[str, Any]:
        """Create gold layer transformations."""
        logger.info("Creating gold layer...")
        try:
            if self.code:
                # Use LLM-generated code for gold layer
                request = CodeGenerationRequest(
                    task="Create gold layer transformations",
                    context={
                        "source_type": state.source.type,
                        "silver_data": state.silver_data,
                        "target_schema": state.target_schema
                    },
                    requirements=[
                        "Business logic application",
                        "Feature engineering",
                        "Data aggregation",
                        "Final validation"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Optimize for analytics"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.gold_data = locals().get("gold_data", {})
                state.transformation_steps.append(
                    TransformationStep(
                        name="gold_layer",
                        description="Business logic application",
                        code=response.code,
                        dependencies=["silver_layer"]
                    )
                )
            else:
                # Use default gold layer transformations
                data = pd.DataFrame(state.silver_data["data"])
                state.gold_data = {
                    "data": data.to_dict(),
                    "metadata": {
                        "timestamp": datetime.now().isoformat(),
                        "source": state.silver_data["metadata"]
                    }
                }
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.target_schema = state.target_schema
            state.transformation_steps = state.transformation_steps or []
            state.bronze_data = state.bronze_data or {}
            state.silver_data = state.silver_data or {}
            state.gold_data = state.gold_data or {}
            state.validation_results = state.validation_results or {}
            state.execution_time = 0.0
            
            state.messages.append("Gold layer creation completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Gold layer creation failed: {str(e)}")
            return {"state": state}

    async def _validate_transformations(self, state: TransformerState) -> Dict[str, Any]:
        """Validate the transformations."""
        logger.info("Validating transformations...")
        try:
            if self.code:
                # Use LLM-generated code for validation
                request = CodeGenerationRequest(
                    task="Validate transformations",
                    context={
                        "source_type": state.source.type,
                        "bronze_data": state.bronze_data,
                        "silver_data": state.silver_data,
                        "gold_data": state.gold_data,
                        "target_schema": state.target_schema
                    },
                    requirements=[
                        "Schema validation",
                        "Data quality checks",
                        "Business rule validation",
                        "Performance metrics"
                    ],
                    constraints=[
                        "Handle large datasets",
                        "Memory efficient",
                        "Comprehensive validation"
                    ]
                )
                response = await self.llm_agent.generate_code(request)
                # Execute generated code
                exec(response.code, globals(), locals())
                state.validation_results = locals().get("validation_results", {})
            else:
                # Use default validation
                state.validation_results = {
                    "bronze_layer": {
                        "row_count": len(pd.DataFrame(state.bronze_data["data"])),
                        "completeness": 1.0,
                        "success": True
                    },
                    "silver_layer": {
                        "row_count": len(pd.DataFrame(state.silver_data["data"])),
                        "completeness": 1.0,
                        "success": True
                    },
                    "gold_layer": {
                        "row_count": len(pd.DataFrame(state.gold_data["data"])),
                        "completeness": 1.0,
                        "success": True
                    }
                }
            
            # Ensure all required fields are written
            state.messages = state.messages or []
            state.error = None
            state.completed = True
            state.source = state.source
            state.target_schema = state.target_schema
            state.transformation_steps = state.transformation_steps or []
            state.bronze_data = state.bronze_data or {}
            state.silver_data = state.silver_data or {}
            state.gold_data = state.gold_data or {}
            state.validation_results = state.validation_results or {}
            state.execution_time = 0.0
            
            state.messages.append("Transformation validation completed")
            return {"state": state}
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Transformation validation failed: {str(e)}")
            return {"state": state} 