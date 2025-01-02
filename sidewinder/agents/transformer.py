"""
Transformation designer agent for creating data transformation pipelines.
"""

from typing import Dict, Any, List, Optional
import logging
from pydantic import BaseModel, Field
import json
import pandas as pd
import numpy as np
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

from sidewinder.agents.base import BaseAgent, BaseAgentState
from sidewinder.core.config import Source, SourceType, TransformationConfig
from sidewinder.core.llm import DataEngineeringAgent, CodeGenerationRequest

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


class TransformationDesigner(BaseAgent[TransformerState]):
    """Agent responsible for designing and executing data transformations."""
    
    def __init__(self):
        super().__init__()
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
            # Create transformation workflow
            workflow = StateGraph(TransformerState)
            
            # Add nodes for each transformation layer
            workflow.add_node("bronze", self._create_bronze_layer)
            workflow.add_node("silver", self._create_silver_layer)
            workflow.add_node("gold", self._create_gold_layer)
            workflow.add_node("validate", self._validate_transformations)
            
            # Add edges
            workflow.add_edge("bronze", "silver")
            workflow.add_edge("silver", "gold")
            workflow.add_edge("gold", "validate")
            
            # Set entry point
            workflow.set_entry_point("bronze")
            
            # Set end point conditions
            workflow.add_edge("validate", END)
            
            # Execute workflow
            state = await workflow.arun(state)
            
            state.completed = True
            return state
            
        except Exception as e:
            logger.error(f"Transformation failed: {str(e)}")
            state.error = str(e)
            return state
    
    async def _create_bronze_layer(self, state: TransformerState) -> TransformerState:
        """Create bronze layer transformations for raw data ingestion."""
        try:
            # Generate bronze layer code using LLM
            bronze_code = await self.llm_agent.generate_transformation(
                source_type=state.source.type,
                source_data=state.source.sample_data,
                target_schema=state.target_schema,
                context={
                    "source_config": state.source.dict(),
                    "layer": "bronze",
                    "purpose": "standardize and validate raw data"
                }
            )
            
            # Add bronze step
            state.transformation_steps.append(
                TransformationStep(
                    name="create_bronze_layer",
                    description="Standardize and validate raw data",
                    code=bronze_code
                )
            )
            
            # Execute bronze code
            exec_globals = {"source_data": state.source.sample_data}
            exec(bronze_code, exec_globals)
            bronze_func = exec_globals.get("create_bronze_layer")
            
            if bronze_func:
                state.bronze_data = bronze_func(state.source.sample_data)
            
            return state
            
        except Exception as e:
            logger.error(f"Bronze layer creation failed: {str(e)}")
            raise
    
    async def _create_silver_layer(self, state: TransformerState) -> TransformerState:
        """Create silver layer transformations for data cleaning and enrichment."""
        try:
            # Generate silver layer code using LLM
            silver_code = await self.llm_agent.generate_transformation(
                source_type=state.source.type,
                source_data=state.bronze_data,
                target_schema=state.target_schema,
                context={
                    "source_config": state.source.dict(),
                    "layer": "silver",
                    "purpose": "clean and enrich data"
                }
            )
            
            # Add silver step
            state.transformation_steps.append(
                TransformationStep(
                    name="create_silver_layer",
                    description="Clean and enrich data",
                    code=silver_code,
                    dependencies=["create_bronze_layer"]
                )
            )
            
            # Execute silver code
            exec_globals = {"bronze_data": state.bronze_data}
            exec(silver_code, exec_globals)
            silver_func = exec_globals.get("create_silver_layer")
            
            if silver_func:
                state.silver_data = silver_func(state.bronze_data)
            
            return state
            
        except Exception as e:
            logger.error(f"Silver layer creation failed: {str(e)}")
            raise
    
    async def _create_gold_layer(self, state: TransformerState) -> TransformerState:
        """Create gold layer transformations for business logic application."""
        try:
            # Generate gold layer code using LLM
            gold_code = await self.llm_agent.generate_transformation(
                source_type=state.source.type,
                source_data=state.silver_data,
                target_schema=state.target_schema,
                context={
                    "source_config": state.source.dict(),
                    "layer": "gold",
                    "purpose": "apply business logic and create final output"
                }
            )
            
            # Add gold step
            state.transformation_steps.append(
                TransformationStep(
                    name="create_gold_layer",
                    description="Apply business logic and create final output",
                    code=gold_code,
                    dependencies=["create_silver_layer"]
                )
            )
            
            # Execute gold code
            exec_globals = {"silver_data": state.silver_data}
            exec(gold_code, exec_globals)
            gold_func = exec_globals.get("create_gold_layer")
            
            if gold_func:
                state.gold_data = gold_func(state.silver_data)
            
            return state
            
        except Exception as e:
            logger.error(f"Gold layer creation failed: {str(e)}")
            raise
    
    async def _validate_transformations(self, state: TransformerState) -> TransformerState:
        """Validate transformation results against target schema."""
        try:
            # Generate validation code using LLM
            validation_code = await self.llm_agent.generate_transformation(
                source_type=state.source.type,
                source_data=state.gold_data,
                target_schema=state.target_schema,
                context={
                    "source_config": state.source.dict(),
                    "layer": "validation",
                    "purpose": "validate final output against target schema"
                }
            )
            
            # Add validation step
            state.transformation_steps.append(
                TransformationStep(
                    name="validate_transformations",
                    description="Validate final output against target schema",
                    code=validation_code,
                    dependencies=["create_gold_layer"]
                )
            )
            
            # Execute validation code
            exec_globals = {
                "gold_data": state.gold_data,
                "target_schema": state.target_schema
            }
            exec(validation_code, exec_globals)
            validate_func = exec_globals.get("validate_transformations")
            
            if validate_func:
                state.validation_results = validate_func(
                    state.gold_data,
                    state.target_schema
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            raise 