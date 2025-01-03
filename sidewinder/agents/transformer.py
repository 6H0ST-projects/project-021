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
from sidewinder.agents.analyzer import DiscoveredSource, InferredRelationship

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


class TransformationDesigner(BaseAgent[TransformerState]):
    """Agent for designing data transformations."""
    
    async def run(self, state: TransformerState) -> TransformerState:
        """
        Design and execute the transformation pipeline for discovered sources:
        1. Load discovered source data into bronze layer
        2. Apply transformations based on inferred relationships for silver layer
        3. Generate gold layer output matching target schema
        
        Args:
            state: Current transformer state
            
        Returns:
            Updated transformer state with transformation results
        """
        try:
            logger.info("Starting transformation pipeline design...")
            start_time = time.time()
            
            # Bronze Layer: Load and standardize raw data
            await self._create_bronze_layer(state)
            
            # Silver Layer: Apply relationships and transformations
            await self._create_silver_layer(state)
            
            # Gold Layer: Final transformations to match target schema
            if state.target_schema:
                await self._create_gold_layer(state)
            
            # Validate results
            state.validation_results = await self._validate_results(state)
            
            state.completed = True
            state.messages.append("Transformation pipeline completed successfully")
            
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Transformation pipeline failed: {str(e)}")
            
        finally:
            state.execution_time = time.time() - start_time
            
        return state

    async def _create_bronze_layer(self, state: TransformerState) -> None:
        """Create bronze layer by loading and standardizing raw data."""
        try:
            for source in state.discovered_sources.values():
                # Load raw data
                df = await self._load_source_data(source)
                
                # Basic standardization
                df = self._standardize_column_names(df)
                df = self._add_metadata_columns(df, source)
                
                # Store in bronze layer
                state.bronze_data[source.name] = df
                
                # Add transformation step
                state.transformation_steps.append(
                    TransformationStep(
                        name=f"bronze_{source.name}",
                        description=f"Load and standardize {source.name} data",
                        code=self._generate_bronze_code(source)
                    )
                )
                
        except Exception as e:
            logger.error(f"Error creating bronze layer: {str(e)}")
            raise

    async def _create_silver_layer(self, state: TransformerState) -> None:
        """Create silver layer by applying relationships and business logic."""
        try:
            # Sort relationships by confidence
            sorted_relationships = sorted(
                state.inferred_relationships,
                key=lambda r: r.confidence,
                reverse=True
            )
            
            # Apply transformations based on relationships
            for relationship in sorted_relationships:
                result_name = await self._apply_relationship_transformation(
                    state,
                    relationship
                )
                
                # Store in silver layer
                if result_name:
                    state.silver_data[result_name] = state.intermediate_results[result_name]
            
        except Exception as e:
            logger.error(f"Error creating silver layer: {str(e)}")
            raise

    async def _create_gold_layer(self, state: TransformerState) -> None:
        """Create gold layer by transforming data to match target schema."""
        try:
            if not state.target_schema:
                return
                
            # Start with the most complete silver dataset
            base_df = self._get_most_complete_silver_df(state)
            
            # Apply schema transformations
            final_df = self._apply_schema_transformations(
                base_df,
                state.target_schema
            )
            
            # Store in gold layer
            state.gold_data["final"] = final_df
            
            # Add transformation step
            state.transformation_steps.append(
                TransformationStep(
                    name="gold_layer",
                    description="Transform data to match target schema",
                    code=self._generate_gold_code(state.target_schema)
                )
            )
            
        except Exception as e:
            logger.error(f"Error creating gold layer: {str(e)}")
            raise

    def _standardize_column_names(self, df: DataFrame) -> DataFrame:
        """Standardize column names to snake_case."""
        for col in df.columns:
            new_col = col.lower().replace(" ", "_")
            df = df.withColumnRenamed(col, new_col)
        return df

    def _add_metadata_columns(self, df: DataFrame, source: DiscoveredSource) -> DataFrame:
        """Add metadata columns to track data lineage."""
        return df.withColumns({
            "source_system": F.lit(source.name),
            "ingestion_timestamp": F.current_timestamp(),
            "batch_id": F.lit(datetime.now().strftime("%Y%m%d_%H%M%S"))
        })

    def _generate_bronze_code(self, source: DiscoveredSource) -> str:
        """Generate code for bronze layer transformation."""
        return f"""
        # Load raw data from {source.name}
        df = spark.read.format("{source.format}").load("{source.path}")
        
        # Standardize column names
        for col in df.columns:
            new_col = col.lower().replace(" ", "_")
            df = df.withColumnRenamed(col, new_col)
            
        # Add metadata columns
        df = df.withColumns({{
            "source_system": F.lit("{source.name}"),
            "ingestion_timestamp": F.current_timestamp(),
            "batch_id": F.lit(datetime.now().strftime("%Y%m%d_%H%M%S"))
        }})
        """

    def _get_most_complete_silver_df(self, state: TransformerState) -> DataFrame:
        """Get the most complete silver layer DataFrame."""
        if not state.silver_data:
            raise ValueError("No silver layer data available")
            
        # Find the DataFrame with the most columns
        most_complete_df = None
        max_columns = 0
        
        for df in state.silver_data.values():
            num_columns = len(df.columns)
            if num_columns > max_columns:
                max_columns = num_columns
                most_complete_df = df
                
        return most_complete_df

    def _apply_schema_transformations(
        self,
        df: DataFrame,
        target_schema: Dict[str, Any]
    ) -> DataFrame:
        """Apply transformations to match target schema."""
        for col_name, col_type in target_schema.items():
            if col_name not in df.columns:
                # Handle missing columns based on type
                if col_type == "string":
                    df = df.withColumn(col_name, F.lit(None).cast("string"))
                elif col_type in ["int", "long"]:
                    df = df.withColumn(col_name, F.lit(None).cast(col_type))
                elif col_type == "double":
                    df = df.withColumn(col_name, F.lit(None).cast("double"))
                elif col_type == "boolean":
                    df = df.withColumn(col_name, F.lit(None).cast("boolean"))
                elif col_type == "timestamp":
                    df = df.withColumn(col_name, F.lit(None).cast("timestamp"))
            else:
                # Cast existing columns to target type
                df = df.withColumn(col_name, F.col(col_name).cast(col_type))
                
        return df

    def _generate_gold_code(self, target_schema: Dict[str, Any]) -> str:
        """Generate code for gold layer transformation."""
        code_lines = ["# Transform data to match target schema"]
        
        for col_name, col_type in target_schema.items():
            code_lines.append(f"""
            if "{col_name}" not in df.columns:
                df = df.withColumn("{col_name}", F.lit(None).cast("{col_type}"))
            else:
                df = df.withColumn("{col_name}", F.col("{col_name}").cast("{col_type}"))
            """)
            
        return "\n".join(code_lines)

    async def _load_source_data(self, source: DiscoveredSource) -> DataFrame:
        """Load data from a discovered source."""
        try:
            return spark.read.format(source.format).load(source.path)
        except Exception as e:
            logger.error(f"Error loading source {source.name}: {str(e)}")
            raise

    async def _apply_relationship_transformation(
        self,
        state: TransformerState,
        relationship: InferredRelationship
    ) -> None:
        """Apply transformations based on inferred relationship."""
        try:
            from_df = state.source_dataframes[relationship.from_source]
            to_df = state.source_dataframes[relationship.to_source]
            
            # Generate intermediate transformation name
            transform_name = f"{relationship.from_source}_{relationship.to_source}_join"
            
            # Apply join transformation based on relationship type
            if relationship.relationship_type == "one_to_many":
                result = from_df.join(
                    to_df,
                    from_df[relationship.keys["from"]] == to_df[relationship.keys["to"]],
                    "left"
                )
            elif relationship.relationship_type == "many_to_one":
                result = from_df.join(
                    to_df,
                    from_df[relationship.keys["from"]] == to_df[relationship.keys["to"]],
                    "right"
                )
            elif relationship.relationship_type == "one_to_one":
                result = from_df.join(
                    to_df,
                    from_df[relationship.keys["from"]] == to_df[relationship.keys["to"]],
                    "inner"
                )
            else:  # many_to_many or unknown
                result = from_df.join(
                    to_df,
                    from_df[relationship.keys["from"]] == to_df[relationship.keys["to"]],
                    "inner"
                )
            
            # Store intermediate result
            state.intermediate_results[transform_name] = result
            
            # Add transformation step
            state.transformation_steps.append(
                TransformationStep(
                    name=transform_name,
                    description=f"Join {relationship.from_source} with {relationship.to_source} ({relationship.relationship_type})",
                    code=f"""
                    # Join {relationship.from_source} with {relationship.to_source}
                    {transform_name} = {relationship.from_source}_df.join(
                        {relationship.to_source}_df,
                        {relationship.from_source}_df['{relationship.keys["from"]}'] == {relationship.to_source}_df['{relationship.keys["to"]}'],
                        "{relationship.relationship_type}"
                    )
                    """
                )
            )
            
        except Exception as e:
            logger.error(f"Error applying relationship transformation: {str(e)}")
            raise

    async def _validate_results(self, state: TransformerState) -> Dict[str, Any]:
        """Validate transformation results."""
        try:
            validation_results = {}
            
            # Validate bronze layer
            if not state.bronze_data:
                raise ValueError("No data found in bronze layer")
                
            # Validate silver layer
            if not state.silver_data:
                raise ValueError("No data found in silver layer")
                
            # Validate gold layer
            if state.target_schema and not state.gold_data.get("final"):
                raise ValueError("No data found in gold layer")
                
            validation_results["status"] = "success"
            validation_results["message"] = "All validations passed"
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating results: {str(e)}")
            raise 