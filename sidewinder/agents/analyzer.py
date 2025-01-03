"""
Data analyzer agent for analyzing source data.
"""

from typing import Dict, Any, Optional, List, Tuple
import logging
from pydantic import BaseModel, Field
import json
import os
import glob
from pathlib import Path
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


class DiscoveredSource(BaseModel):
    """Information about a discovered source file."""
    name: str
    path: str
    format: str
    schema: Dict[str, str]
    sample_data: Any = None
    
    model_config = {
        "arbitrary_types_allowed": True
    }


class InferredRelationship(BaseModel):
    """Information about an inferred relationship between sources."""
    from_source: str
    to_source: str
    relationship_type: str
    keys: Dict[str, str]
    confidence: float


class AnalyzerState(BaseAgentState):
    """State for the analyzer agent."""
    source_directory: str
    file_patterns: List[str]
    discovered_sources: Dict[str, DiscoveredSource] = Field(default_factory=dict)
    inferred_relationships: List[InferredRelationship] = Field(default_factory=list)
    analysis_results: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    execution_time: float = 0.0


class DataAnalyzer(BaseAgent[AnalyzerState]):
    """Agent for analyzing source data."""
    
    async def run(self, state: AnalyzerState) -> AnalyzerState:
        """
        Run automated source discovery and analysis:
        1. Discover source files in directory
        2. Analyze each source
        3. Infer relationships between sources
        4. Generate comprehensive analysis
        
        Args:
            state: Current analyzer state
            
        Returns:
            Updated analyzer state with analysis results
        """
        try:
            logger.info("Starting automated source discovery and analysis...")
            start_time = time.time()
            
            # Discover source files
            await self._discover_sources(state)
            
            # Analyze each source
            for source in state.discovered_sources.values():
                analysis = await self._analyze_source(source)
                state.analysis_results[source.name] = analysis
            
            # Infer relationships between sources
            await self._infer_relationships(state)
            
            state.completed = True
            state.messages.append("Source discovery and analysis completed successfully")
            
        except Exception as e:
            state.error = str(e)
            state.messages.append(f"Analysis failed: {str(e)}")
            
        finally:
            state.execution_time = time.time() - start_time
            
        return state

    async def _discover_sources(self, state: AnalyzerState) -> None:
        """Discover and analyze source files in directory."""
        try:
            for pattern in state.file_patterns:
                search_pattern = os.path.join(state.source_directory, "**", pattern)
                for file_path in glob.glob(search_pattern, recursive=True):
                    # Get file format from extension
                    file_format = Path(file_path).suffix.lstrip('.')
                    
                    # Generate source name from filename
                    source_name = Path(file_path).stem
                    
                    # Load sample data and infer schema based on format
                    if file_format == 'json':
                        # First read the JSON to infer schema
                        with open(file_path, 'r') as f:
                            sample_json = json.load(f)
                            if isinstance(sample_json, list) and len(sample_json) > 0:
                                sample_json = sample_json[0]
                        
                        # Create schema from sample
                        spark_schema = self._infer_json_schema(sample_json)
                        schema = self._convert_schema_to_dict(spark_schema)
                        
                        # Read with inferred schema
                        df = spark.read.schema(spark_schema).json(file_path)
                    else:
                        df = spark.read.format(file_format).load(file_path)
                        schema = {
                            field.name: str(field.dataType)
                            for field in df.schema.fields
                        }
                    
                    # Store discovered source
                    state.discovered_sources[source_name] = DiscoveredSource(
                        name=source_name,
                        path=file_path,
                        format=file_format,
                        schema=schema,
                        sample_data=df
                    )
                    
            logger.info(f"Discovered {len(state.discovered_sources)} source files")
            
        except Exception as e:
            logger.error(f"Error discovering sources: {str(e)}")
            raise

    def _infer_json_schema(self, json_obj: Dict) -> StructType:
        """Infer Spark schema from JSON object."""
        fields = []
        for key, value in json_obj.items():
            if isinstance(value, bool):
                field_type = BooleanType()
            elif isinstance(value, int):
                field_type = LongType()
            elif isinstance(value, float):
                field_type = DoubleType()
            elif isinstance(value, str):
                field_type = StringType()
            elif isinstance(value, dict):
                field_type = StructType(self._infer_json_schema(value))
            elif isinstance(value, list):
                if value and isinstance(value[0], dict):
                    field_type = ArrayType(self._infer_json_schema(value[0]))
                else:
                    field_type = ArrayType(StringType())
            else:
                field_type = StringType()
            
            fields.append(StructField(key, field_type, True))
        
        return StructType(fields)

    def _convert_schema_to_dict(self, schema: StructType) -> Dict[str, str]:
        """Convert Spark schema to dictionary format with flattened string representations."""
        result = {}
        for field in schema.fields:
            if isinstance(field.dataType, StructType):
                nested_fields = [f"{field.name}.{k}: {v}" for k, v in self._convert_schema_to_dict(field.dataType).items()]
                result[field.name] = f"STRUCT<{', '.join(nested_fields)}>"
            elif isinstance(field.dataType, ArrayType):
                if isinstance(field.dataType.elementType, StructType):
                    nested_fields = [f"{k}: {v}" for k, v in self._convert_schema_to_dict(field.dataType.elementType).items()]
                    result[field.name] = f"ARRAY<STRUCT<{', '.join(nested_fields)}>>"
                else:
                    result[field.name] = f"ARRAY<{str(field.dataType.elementType)}>"
            else:
                result[field.name] = str(field.dataType)
        return result

    async def _analyze_source(self, source: DiscoveredSource) -> Dict[str, Any]:
        """Analyze a single source file."""
        try:
            # Analyze data quality
            quality_metrics = self._analyze_data_quality(source.sample_data)
            
            # Analyze value distributions
            distributions = self._analyze_distributions(source.sample_data)
            
            # Identify potential key columns
            key_candidates = self._identify_key_candidates(source.sample_data)
            
            return {
                "schema": source.schema,
                "quality_metrics": quality_metrics,
                "distributions": distributions,
                "key_candidates": key_candidates
            }
            
        except Exception as e:
            logger.error(f"Error analyzing source {source.name}: {str(e)}")
            raise

    def _analyze_data_quality(self, df: DataFrame) -> Dict[str, Any]:
        """Analyze data quality metrics."""
        metrics = {}
        for col in df.columns:
            col_metrics = {
                "null_count": df.filter(F.col(col).isNull()).count(),
                "distinct_count": df.select(col).distinct().count(),
                "total_count": df.count()
            }
            
            # Add numeric metrics if applicable
            if isinstance(df.schema[col].dataType, (IntegerType, LongType, FloatType, DoubleType)):
                stats = df.select(
                    F.min(col).alias("min"),
                    F.max(col).alias("max"),
                    F.mean(col).alias("mean"),
                    F.stddev(col).alias("stddev")
                ).collect()[0]
                col_metrics.update({
                    "min": stats["min"],
                    "max": stats["max"],
                    "mean": stats["mean"],
                    "stddev": stats["stddev"]
                })
            
            metrics[col] = col_metrics
            
        return metrics

    def _analyze_distributions(self, df: DataFrame) -> Dict[str, Any]:
        """Analyze value distributions for each column."""
        distributions = {}
        for col in df.columns:
            # Get top N most frequent values
            value_counts = df.groupBy(col) \
                .count() \
                .orderBy(F.desc("count")) \
                .limit(10) \
                .collect()
            
            distributions[col] = [
                {"value": row[col], "count": row["count"]}
                for row in value_counts
            ]
            
        return distributions

    def _identify_key_candidates(self, df: DataFrame) -> List[Dict[str, Any]]:
        """Identify potential key columns based on uniqueness and nullability."""
        candidates = []
        total_rows = df.count()
        
        for col in df.columns:
            distinct_count = df.select(col).distinct().count()
            null_count = df.filter(F.col(col).isNull()).count()
            
            # Calculate metrics
            uniqueness_ratio = distinct_count / total_rows if total_rows > 0 else 0
            null_ratio = null_count / total_rows if total_rows > 0 else 0
            
            # Score the column as a key candidate
            if uniqueness_ratio > 0.9 and null_ratio < 0.01:
                candidates.append({
                    "column": col,
                    "uniqueness_ratio": uniqueness_ratio,
                    "null_ratio": null_ratio,
                    "confidence": (uniqueness_ratio * (1 - null_ratio))
                })
        
        return sorted(candidates, key=lambda x: x["confidence"], reverse=True)

    async def _infer_relationships(self, state: AnalyzerState) -> None:
        """Infer relationships between sources based on schema and data analysis."""
        try:
            # Get all pairs of sources
            source_pairs = [
                (s1, s2) for i, s1 in enumerate(state.discovered_sources.values())
                for s2 in list(state.discovered_sources.values())[i+1:]
            ]
            
            for source1, source2 in source_pairs:
                # Find potential joining keys
                relationships = self._find_relationships(
                    source1,
                    source2,
                    state.analysis_results[source1.name],
                    state.analysis_results[source2.name]
                )
                
                state.inferred_relationships.extend(relationships)
                
            logger.info(f"Inferred {len(state.inferred_relationships)} relationships")
            
        except Exception as e:
            logger.error(f"Error inferring relationships: {str(e)}")
            raise

    def _find_relationships(
        self,
        source1: DiscoveredSource,
        source2: DiscoveredSource,
        analysis1: Dict[str, Any],
        analysis2: Dict[str, Any]
    ) -> List[InferredRelationship]:
        """Find potential relationships between two sources."""
        relationships = []
        
        # Get key candidates from both sources
        keys1 = {c["column"]: c for c in analysis1["key_candidates"]}
        keys2 = {c["column"]: c for c in analysis2["key_candidates"]}
        
        # Find matching column names that are key candidates
        for col1, candidate1 in keys1.items():
            for col2, candidate2 in keys2.items():
                # Check if columns have matching names or patterns
                name_match = (
                    col1 == col2 or
                    col1.endswith(f"_{col2}") or
                    col2.endswith(f"_{col1}")
                )
                
                if name_match:
                    # Compare data distributions to determine relationship type
                    relationship_type = self._determine_relationship_type(
                        source1.sample_data,
                        source2.sample_data,
                        col1,
                        col2
                    )
                    
                    # Calculate confidence score
                    confidence = (
                        candidate1["confidence"] *
                        candidate2["confidence"] *
                        (1.0 if col1 == col2 else 0.8)
                    )
                    
                    relationships.append(
                        InferredRelationship(
                            from_source=source1.name,
                            to_source=source2.name,
                            relationship_type=relationship_type,
                            keys={"from": col1, "to": col2},
                            confidence=confidence
                        )
                    )
        
        return relationships

    def _determine_relationship_type(
        self,
        df1: DataFrame,
        df2: DataFrame,
        col1: str,
        col2: str
    ) -> str:
        """Determine the type of relationship between two columns."""
        try:
            # Count distinct values in both columns
            distinct1 = df1.select(col1).distinct().count()
            distinct2 = df2.select(col2).distinct().count()
            
            # Count occurrences per value
            counts1 = df1.groupBy(col1).count()
            counts2 = df2.groupBy(col2).count()
            
            # Get maximum occurrences
            max1 = counts1.agg(F.max("count")).collect()[0][0]
            max2 = counts2.agg(F.max("count")).collect()[0][0]
            
            # Determine relationship type based on cardinality
            if max1 == 1 and max2 > 1:
                return "one_to_many"
            elif max1 > 1 and max2 == 1:
                return "many_to_one"
            elif max1 == 1 and max2 == 1:
                return "one_to_one"
            else:
                return "many_to_many"
                
        except Exception as e:
            logger.error(f"Error determining relationship type: {str(e)}")
            return "unknown" 