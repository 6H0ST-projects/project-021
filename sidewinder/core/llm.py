"""
Core LLM integration layer for Sidewinder.
"""

from typing import Dict, Any, Optional, List, Tuple
from pydantic import BaseModel, Field
import json
import logging
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
import instructor
from openai import OpenAI
from pyspark.sql import SparkSession
import numpy as np
import traceback
from sidewinder.core.analyzer import analyze_data
from sidewinder.core.config import TargetSchema

# Setup logging
logger = logging.getLogger(__name__)

# Initialize instructor-enabled client
client = instructor.patch(OpenAI())

class CodeGenerationRequest(BaseModel):
    """Request for code generation."""
    task: str
    context: Dict[str, Any]
    requirements: List[str]
    constraints: Optional[List[str]] = None
    examples: Optional[List[Dict[str, str]]] = None

class CodeGenerationResponse(BaseModel):
    """Response from code generation."""
    code: str
    explanation: str
    imports: List[str]
    tests: Optional[List[str]] = None

class CodeExecutionError(BaseModel):
    """Details about a code execution error."""
    error_type: str
    error_message: str
    traceback: Optional[str] = None
    context: Dict[str, Any]

    @classmethod
    def from_exception(cls, e: Exception, context: Dict[str, Any]) -> "CodeExecutionError":
        """Create error details from an exception."""
        import traceback as tb
        return cls(
            error_type=type(e).__name__,
            error_message=str(e),
            traceback=tb.format_exc(),
            context=context
        )

class LLMCodeGenerator:
    """LLM code generator for data engineering tasks."""
    
    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 0.2,
        max_tokens: int = 4000,
        max_retries: int = 3
    ):
        """Initialize the code generator."""
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
    async def generate_code(
        self,
        task: str,
        context: Dict[str, Any],
        requirements: List[str],
        constraints: Optional[List[str]] = None
    ) -> str:
        """
        Generate code using the LLM.
        
        Args:
            task: The task description
            context: The context for code generation
            requirements: List of requirements
            constraints: Optional list of constraints
            
        Returns:
            The generated code
        """
        try:
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a data engineering code generator "
                          "that follows best practices. The code should be production-ready and include "
                          "proper error handling, logging, and type hints.\n\n"
                          "IMPORTANT: Always wrap your code in ```python and ``` markers.\n"
                          "IMPORTANT: Make sure to implement the exact function name as specified in the context.\n"
                          "IMPORTANT: Your code must include:\n"
                          "1. The required function with the exact name\n"
                          "2. Use of Spark SQL for data manipulation with these specific features:\n"
                          "   - Import pyspark.sql.functions as F\n"
                          "   - Use DataFrame.explain() for query plans\n"
                          "   - Use broadcast hints for small tables\n"
                          "   - Use window functions efficiently\n"
                          "   - Cache intermediate results with df.cache()\n"
                          "   - Monitor performance with spark.sparkContext.statusTracker()\n"
                          "3. Proper function signature with type hints\n"
                          "4. Error handling with try/except\n"
                          "5. Logging for important steps\n"
                          "6. Return values in the expected format"),
                ("human", "Task: {task}\n"
                         "Requirements:\n{requirements_str}\n"
                         "Constraints:\n{constraints_str}\n"
                         "Context:\n{context_str}\n\n"
                         "Generate the code following these guidelines:\n"
                         "1. Include necessary imports (especially pyspark.sql.functions)\n"
                         "2. Add comprehensive docstrings\n"
                         "3. Implement proper error handling\n"
                         "4. Add type hints\n"
                         "5. Include logging\n"
                         "6. Use Spark SQL for all data transformations with:\n"
                         "   - DataFrame operations\n"
                         "   - Window functions\n"
                         "   - Query plan analysis\n"
                         "   - Performance monitoring\n"
                         "7. Follow PEP 8 style guidelines\n\n"
                         "Remember to wrap your code in ```python and ``` markers.\n"
                         "Remember to implement the exact function name as specified in the context.")
            ])
            
            # Format requirements and constraints
            requirements_str = "\n".join(f"- {r}" for r in requirements)
            constraints_str = "\n".join(f"- {c}" for c in (constraints or []))
            context_str = json.dumps(context, indent=2)
            
            # Format prompt
            formatted_prompt = prompt.format(
                task=task,
                requirements_str=requirements_str,
                constraints_str=constraints_str,
                context_str=context_str
            )
            
            # Generate code
            response = await self.llm.ainvoke(formatted_prompt)
            
            # Extract code from response
            code = response.content
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            
            return code
            
        except Exception as e:
            logger.error(f"Code generation failed: {str(e)}")
            raise

class TransformationCode(BaseModel):
    """Container for transformation code across layers."""
    bronze_layer: str
    silver_layer: str
    gold_layer: str
    analyzer_findings: Dict[str, Any]


class DataEngineeringAgent:
    """LLM-powered data engineering agent."""
    
    def __init__(self):
        self.llm = LLMCodeGenerator()
    
    async def generate_analyzer_code(
        self,
        source_type: str,
        source_data: str
    ) -> Dict[str, Any]:
        """Generate code to analyze the source data."""
        prompt = self._create_analyzer_prompt(source_type, source_data)
        response = await self.llm.generate_code(
            task="Generate data analyzer code",
            context={
                "source_type": source_type,
                "source_data": source_data
            },
            requirements=[
                "Analyze data types and patterns",
                "Identify data quality issues",
                "Detect relationships between data sources",
                "Calculate basic statistics",
                "Provide transformation recommendations"
            ]
        )
        return self._parse_analyzer_response(response)
    
    async def generate_transformer_code(
        self,
        source_type: str,
        source_data: str,
        analyzer_output: Dict[str, Any],
        target_schema: Optional[Dict[str, Any]] = None
    ) -> TransformationCode:
        """Generate transformation code for all layers."""
        # Generate bronze layer code
        bronze_code = await self.llm.generate_code(
            task="Generate bronze layer ETL code",
            context={
                "source_type": source_type,
                "source_data": source_data,
                "analyzer_output": analyzer_output
            },
            requirements=[
                "Ingest raw data with minimal transformations",
                "Preserve original data",
                "Add metadata columns",
                "Handle basic error cases"
            ]
        )
        
        # Generate silver layer code
        silver_code = await self.llm.generate_code(
            task="Generate silver layer ETL code",
            context={
                "source_type": source_type,
                "source_data": source_data,
                "analyzer_output": analyzer_output,
                "bronze_code": bronze_code
            },
            requirements=[
                "Clean and standardize data",
                "Handle data quality issues",
                "Implement efficient joins",
                "Apply data engineering best practices",
                "Create normalized data model"
            ]
        )
        
        # Generate gold layer code
        gold_code = await self.llm.generate_code(
            task="Generate gold layer ETL code",
            context={
                "source_type": source_type,
                "source_data": source_data,
                "analyzer_output": analyzer_output,
                "silver_code": silver_code,
                "target_schema": target_schema
            },
            requirements=[
                "Create final data model",
                "Implement feature engineering",
                "Ensure high query performance",
                "Apply business logic and aggregations",
                "Validate final output"
            ]
        )
        
        return TransformationCode(
            bronze_layer=bronze_code,
            silver_layer=silver_code,
            gold_layer=gold_code,
            analyzer_findings=analyzer_output
        )
    
    async def generate_tester_code(
        self,
        source_type: str,
        source_data: str,
        transformation_code: TransformationCode,
        target_schema: Optional[Dict[str, Any]] = None,
        test_types: List[str] = []
    ) -> str:
        """Generate code to test the transformations."""
        return await self.llm.generate_code(
            task="Generate test code",
            context={
                "source_type": source_type,
                "source_data": source_data,
                "transformation_code": transformation_code.model_dump(),
                "target_schema": target_schema,
                "test_types": test_types
            },
            requirements=[
                "Evaluate join performance",
                "Validate data engineering practices",
                "Check data quality",
                "Verify schema compliance",
                "Run user-specified tests"
            ]
        )
    
    def _create_analyzer_prompt(self, source_type: str, source_data: str) -> str:
        """Create prompt for data analysis."""
        return f"""Analyze the following {source_type} data and provide detailed information about:
1. Data types and patterns for each field
2. Data quality issues (nulls, duplicates, inconsistencies)
3. Relationships between data sources
4. Basic statistics (counts, distributions, etc.)
5. Recommendations for cleaning and transformation

Data:
{source_data}

Return the analysis as a structured dictionary with the above categories.
"""
    
    def _create_bronze_layer_prompt(
        self,
        source_type: str,
        source_data: str,
        analyzer_output: Dict[str, Any]
    ) -> str:
        """Create prompt for bronze layer transformation."""
        return f"""Generate code for the bronze layer ETL pipeline.
The bronze layer should:
1. Ingest raw data with minimal transformations
2. Preserve the original data as much as possible
3. Add metadata columns (load_time, source_file, etc.)
4. Handle basic error cases

Source Type: {source_type}
Analyzer Findings: {json.dumps(analyzer_output, indent=2)}

Sample Data:
{source_data}

Return only the code for the bronze layer transformation.
"""
    
    def _create_silver_layer_prompt(
        self,
        source_type: str,
        source_data: str,
        analyzer_output: Dict[str, Any],
        bronze_code: str
    ) -> str:
        """Create prompt for silver layer transformation."""
        return f"""Generate code for the silver layer ETL pipeline.
The silver layer should:
1. Clean and standardize data
2. Handle data quality issues identified by the analyzer
3. Implement efficient join strategies
4. Apply data engineering best practices
5. Create a normalized data model

Source Type: {source_type}
Analyzer Findings: {json.dumps(analyzer_output, indent=2)}
Bronze Layer Code: {bronze_code}

Sample Data:
{source_data}

Return only the code for the silver layer transformation.
"""
    
    def _create_gold_layer_prompt(
        self,
        source_type: str,
        source_data: str,
        analyzer_output: Dict[str, Any],
        silver_code: str,
        target_schema: Optional[Dict[str, Any]]
    ) -> str:
        """Create prompt for gold layer transformation."""
        schema_text = "using the provided schema" if target_schema else "by engineering relevant features"
        return f"""Generate code for the gold layer ETL pipeline.
The gold layer should:
1. Create the final data model {schema_text}
2. Implement feature engineering
3. Ensure high query performance
4. Apply business logic and aggregations
5. Validate final output

Source Type: {source_type}
Analyzer Findings: {json.dumps(analyzer_output, indent=2)}
Silver Layer Code: {silver_code}
Target Schema: {json.dumps(target_schema, indent=2) if target_schema else "Not specified - engineer relevant features"}

Sample Data:
{source_data}

Return only the code for the gold layer transformation.
"""
    
    def _create_tester_prompt(
        self,
        source_type: str,
        source_data: str,
        transformation_code: TransformationCode,
        target_schema: Optional[Dict[str, Any]],
        test_types: List[str]
    ) -> str:
        """Create prompt for test generation."""
        return f"""Generate comprehensive tests for the ETL pipeline.
Tests should cover:
1. Join performance evaluation
2. Data engineering best practices validation
3. Data quality checks
4. Schema compliance
5. User-specified tests: {', '.join(test_types) if test_types else 'None specified'}

Source Type: {source_type}
Transformation Code:
Bronze Layer:
{transformation_code.bronze_layer}

Silver Layer:
{transformation_code.silver_layer}

Gold Layer:
{transformation_code.gold_layer}

Target Schema: {json.dumps(target_schema, indent=2) if target_schema else "Not specified"}
Analyzer Findings: {json.dumps(transformation_code.analyzer_findings, indent=2)}

Sample Data:
{source_data}

Return the test code that validates all aspects of the transformation pipeline.
"""
    
    def _parse_analyzer_response(self, response: str) -> Dict[str, Any]:
        """Parse the analyzer's response into a structured format."""
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            # If the response isn't valid JSON, try to extract structured data
            # This is a simplified version - you might want to add more robust parsing
            sections = {
                "data_types": {},
                "quality_issues": [],
                "relationships": [],
                "statistics": {},
                "recommendations": []
            }
            
            current_section = None
            for line in response.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.endswith(":"):
                    current_section = line[:-1].lower().replace(" ", "_")
                    continue
                
                if current_section:
                    if isinstance(sections.get(current_section), list):
                        sections[current_section].append(line)
                    elif isinstance(sections.get(current_section), dict):
                        if ":" in line:
                            key, value = line.split(":", 1)
                            sections[current_section][key.strip()] = value.strip()
            
            return sections 