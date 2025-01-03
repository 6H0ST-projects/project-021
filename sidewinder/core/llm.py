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
                          "2. Use of pandas DataFrame for data manipulation\n"
                          "3. Proper function signature with type hints\n"
                          "4. Error handling with try/except\n"
                          "5. Logging for important steps\n"
                          "6. Return values in the expected format"),
                ("human", "Task: {task}\n"
                         "Requirements:\n{requirements_str}\n"
                         "Constraints:\n{constraints_str}\n"
                         "Context:\n{context_str}\n\n"
                         "Generate the code following these guidelines:\n"
                         "1. Include necessary imports\n"
                         "2. Add comprehensive docstrings\n"
                         "3. Implement proper error handling\n"
                         "4. Add type hints\n"
                         "5. Include logging\n"
                         "6. Use pandas for all data transformations\n"
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

class DataEngineeringAgent:
    """Agent for generating data engineering code."""
    
    def __init__(self):
        """Initialize the agent."""
        self.generator = LLMCodeGenerator()
        
    async def generate_analyzer_code(
        self,
        source_type: str,
        source_data: str
    ) -> str:
        """Generate code for data analysis."""
        try:
            code = await self.generator.generate_code(
                task="Generate data analyzer code",
                context={
                    "source_type": source_type,
                    "source_data": source_data
                },
                requirements=[
                    "Define analyze_data function that takes a source_data parameter",
                    "Use pandas DataFrame for data processing",
                    "Return a dictionary containing analysis results",
                    "Include schema detection, data quality metrics, and patterns"
                ],
                constraints=[
                    "Function must be named exactly 'analyze_data'",
                    "Must use pd.DataFrame for data processing",
                    "Include proper error handling",
                    "Add logging for important steps"
                ]
            )
            return code
        except Exception as e:
            logger.error(f"Analyzer code generation failed: {str(e)}")
            raise
            
    async def generate_transformer_code(
        self,
        source_type: str,
        source_data: str,
        target_schema: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate code for data transformation."""
        try:
            code = await self.generator.generate_code(
                task="Generate data transformer code",
                context={
                    "source_type": source_type,
                    "source_data": source_data,
                    "target_schema": target_schema
                },
                requirements=[
                    "Define transform_data function that takes source_data parameter",
                    "Use pandas DataFrame for data processing",
                    "Apply transformations according to target schema",
                    "Return transformed DataFrame"
                ],
                constraints=[
                    "Function must be named exactly 'transform_data'",
                    "Must use pd.DataFrame for data processing",
                    "Include proper error handling",
                    "Add logging for important steps"
                ]
            )
            return code
        except Exception as e:
            logger.error(f"Transformer code generation failed: {str(e)}")
            raise
            
    async def generate_tester_code(
        self,
        source_type: str,
        source_data: str,
        target_schema: Optional[Dict[str, Any]] = None,
        test_types: Optional[List[Dict[str, Any]]] = None
    ) -> str:
        """Generate code for data testing."""
        try:
            code = await self.generator.generate_code(
                task="Generate data tester code",
                context={
                    "source_type": source_type,
                    "source_data": source_data,
                    "target_schema": target_schema,
                    "test_types": test_types
                },
                requirements=[
                    "Define test_data function that takes source_data parameter",
                    "Use pandas DataFrame for data validation",
                    "Implement all specified test types",
                    "Return test results dictionary"
                ],
                constraints=[
                    "Function must be named exactly 'test_data'",
                    "Must use pd.DataFrame for data processing",
                    "Include proper error handling",
                    "Add logging for important steps"
                ]
            )
            return code
        except Exception as e:
            logger.error(f"Tester code generation failed: {str(e)}")
            raise 