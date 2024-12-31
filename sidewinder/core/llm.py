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
import pandas as pd
import numpy as np

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
    traceback: str
    context: Dict[str, Any]

class LLMCodeGenerator:
    """Core LLM code generation service."""
    
    def __init__(
        self,
        model: str = "gpt-4-turbo-preview",
        temperature: float = 0.2,
        max_tokens: int = 4000,
        max_retries: int = 3
    ):
        self.llm = ChatOpenAI(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        self.max_retries = max_retries
        
        # Create base prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an expert Python developer specializing in data engineering. "
                "Your task is to generate high-quality, efficient, and well-documented code "
                "that follows best practices. The code should be production-ready and include "
                "proper error handling, logging, and type hints."
            )),
            MessagesPlaceholder(key="context"),
            HumanMessage(content=(
                "Task: {task}\n"
                "Requirements: {requirements}\n"
                "Constraints: {constraints}\n"
                "Context: {context}\n\n"
                "Generate the code following these guidelines:\n"
                "1. Include necessary imports\n"
                "2. Add comprehensive docstrings\n"
                "3. Implement proper error handling\n"
                "4. Add type hints\n"
                "5. Include logging\n"
                "6. Follow PEP 8 style guidelines"
            ))
        ])
        
        # Create refinement prompt template
        self.refinement_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an expert Python developer specializing in data engineering. "
                "Your task is to fix and improve code that encountered errors during execution. "
                "Analyze the error, identify the root cause, and generate an improved version "
                "that addresses the issue while maintaining all functionality."
            )),
            MessagesPlaceholder(key="history"),
            HumanMessage(content=(
                "The following code encountered an error:\n"
                "```python\n{code}\n```\n\n"
                "Error details:\n"
                "Type: {error_type}\n"
                "Message: {error_message}\n"
                "Traceback: {traceback}\n\n"
                "Context: {context}\n\n"
                "Please fix the code and explain your changes."
            ))
        ])
    
    async def generate_code(
        self,
        request: CodeGenerationRequest,
        execution_context: Optional[Dict[str, Any]] = None
    ) -> CodeGenerationResponse:
        """
        Generate code using the LLM with iterative refinement.
        
        Args:
            request: Code generation request containing task and context
            execution_context: Optional context for code execution and testing
            
        Returns:
            Generated code and metadata
        """
        conversation_history = []
        attempts = 0
        
        while attempts < self.max_retries:
            try:
                # Format prompt with request details and conversation history
                formatted_prompt = self.prompt.format_messages(
                    task=request.task,
                    requirements=request.requirements,
                    constraints=request.constraints or [],
                    context=json.dumps(request.context, indent=2),
                    context_messages=conversation_history
                )
                
                # Get LLM response
                response = await self.llm.ainvoke(formatted_prompt)
                conversation_history.append(response)
                
                # Parse code blocks from response
                code_blocks = self._extract_code_blocks(response.content)
                
                if not code_blocks:
                    raise ValueError("No code blocks found in response")
                
                # Extract imports
                imports = self._extract_imports(code_blocks[0])
                
                # If execution context provided, test the code
                if execution_context:
                    try:
                        await self._test_code_execution(
                            code_blocks[0],
                            execution_context
                        )
                    except Exception as e:
                        # Create error details for refinement
                        error_details = CodeExecutionError(
                            error_type=type(e).__name__,
                            error_message=str(e),
                            traceback=e.__traceback__.format(),
                            context=execution_context
                        )
                        
                        # Log the error
                        logger.error(
                            f"Code execution failed (attempt {attempts + 1}/{self.max_retries}): "
                            f"{error_details.error_type}: {error_details.error_message}"
                        )
                        
                        if attempts < self.max_retries - 1:
                            # Try to refine the code
                            refined_code = await self._refine_code(
                                code_blocks[0],
                                error_details,
                                conversation_history
                            )
                            code_blocks[0] = refined_code
                            attempts += 1
                            continue
                        else:
                            raise RuntimeError(
                                f"Failed to generate working code after {self.max_retries} attempts"
                            )
                
                return CodeGenerationResponse(
                    code=code_blocks[0],
                    explanation=response.content.split("```")[0].strip(),
                    imports=imports,
                    tests=code_blocks[1:] if len(code_blocks) > 1 else None
                )
                
            except Exception as e:
                if attempts == self.max_retries - 1:
                    raise
                attempts += 1
                logger.warning(
                    f"Code generation attempt {attempts} failed: {str(e)}. Retrying..."
                )
    
    async def _test_code_execution(
        self,
        code: str,
        context: Dict[str, Any]
    ) -> None:
        """
        Test code execution in a safe environment.
        
        Args:
            code: Code to test
            context: Execution context with test data and dependencies
            
        Raises:
            Exception if code execution fails
        """
        try:
            # Create isolated globals dictionary
            exec_globals = {
                "__builtins__": __builtins__,
                **context
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Verify expected functions exist
            for expected_func in context.get("expected_functions", []):
                if expected_func not in exec_globals:
                    raise ValueError(f"Expected function '{expected_func}' not found in generated code")
            
            # Run basic tests if provided
            if "test_cases" in context:
                for test_case in context["test_cases"]:
                    result = eval(test_case, exec_globals)
                    if not result:
                        raise ValueError(f"Test case failed: {test_case}")
                        
        except Exception as e:
            logger.error(f"Code execution failed: {str(e)}")
            raise
    
    async def _refine_code(
        self,
        code: str,
        error: CodeExecutionError,
        history: List[AIMessage]
    ) -> str:
        """
        Refine code based on execution error.
        
        Args:
            code: Original code that failed
            error: Error details
            history: Conversation history
            
        Returns:
            Refined code
        """
        # Format refinement prompt
        formatted_prompt = self.refinement_prompt.format_messages(
            code=code,
            error_type=error.error_type,
            error_message=error.error_message,
            traceback=error.traceback,
            context=json.dumps(error.context, indent=2),
            history=history
        )
        
        # Get LLM response
        response = await self.llm.ainvoke(formatted_prompt)
        
        # Extract refined code
        code_blocks = self._extract_code_blocks(response.content)
        if not code_blocks:
            raise ValueError("No code blocks found in refinement response")
        
        return code_blocks[0]
    
    def _extract_code_blocks(self, text: str) -> List[str]:
        """Extract code blocks from markdown-formatted text."""
        blocks = []
        parts = text.split("```python")
        
        for part in parts[1:]:  # Skip the first part (before first code block)
            if "```" in part:
                code = part.split("```")[0].strip()
                blocks.append(code)
        
        return blocks
    
    def _extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        for line in code.split("\n"):
            if line.startswith(("import ", "from ")):
                imports.append(line.strip())
        return imports

class DataEngineeringAgent:
    """Agent for generating data engineering code."""
    
    def __init__(self):
        self.code_generator = LLMCodeGenerator()
    
    async def analyze_data_source(
        self,
        source_type: str,
        sample_data: Any,
        context: Dict[str, Any]
    ) -> str:
        """
        Generate data analysis code for a given source.
        
        Args:
            source_type: Type of data source
            sample_data: Sample of the data to analyze
            context: Additional context about the source
            
        Returns:
            Generated analysis code
        """
        # Create execution context with test data and expected functions
        execution_context = {
            "source_data": sample_data,
            "pandas": pd,
            "numpy": np,
            "expected_functions": [
                context.get("purpose", "analyze_data")
            ],
            "test_cases": [
                # Basic function existence and signature
                f"callable(locals().get('{context.get('purpose', 'analyze_data')}')",
                # Input validation
                f"hasattr(locals().get('{context.get('purpose', 'analyze_data')}', '__code__'), 'co_varnames')",
                # Return type validation
                f"isinstance({context.get('purpose', 'analyze_data')}(source_data), dict)"
            ]
        }
        
        request = CodeGenerationRequest(
            task=f"Generate comprehensive data analysis code for {source_type} data source",
            context={
                "source_type": source_type,
                "sample_data": str(sample_data)[:1000] if sample_data is not None else None,  # Truncate for LLM context
                **context
            },
            requirements=[
                "Implement data profiling",
                "Detect data quality issues",
                "Infer schema and data types",
                "Generate summary statistics",
                "Identify patterns and anomalies"
            ],
            constraints=[
                "Handle large datasets efficiently",
                "Use pandas and numpy for analysis",
                "Include proper error handling",
                "Add logging for important steps"
            ]
        )
        
        response = await self.code_generator.generate_code(request, execution_context)
        return response.code
    
    async def generate_transformation(
        self,
        source_type: str,
        source_data: Any,
        target_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate transformation code based on source data and target schema.
        
        Args:
            source_type: Type of data source
            source_data: Source data to transform
            target_schema: Target schema to transform to
            context: Additional context
            
        Returns:
            Generated transformation code
        """
        layer = context.get("layer", "transform")
        purpose = context.get("purpose", "transform_data")
        
        # Create execution context with test data and expected functions
        execution_context = {
            "source_data": source_data,
            "target_schema": target_schema,
            "pandas": pd,
            "numpy": np,
            "expected_functions": [
                f"create_{layer}_layer" if layer != "transform" else purpose
            ],
            "test_cases": [
                # Basic function existence and signature
                f"callable(locals().get('create_{layer}_layer' if '{layer}' != 'transform' else '{purpose}'))",
                # Input validation
                f"hasattr(locals().get('create_{layer}_layer' if '{layer}' != 'transform' else '{purpose}'), '__code__')",
                # Schema validation if target schema provided
                f"all(field in create_{layer}_layer(source_data).columns for field in {list(target_schema.keys())})" if target_schema else "True"
            ]
        }
        
        request = CodeGenerationRequest(
            task=f"Generate {layer} layer transformation code",
            context={
                "source_type": source_type,
                "source_data": str(source_data)[:1000] if source_data is not None else None,
                "target_schema": target_schema,
                **context
            },
            requirements=[
                "Implement data cleaning and standardization",
                "Handle data type conversions",
                "Apply business rules and transformations",
                "Validate against target schema",
                "Maintain data quality"
            ],
            constraints=[
                "Optimize for performance",
                "Handle errors gracefully",
                "Support incremental processing",
                "Maintain data lineage"
            ]
        )
        
        response = await self.code_generator.generate_code(request, execution_context)
        return response.code
    
    async def generate_test_data(
        self,
        source_type: str,
        source_data: Any,
        target_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate test data based on source schema and constraints.
        
        Args:
            source_type: Type of data source
            source_data: Sample source data
            target_schema: Target schema for validation
            context: Additional context
            
        Returns:
            Generated test data generation code
        """
        # Create execution context with dependencies and expected functions
        execution_context = {
            "source_data": source_data,
            "target_schema": target_schema,
            "pandas": pd,
            "numpy": np,
            "expected_functions": ["generate_test_data"],
            "test_cases": [
                # Basic function existence and signature
                "callable(locals().get('generate_test_data'))",
                # Input validation
                "hasattr(locals().get('generate_test_data'), '__code__')",
                # Output validation
                "isinstance(generate_test_data(source_data), (pd.DataFrame, dict))",
                # Schema validation if target schema provided
                f"all(field in generate_test_data(source_data).columns for field in {list(target_schema.keys())})" if target_schema else "True"
            ]
        }
        
        request = CodeGenerationRequest(
            task="Generate comprehensive test data",
            context={
                "source_type": source_type,
                "source_data": str(source_data)[:1000] if source_data is not None else None,
                "target_schema": target_schema,
                **context
            },
            requirements=[
                "Generate representative test data",
                "Include edge cases and special values",
                "Match source data distribution",
                "Validate against schema",
                "Include data quality issues"
            ],
            constraints=[
                "Maintain data relationships",
                "Generate reasonable volume",
                "Include all required fields",
                "Match data types and formats"
            ]
        )
        
        response = await self.code_generator.generate_code(request, execution_context)
        return response.code
    
    async def generate_test_cases(
        self,
        source_type: str,
        source_data: Any,
        target_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate test cases for data validation.
        
        Args:
            source_type: Type of data source
            source_data: Data to test
            target_schema: Target schema for validation
            context: Additional context
            
        Returns:
            Generated test case code
        """
        # Create execution context with dependencies and expected functions
        execution_context = {
            "test_data": source_data,
            "target_schema": target_schema,
            "pandas": pd,
            "numpy": np,
            "expected_functions": ["execute_test_cases"],
            "test_cases": [
                # Basic function existence and signature
                "callable(locals().get('execute_test_cases'))",
                # Input validation
                "hasattr(locals().get('execute_test_cases'), '__code__')",
                # Output validation
                "isinstance(execute_test_cases(test_data), dict)",
                # Results structure
                "all(k in execute_test_cases(test_data) for k in ['passed', 'failed', 'errors'])"
            ]
        }
        
        request = CodeGenerationRequest(
            task="Generate comprehensive test cases",
            context={
                "source_type": source_type,
                "source_data": str(source_data)[:1000] if source_data is not None else None,
                "target_schema": target_schema,
                **context
            },
            requirements=[
                "Test data quality rules",
                "Validate schema compliance",
                "Check data relationships",
                "Verify transformations",
                "Test edge cases"
            ],
            constraints=[
                "Include all critical checks",
                "Report detailed results",
                "Handle errors gracefully",
                "Support parallel execution"
            ]
        )
        
        response = await self.code_generator.generate_code(request, execution_context)
        return response.code
    
    async def validate_test_results(
        self,
        source_type: str,
        source_data: Any,
        target_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate validation code for test results.
        
        Args:
            source_type: Type of data source
            source_data: Test results to validate
            target_schema: Target schema for validation
            context: Additional context
            
        Returns:
            Generated validation code
        """
        # Create execution context with dependencies and expected functions
        execution_context = {
            "test_results": source_data,
            "target_schema": target_schema,
            "pandas": pd,
            "numpy": np,
            "expected_functions": ["validate_test_results"],
            "test_cases": [
                # Basic function existence and signature
                "callable(locals().get('validate_test_results'))",
                # Input validation
                "hasattr(locals().get('validate_test_results'), '__code__')",
                # Output validation
                "isinstance(validate_test_results(test_results, target_schema), dict)",
                # Validation structure
                "all(k in validate_test_results(test_results, target_schema) for k in ['valid', 'issues', 'metrics'])"
            ]
        }
        
        request = CodeGenerationRequest(
            task="Generate test results validation code",
            context={
                "source_type": source_type,
                "source_data": str(source_data)[:1000] if source_data is not None else None,
                "target_schema": target_schema,
                **context
            },
            requirements=[
                "Validate test coverage",
                "Verify success criteria",
                "Calculate quality metrics",
                "Generate summary report",
                "Identify improvement areas"
            ],
            constraints=[
                "Consider all test types",
                "Provide actionable insights",
                "Include trend analysis",
                "Support result aggregation"
            ]
        )
        
        response = await self.code_generator.generate_code(request, execution_context)
        return response.code
    
    async def measure_performance(
        self,
        source_type: str,
        source_data: Any,
        target_schema: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate performance measurement code.
        
        Args:
            source_type: Type of data source
            source_data: Test results to measure
            target_schema: Target schema for validation
            context: Additional context
            
        Returns:
            Generated performance measurement code
        """
        # Create execution context with dependencies and expected functions
        execution_context = {
            "test_results": source_data,
            "validation_results": context.get("validation_results"),
            "pandas": pd,
            "numpy": np,
            "expected_functions": ["measure_performance"],
            "test_cases": [
                # Basic function existence and signature
                "callable(locals().get('measure_performance'))",
                # Input validation
                "hasattr(locals().get('measure_performance'), '__code__')",
                # Output validation
                "isinstance(measure_performance(test_results, validation_results), dict)",
                # Metrics structure
                "all(k in measure_performance(test_results, validation_results) for k in ['execution_time', 'memory_usage', 'throughput'])"
            ]
        }
        
        request = CodeGenerationRequest(
            task="Generate performance measurement code",
            context={
                "source_type": source_type,
                "source_data": str(source_data)[:1000] if source_data is not None else None,
                "target_schema": target_schema,
                **context
            },
            requirements=[
                "Measure execution time",
                "Track resource usage",
                "Calculate throughput",
                "Monitor bottlenecks",
                "Generate performance report"
            ],
            constraints=[
                "Minimize overhead",
                "Handle large datasets",
                "Support distributed execution",
                "Include benchmarks"
            ]
        )
        
        response = await self.code_generator.generate_code(request, execution_context)
        return response.code 