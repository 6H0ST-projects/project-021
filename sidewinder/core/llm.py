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
    """Core LLM code generation service."""
    
    def __init__(
        self,
        model: str = "gpt-4",
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
                "proper error handling, logging, and type hints.\n\n"
                "IMPORTANT: Always wrap your code in ```python and ``` markers.\n"
                "IMPORTANT: Make sure to implement the exact function name as specified in the context."
            )),
            MessagesPlaceholder(variable_name="context_messages"),
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
                "6. Follow PEP 8 style guidelines\n\n"
                "Remember to wrap your code in ```python and ``` markers.\n"
                "Remember to implement the exact function name as specified in the context."
            ))
        ])
        
        # Create SQL-specific prompt template
        self.sql_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an expert data engineer specializing in writing performant SQL for large-scale data processing. "
                "Your task is to generate high-performance SQL code that follows best practices for scalability. "
                "Consider the following performance guidelines:\n"
                "1. Minimize expensive operations (e.g., CROSS JOIN, correlated subqueries)\n"
                "2. Use window functions instead of self-joins where possible\n"
                "3. Optimize JOIN order and types (prefer INNER JOIN over LEFT JOIN when possible)\n"
                "4. Leverage partitioning and clustering when available\n"
                "5. Use appropriate indexing strategies\n"
                "6. Consider data skew and implement appropriate handling\n"
                "7. Implement efficient aggregations (e.g., pre-aggregate when possible)\n"
                "8. Use CTEs for better readability and optimization\n"
                "9. Implement proper error handling and NULL value management\n"
                "10. Add clear comments explaining complex logic and performance considerations\n\n"
                "IMPORTANT: Always wrap your code in ```python and ``` markers.\n"
                "IMPORTANT: Make sure to implement the exact function name as specified in the context."
            )),
            MessagesPlaceholder(variable_name="context_messages"),
            HumanMessage(content=(
                "Task: {task}\n"
                "Requirements: {requirements}\n"
                "Constraints: {constraints}\n"
                "Context: {context}\n\n"
                "Generate performant SQL code that scales to large datasets.\n"
                "Remember to wrap your code in ```python and ``` markers.\n"
                "Remember to implement the exact function name as specified in the context."
            ))
        ])
        
        # Create refinement prompt template
        self.refinement_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=(
                "You are an expert Python developer specializing in data engineering. "
                "Your task is to fix and improve code that encountered errors during execution. "
                "Analyze the error, identify the root cause, and generate an improved version "
                "that addresses the issue while maintaining all functionality.\n\n"
                "IMPORTANT: Always wrap your code in ```python and ``` markers.\n"
                "IMPORTANT: Make sure to implement the exact function name as specified in the context."
            )),
            MessagesPlaceholder(variable_name="history"),
            HumanMessage(content=(
                "The following code encountered an error:\n"
                "```python\n{code}\n```\n\n"
                "Error details:\n"
                "Type: {error_type}\n"
                "Message: {error_message}\n"
                "Traceback: {traceback}\n\n"
                "Context: {context}\n\n"
                "Please fix the code and explain your changes.\n"
                "Remember to wrap your code in ```python and ``` markers.\n"
                "Remember to implement the exact function name as specified in the context."
            ))
        ])
    
    async def generate_code(
        self,
        request: CodeGenerationRequest,
        execution_context: Optional[Dict[str, Any]] = None,
        use_sql_prompt: bool = False
    ) -> CodeGenerationResponse:
        """
        Generate code using the LLM with iterative refinement.
        
        Args:
            request: Code generation request containing task and context
            execution_context: Optional context for code execution and testing
            use_sql_prompt: Whether to use the SQL-specific prompt template
            
        Returns:
            Generated code and metadata
        """
        conversation_history = []
        attempts = 0
        
        while attempts < self.max_retries:
            try:
                # Format prompt with request details and conversation history
                prompt_template = self.sql_prompt if use_sql_prompt else self.prompt
                
                # Format requirements and constraints as bullet points
                formatted_requirements = "\n".join(f"- {req}" for req in request.requirements)
                formatted_constraints = "\n".join(f"- {const}" for const in (request.constraints or []))
                
                # Format context with pretty indentation
                formatted_context = json.dumps(request.context, indent=2)
                
                # Create human message with filled placeholders
                human_message = HumanMessage(content=(
                    f"Task: {request.task}\n"
                    f"Requirements:\n{formatted_requirements}\n"
                    f"Constraints:\n{formatted_constraints}\n"
                    f"Context:\n{formatted_context}\n\n"
                    "Generate the code following these guidelines:\n"
                    "1. Include necessary imports\n"
                    "2. Add comprehensive docstrings\n"
                    "3. Implement proper error handling\n"
                    "4. Add type hints\n"
                    "5. Include logging\n"
                    "6. Follow PEP 8 style guidelines\n\n"
                    "Remember to wrap your code in ```python and ``` markers.\n"
                    "Remember to implement the exact function name as specified in the context."
                ))
                
                # Create message list
                messages = [
                    SystemMessage(content=(
                        "You are an expert Python developer specializing in data engineering. "
                        "Your task is to generate high-quality, efficient, and well-documented code "
                        "that follows best practices. The code should be production-ready and include "
                        "proper error handling, logging, and type hints.\n\n"
                        "IMPORTANT: Always wrap your code in ```python and ``` markers.\n"
                        "IMPORTANT: Make sure to implement the exact function name as specified in the context."
                    )),
                    *conversation_history,
                    human_message
                ]
                
                # Print formatted prompt for debugging
                print("\n=== Formatted Prompt ===")
                for msg in messages:
                    print(f"{msg.type}: {msg.content}")
                print("=== End Prompt ===\n")
                
                # Get LLM response
                response = await self.llm.ainvoke(messages)
                
                # Print response for debugging
                print("\n=== Model Response ===")
                print(response.content)
                print("=== End Response ===\n")
                
                conversation_history.append(response)
                
                # Parse code blocks from response
                code_blocks = self._extract_code_blocks(response.content)
                
                if not code_blocks:
                    print("No code blocks found in response. Attempting fallback extraction...")
                    # Try fallback extraction
                    code_blocks = self._extract_code_blocks_fallback(response.content)
                    if not code_blocks:
                        raise ValueError("No code blocks found in response")
                    else:
                        print("Found code blocks using fallback extraction.")
                
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
                        error_details = CodeExecutionError.from_exception(e, execution_context)
                        
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
    
    def _extract_code_blocks_fallback(self, text: str) -> List[str]:
        """Fallback method to extract code blocks using more lenient parsing."""
        blocks = []
        
        # First try to find code between triple backticks
        parts = text.split("```")
        if len(parts) > 1:
            # Extract code from every other part (between backticks)
            for i in range(1, len(parts), 2):
                code = parts[i].strip()
                if code.startswith("python"):
                    code = code[6:].strip()  # Remove "python" prefix
                if code:
                    blocks.append(code)
        
        # If no blocks found, try to extract Python-like code
        if not blocks:
            lines = text.split("\n")
            current_block = []
            in_code = False
            
            for line in lines:
                stripped = line.strip()
                
                # Check for Python code indicators
                if (stripped.startswith(("def ", "class ", "import ", "from ")) or
                    stripped.endswith(":") or
                    "=" in stripped or
                    stripped.startswith(("@", "#", "    ", "\t"))):
                    in_code = True
                    current_block.append(line)
                elif in_code:
                    if stripped:
                        current_block.append(line)
                    else:
                        if current_block:
                            blocks.append("\n".join(current_block))
                            current_block = []
                        in_code = False
            
            if current_block:
                blocks.append("\n".join(current_block))
        
        return blocks
    
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
        try:
            # Format refinement prompt
            formatted_prompt = self.refinement_prompt.format_messages(
                code=code,
                error_type=error.error_type,
                error_message=error.error_message,
                traceback="\n".join(traceback.format_tb(error.traceback)) if error.traceback else "No traceback available",
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
        except Exception as e:
            logger.error(f"Code refinement failed: {str(e)}")
            raise
    
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
    """Core LLM code generation service."""
    
    def __init__(
        self,
        model: str = "gpt-4",
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
    
    async def analyze_data_source(
        self,
        source_type: str,
        sample_data: pd.DataFrame,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze a data source using our existing analyzer.
        
        Args:
            source_type: Type of data source
            sample_data: Sample data to analyze
            context: Additional context for analysis
            
        Returns:
            Analysis results
        """
        try:
            # Use our existing analyzer
            results = analyze_data(sample_data)
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}")
            raise
    
    async def generate_transformation(
        self,
        source_type: str,
        source_data: pd.DataFrame,
        target_schema: Optional[TargetSchema] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate transformation code.
        
        Args:
            source_type: Type of data source
            source_data: Sample data to transform
            target_schema: Optional target schema
            context: Additional context for transformation
            
        Returns:
            Generated transformation code
        """
        try:
            # Convert target schema to dictionary if present
            schema_dict = target_schema.dict() if target_schema else None
            
            # Create transformation request
            request = CodeGenerationRequest(
                task="Generate data transformation code",
                context={
                    "source_type": source_type,
                    "sample_data": source_data.head().to_string(),
                    "target_schema": schema_dict,
                    **(context or {})
                },
                requirements=[
                    "Implement transform_data function that takes source_data as input",
                    "Apply necessary data transformations",
                    "Validate against target schema if provided",
                    "Handle data type conversions",
                    "Implement error handling"
                ],
                constraints=[
                    "Handle large datasets efficiently",
                    "Use pandas and numpy for transformations",
                    "Include proper error handling",
                    "Add logging for important steps"
                ]
            )
            
            # Generate code
            generator = LLMCodeGenerator()
            response = await generator.generate_code(request, use_sql_prompt=True)
            return response.code
            
        except Exception as e:
            logger.error(f"Transformation generation failed: {str(e)}")
            raise
    
    async def generate_test_cases(
        self,
        source_type: str,
        source_data: pd.DataFrame,
        target_schema: Optional[TargetSchema] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate test cases.
        
        Args:
            source_type: Type of data source
            source_data: Sample data to test
            target_schema: Optional target schema
            context: Additional context for testing
            
        Returns:
            Generated test code
        """
        try:
            # Convert target schema to dictionary if present
            schema_dict = target_schema.dict() if target_schema else None
            
            # Create test request
            request = CodeGenerationRequest(
                task="Generate test cases",
                context={
                    "source_type": source_type,
                    "sample_data": source_data.head().to_string(),
                    "target_schema": schema_dict,
                    **(context or {})
                },
                requirements=[
                    "Implement test_data function that takes source_data as input",
                    "Validate data quality",
                    "Check schema compliance",
                    "Verify transformations",
                    "Test error handling"
                ],
                constraints=[
                    "Use pytest for testing",
                    "Include data validation tests",
                    "Add performance tests",
                    "Test error cases"
                ]
            )
            
            # Generate code
            generator = LLMCodeGenerator()
            response = await generator.generate_code(request)
            return response.code
            
        except Exception as e:
            logger.error(f"Test generation failed: {str(e)}")
            raise 