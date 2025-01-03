"""
Test execution environment for Sidewinder.
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import asyncio
from pydantic import BaseModel, Field
from pyspark.sql import SparkSession
import numpy as np

from sidewinder.core.config import Source, Target
from sidewinder.core.pipeline import Pipeline
from sidewinder.core.llm import LLMCodeGenerator
from sidewinder.testing.metrics import PerformanceMetrics

# Initialize Spark session for local mode
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("SidewinderTests") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()


class GlobalTestConfig(BaseModel):
    """Global test configuration."""
    environment: str
    output_dir: str
    max_parallel_tests: int = 4
    max_parallel_suites: int = 2
    max_memory_gb: float = 4.0
    max_cpu_percent: float = 80.0
    timeout_seconds: int = 300
    llm_config: Optional[Dict[str, Any]] = None
    scenarios: List[Dict[str, Any]] = Field(default_factory=list)


class TestResult(BaseModel):
    """Result of a single test."""
    success: bool
    test_name: str
    duration_seconds: float
    memory_usage_gb: float
    cpu_percent: float
    error_message: Optional[str] = None
    join_performance: Optional[float] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        validate_assignment = True
        extra = "allow"
        arbitrary_types_allowed = True


class TestSuiteResult(BaseModel):
    """Result of a test suite."""
    suite_name: str
    results: List[TestResult]
    total_duration_seconds: float
    peak_memory_gb: float
    avg_cpu_percent: float
    success_rate: float
    join_metrics: Optional[Dict[str, Any]] = None


class TestReport(BaseModel):
    """Complete test execution report."""
    suites: List[TestSuiteResult]


class TestContext:
    """Context for test execution."""
    
    def __init__(self, config: GlobalTestConfig, test_data: Optional[Any] = None):
        self.config = config
        self.start_time = time.time()
        self.metrics = {}
        self.test_data = test_data
        
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor resource usage."""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "elapsed_seconds": time.time() - self.start_time,
            "memory_gb": memory_info.rss / (1024 * 1024 * 1024),
            "cpu_percent": process.cpu_percent()
        }
        
    def load_data(self, path: str) -> "pyspark.sql.DataFrame":
        """Load test data."""
        if self.test_data is not None:
            return self.test_data
            
        if path.endswith(".json"):
            return spark.read.json(path)
        elif path.endswith(".parquet"):
            return spark.read.parquet(path)
        elif path.endswith(".csv"):
            return spark.read.csv(path, header=True, inferSchema=True)
        else:
            raise ValueError(f"Unsupported file format: {path}")


class TestExecutor:
    """Executes tests based on configuration."""
    
    def __init__(self, config: GlobalTestConfig):
        self.config = config
        self.logger = logging.getLogger("sidewinder.testing")
        
        # Initialize LLM agents
        self.llm_config = config.llm_config or {}
        self.llm_generator = LLMCodeGenerator(
            model=self.llm_config.get("model", "gpt-4o"),
            temperature=self.llm_config.get("temperature", 0.2),
            max_tokens=self.llm_config.get("max_tokens", 4000),
            max_retries=self.llm_config.get("max_retries", 3)
        )
        
    async def run_tests(self, test_data: Optional[Any] = None) -> TestReport:
        """
        Run all tests.
        
        Args:
            test_data: Optional test data to use instead of loading from files
            
        Returns:
            Test execution report
        """
        try:
            # Create test suites
            suites = []
            for scenario in self.config.scenarios:
                suite_result = await self._run_test_suite(scenario, test_data)
                suites.append(suite_result)
            
            return TestReport(suites=suites)
            
        except Exception as e:
            self.logger.error(f"Test execution failed: {str(e)}")
            raise
            
    async def _run_test_suite(
        self,
        scenario: Dict[str, Any],
        test_data: Optional[Any] = None
    ) -> TestSuiteResult:
        """Run a test suite."""
        try:
            # Run tests in parallel
            tasks = []
            for test_config in scenario["tests"]:
                context = TestContext(self.config, test_data)
                task = asyncio.create_task(self._run_single_test(test_config, context))
                tasks.append(task)
                
                # Limit parallel tests
                if len(tasks) >= self.config.max_parallel_tests:
                    await asyncio.gather(*tasks)
                    tasks = []
            
            # Wait for remaining tests
            if tasks:
                await asyncio.gather(*tasks)
            
            # Create test suite result
            results = [task.result() for task in tasks]
            
            # Calculate join metrics if available
            join_metrics = None
            if test_data is not None and hasattr(test_data, "join_metrics"):
                join_metrics = test_data.join_metrics
            
            return TestSuiteResult(
                suite_name=scenario["name"],
                results=results,
                total_duration_seconds=sum(r.duration_seconds for r in results),
                peak_memory_gb=max(r.memory_usage_gb for r in results),
                avg_cpu_percent=sum(r.cpu_percent for r in results) / len(results),
                success_rate=len([r for r in results if r.success]) / len(results),
                join_metrics=join_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Test suite execution failed: {str(e)}")
            raise
            
    async def _run_code_generation_test(
        self,
        test_config: Dict[str, Any],
        context: TestContext
    ) -> Dict[str, Any]:
        """Run a code generation test."""
        try:
            # Generate code
            code = await self.llm_generator.generate_code(
                task=test_config["task"],
                context=test_config["context"],
                requirements=test_config["requirements"],
                constraints=test_config.get("constraints", [])
            )
            
            # Validate generated code
            validation_results = self._validate_generated_code(
                code,
                test_config["validation_rules"]
            )
            
            # Get metrics
            metrics = context.monitor_resources()
            
            # Extract join performance if available
            join_performance = None
            test_data_metrics = None
            if context.test_data is not None:
                if hasattr(context.test_data, "join_metrics"):
                    test_data_metrics = context.test_data.join_metrics
                if hasattr(context.test_data, "join_performance"):
                    join_performance = float(context.test_data.join_performance)
            
            return {
                "success": validation_results["success"],
                "test_name": test_config["name"],
                "duration_seconds": metrics["elapsed_seconds"],
                "memory_usage_gb": metrics["memory_gb"],
                "cpu_percent": metrics["cpu_percent"],
                "join_performance": join_performance,
                "metrics": {
                    **validation_results["metrics"],
                    "test_data_metrics": test_data_metrics
                }
            }
            
        except Exception as e:
            self.logger.error(f"Code generation test failed: {str(e)}")
            metrics = context.monitor_resources()
            
            return {
                "success": False,
                "test_name": test_config["name"],
                "duration_seconds": metrics["elapsed_seconds"],
                "memory_usage_gb": metrics["memory_gb"],
                "cpu_percent": metrics["cpu_percent"],
                "error": str(e),
                "metrics": {}
            }
            
    async def _run_code_execution_test(
        self,
        test_config: Dict[str, Any],
        context: TestContext
    ) -> Dict[str, Any]:
        """Run a code execution test."""
        try:
            # Get test parameters
            code = test_config["parameters"]["code"]
            test_data = context.load_data(test_config["parameters"]["test_data_path"])
            
            # Create execution context
            execution_context = {
                "test_data": test_data,
                "spark": spark,
                "expected_functions": test_config["parameters"].get("expected_functions", []),
                "test_cases": test_config["parameters"].get("test_cases", [])
            }
            
            # Execute code
            await self.llm_generator._test_code_execution(code, execution_context)
            
            # Get metrics
            metrics = context.monitor_resources()
            
            # Extract join performance if available
            join_performance = None
            test_data_metrics = None
            if context.test_data is not None:
                if hasattr(context.test_data, "join_metrics"):
                    test_data_metrics = context.test_data.join_metrics
                if hasattr(context.test_data, "join_performance"):
                    join_performance = float(context.test_data.join_performance)
            
            return {
                "success": True,
                "test_name": test_config["name"],
                "duration_seconds": metrics["elapsed_seconds"],
                "memory_usage_gb": metrics["memory_gb"],
                "cpu_percent": metrics["cpu_percent"],
                "metrics": {
                    **metrics,
                    "join_performance": join_performance,
                    "test_data_metrics": test_data_metrics
                }
            }
            
        except Exception as e:
            self.logger.error(f"Code execution test failed: {str(e)}")
            metrics = context.monitor_resources()
            
            return {
                "success": False,
                "test_name": test_config["name"],
                "duration_seconds": metrics["elapsed_seconds"],
                "memory_usage_gb": metrics["memory_gb"],
                "cpu_percent": metrics["cpu_percent"],
                "error": str(e),
                "metrics": {}
            }
            
    async def _run_single_test(
        self,
        test_config: Dict[str, Any],
        context: TestContext
    ) -> TestResult:
        """Run a single test."""
        try:
            # Run test based on type
            if test_config["type"] == "code_generation":
                result = await self._run_code_generation_test(test_config, context)
            elif test_config["type"] == "code_execution":
                result = await self._run_code_execution_test(test_config, context)
            else:
                raise ValueError(f"Unknown test type: {test_config['type']}")
            
            # Get resource metrics
            metrics = context.monitor_resources()
            
            # Extract join performance if available
            join_performance = None
            test_data_metrics = None
            if context.test_data is not None:
                if hasattr(context.test_data, "join_metrics"):
                    test_data_metrics = context.test_data.join_metrics
                if hasattr(context.test_data, "join_performance"):
                    join_performance = float(context.test_data.join_performance)
            
            return TestResult(
                success=result["success"],
                test_name=result["test_name"],
                duration_seconds=result["duration_seconds"],
                memory_usage_gb=result["memory_usage_gb"],
                cpu_percent=result["cpu_percent"],
                error_message=result.get("error"),
                join_performance=join_performance,
                metrics=result.get("metrics", {}),
                test_data_metrics=test_data_metrics
            )
            
        except Exception as e:
            self.logger.error(f"Test {test_config['name']} failed: {str(e)}")
            metrics = context.monitor_resources()
            
            return TestResult(
                success=False,
                test_name=test_config["name"],
                duration_seconds=metrics["elapsed_seconds"],
                memory_usage_gb=metrics["memory_gb"],
                cpu_percent=metrics["cpu_percent"],
                error_message=str(e),
                join_performance=None,
                metrics={},
                test_data_metrics=None
            )
            
    def _validate_generated_code(
        self,
        code_response: str,
        validation_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate generated code against rules."""
        try:
            results = {
                "success": True,
                "metrics": {
                    "rules_passed": 0,
                    "rules_failed": 0,
                    "coverage_percent": 0
                }
            }
            
            # Check each validation rule
            for rule in validation_rules:
                if rule["type"] == "contains":
                    if rule["pattern"] in code_response:
                        results["metrics"]["rules_passed"] += 1
                    else:
                        results["metrics"]["rules_failed"] += 1
                        results["success"] = False
                        
                elif rule["type"] == "regex":
                    import re
                    if re.search(rule["pattern"], code_response):
                        results["metrics"]["rules_passed"] += 1
                    else:
                        results["metrics"]["rules_failed"] += 1
                        results["success"] = False
                        
                elif rule["type"] == "function_exists":
                    if f"def {rule['name']}" in code_response:
                        results["metrics"]["rules_passed"] += 1
                    else:
                        results["metrics"]["rules_failed"] += 1
                        results["success"] = False
                        
                else:
                    raise ValueError(f"Unknown validation rule type: {rule['type']}")
            
            # Calculate coverage
            total_rules = len(validation_rules)
            results["metrics"]["coverage_percent"] = (
                results["metrics"]["rules_passed"] / total_rules * 100
            )
            
            return results
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metrics": {
                    "rules_passed": 0,
                    "rules_failed": len(validation_rules),
                    "coverage_percent": 0
                }
            } 