"""
Test execution environment for running Sidewinder tests.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import tempfile
import os
import json
import logging
from contextlib import contextmanager
import time
import psutil

from sidewinder.agents.transformer import TransformationStep
from sidewinder.core.config import Source, Target


class TestEnvironment(BaseModel):
    """Test execution environment configuration."""
    spark_config: Dict[str, str] = {
        "spark.sql.execution.arrow.pyspark.enabled": "true",
        "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
        "spark.sql.shuffle.partitions": "4",
        "spark.default.parallelism": "4",
        "spark.driver.memory": "2g",
        "spark.executor.memory": "2g"
    }
    temp_dir: Optional[str] = None
    log_level: str = "INFO"
    max_memory_gb: float = 4.0
    timeout_seconds: int = 300


class TestContext:
    """Context for test execution with resource management."""
    
    def __init__(self, config: TestEnvironment):
        self.config = config
        self.spark = None
        self.temp_dir = None
        self.start_time = None
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        """Set up logging for the test environment."""
        logger = logging.getLogger("sidewinder.testing")
        logger.setLevel(self.config.log_level)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _setup_spark(self):
        """Set up Spark session for testing."""
        from pyspark.sql import SparkSession
        
        builder = SparkSession.builder.appName("SidewinderTest")
        
        # Apply configuration
        for key, value in self.config.spark_config.items():
            builder = builder.config(key, value)
        
        self.spark = builder.getOrCreate()
        self.logger.info("Spark session initialized")
    
    def _create_temp_dir(self):
        """Create temporary directory for test artifacts."""
        self.temp_dir = self.config.temp_dir or tempfile.mkdtemp(prefix="sidewinder_test_")
        os.makedirs(self.temp_dir, exist_ok=True)
        self.logger.info(f"Created temp directory: {self.temp_dir}")
    
    def _cleanup(self):
        """Clean up resources."""
        if self.spark:
            self.spark.stop()
            self.spark = None
        
        if self.temp_dir and not self.config.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    @contextmanager
    def session(self):
        """Create a test session with managed resources."""
        try:
            self._setup_spark()
            self._create_temp_dir()
            self.start_time = time.time()
            yield self
        finally:
            self._cleanup()
    
    def monitor_resources(self) -> Dict[str, float]:
        """Monitor resource usage during test execution."""
        process = psutil.Process()
        
        return {
            "cpu_percent": process.cpu_percent(),
            "memory_gb": process.memory_info().rss / (1024 * 1024 * 1024),
            "elapsed_seconds": time.time() - self.start_time
        }
    
    def check_resource_limits(self):
        """Check if resource usage is within limits."""
        resources = self.monitor_resources()
        
        if resources["memory_gb"] > self.config.max_memory_gb:
            raise MemoryError(
                f"Memory usage ({resources['memory_gb']:.2f}GB) exceeds limit "
                f"({self.config.max_memory_gb}GB)"
            )
        
        if resources["elapsed_seconds"] > self.config.timeout_seconds:
            raise TimeoutError(
                f"Test execution time ({resources['elapsed_seconds']:.2f}s) exceeds "
                f"timeout ({self.config.timeout_seconds}s)"
            )


class TestExecutor:
    """Executes tests in a managed environment."""
    
    def __init__(self, environment: TestEnvironment):
        self.environment = environment
    
    def execute_test(self, test_code: str, test_data: Any) -> Dict[str, Any]:
        """
        Execute a single test in an isolated environment.
        
        Args:
            test_code: The test code to execute
            test_data: Test data to use
            
        Returns:
            Test results including metrics
        """
        with TestContext(self.environment).session() as context:
            try:
                # Create test globals
                test_globals = {
                    "spark": context.spark,
                    "test_data": test_data,
                    "temp_dir": context.temp_dir,
                    "logger": context.logger
                }
                
                # Execute test
                exec(test_code, test_globals)
                
                # Collect metrics
                metrics = context.monitor_resources()
                
                return {
                    "success": True,
                    "metrics": metrics
                }
                
            except Exception as e:
                context.logger.error(f"Test failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "metrics": context.monitor_resources()
                }
    
    def execute_transformation(
        self,
        step: TransformationStep,
        input_data: Any,
        source: Source,
        target: Target
    ) -> Dict[str, Any]:
        """
        Execute a transformation step in the test environment.
        
        Args:
            step: The transformation step to execute
            input_data: Input data for the transformation
            source: Source configuration
            target: Target configuration
            
        Returns:
            Transformation results and metrics
        """
        with TestContext(self.environment).session() as context:
            try:
                # Create transformation globals
                transform_globals = {
                    "spark": context.spark,
                    "input_data": input_data,
                    "source": source,
                    "target": target,
                    "temp_dir": context.temp_dir,
                    "logger": context.logger
                }
                
                # Execute transformation
                exec(step.code, transform_globals)
                
                # Get output data
                output_data = transform_globals.get("output_data")
                
                # Validate schema
                if output_data is not None:
                    self._validate_schema(output_data, step.output_schema)
                
                return {
                    "success": True,
                    "output_data": output_data,
                    "metrics": context.monitor_resources()
                }
                
            except Exception as e:
                context.logger.error(f"Transformation failed: {str(e)}")
                return {
                    "success": False,
                    "error": str(e),
                    "metrics": context.monitor_resources()
                }
    
    def _validate_schema(self, data: Any, expected_schema: Dict[str, Any]):
        """Validate data against expected schema."""
        if hasattr(data, "schema"):
            actual_schema = {
                field.name: str(field.dataType)
                for field in data.schema.fields
            }
            
            for field_name, field_type in expected_schema.items():
                if field_name not in actual_schema:
                    raise ValueError(f"Missing field: {field_name}")
                if actual_schema[field_name] != field_type:
                    raise ValueError(
                        f"Schema mismatch for {field_name}: "
                        f"expected {field_type}, got {actual_schema[field_name]}"
                    )


class TestResult(BaseModel):
    """Results from test execution."""
    success: bool
    test_name: str
    duration_seconds: float
    memory_usage_gb: float
    cpu_percent: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = {}


class TestSuiteResult(BaseModel):
    """Results from test suite execution."""
    suite_name: str
    results: List[TestResult]
    total_duration_seconds: float
    peak_memory_gb: float
    avg_cpu_percent: float
    success_rate: float 