"""
Test executor for running tests based on configuration.
"""

from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging

from sidewinder.core.test_config import GlobalTestConfig, TestConfig
from sidewinder.testing.environment import TestEnvironment, TestContext
from sidewinder.testing.registry import registry
from sidewinder.testing.reporting import TestReport, TestResult, TestSuiteResult


class TestExecutor:
    """Executes tests based on configuration."""
    
    def __init__(self, config: GlobalTestConfig):
        self.config = config
        self.environment = TestEnvironment(
            spark_config={
                "spark.sql.execution.arrow.pyspark.enabled": "true",
                "spark.sql.execution.arrow.maxRecordsPerBatch": "10000",
                "spark.sql.shuffle.partitions": "200",
                "spark.default.parallelism": "100",
                "spark.driver.memory": "8g",
                "spark.executor.memory": "16g"
            },
            max_memory_gb=config.max_memory_gb,
            max_cpu_percent=config.max_cpu_percent,
            timeout_seconds=config.timeout_seconds
        )
        self.logger = logging.getLogger("sidewinder.testing")
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build graph of test dependencies."""
        return {
            test.name: test.dependencies
            for test in self.config.tests
            if test.enabled
        }
    
    def _get_ready_tests(
        self,
        graph: Dict[str, List[str]],
        completed: set
    ) -> List[str]:
        """Get tests that are ready to run."""
        ready = []
        for test_name, deps in graph.items():
            if test_name not in completed and all(d in completed for d in deps):
                ready.append(test_name)
        return ready
    
    async def run_tests(self) -> TestReport:
        """Run all enabled tests."""
        # Build dependency graph
        graph = self._build_dependency_graph()
        completed = set()
        results: List[TestResult] = []
        
        # Create thread pool
        executor = ThreadPoolExecutor(max_workers=self.config.max_parallel_tests)
        loop = asyncio.get_event_loop()
        
        while len(completed) < len(graph):
            # Get tests that are ready to run
            ready_tests = self._get_ready_tests(graph, completed)
            
            if not ready_tests:
                raise RuntimeError("Circular dependency detected in test configuration")
            
            # Run ready tests in parallel
            tasks = []
            for test_name in ready_tests:
                test_config = next(
                    t for t in self.config.tests
                    if t.name == test_name and t.enabled
                )
                
                task = loop.run_in_executor(
                    executor,
                    self._run_single_test,
                    test_config
                )
                tasks.append((test_name, task))
            
            # Wait for all tasks to complete
            for test_name, task in tasks:
                try:
                    result = await task
                    results.append(result)
                    completed.add(test_name)
                    
                except Exception as e:
                    self.logger.error(f"Test {test_name} failed: {str(e)}")
                    results.append(
                        TestResult(
                            success=False,
                            test_name=test_name,
                            duration_seconds=0,
                            memory_usage_gb=0,
                            cpu_percent=0,
                            error_message=str(e)
                        )
                    )
                    completed.add(test_name)
        
        # Create test suite result
        suite_result = TestSuiteResult(
            suite_name=f"Sidewinder Tests - {self.config.environment}",
            results=results,
            total_duration_seconds=sum(r.duration_seconds for r in results),
            peak_memory_gb=max(r.memory_usage_gb for r in results),
            avg_cpu_percent=sum(r.cpu_percent for r in results) / len(results),
            success_rate=len([r for r in results if r.success]) / len(results)
        )
        
        return TestReport([suite_result])
    
    def _run_single_test(self, test_config: TestConfig) -> TestResult:
        """Run a single test."""
        with TestContext(self.environment).session() as context:
            try:
                # Get test implementation
                test_func = registry.get_test(test_config.type, test_config.name)
                if not test_func:
                    raise ValueError(
                        f"No implementation found for test {test_config.name} "
                        f"of type {test_config.type}"
                    )
                
                # Add source and target info to parameters
                parameters = {
                    **test_config.parameters,
                    "source": self.config.source,
                    "target": self.config.target,
                    "input_path": f"{self.config.test_data_location}/{test_config.name}"
                }
                
                # Run test
                start_time = context.start_time
                test_func(context, test_config)
                
                # Get metrics
                metrics = context.monitor_resources()
                
                return TestResult(
                    success=True,
                    test_name=test_config.name,
                    duration_seconds=metrics["elapsed_seconds"],
                    memory_usage_gb=metrics["memory_gb"],
                    cpu_percent=metrics["cpu_percent"]
                )
                
            except Exception as e:
                self.logger.error(f"Test {test_config.name} failed: {str(e)}")
                metrics = context.monitor_resources()
                
                return TestResult(
                    success=False,
                    test_name=test_config.name,
                    duration_seconds=metrics["elapsed_seconds"],
                    memory_usage_gb=metrics["memory_gb"],
                    cpu_percent=metrics["cpu_percent"],
                    error_message=str(e)
                )


async def run_tests(config_file: str) -> TestReport:
    """Run tests from configuration file."""
    # Load configuration
    config = GlobalTestConfig.from_json(config_file)
    
    # Create and run executor
    executor = TestExecutor(config)
    return await executor.run_tests()


def run_tests_sync(config_file: str) -> TestReport:
    """Synchronous version of run_tests."""
    return asyncio.run(run_tests(config_file)) 