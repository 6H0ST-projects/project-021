"""
Test runner for executing Sidewinder test suites.
"""

from typing import List, Dict, Any, Optional
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import statistics

from sidewinder.testing.environment import (
    TestEnvironment,
    TestExecutor,
    TestResult,
    TestSuiteResult
)
from sidewinder.agents.transformer import TransformationStep
from sidewinder.core.config import Source, Target


class TestSuite:
    """A collection of related tests to be executed together."""
    
    def __init__(
        self,
        name: str,
        environment: Optional[TestEnvironment] = None
    ):
        self.name = name
        self.environment = environment or TestEnvironment()
        self.tests: List[Dict[str, Any]] = []
        self.executor = TestExecutor(self.environment)
    
    def add_test(
        self,
        name: str,
        test_code: str,
        test_data: Any,
        dependencies: Optional[List[str]] = None
    ):
        """Add a test to the suite."""
        self.tests.append({
            "name": name,
            "code": test_code,
            "data": test_data,
            "dependencies": dependencies or []
        })
    
    def add_transformation_test(
        self,
        name: str,
        step: TransformationStep,
        input_data: Any,
        source: Source,
        target: Target,
        dependencies: Optional[List[str]] = None
    ):
        """Add a transformation test to the suite."""
        self.tests.append({
            "name": name,
            "type": "transformation",
            "step": step,
            "input_data": input_data,
            "source": source,
            "target": target,
            "dependencies": dependencies or []
        })
    
    def _build_dependency_graph(self) -> Dict[str, List[str]]:
        """Build a graph of test dependencies."""
        graph = {}
        for test in self.tests:
            graph[test["name"]] = test["dependencies"]
        return graph
    
    def _get_ready_tests(
        self,
        graph: Dict[str, List[str]],
        completed: set
    ) -> List[str]:
        """Get tests that are ready to run (all dependencies satisfied)."""
        ready = []
        for test_name, deps in graph.items():
            if test_name not in completed and all(d in completed for d in deps):
                ready.append(test_name)
        return ready
    
    async def run(self, max_parallel: int = 4) -> TestSuiteResult:
        """
        Run all tests in the suite, respecting dependencies.
        
        Args:
            max_parallel: Maximum number of tests to run in parallel
            
        Returns:
            TestSuiteResult containing all test results and metrics
        """
        start_time = time.time()
        results: List[TestResult] = []
        completed = set()
        
        # Build dependency graph
        graph = self._build_dependency_graph()
        
        # Create thread pool for test execution
        executor = ThreadPoolExecutor(max_workers=max_parallel)
        loop = asyncio.get_event_loop()
        
        while len(completed) < len(self.tests):
            # Get tests that are ready to run
            ready_tests = self._get_ready_tests(graph, completed)
            
            if not ready_tests:
                raise RuntimeError("Circular dependency detected in test suite")
            
            # Run ready tests in parallel
            tasks = []
            for test_name in ready_tests:
                test = next(t for t in self.tests if t["name"] == test_name)
                
                if test.get("type") == "transformation":
                    task = loop.run_in_executor(
                        executor,
                        self.executor.execute_transformation,
                        test["step"],
                        test["input_data"],
                        test["source"],
                        test["target"]
                    )
                else:
                    task = loop.run_in_executor(
                        executor,
                        self.executor.execute_test,
                        test["code"],
                        test["data"]
                    )
                
                tasks.append((test_name, task))
            
            # Wait for all tasks to complete
            for test_name, task in tasks:
                try:
                    result = await task
                    
                    test_result = TestResult(
                        success=result["success"],
                        test_name=test_name,
                        duration_seconds=result["metrics"]["elapsed_seconds"],
                        memory_usage_gb=result["metrics"]["memory_gb"],
                        cpu_percent=result["metrics"]["cpu_percent"],
                        error_message=result.get("error"),
                        metrics=result.get("metrics", {})
                    )
                    
                    results.append(test_result)
                    completed.add(test_name)
                    
                except Exception as e:
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
        
        # Calculate suite-level metrics
        total_duration = time.time() - start_time
        peak_memory = max(r.memory_usage_gb for r in results)
        avg_cpu = statistics.mean(r.cpu_percent for r in results)
        success_rate = len([r for r in results if r.success]) / len(results)
        
        return TestSuiteResult(
            suite_name=self.name,
            results=results,
            total_duration_seconds=total_duration,
            peak_memory_gb=peak_memory,
            avg_cpu_percent=avg_cpu,
            success_rate=success_rate
        )


class TestRunner:
    """Runs test suites and aggregates results."""
    
    def __init__(self, environment: Optional[TestEnvironment] = None):
        self.environment = environment or TestEnvironment()
        self.suites: List[TestSuite] = []
    
    def add_suite(self, suite: TestSuite):
        """Add a test suite to the runner."""
        self.suites.append(suite)
    
    async def run_all(self, max_parallel_suites: int = 2) -> List[TestSuiteResult]:
        """
        Run all test suites.
        
        Args:
            max_parallel_suites: Maximum number of test suites to run in parallel
            
        Returns:
            List of TestSuiteResult for each suite
        """
        tasks = []
        for suite in self.suites:
            tasks.append(suite.run())
        
        # Run suites with limited concurrency
        results = []
        for i in range(0, len(tasks), max_parallel_suites):
            batch = tasks[i:i + max_parallel_suites]
            results.extend(await asyncio.gather(*batch))
        
        return results 