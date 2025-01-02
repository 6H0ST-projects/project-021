"""
Test executor for running tests based on configuration.
"""

from typing import List, Dict, Any
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import json
from pathlib import Path
import ast

from sidewinder.core.test_config import GlobalTestConfig, TestConfig, TestType
from sidewinder.testing.environment import TestEnvironment, TestContext
from sidewinder.testing.registry import registry
from sidewinder.testing.reporting import TestReport, TestResult, TestSuiteResult
from sidewinder.core.llm import DataEngineeringAgent, LLMCodeGenerator


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
        
        # Initialize LLM agents
        self.llm_config = config.llm_config
        self.llm_generator = LLMCodeGenerator(
            model=self.llm_config.model if self.llm_config else "gpt-4-turbo-preview",
            temperature=self.llm_config.temperature if self.llm_config else 0.2,
            max_tokens=self.llm_config.max_tokens if self.llm_config else 4000,
            max_retries=self.llm_config.max_retries if self.llm_config else 3
        )
        self.data_engineering_agent = DataEngineeringAgent()
    
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
                # Handle different test types
                if test_config.type == TestType.CODE_GENERATION:
                    return self._run_code_generation_test(test_config, context)
                elif test_config.type == TestType.CODE_EXECUTION:
                    return self._run_code_execution_test(test_config, context)
                else:
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
    
    async def _run_code_generation_test(
        self,
        test_config: TestConfig,
        context: TestContext
    ) -> TestResult:
        """Run a code generation test."""
        try:
            # Get test parameters
            task = test_config.parameters["task"]
            requirements = test_config.parameters["requirements"]
            test_data = context.load_data(test_config.parameters["test_data_path"])
            
            # Generate code
            code_response = await self.data_engineering_agent.generate_transformation(
                source_type=self.config.source.type,
                source_data=test_data,
                target_schema=test_config.parameters.get("target_schema"),
                context={
                    "task": task,
                    "requirements": requirements
                }
            )
            
            # Validate code meets requirements
            validation_results = self._validate_generated_code(
                code_response,
                test_config.parameters["validation_rules"]
            )
            
            # Get metrics
            metrics = context.monitor_resources()
            metrics.update(validation_results["metrics"])
            
            return TestResult(
                success=validation_results["success"],
                test_name=test_config.name,
                duration_seconds=metrics["elapsed_seconds"],
                memory_usage_gb=metrics["memory_gb"],
                cpu_percent=metrics["cpu_percent"],
                error_message=validation_results.get("error"),
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Code generation test failed: {str(e)}")
            metrics = context.monitor_resources()
            
            return TestResult(
                success=False,
                test_name=test_config.name,
                duration_seconds=metrics["elapsed_seconds"],
                memory_usage_gb=metrics["memory_gb"],
                cpu_percent=metrics["cpu_percent"],
                error_message=str(e)
            )
    
    async def _run_code_execution_test(
        self,
        test_config: TestConfig,
        context: TestContext
    ) -> TestResult:
        """Run a code execution test."""
        try:
            # Get test parameters
            code = test_config.parameters["code"]
            test_data = context.load_data(test_config.parameters["test_data_path"])
            
            # Create execution context
            execution_context = {
                "test_data": test_data,
                "pandas": pd,
                "numpy": np,
                "expected_functions": test_config.parameters.get("expected_functions", []),
                "test_cases": test_config.parameters.get("test_cases", [])
            }
            
            # Execute code
            await self.llm_generator._test_code_execution(code, execution_context)
            
            # Get metrics
            metrics = context.monitor_resources()
            
            return TestResult(
                success=True,
                test_name=test_config.name,
                duration_seconds=metrics["elapsed_seconds"],
                memory_usage_gb=metrics["memory_gb"],
                cpu_percent=metrics["cpu_percent"],
                metrics=metrics
            )
            
        except Exception as e:
            self.logger.error(f"Code execution test failed: {str(e)}")
            metrics = context.monitor_resources()
            
            return TestResult(
                success=False,
                test_name=test_config.name,
                duration_seconds=metrics["elapsed_seconds"],
                memory_usage_gb=metrics["memory_gb"],
                cpu_percent=metrics["cpu_percent"],
                error_message=str(e)
            )
    
    def _validate_generated_code(
        self,
        code_response: str,
        validation_rules: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Validate generated code against rules."""
        results = {
            "success": True,
            "metrics": {
                "rules_passed": 0,
                "rules_failed": 0,
                "coverage_percent": 0
            }
        }
        
        try:
            # Parse code
            import ast
            tree = ast.parse(code_response)
            
            # Analyze code
            analyzer = CodeAnalyzer()
            analyzer.visit(tree)
            
            # Check each validation rule
            for rule in validation_rules:
                rule_type = rule["type"]
                rule_params = rule["parameters"]
                
                if rule_type == "imports":
                    if not all(imp in analyzer.imports for imp in rule_params["required"]):
                        results["success"] = False
                        results["metrics"]["rules_failed"] += 1
                        continue
                
                elif rule_type == "functions":
                    if not all(func in analyzer.functions for func in rule_params["required"]):
                        results["success"] = False
                        results["metrics"]["rules_failed"] += 1
                        continue
                
                elif rule_type == "complexity":
                    if analyzer.complexity > rule_params["max_complexity"]:
                        results["success"] = False
                        results["metrics"]["rules_failed"] += 1
                        continue
                
                elif rule_type == "documentation":
                    if analyzer.doc_coverage < rule_params["min_coverage"]:
                        results["success"] = False
                        results["metrics"]["rules_failed"] += 1
                        continue
                
                results["metrics"]["rules_passed"] += 1
            
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


class CodeAnalyzer(ast.NodeVisitor):
    """Analyzes Python code for validation."""
    
    def __init__(self):
        self.imports = set()
        self.functions = set()
        self.complexity = 0
        self.doc_strings = 0
        self.total_nodes = 0
    
    def visit_Import(self, node):
        for name in node.names:
            self.imports.add(name.name)
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        for name in node.names:
            self.imports.add(f"{node.module}.{name.name}")
        self.generic_visit(node)
    
    def visit_FunctionDef(self, node):
        self.functions.add(node.name)
        if ast.get_docstring(node):
            self.doc_strings += 1
        self.complexity += 1  # Basic complexity metric
        self.generic_visit(node)
    
    def visit(self, node):
        self.total_nodes += 1
        super().visit(node)
    
    @property
    def doc_coverage(self):
        """Calculate documentation coverage."""
        return self.doc_strings / len(self.functions) if self.functions else 0


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