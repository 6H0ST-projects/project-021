"""
End-to-end test runner for Sidewinder.
"""

import json
import logging
from pathlib import Path
import asyncio
from typing import Dict, Any, List
import time
from dotenv import load_dotenv
import pandas as pd
from datetime import datetime
import webbrowser

# Load environment variables from .env.local
load_dotenv('.env.local')

from sidewinder.core.config import Source, Target
from sidewinder.core.pipeline import Pipeline
from sidewinder.testing.executor import TestExecutor, GlobalTestConfig
from sidewinder.testing.data_gen import generate_test_data, generate_scalability_test_data
from sidewinder.core.llm import DataEngineeringAgent, LLMCodeGenerator
from sidewinder.testing.metrics import PerformanceMetrics
from sidewinder.testing.reporting import TestReport, TestSuiteResult, TestResult, RequirementStatus


def load_config(config_path: str) -> Dict[str, Any]:
    """Load test configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def setup_logging(output_dir: str) -> logging.Logger:
    """Set up logging for test execution."""
    logger = logging.getLogger("sidewinder.testing")
    logger.setLevel(logging.INFO)
    
    # Create handlers
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(f"{output_dir}/test_execution.log")
    console_handler = logging.StreamHandler()
    
    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


async def run_scenario(
    scenario: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run a single test scenario."""
    start_time = datetime.now()
    try:
        # Create source and target objects
        source = Source(**scenario["source"])
        target = Target(**scenario["target"])
        
        # Generate test data
        logger.info("Generating sample test data...")
        test_data = generate_test_data(source)
        logger.info("Sample data generation complete")
        
        # Generate scalability test data if specified
        scalability_data = None
        if scenario.get("scalability_testing", False):
            logger.info("Generating scalability test data...")
            scale_factors = scenario.get("scale_factors", [1, 10, 100])
            base_size = scenario.get("base_size", 1000)
            scalability_data = generate_scalability_test_data(
                source,
                scale_factors=scale_factors,
                base_size=base_size
            )
            logger.info(f"Generated test data at scales: {list(scalability_data.keys())}x")
        
        # Create pipeline
        logger.info("Starting test scenarios...")
        logger.info(f"Running scenario: {scenario['name']}")
        
        # Initialize LLM pipeline
        logger.info("Starting LLM code generation pipeline...")
        llm_agent = DataEngineeringAgent()
        
        # Convert test data to string
        test_data_str = test_data.to_string() if isinstance(test_data, pd.DataFrame) else str(test_data)
        
        # Generate code for each component
        # First, get the analyzer's description of the data
        analyzer_result = await llm_agent.generate_analyzer_code(
            source_type=source.type,
            source_data=test_data_str
        )
        
        # Then, use the analyzer's output to generate all layer transformations
        transformation_code = await llm_agent.generate_transformer_code(
            source_type=source.type,
            source_data=test_data_str,
            analyzer_output=analyzer_result,
            target_schema=target.target_schema if hasattr(target, 'target_schema') else None
        )
        
        # Finally, generate tests to validate the transformations
        tester_code = await llm_agent.generate_tester_code(
            source_type=source.type,
            source_data=test_data_str,
            transformation_code=transformation_code,
            target_schema=target.target_schema if hasattr(target, 'target_schema') else None,
            test_types=scenario.get("test_types", [])
        )
        
        # Create pipeline
        logger.info("Generating pipeline...")
        pipeline = Pipeline(
            source=source,
            target=target,
            environment=config["environment"],
            analyzer_code=analyzer_result,
            transformer_code=transformation_code,
            tester_code=tester_code
        )
        
        # Generate pipeline
        await pipeline.generate()
        
        # Run tests
        logger.info("Running tests...")
        test_executor = TestExecutor(GlobalTestConfig(**config))
        test_report = await test_executor.run_tests()
        
        # Run scalability tests if data is available
        scalability_results = None
        if scalability_data:
            logger.info("Running scalability tests...")
            scalability_results = {}
            for scale, data in scalability_data.items():
                logger.info(f"Testing at scale {scale}x...")
                scale_report = await test_executor.run_tests(test_data=data)
                scalability_results[scale] = {
                    "success_rate": scale_report.suites[0].success_rate,
                    "duration": scale_report.suites[0].total_duration_seconds,
                    "peak_memory_gb": scale_report.suites[0].peak_memory_gb,
                    "avg_cpu_percent": scale_report.suites[0].avg_cpu_percent,
                    "join_metrics": scale_report.suites[0].join_metrics if hasattr(scale_report.suites[0], 'join_metrics') else None
                }
        
        # Create test result
        test_result = TestResult(
            name=scenario['name'],
            type="e2e",
            status="passed" if test_report.suites[0].success_rate == 1.0 else "failed",
            duration=test_report.suites[0].total_duration_seconds,
            error=next((r.error_message for r in test_report.suites[0].results if r.error_message), None),
            details={
                "success_rate": test_report.suites[0].success_rate,
                "analyzer_findings": transformation_code.analyzer_findings if hasattr(transformation_code, 'analyzer_findings') else None,
                "scalability_results": scalability_results
            },
            metrics={
                "peak_memory_gb": test_report.suites[0].peak_memory_gb,
                "avg_cpu_percent": test_report.suites[0].avg_cpu_percent,
                "join_performance": test_report.suites[0].join_metrics if hasattr(test_report.suites[0], 'join_metrics') else None
            },
            requirements=[
                RequirementStatus(
                    requirement=result.test_name,
                    status="passed" if result.success else "failed",
                    details=result.error_message if not result.success else None,
                    code_snippet=result.test_code if hasattr(result, 'test_code') else None
                )
                for result in test_report.suites[0].results
            ],
            generated_code={
                "bronze": transformation_code.bronze_layer if hasattr(transformation_code, 'bronze_layer') else None,
                "silver": transformation_code.silver_layer if hasattr(transformation_code, 'silver_layer') else None,
                "gold": transformation_code.gold_layer if hasattr(transformation_code, 'gold_layer') else None
            },
            test_results={
                result.test_name: {
                    "passed": result.success,
                    "error": result.error_message if not result.success else None,
                    "duration": result.duration_seconds if hasattr(result, 'duration_seconds') else None,
                    "assertions": result.assertions if hasattr(result, 'assertions') else None,
                    "performance_metrics": result.performance_metrics if hasattr(result, 'performance_metrics') else None
                }
                for result in test_report.suites[0].results
            }
        )
        
        return test_result
        
    except Exception as e:
        logger.error(f"Scenario {scenario['name']} failed: {str(e)}")
        return TestResult(
            name=scenario['name'],
            type="e2e",
            status="failed",
            duration=time.time() - start_time.timestamp(),
            error=str(e)
        )


async def run_tests() -> None:
    """Run all test scenarios."""
    try:
        # Load configuration
        config = load_config("tests/config/test_config.json")
        
        # Setup logging
        logger = setup_logging(config["output_dir"])
        
        # Run scenarios
        start_time = datetime.now()
        test_results = []
        for scenario in config["scenarios"]:
            result = await run_scenario(scenario, config, logger)
            test_results.append(result)
        end_time = datetime.now()
        
        # Create test suite result
        suite_result = TestSuiteResult(
            name="E2E Test Suite",
            start_time=start_time,
            end_time=end_time,
            tests=test_results,
            success_rate=len([t for t in test_results if t.status == "passed"]) / len(test_results),
            total_duration=(end_time - start_time).total_seconds()
        )
        
        # Generate report
        report = TestReport(results=[suite_result])
        output_dir = Path(config["output_dir"])
        report.generate_html_report(str(output_dir))
        
        # Display results to user
        main_report_path = output_dir / "test_report.html"
        req_report_path = output_dir / "requirements_report.html"
        code_report_path = output_dir / "code_report.html"
        
        print("\nTest execution completed!")
        print(f"Success Rate: {suite_result.success_rate * 100:.1f}%")
        print(f"Total Duration: {suite_result.total_duration:.2f}s")
        print(f"\nDetailed reports have been generated:")
        print(f"- Main Report: {main_report_path}")
        print(f"- Requirements Report: {req_report_path}")
        print(f"- Code Report: {code_report_path}")
        
        # Open reports in browser
        webbrowser.open(f"file://{main_report_path.absolute()}")
        webbrowser.open(f"file://{req_report_path.absolute()}")
        webbrowser.open(f"file://{code_report_path.absolute()}")
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_tests()) 