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

# Load environment variables from .env.local
load_dotenv('.env.local')

from sidewinder.core.config import Source, Target
from sidewinder.core.pipeline import Pipeline
from sidewinder.testing.executor import run_tests_sync
from sidewinder.testing.data_gen import generate_test_data
from sidewinder.core.llm import DataEngineeringAgent, LLMCodeGenerator
from sidewinder.testing.metrics import PerformanceMetrics


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


def generate_sample_data(config: Dict[str, Any], logger: logging.Logger):
    """Generate sample data if enabled."""
    if not config["test_data"]["generation"]["enabled"]:
        return
    
    gen_config = config["test_data"]["generation"]
    output_dir = gen_config["output_dir"]
    
    logger.info("Generating sample test data...")
    
    # Generate data for each scenario
    for scenario in config["scenarios"]:
        source_config = Source(**scenario["source"])
        
        # Generate test data
        generate_test_data(
            source=source_config,
            output_path=f"{output_dir}/{scenario['name']}.{source_config.format}",
            sample_size=gen_config["sample_size"],
            seed=gen_config["seed"]
        )
    
    logger.info("Sample data generation complete")


async def run_llm_pipeline(
    scenario: Dict[str, Any],
    logger: logging.Logger,
    metrics: PerformanceMetrics
) -> Dict[str, Any]:
    """Run LLM-based pipeline generation and execution."""
    agent = DataEngineeringAgent()
    source = Source(**scenario["source"])
    target = Target(**scenario["target"])
    
    logger.info("Starting LLM code generation pipeline...")
    
    # Load sample data
    if source.type == "file":
        if source.format == "json":
            with open(source.location) as f:
                json_data = json.load(f)
            # Handle nested JSON structures
            if isinstance(json_data, dict):
                for key, value in json_data.items():
                    if isinstance(value, list):
                        json_data = value
                        break
            sample_data = pd.DataFrame(json_data)
        elif source.format == "parquet":
            sample_data = pd.read_parquet(source.location)
        elif source.format == "csv":
            sample_data = pd.read_csv(source.location)
        else:
            raise ValueError(f"Unsupported file format: {source.format}")
    else:
        raise ValueError(f"Unsupported source type: {source.type}")
    
    # Track LLM performance metrics
    llm_metrics = {
        "analyzer": {"time": 0, "tokens": 0},
        "transformer": {"time": 0, "tokens": 0},
        "tester": {"time": 0, "tokens": 0}
    }
    
    try:
        # Generate analyzer code
        start_time = time.time()
        analyzer_code = await agent.analyze_data_source(
            source_type=source.type,
            sample_data=sample_data,
            context={"purpose": "analyze_data"}
        )
        llm_metrics["analyzer"]["time"] = time.time() - start_time
        metrics.record_llm_metrics("analyzer", llm_metrics["analyzer"])
        
        # Generate transformer code
        start_time = time.time()
        transformer_code = await agent.generate_transformation(
            source_type=source.type,
            source_data=sample_data,
            target_schema=target.target_schema,
            context={"layer": "gold", "purpose": "transform_data"}
        )
        llm_metrics["transformer"]["time"] = time.time() - start_time
        metrics.record_llm_metrics("transformer", llm_metrics["transformer"])
        
        # Generate test code
        start_time = time.time()
        test_code = await agent.generate_test_cases(
            source_type=source.type,
            source_data=sample_data,
            target_schema=target.target_schema,
            context={"test_types": scenario["tests"]}
        )
        llm_metrics["tester"]["time"] = time.time() - start_time
        metrics.record_llm_metrics("tester", llm_metrics["tester"])
        
        return {
            "analyzer_code": analyzer_code,
            "transformer_code": transformer_code,
            "test_code": test_code,
            "metrics": llm_metrics
        }
        
    except Exception as e:
        logger.error(f"LLM code generation failed: {str(e)}")
        raise


async def run_scenario(
    scenario: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run a single test scenario."""
    logger.info(f"Running scenario: {scenario['name']}")
    
    # Initialize performance metrics
    metrics = PerformanceMetrics()
    
    try:
        # Run LLM pipeline
        llm_results = await run_llm_pipeline(scenario, logger, metrics)
        
        # Create pipeline with generated code
        pipeline = Pipeline(
            source=Source(**scenario["source"]),
            target=Target(**scenario["target"]),
            analyzer_code=llm_results["analyzer_code"],
            transformer_code=llm_results["transformer_code"]
        )
        
        # Generate pipeline
        logger.info("Generating pipeline...")
        pipeline.generate()
        
        # Run tests with generated test code
        logger.info("Running tests...")
        test_config = {
            "source": scenario["source"],
            "target": scenario["target"],
            "tests": scenario["tests"],
            "test_code": llm_results["test_code"]
        }
        
        with open("temp_test_config.json", "w") as f:
            json.dump(test_config, f)
        
        results = run_tests_sync("temp_test_config.json")
        Path("temp_test_config.json").unlink()
        
        # Add performance metrics
        metrics.record_execution_metrics(results)
        
        return {
            "scenario": scenario["name"],
            "results": results.to_dict(),
            "llm_metrics": llm_results["metrics"],
            "performance_metrics": metrics.to_dict()
        }
        
    except Exception as e:
        logger.error(f"Scenario {scenario['name']} failed: {str(e)}")
        return {
            "scenario": scenario["name"],
            "error": str(e),
            "performance_metrics": metrics.to_dict()
        }


async def run_all_scenarios(
    config: Dict[str, Any],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Run all test scenarios."""
    results = []
    
    for scenario in config["scenarios"]:
        result = await run_scenario(scenario, logger)
        results.append(result)
    
    return results


def save_results(
    results: List[Dict[str, Any]],
    config: Dict[str, Any],
    logger: logging.Logger
):
    """Save test results."""
    output_dir = config["reporting"]["output_dir"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save JSON results
    if "json" in config["reporting"]["formats"]:
        with open(f"{output_dir}/test_results.json", "w") as f:
            json.dump(results, f, indent=2)
    
    # Save HTML report
    if "html" in config["reporting"]["formats"]:
        from sidewinder.testing.reporting import TestReport
        report = TestReport([r["results"] for r in results if "results" in r])
        report.generate_html_report(output_dir)
        
        # Generate performance report
        report.generate_performance_report(
            [r["performance_metrics"] for r in results],
            output_dir
        )
    
    logger.info(f"Results saved to {output_dir}")


def main():
    """Run end-to-end tests."""
    # Load configuration
    config = load_config("tests/config/test_scenarios.json")
    
    # Setup logging
    logger = setup_logging(config["reporting"]["output_dir"])
    
    try:
        # Generate sample data
        generate_sample_data(config, logger)
        
        # Run scenarios
        logger.info("Starting test scenarios...")
        results = asyncio.run(run_all_scenarios(config, logger))
        
        # Save results
        save_results(results, config, logger)
        
        # Check for failures
        failed = any(
            "error" in r or 
            ("results" in r and not r["results"]["success_rate"] == 1.0)
            for r in results
        )
        
        if failed:
            logger.error("Some tests failed. Check results for details.")
            exit(1)
        else:
            logger.info("All tests passed successfully!")
            exit(0)
        
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main() 