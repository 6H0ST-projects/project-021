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
from sidewinder.testing.executor import TestExecutor, GlobalTestConfig
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


async def run_scenario(
    scenario: Dict[str, Any],
    config: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run a single test scenario."""
    try:
        # Create source and target objects
        source = Source(**scenario["source"])
        target = Target(**scenario["target"])
        
        # Generate test data
        logger.info("Generating sample test data...")
        test_data = generate_test_data(source)
        logger.info("Sample data generation complete")
        
        # Create pipeline
        logger.info("Starting test scenarios...")
        logger.info(f"Running scenario: {scenario['name']}")
        
        # Initialize LLM pipeline
        logger.info("Starting LLM code generation pipeline...")
        llm_agent = DataEngineeringAgent()
        
        # Convert test data to string
        test_data_str = test_data.to_string() if isinstance(test_data, pd.DataFrame) else str(test_data)
        
        # Generate code for each component
        analyzer_code = await llm_agent.generate_analyzer_code(
            source_type=source.type,
            source_data=test_data_str
        )
        
        transformer_code = await llm_agent.generate_transformer_code(
            source_type=source.type,
            source_data=test_data_str,
            target_schema=target.target_schema
        )
        
        tester_code = await llm_agent.generate_tester_code(
            source_type=source.type,
            source_data=test_data_str,
            target_schema=target.target_schema,
            test_types=scenario.get("test_types", [])
        )
        
        # Create pipeline
        logger.info("Generating pipeline...")
        pipeline = Pipeline(
            source=source,
            target=target,
            environment=config["environment"],
            analyzer_code=analyzer_code,
            transformer_code=transformer_code,
            tester_code=tester_code
        )
        
        # Generate pipeline
        await pipeline.generate()
        
        # Run tests
        logger.info("Running tests...")
        test_executor = TestExecutor(GlobalTestConfig(**config))
        test_report = await test_executor.run_tests()
        
        return test_report.model_dump()
        
    except Exception as e:
        logger.error(f"Scenario {scenario['name']} failed: {str(e)}")
        raise


async def run_tests() -> None:
    """Run all test scenarios."""
    try:
        # Load configuration
        config = load_config("tests/config/test_config.json")
        
        # Setup logging
        logger = setup_logging(config["output_dir"])
        
        # Run scenarios
        results = []
        for scenario in config["scenarios"]:
            result = await run_scenario(scenario, config, logger)
            results.append(result)
        
        # Save results
        output_path = Path(config["output_dir"]) / "test_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
            
    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(run_tests()) 