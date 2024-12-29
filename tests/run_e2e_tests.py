"""
End-to-end test runner for Sidewinder.
"""

import json
import logging
from pathlib import Path
import asyncio
from typing import Dict, Any, List

from sidewinder.core.config import Source, Target
from sidewinder.core.pipeline import Pipeline
from sidewinder.testing.executor import run_tests_sync
from sidewinder.testing.data_gen import generate_test_data


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
    
    # Generate data for each source
    for scenario in config["scenarios"]:
        for source in scenario["source"]["sources"]:
            source_config = Source(**source)
            
            generate_test_data(
                source=source_config,
                output_path=f"{output_dir}/{source['name']}.parquet",
                sample_size=gen_config["sample_size"],
                seed=gen_config["seed"]
            )
    
    logger.info("Sample data generation complete")


def run_scenario(
    scenario: Dict[str, Any],
    logger: logging.Logger
) -> Dict[str, Any]:
    """Run a single test scenario."""
    logger.info(f"Running scenario: {scenario['name']}")
    
    # Create pipeline
    pipeline = Pipeline(
        source=Source(**scenario["source"]),
        target=Target(**scenario["target"])
    )
    
    # Generate pipeline
    logger.info("Generating pipeline...")
    pipeline.generate()
    
    # Run tests
    logger.info("Running tests...")
    test_config = {
        "source": scenario["source"],
        "target": scenario["target"],
        "tests": scenario["tests"]
    }
    
    with open("temp_test_config.json", "w") as f:
        json.dump(test_config, f)
    
    results = run_tests_sync("temp_test_config.json")
    Path("temp_test_config.json").unlink()
    
    return {
        "scenario": scenario["name"],
        "results": results.to_dict()
    }


async def run_all_scenarios(
    config: Dict[str, Any],
    logger: logging.Logger
) -> List[Dict[str, Any]]:
    """Run all test scenarios."""
    results = []
    
    for scenario in config["scenarios"]:
        try:
            result = await asyncio.to_thread(run_scenario, scenario, logger)
            results.append(result)
        except Exception as e:
            logger.error(f"Scenario {scenario['name']} failed: {str(e)}")
            results.append({
                "scenario": scenario["name"],
                "error": str(e)
            })
    
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
            "error" in r or not r["results"]["success_rate"] == 1.0
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