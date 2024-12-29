"""
Test registry mapping test names to their implementations.
"""

from typing import Dict, Any, Callable, Optional
from functools import wraps

from sidewinder.core.test_config import TestType, TestConfig
from sidewinder.testing.environment import TestContext


class TestRegistry:
    """Registry for test implementations."""
    
    def __init__(self):
        self._tests: Dict[TestType, Dict[str, Callable]] = {
            test_type: {} for test_type in TestType
        }
    
    def register(
        self,
        test_type: TestType,
        name: Optional[str] = None
    ):
        """
        Register a test implementation.
        
        Args:
            test_type: Type of test to register
            name: Optional name for the test (defaults to function name)
        """
        def decorator(func: Callable):
            @wraps(func)
            def wrapper(context: TestContext, config: TestConfig, **kwargs):
                return func(context, config, **kwargs)
            
            test_name = name or func.__name__
            self._tests[test_type][test_name] = wrapper
            return wrapper
        
        return decorator
    
    def get_test(
        self,
        test_type: TestType,
        name: str
    ) -> Optional[Callable]:
        """Get test implementation by type and name."""
        return self._tests.get(test_type, {}).get(name)
    
    def list_tests(self, test_type: Optional[TestType] = None) -> Dict[str, Any]:
        """List available tests."""
        if test_type:
            return {
                name: func.__doc__ for name, func in self._tests[test_type].items()
            }
        
        return {
            t.value: {
                name: func.__doc__
                for name, func in tests.items()
            }
            for t, tests in self._tests.items()
        }


# Create global registry
registry = TestRegistry()


# Example test implementations
@registry.register(TestType.DATA_QUALITY, "null_check")
def check_null_values(context: TestContext, config: TestConfig):
    """Check for null values in specified columns."""
    from pyspark.sql.functions import col, count, when
    
    df = context.spark.read.parquet(config.parameters["input_path"])
    threshold = config.parameters.get("threshold", 0.1)
    
    for column in config.parameters.get("columns", df.columns):
        null_count = df.filter(col(column).isNull()).count()
        null_ratio = null_count / df.count()
        
        assert null_ratio <= threshold, \
            f"Column {column} has {null_ratio:.1%} null values (threshold: {threshold:.1%})"


@registry.register(TestType.SCHEMA_VALIDATION, "required_fields")
def validate_required_fields(context: TestContext, config: TestConfig):
    """Validate presence and types of required fields."""
    df = context.spark.read.parquet(config.parameters["input_path"])
    required_fields = config.parameters["required_fields"]
    schema = config.parameters["schema"]
    
    # Check field presence
    missing_fields = [f for f in required_fields if f not in df.columns]
    assert not missing_fields, f"Missing required fields: {missing_fields}"
    
    # Check field types
    for field, expected_type in schema.items():
        actual_type = str(df.schema[field].dataType)
        assert actual_type == expected_type, \
            f"Field {field} has type {actual_type}, expected {expected_type}"


@registry.register(TestType.PERFORMANCE, "load_test")
def run_load_test(context: TestContext, config: TestConfig):
    """Run load test with concurrent users."""
    import time
    import concurrent.futures
    from pyspark.sql.functions import current_timestamp, lit
    
    def process_batch(batch_id: int):
        start_time = time.time()
        
        df = context.spark.read.parquet(
            f"{config.parameters['input_path']}/batch_{batch_id}"
        )
        
        # Apply test transformations
        df = df.withColumn("processed_at", current_timestamp())
        df = df.withColumn("batch_id", lit(batch_id))
        
        # Force computation
        count = df.count()
        
        return {
            "batch_id": batch_id,
            "duration_seconds": time.time() - start_time,
            "records_processed": count
        }
    
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.parameters["concurrent_users"]
    ) as executor:
        futures = [
            executor.submit(process_batch, i)
            for i in range(config.parameters["concurrent_users"])
        ]
        
        results = [f.result() for f in futures]
    
    # Calculate metrics
    durations = [r["duration_seconds"] for r in results]
    records = [r["records_processed"] for r in results]
    
    metrics = {
        "total_duration": max(durations),
        "avg_duration": sum(durations) / len(durations),
        "total_records": sum(records),
        "records_per_second": sum(records) / max(durations)
    }
    
    # Validate metrics
    assert metrics["records_per_second"] >= config.parameters["min_throughput"], \
        f"Throughput {metrics['records_per_second']:.0f} rps below threshold"


@registry.register(TestType.BUSINESS_RULES, "value_rules")
def check_business_rules(context: TestContext, config: TestConfig):
    """Check business rule conditions."""
    df = context.spark.read.parquet(config.parameters["input_path"])
    
    for rule in config.parameters["rules"]:
        invalid_count = df.filter(rule["condition"]).count()
        assert invalid_count == 0, \
            f"Rule '{rule['name']}' violated {invalid_count} times"


@registry.register(TestType.REFERENTIAL_INTEGRITY, "foreign_keys")
def check_referential_integrity(context: TestContext, config: TestConfig):
    """Check referential integrity between tables."""
    for rel in config.parameters["relationships"]:
        from_df = context.spark.read.parquet(rel["from_table"])
        to_df = context.spark.read.parquet(rel["to_table"])
        
        # Get distinct keys from both tables
        from_keys = from_df.select(rel["keys"]).distinct()
        to_keys = to_df.select(rel["keys"]).distinct()
        
        # Find invalid references
        invalid_keys = from_keys.subtract(to_keys)
        invalid_count = invalid_keys.count()
        
        assert invalid_count == 0, \
            f"Found {invalid_count} invalid references in {rel['from_table']}" 