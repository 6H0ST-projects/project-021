"""
High-level test configuration system for Sidewinder.
"""

from typing import Dict, Any, List, Optional, Union
from enum import Enum
from pydantic import BaseModel, Field

from sidewinder.core.config import Source, Target


class TestType(str, Enum):
    """Types of tests available in Sidewinder."""
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"
    TRANSFORMATION = "transformation"
    SCHEMA_VALIDATION = "schema_validation"
    DATA_CONSISTENCY = "data_consistency"
    REFERENTIAL_INTEGRITY = "referential_integrity"
    BUSINESS_RULES = "business_rules"
    LOAD_TEST = "load_test"
    INTEGRATION = "integration"
    CODE_GENERATION = "code_generation"
    CODE_EXECUTION = "code_execution"


class DataQualityPreset(str, Enum):
    """Preset data quality test configurations."""
    BASIC = "basic"  # Null checks, data types
    STANDARD = "standard"  # Basic + uniqueness, patterns
    STRICT = "strict"  # Standard + advanced validation
    CUSTOM = "custom"  # User-defined rules


class PerformancePreset(str, Enum):
    """Preset performance test configurations."""
    SMOKE = "smoke"  # Quick validation
    LOAD = "load"  # Moderate load testing
    STRESS = "stress"  # Heavy load testing
    ENDURANCE = "endurance"  # Long-running test
    CUSTOM = "custom"  # User-defined configuration


class CodeGenerationPreset(str, Enum):
    """Preset code generation test configurations."""
    BASIC = "basic"  # Basic functionality
    COMPREHENSIVE = "comprehensive"  # Full feature set
    EDGE_CASES = "edge_cases"  # Test edge cases
    CUSTOM = "custom"  # User-defined configuration


class LLMConfig(BaseModel):
    """LLM-specific configuration."""
    model: str = "gpt-4o"
    temperature: float = 0.2
    max_tokens: int = 4000
    max_retries: int = 3
    cache_dir: Optional[str] = None


class TestConfig(BaseModel):
    """Individual test configuration."""
    name: str
    type: TestType
    enabled: bool = True
    description: Optional[str] = None
    preset: Optional[Union[DataQualityPreset, PerformancePreset, CodeGenerationPreset]] = None
    parameters: Dict[str, Any] = {}
    dependencies: List[str] = []
    llm_config: Optional[LLMConfig] = None


class GlobalTestConfig(BaseModel):
    """Global test configuration for a Sidewinder pipeline."""
    
    # Pipeline configuration
    source: Source
    target: Optional[Target] = None
    environment: str = "development"
    
    # Test configuration
    test_data_location: str = "data/test"
    max_parallel_tests: int = 4
    max_parallel_suites: int = 2
    
    # Resource limits
    max_memory_gb: float = 4.0
    max_cpu_percent: float = 80
    timeout_seconds: int = 300
    
    # LLM configuration
    llm_config: Optional[LLMConfig] = None
    
    # Test definitions
    tests: List[TestConfig] = []
    
    class Config:
        use_enum_values = True

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "GlobalTestConfig":
        """Create configuration from dictionary."""
        return cls(**config)
    
    @classmethod
    def from_json(cls, json_file: str) -> "GlobalTestConfig":
        """Load configuration from JSON file."""
        import json
        with open(json_file, "r") as f:
            return cls.from_dict(json.load(f))


def create_example_config() -> Dict[str, Any]:
    """Create an example configuration showing all capabilities."""
    return {
        "source": {
            "name": "customer_data",
            "type": "parquet",
            "path": "data/customers.parquet"
        },
        "target": {
            "name": "enriched_customers",
            "type": "delta",
            "path": "data/enriched/customers"
        },
        "environment": "development",
        "test_data_location": "data/test",
        "max_parallel_tests": 4,
        "max_parallel_suites": 2,
        "tests": [
            {
                "name": "basic_quality",
                "type": "data_quality",
                "enabled": True,
                "preset": "standard",
                "parameters": {
                    "null_threshold": 0.1,
                    "unique_columns": ["id", "email"],
                    "pattern_rules": {
                        "email": "email",
                        "phone": "phone"
                    }
                }
            },
            {
                "name": "referential_check",
                "type": "referential_integrity",
                "enabled": True,
                "parameters": {
                    "relationships": [
                        {
                            "from_table": "orders",
                            "to_table": "customers",
                            "keys": ["customer_id"]
                        }
                    ]
                }
            },
            {
                "name": "business_rules",
                "type": "business_rules",
                "enabled": True,
                "parameters": {
                    "rules": [
                        {
                            "name": "valid_age",
                            "condition": "age >= 0 AND age <= 120"
                        },
                        {
                            "name": "valid_total",
                            "condition": "total_spend >= 0"
                        }
                    ]
                }
            },
            {
                "name": "load_test",
                "type": "performance",
                "enabled": True,
                "preset": "load",
                "parameters": {
                    "duration_seconds": 600,
                    "concurrent_users": 20,
                    "data_size": "10GB"
                }
            },
            {
                "name": "schema_check",
                "type": "schema_validation",
                "enabled": True,
                "parameters": {
                    "required_fields": [
                        "customer_id",
                        "email",
                        "total_orders"
                    ],
                    "schema": {
                        "customer_id": "string",
                        "email": "string",
                        "total_orders": "long"
                    }
                }
            }
        ]
    }


def save_example_config(output_file: str = "examples/test_config.json"):
    """Save example configuration to file."""
    import json
    from pathlib import Path
    
    config = create_example_config()
    
    # Create directory if it doesn't exist
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_file, "w") as f:
        json.dump(config, f, indent=2) 