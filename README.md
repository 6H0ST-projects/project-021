# Sidewinder

A powerful ETL pipeline generator with intelligent data analysis and transformation capabilities.

## Testing Framework

Sidewinder provides a comprehensive testing framework that allows you to define tests through configuration. Tests are automatically executed in parallel while respecting dependencies and resource constraints.

### Test Configuration

Tests are defined in a JSON configuration file that specifies both the data pipeline and testing parameters:

```json
{
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
  "max_memory_gb": 4.0,
  "max_cpu_percent": 80,
  "timeout_seconds": 300,
  "tests": [
    // Test definitions here
  ]
}
```

### Available Test Types

1. **Data Quality Tests** (`data_quality`)
   - Validates data quality aspects like null values, uniqueness, and patterns
   - Presets: basic, standard, strict, custom
   ```json
   {
     "name": "customer_quality",
     "type": "data_quality",
     "preset": "standard",
     "parameters": {
       "null_threshold": 0.1,
       "unique_columns": ["id", "email"],
       "pattern_rules": {
         "email": "email",
         "phone": "phone"
       }
     }
   }
   ```

2. **Schema Validation** (`schema_validation`)
   - Validates data schema, required fields, and data types
   ```json
   {
     "name": "schema_check",
     "type": "schema_validation",
     "parameters": {
       "required_fields": ["customer_id", "email"],
       "schema": {
         "customer_id": "string",
         "email": "string",
         "total_orders": "long"
       }
     }
   }
   ```

3. **Referential Integrity** (`referential_integrity`)
   - Checks relationships between tables
   ```json
   {
     "name": "foreign_keys",
     "type": "referential_integrity",
     "parameters": {
       "relationships": [
         {
           "from_table": "orders",
           "to_table": "customers",
           "keys": ["customer_id"]
         }
       ]
     }
   }
   ```

4. **Business Rules** (`business_rules`)
   - Validates business logic and constraints
   ```json
   {
     "name": "business_validation",
     "type": "business_rules",
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
   }
   ```

5. **Performance Tests** (`performance`)
   - Tests performance under various conditions
   - Presets: smoke, load, stress, endurance, custom
   ```json
   {
     "name": "load_test",
     "type": "performance",
     "preset": "load",
     "parameters": {
       "duration_seconds": 600,
       "concurrent_users": 20,
       "data_size": "10GB",
       "min_throughput": 1000
     }
   }
   ```

6. **Data Consistency** (`data_consistency`)
   - Validates data consistency across transformations
   ```json
   {
     "name": "consistency_check",
     "type": "data_consistency",
     "parameters": {
       "checkpoints": [
         {
           "name": "bronze_to_silver",
           "source_query": "SELECT COUNT(*) FROM bronze",
           "target_query": "SELECT COUNT(*) FROM silver",
           "expected": "equal"
         }
       ]
     }
   }
   ```

### Test Dependencies

Tests can specify dependencies to ensure proper execution order:

```json
{
  "name": "final_validation",
  "type": "data_quality",
  "dependencies": ["schema_check", "business_validation"],
  "parameters": {
    // Test parameters
  }
}
```

### Running Tests

```python
from sidewinder.testing.executor import run_tests_sync

# Run tests and get report
report = run_tests_sync("test_config.json")

# Print summary
report.print_summary()

# Generate detailed HTML report
report.generate_html_report("test_results")
```

### Test Reports

Test results include:
- Success/failure status
- Execution duration
- Resource usage (memory, CPU)
- Detailed error messages
- Performance metrics
- Interactive visualizations

### Best Practices

1. **Test Organization**
   - Group related tests using dependencies
   - Use presets for common scenarios
   - Create reusable test configurations

2. **Resource Management**
   - Configure appropriate resource limits
   - Monitor performance metrics
   - Adjust parallelism based on system capacity

3. **Test Data**
   - Use representative test data
   - Maintain test data versioning
   - Document data generation procedures

4. **Maintenance**
   - Review and update test configurations regularly
   - Monitor test execution times
   - Analyze failure patterns
   - Update thresholds based on changing requirements
