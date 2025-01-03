# Sidewinder

A data engineering pipeline generator powered by LLMs. Sidewinder analyzes, transforms, and tests data pipelines using natural language instructions.

## Features

- 🔍 Automated data analysis and profiling
- 🔄 Smart data transformation generation
- ✅ Automated test case generation and execution
- 📊 Comprehensive metrics and reporting
- 🚀 Docker-based execution environment

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sidewinder.git
   cd sidewinder
   ```

2. Set up your environment:
   ```bash
   # Create .env.local file with your OpenAI API key
   echo "OPENAI_API_KEY=your-api-key-here" > .env.local
   ```

## Development Environment

### Local Setup (Optional)

If you want to develop or run tests locally without Docker:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Docker Environment (Recommended)

The project uses Docker to ensure consistent environments and handle dependencies:

1. **Environment Configuration**:
   - Java 17 (for Spark)
   - Python 3.11
   - PySpark 3.4.1
   - All Python dependencies

2. **Resource Configuration** (in `docker-compose.yml`):
   ```yaml
   environment:
     - SPARK_WORKER_CORES=2
     - SPARK_WORKER_MEMORY=4g
     - SPARK_DRIVER_MEMORY=4g
     - SPARK_EXECUTOR_MEMORY=4g
   ```

3. **Volume Mappings**:
   ```yaml
   volumes:
     - .:/app                        # Project root
     - ./logs:/app/logs              # Logs directory
     - ./tests/output:/app/tests/output  # Test results
   ```

4. **Ports**:
   - Spark UI: http://localhost:4040

5. **Build and Run**:
   ```bash
   # Build the Docker image
   docker-compose build

   # Start the services
   docker-compose up

   # Run in detached mode
   docker-compose up -d

   # View logs
   docker-compose logs -f

   # Stop services
   docker-compose down
   ```

## Configuration

### Test Configuration

Create or modify test configurations in `tests/config/test_config.json`:

```json
{
    "environment": "local",
    "output_dir": "tests/output",
    "max_parallel_tests": 4,
    "max_parallel_suites": 2,
    "max_memory_gb": 4.0,
    "max_cpu_percent": 80.0,
    "timeout_seconds": 300,
    "llm_config": {
        "model": "gpt-4o",
        "temperature": 0.2,
        "max_tokens": 4000,
        "max_retries": 3
    },
    "scenarios": [
        {
            "name": "your_scenario_name",
            "source": {
                "type": "file",
                "format": "json",
                "location": "path/to/source.json"
            },
            "target": {
                "type": "file",
                "format": "json",
                "location": "path/to/target_schema.json"
            },
            "tests": [
                {
                    "name": "your_test_name",
                    "type": "code_generation",
                    "task": "Your task description",
                    "context": {
                        "source_type": "file",
                        "source_data": "path/to/source.json"
                    },
                    "requirements": [
                        "List your requirements here"
                    ],
                    "validation_rules": [
                        {
                            "type": "function_exists",
                            "name": "your_function_name"
                        },
                        {
                            "type": "contains",
                            "pattern": "import pandas"
                        }
                    ]
                }
            ]
        }
    ]
}
```

### Directory Structure

```
sidewinder/
├── tests/
│   ├── config/          # Test configurations
│   ├── data/           # Test data files
│   │   ├── source/     # Source data for tests
│   │   └── target/     # Target schemas
│   └── output/         # Test results and logs
├── sidewinder/         # Main package code
└── docs/              # Documentation
```

## Running Tests

The project includes a test runner script that handles Docker operations:

```bash
chmod +x run_tests.sh
./run_tests.sh
```

This script will:
1. Check for `.env.local` file
2. Copy environment variables for Docker
3. Build and start the Docker container
4. Run the test suite
5. Save results to `tests/output/`
6. Clean up containers

Test results will be saved in `tests/output/`:
- `test_results.json`: Detailed test results
- `test_execution.log`: Execution logs

## Troubleshooting

1. **Docker Issues**:
   - Ensure Docker daemon is running
   - Check resource allocation in Docker Desktop
   - Verify port 4040 is available for Spark UI

2. **Environment Issues**:
   - Verify `.env.local` exists with valid API key
   - Check Docker logs: `docker-compose logs`
   - Inspect container: `docker-compose exec sidewinder bash`

3. **Test Failures**:
   - Check `tests/output/test_execution.log`
   - View Spark UI at http://localhost:4040
   - Verify test data exists in correct locations

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
