# Sidewinder

A data engineering pipeline generator using LLMs and Spark SQL.

## Prerequisites

- Docker
- Docker Compose
- OpenAI API key

## Running Tests

1. Clone the repository:
```bash
git clone <repository-url>
cd project-021
```

2. Add your OpenAI API key to `.env` file:
```bash
echo "OPENAI_API_KEY=your-api-key-here" > .env
```

3. Run the tests:
```bash
./run_tests.sh
```

This will:
- Build the Docker image with all required dependencies
- Start the container with proper Spark configuration
- Run the end-to-end tests
- Clean up the containers when done

## Development

The project uses Docker to ensure consistent development environments. The setup includes:

- Python 3.11
- PySpark 3.4.1
- All required Python packages
- Java 17 (required for Spark)

### Project Structure

```
.
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker services configuration
├── requirements.txt        # Python dependencies
├── run_tests.sh           # Test runner script
├── sidewinder/            # Main package directory
│   ├── agents/           # Agent implementations
│   ├── core/             # Core functionality
│   └── testing/          # Test utilities
└── tests/                # Test files
```

### Spark Configuration

The Docker environment comes pre-configured with:
- 2 worker cores
- 4GB worker memory
- 4GB driver memory
- 4GB executor memory
- Spark UI accessible at http://localhost:4040

### Adding New Tests

1. Add your test scenarios to `tests/config/test_scenarios.json`
2. Add test data to `tests/data/`
3. Run the tests using `./run_tests.sh`

## Troubleshooting

If you encounter issues:

1. Check the logs in `logs/` directory
2. Verify your OpenAI API key in `.env`
3. Ensure Docker has enough resources allocated
4. Check Spark UI at http://localhost:4040 for execution details
