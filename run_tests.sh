#!/bin/bash

# Check if .env.local file exists
if [ ! -f .env.local ]; then
    echo "Error: .env.local file not found"
    echo "Please create .env.local with your OpenAI API key"
    exit 1
fi

# Copy .env.local to .env for Docker
echo "Copying .env.local to .env for Docker..."
cp .env.local .env

# Build and run the Docker containers
echo "Building and starting Docker containers..."
docker-compose up --build

# Check if the tests were successful
EXIT_CODE=$?

# Clean up
echo "Cleaning up Docker containers..."
docker-compose down

# Remove the temporary .env file
rm .env

exit $EXIT_CODE 