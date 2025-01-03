FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    curl \
    gcc \
    python3-dev \
    procps \
    net-tools \
    && rm -rf /var/lib/apt/lists/* \
    && java -version

# Set Java environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Set environment variables for Spark
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.7-src.zip:$PYTHONPATH
ENV PATH=$SPARK_HOME/bin:$PATH

# Set up logging directory
RUN mkdir -p /app/logs

# Default command
CMD ["python", "tests/run_e2e_tests.py"] 