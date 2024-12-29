"""
Data analyzer agent for inspecting and understanding source data.
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field
import json
import importlib
import inspect
import pandas as pd

from sidewinder.agents.base import BaseAgent, BaseAgentState
from sidewinder.core.config import Source, SourceType


class AnalysisStep(BaseModel):
    """A single step in the data analysis process."""
    name: str
    description: str
    code: str
    dependencies: List[str] = Field(default_factory=list)
    results: Optional[Dict[str, Any]] = None


class AnalyzerState(BaseAgentState):
    """State for the analyzer agent."""
    source: Source
    analysis_steps: List[AnalysisStep] = Field(default_factory=list)
    schema: Optional[Dict[str, Any]] = None
    sample_data: Optional[Dict[str, Any]] = None
    data_quality_issues: list[str] = Field(default_factory=list)
    inferred_data_types: Optional[Dict[str, str]] = None
    statistics: Optional[Dict[str, Dict[str, Any]]] = None


class DataAnalyzer(BaseAgent[AnalyzerState]):
    """Agent responsible for analyzing source data and determining its characteristics."""
    
    def _generate_connection_code(self, source: Source) -> str:
        """Generate source-specific connection code."""
        # File-based sources
        if source.type == SourceType.FILE:
            return """
import pandas as pd
import json
from pathlib import Path

def connect_to_source(source):
    path = Path(source.location)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
        
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    elif path.suffix.lower() == '.parquet':
        return pd.read_parquet(path)
    elif path.suffix.lower() == '.json':
        return pd.read_json(path)
    elif path.suffix.lower() == '.avro':
        import avro.datafile
        with avro.datafile.DataFileReader(path.open('rb')) as reader:
            return pd.DataFrame.from_records(list(reader))
    else:
        with open(path, 'r') as f:
            return json.load(f)
"""
        elif source.type == SourceType.DIRECTORY:
            return """
import pandas as pd
import glob
from pathlib import Path

def connect_to_source(source):
    path = Path(source.location)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")
    
    pattern = f"*.{source.format}" if source.format else "*"
    files = glob.glob(str(path / pattern))
    
    dfs = []
    for file in files[:10]:  # Limit initial load to 10 files
        if file.endswith('.csv'):
            dfs.append(pd.read_csv(file))
        elif file.endswith('.parquet'):
            dfs.append(pd.read_parquet(file))
        elif file.endswith('.json'):
            dfs.append(pd.read_json(file))
    
    return pd.concat(dfs) if dfs else pd.DataFrame()
"""
        # Cloud Storage
        elif source.type in [SourceType.S3, SourceType.GCS, SourceType.AZURE_BLOB, SourceType.DBFS]:
            cloud_imports = {
                SourceType.S3: "import boto3",
                SourceType.GCS: "from google.cloud import storage",
                SourceType.AZURE_BLOB: "from azure.storage.blob import BlobServiceClient",
                SourceType.DBFS: "import databricks.sdk"
            }
            
            return f"""
{cloud_imports[source.type]}
import pandas as pd
import io

def connect_to_source(source):
    if source.type == 'S3':
        s3 = boto3.client('s3')
        bucket, key = source.location.replace('s3://', '').split('/', 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        data = obj['Body'].read()
    elif source.type == 'GCS':
        client = storage.Client()
        bucket_name, blob_name = source.location.replace('gs://', '').split('/', 1)
        bucket = client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        data = blob.download_as_bytes()
    elif source.type == 'AZURE_BLOB':
        account_url = source.options.get('account_url')
        container, blob_name = source.location.replace('wasbs://', '').split('@')[0].split('/', 1)
        client = BlobServiceClient.from_connection_string(source.options['connection_string'])
        container_client = client.get_container_client(container)
        blob_client = container_client.get_blob_client(blob_name)
        data = blob_client.download_blob().readall()
    
    # Handle different file formats
    if source.location.endswith('.csv'):
        return pd.read_csv(io.BytesIO(data))
    elif source.location.endswith('.parquet'):
        return pd.read_parquet(io.BytesIO(data))
    elif source.location.endswith('.json'):
        return pd.read_json(io.BytesIO(data))
    else:
        return json.loads(data.decode('utf-8'))
"""
        # Databases
        elif source.type in [SourceType.POSTGRES, SourceType.MYSQL, SourceType.SNOWFLAKE, 
                           SourceType.BIGQUERY, SourceType.REDSHIFT, SourceType.DATABRICKS_SQL]:
            return """
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse

def connect_to_source(source):
    config = source.database_config
    
    # Build connection string based on database type
    if source.type == 'POSTGRES':
        conn_str = f"postgresql://{config.host}:{config.port}/{config.database}"
    elif source.type == 'MYSQL':
        conn_str = f"mysql+pymysql://{config.host}:{config.port}/{config.database}"
    elif source.type == 'SNOWFLAKE':
        conn_str = f"snowflake://{config.host}/{config.database}"
    elif source.type == 'BIGQUERY':
        conn_str = f"bigquery://{config.project}/{config.database}"
    elif source.type == 'REDSHIFT':
        conn_str = f"redshift+psycopg2://{config.host}:{config.port}/{config.database}"
    
    engine = create_engine(conn_str, **(config.connection_params or {}))
    
    # Use query if provided, otherwise use table
    query = config.query if config.query else f"SELECT * FROM {config.table}"
    if source.filter_condition:
        query += f" WHERE {source.filter_condition}"
    if source.partition_column:
        query += f" ORDER BY {source.partition_column}"
    query += f" LIMIT {source.sample_size}"
    
    return pd.read_sql(query, engine)
"""
        # MongoDB needs special handling
        elif source.type == SourceType.MONGODB:
            return """
import pandas as pd
from pymongo import MongoClient

def connect_to_source(source):
    config = source.database_config
    client = MongoClient(config.host, config.port, **(config.connection_params or {}))
    db = client[config.database]
    collection = db[config.table]
    
    # Build query
    query = {}
    if source.filter_condition:
        query = json.loads(source.filter_condition)
    
    cursor = collection.find(
        query,
        limit=source.sample_size
    )
    
    return pd.DataFrame(list(cursor))
"""
        # Streaming sources
        elif source.type in [SourceType.KAFKA, SourceType.KINESIS, SourceType.PUBSUB, SourceType.EVENT_HUB]:
            return """
import pandas as pd
from confluent_kafka import Consumer
import json

def connect_to_source(source):
    config = source.stream_config
    messages = []
    
    if source.type == 'KAFKA':
        consumer = Consumer({
            'bootstrap.servers': ','.join(config.brokers),
            'group.id': config.group_id,
            'auto.offset.reset': config.offset,
            **(config.consumer_config or {})
        })
        
        consumer.subscribe([config.topic])
        
        # Collect sample messages
        for _ in range(source.sample_size):
            msg = consumer.poll(1.0)
            if msg is None:
                continue
            if msg.error():
                continue
            try:
                messages.append(json.loads(msg.value().decode('utf-8')))
            except:
                messages.append(msg.value().decode('utf-8'))
                
        consumer.close()
    
    return pd.DataFrame(messages)
"""
        # API sources
        elif source.type in [SourceType.REST, SourceType.GRAPHQL]:
            return """
import pandas as pd
import requests
import json

def connect_to_source(source):
    config = source.api_config
    
    # Prepare request
    headers = config.headers or {}
    params = config.params or {}
    
    if config.auth_config:
        if config.auth_config.get('type') == 'bearer':
            headers['Authorization'] = f"Bearer {config.auth_config['token']}"
        elif config.auth_config.get('type') == 'basic':
            from requests.auth import HTTPBasicAuth
            auth = HTTPBasicAuth(config.auth_config['username'], config.auth_config['password'])
    else:
        auth = None
    
    # Make request
    response = requests.request(
        method=config.method,
        url=config.url,
        headers=headers,
        params=params,
        json=config.body,
        auth=auth
    )
    response.raise_for_status()
    
    # Parse response
    data = response.json()
    if isinstance(data, list):
        return pd.DataFrame(data)
    elif isinstance(data, dict):
        # Try to find the main data array in the response
        for key, value in data.items():
            if isinstance(value, list):
                return pd.DataFrame(value)
        return pd.DataFrame([data])
    
    return pd.DataFrame()
"""
        
        return ""

    def _generate_sampling_code(self, source: Source) -> str:
        """Generate source-specific sampling code."""
        return """
def extract_sample(data, sample_size=1000):
    if isinstance(data, pd.DataFrame):
        if len(data) > sample_size:
            if 'timestamp' in data.columns:
                # Time-based sampling for time series data
                return data.set_index('timestamp').last(f'{sample_size}rows')
            else:
                # Random sampling for other data
                return data.sample(n=sample_size, random_state=42)
        return data
    elif isinstance(data, dict):
        return data  # For now, return full dict
    return data
"""

    def _generate_analysis_plan(self, source: Source) -> List[AnalysisStep]:
        """Generate a dynamic analysis plan based on the source type."""
        steps = []
        
        # Step 1: Source Connection
        connection_code = self._generate_connection_code(source)
        steps.append(
            AnalysisStep(
                name="connect_to_source",
                description="Establish connection to the data source",
                code=connection_code
            )
        )
        
        # Step 2: Sample Data Extraction
        sample_code = self._generate_sampling_code(source)
        steps.append(
            AnalysisStep(
                name="extract_sample",
                description="Extract a representative data sample",
                code=sample_code,
                dependencies=["connect_to_source"]
            )
        )
        
        # Step 3: Schema Inference
        schema_code = """
def infer_schema(data):
    if isinstance(data, pd.DataFrame):
        return {
            'columns': {
                col: str(dtype) for col, dtype in data.dtypes.items()
            },
            'num_columns': len(data.columns),
            'column_descriptions': {
                col: {
                    'unique_values': data[col].nunique(),
                    'missing_values': data[col].isnull().sum(),
                    'sample_values': data[col].dropna().head(5).tolist()
                } for col in data.columns
            }
        }
    elif isinstance(data, dict):
        return {
            'fields': {k: type(v).__name__ for k, v in data.items()},
            'nested_fields': {
                k: infer_schema(v) for k, v in data.items() 
                if isinstance(v, (dict, list))
            }
        }
    return {'type': type(data).__name__}
"""
        steps.append(
            AnalysisStep(
                name="infer_schema",
                description="Infer the data schema and types",
                code=schema_code,
                dependencies=["extract_sample"]
            )
        )
        
        # Step 4: Data Quality Analysis
        quality_code = """
def analyze_data_quality(data):
    issues = []
    stats = {}
    
    if isinstance(data, pd.DataFrame):
        # Basic statistics
        stats = {
            'row_count': len(data),
            'column_stats': {}
        }
        
        for col in data.columns:
            col_stats = {
                'missing_rate': data[col].isnull().mean(),
                'unique_rate': data[col].nunique() / len(data),
            }
            
            # Numeric analysis
            if pd.api.types.is_numeric_dtype(data[col]):
                col_stats.update({
                    'mean': data[col].mean(),
                    'std': data[col].std(),
                    'min': data[col].min(),
                    'max': data[col].max()
                })
                
                # Check for outliers
                z_scores = (data[col] - data[col].mean()) / data[col].std()
                if abs(z_scores).max() > 3:
                    issues.append(f"Potential outliers detected in {col}")
                    
                # Check for suspicious zeros
                zero_rate = (data[col] == 0).mean()
                if zero_rate > 0.8:
                    issues.append(f"High rate of zeros ({zero_rate:.1%}) in {col}")
            
            # String analysis
            elif pd.api.types.is_string_dtype(data[col]):
                # Check for inconsistent casing
                if data[col].str.upper().nunique() != data[col].nunique():
                    issues.append(f"Inconsistent casing detected in {col}")
                
                # Check for special characters
                if data[col].str.contains('[^a-zA-Z0-9\\s]').any():
                    issues.append(f"Special characters detected in {col}")
                    
                # Check for potential dates in string columns
                date_pattern = r'\\d{4}-\\d{2}-\\d{2}'
                if data[col].str.contains(date_pattern).any():
                    issues.append(f"Potential dates stored as strings in {col}")
            
            # Timestamp analysis
            elif pd.api.types.is_datetime64_any_dtype(data[col]):
                col_stats.update({
                    'min_date': data[col].min(),
                    'max_date': data[col].max(),
                    'date_range_days': (data[col].max() - data[col].min()).days
                })
                
                # Check for future dates
                if data[col].max() > pd.Timestamp.now():
                    issues.append(f"Future dates detected in {col}")
            
            stats['column_stats'][col] = col_stats
            
            # Check for high missing rate
            if col_stats['missing_rate'] > 0.1:
                issues.append(f"High missing rate ({col_stats['missing_rate']:.1%}) in {col}")
            
            # Check for high cardinality
            if col_stats['unique_rate'] > 0.9:
                issues.append(f"High cardinality ({col_stats['unique_rate']:.1%}) in {col}")
            
            # Check for low cardinality
            if col_stats['unique_rate'] < 0.01 and len(data) > 1000:
                issues.append(f"Low cardinality ({col_stats['unique_rate']:.1%}) in {col}")
    
    return issues, stats
"""
        steps.append(
            AnalysisStep(
                name="analyze_quality",
                description="Analyze data quality and generate statistics",
                code=quality_code,
                dependencies=["extract_sample"]
            )
        )
        
        return steps

    async def run(self, state: AnalyzerState) -> AnalyzerState:
        """
        Analyze the source data to determine:
        1. Data schema and types
        2. Data quality issues
        3. Sample data for validation
        
        Args:
            state: Current analyzer state
            
        Returns:
            Updated analyzer state with data insights
        """
        try:
            # Generate analysis plan
            state.analysis_steps = self._generate_analysis_plan(state.source)
            
            # Create a temporary module to execute the analysis
            analysis_module = type('AnalysisModule', (), {})
            
            # Execute each analysis step
            for step in state.analysis_steps:
                # Add step code to module
                exec(step.code, analysis_module.__dict__)
                
                # Execute the step
                if step.name == "connect_to_source":
                    data = analysis_module.connect_to_source(state.source)
                elif step.name == "extract_sample":
                    state.sample_data = analysis_module.extract_sample(data)
                elif step.name == "infer_schema":
                    state.schema = analysis_module.infer_schema(state.sample_data)
                elif step.name == "analyze_quality":
                    issues, stats = analysis_module.analyze_data_quality(state.sample_data)
                    state.data_quality_issues = issues
                    state.statistics = stats
                
                # Store results in step
                step.results = {
                    "completed": True,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
            
            state.completed = True
            return state
            
        except Exception as e:
            state.error = str(e)
            return state 