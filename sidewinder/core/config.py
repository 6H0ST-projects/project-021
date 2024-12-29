"""
Core configuration models for Sidewinder.
"""

from enum import Enum
from typing import Optional, Dict, Any, List, Union
from pydantic import BaseModel, Field


class EnvironmentType(str, Enum):
    """Supported environment types."""
    LOCAL = "local"
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    DATABRICKS = "databricks"


class Environment(BaseModel):
    """Environment configuration for the pipeline."""
    type: EnvironmentType
    credentials: Optional[Dict[str, Any]] = Field(default=None, description="Cloud credentials if needed")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Additional environment configuration")


class SourceType(str, Enum):
    """Supported source types."""
    # File-based sources
    FILE = "file"  # Local files
    DIRECTORY = "directory"  # Directory of files
    
    # Cloud Storage
    S3 = "s3"  # AWS S3
    GCS = "gcs"  # Google Cloud Storage
    AZURE_BLOB = "azure_blob"  # Azure Blob Storage
    DBFS = "dbfs"  # Databricks File System
    
    # Databases
    POSTGRES = "postgres"
    MYSQL = "mysql"
    MONGODB = "mongodb"
    SNOWFLAKE = "snowflake"
    BIGQUERY = "bigquery"
    REDSHIFT = "redshift"
    DATABRICKS_SQL = "databricks_sql"
    
    # Streaming
    KAFKA = "kafka"
    KINESIS = "kinesis"
    PUBSUB = "pubsub"
    EVENT_HUB = "event_hub"
    
    # APIs
    REST = "rest"
    GRAPHQL = "graphql"
    GRPC = "grpc"


class DatabaseConfig(BaseModel):
    """Database-specific configuration."""
    host: str
    port: Optional[int] = None
    database: str
    schema: Optional[str] = None
    table: Optional[str] = None
    query: Optional[str] = None
    connection_params: Optional[Dict[str, Any]] = None


class StreamConfig(BaseModel):
    """Streaming source configuration."""
    brokers: List[str]
    topic: Optional[str] = None
    group_id: Optional[str] = None
    offset: Optional[str] = "latest"
    partition: Optional[int] = None
    consumer_config: Optional[Dict[str, Any]] = None


class APIConfig(BaseModel):
    """API source configuration."""
    url: str
    method: str = "GET"
    headers: Optional[Dict[str, str]] = None
    params: Optional[Dict[str, Any]] = None
    body: Optional[Dict[str, Any]] = None
    auth_config: Optional[Dict[str, Any]] = None


class Source(BaseModel):
    """Data source configuration."""
    type: SourceType
    location: str = Field(..., description="Source location (e.g., file path, bucket URI, connection string)")
    format: Optional[str] = Field(default=None, description="Data format (e.g., csv, parquet, json, avro)")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Source-specific options")
    
    # Specific configurations based on source type
    database_config: Optional[DatabaseConfig] = None
    stream_config: Optional[StreamConfig] = None
    api_config: Optional[APIConfig] = None
    
    # Data sampling and processing options
    sample_size: Optional[int] = Field(default=1000, description="Number of records to sample")
    partition_column: Optional[str] = None
    filter_condition: Optional[str] = None
    
    class Config:
        """Pydantic config."""
        extra = "allow"  # Allow additional fields for flexibility


class TargetType(str, Enum):
    """Supported target types."""
    FILE = "file"
    S3 = "s3"
    GCS = "gcs"
    AZURE_BLOB = "azure_blob"
    DATABASE = "database"
    STREAM = "stream"


class FeatureType(str, Enum):
    """Types of features that can be engineered."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEMPORAL = "temporal"
    TEXT = "text"
    GEOSPATIAL = "geospatial"
    INTERACTION = "interaction"
    SEQUENCE = "sequence"
    WINDOW = "window"


class FeatureConfig(BaseModel):
    """Configuration for feature engineering."""
    type: FeatureType
    source_columns: List[str]
    parameters: Optional[Dict[str, Any]] = None


class SchemaField(BaseModel):
    """Definition of a field in the target schema."""
    name: str
    type: str
    description: Optional[str] = None
    required: bool = True
    validation_rules: Optional[List[str]] = None
    feature_config: Optional[FeatureConfig] = None


class TargetSchema(BaseModel):
    """Schema definition for the target data."""
    fields: Dict[str, SchemaField]
    primary_keys: Optional[List[str]] = None
    partitioning_keys: Optional[List[str]] = None
    clustering_keys: Optional[List[str]] = None
    dependencies: Optional[Dict[str, List[str]]] = None


class AutoFeatureConfig(BaseModel):
    """Configuration for automatic feature engineering."""
    enabled_feature_types: List[FeatureType] = Field(
        default_factory=lambda: list(FeatureType),
        description="Types of features to auto-generate"
    )
    max_interaction_degree: int = Field(default=2, description="Maximum number of features to combine")
    max_window_size: int = Field(default=7, description="Maximum window size for temporal features")
    text_analysis_enabled: bool = Field(default=True, description="Enable text feature extraction")
    geospatial_analysis_enabled: bool = Field(default=True, description="Enable geospatial feature extraction")
    correlation_threshold: float = Field(default=0.95, description="Maximum correlation between features")


class Target(BaseModel):
    """Data target configuration."""
    type: TargetType
    location: str = Field(..., description="Target location (e.g., database URI, bucket path)")
    format: Optional[str] = Field(default=None, description="Output format")
    options: Optional[Dict[str, Any]] = Field(default=None, description="Target-specific options")
    
    # Schema configuration
    target_schema: Optional[TargetSchema] = None
    auto_feature_config: Optional[AutoFeatureConfig] = Field(
        default_factory=AutoFeatureConfig,
        description="Configuration for automatic feature engineering when no target schema is provided"
    )
    
    # Quality thresholds
    quality_thresholds: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {
            "null_rate": 0.1,
            "correlation": 0.95,
            "cardinality_ratio": 0.9
        }
    ) 