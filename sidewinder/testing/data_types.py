"""
Shared data types for test data generation.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class DataType(str, Enum):
    """Supported data types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATE = "date"
    TIMESTAMP = "timestamp"
    ARRAY = "array"
    OBJECT = "object"


class ColumnProfile(BaseModel):
    """Profile of a data column."""
    name: str
    data_type: DataType
    nullable: bool = False
    unique: bool = False
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    pattern: Optional[str] = None
    enum_values: Optional[List[Any]] = None
    distribution: Optional[Dict[str, Any]] = None
    dependencies: Optional[List[str]] = None


class DataPattern(BaseModel):
    """Pattern for generating test data."""
    name: str
    description: str
    columns: List[ColumnProfile]
    relationships: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[List[Dict[str, Any]]] = None 