"""
Data generation utilities for testing.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from faker import Faker

from sidewinder.testing.data_patterns import (
    analyze_column_pattern,
    generate_column_data,
    generate_related_data
)
from sidewinder.testing.data_types import ColumnProfile, DataPattern, DataType
from sidewinder.core.config import Source, Target

# Setup logging
logger = logging.getLogger(__name__)

# Initialize Faker
fake = Faker()


def infer_data_patterns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Infer data patterns from source data.
    
    Args:
        df: Source DataFrame to analyze
        
    Returns:
        Dictionary of inferred patterns and statistics
    """
    patterns = {}
    
    for column in df.columns:
        patterns[column] = analyze_column_pattern(df[column])
    
    return patterns


def generate_test_data(
    source: Source,
    output_path: Optional[str] = None,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate test data based on source configuration.
    
    Args:
        source: Source configuration
        output_path: Path to save generated data (optional)
        sample_size: Number of rows to generate (default: from source config)
        seed: Random seed for reproducibility
        
    Returns:
        Generated test DataFrame
    """
    if seed is not None:
        np.random.seed(seed)
        fake.seed_instance(seed)
    
    # Load source data for pattern analysis
    source_data = None
    if source.type == "file":
        if source.format == "json":
            with open(source.location) as f:
                json_data = json.load(f)
            # Handle nested JSON structures
            if isinstance(json_data, dict):
                # Try to find a list in the JSON structure
                for key, value in json_data.items():
                    if isinstance(value, list):
                        json_data = value
                        break
            source_data = pd.DataFrame(json_data)
        elif source.format == "parquet":
            source_data = pd.read_parquet(source.location)
        elif source.format == "csv":
            source_data = pd.read_csv(source.location)
        else:
            raise ValueError(f"Unsupported file format: {source.format}")
    else:
        raise ValueError(f"Unsupported source type: {source.type}")
    
    # Infer patterns from source data
    patterns = infer_data_patterns(source_data)
    size = sample_size or source.sample_size or len(source_data)
    
    # Generate data for each column
    data = {}
    for column, pattern in patterns.items():
        data[column] = generate_column_data(pattern, size, seed)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Save generated data if output path is provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if str(output_path).endswith('.parquet'):
            df.to_parquet(output_path, index=False)
        elif str(output_path).endswith('.csv'):
            df.to_csv(output_path, index=False)
        elif str(output_path).endswith('.json'):
            if source.options and source.options.get("lines", False):
                df.to_json(output_path, orient="records", lines=True)
            else:
                result = {
                    source.location.split("/")[-1].split(".")[0]: df.to_dict(orient="records")
                }
                with open(output_path, "w") as f:
                    json.dump(result, f, indent=4)
        else:
            raise ValueError(f"Unsupported output format: {output_path}")
        
        logger.info(f"Saved test data to {output_path}")
    
    return df


def load_test_data(data_path: Union[str, Path]) -> pd.DataFrame:
    """Load test data from file."""
    data_path = Path(data_path)
    
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    if str(data_path).endswith('.parquet'):
        data = pd.read_parquet(data_path)
    elif str(data_path).endswith('.csv'):
        data = pd.read_csv(data_path)
    elif str(data_path).endswith('.json'):
        data = pd.read_json(data_path)
    else:
        raise ValueError(f"Unsupported file format: {data_path}")
    
    logger.info(f"Loaded test data from {data_path}")
    return data 