"""
Dynamic data pattern generation based on source data analysis.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta

from sidewinder.testing.data_types import DataType, ColumnProfile

# Setup logging
logger = logging.getLogger(__name__)
fake = Faker()

def analyze_column_pattern(series: pd.Series) -> Dict[str, Any]:
    """
    Analyze a column to detect its pattern.
    
    Args:
        series: pandas Series to analyze
        
    Returns:
        Dictionary containing pattern information
    """
    # Handle empty series
    if series.empty:
        return {
            "type": str(series.dtype),
            "unique_count": 0,
            "null_count": 0,
            "total_count": 0,
            "sample_values": []
        }
    
    # Get non-null values for sampling
    non_null_series = series.dropna()
    sample_size = min(5, len(non_null_series))
    
    pattern = {
        "type": str(series.dtype),
        "unique_count": series.nunique(),
        "null_count": series.isnull().sum(),
        "total_count": len(series),
        "sample_values": non_null_series.sample(n=sample_size, replace=True).tolist() if sample_size > 0 else []
    }
    
    # Detect numerical patterns
    if pd.api.types.is_numeric_dtype(series):
        pattern.update({
            "min": float(series.min()) if not pd.isna(series.min()) else None,
            "max": float(series.max()) if not pd.isna(series.max()) else None,
            "mean": float(series.mean()) if not pd.isna(series.mean()) else None,
            "std": float(series.std()) if not pd.isna(series.std()) else None,
            "distribution": detect_distribution(series)
        })
        
        # Check for sequence patterns
        if is_sequential(series):
            pattern["sequence_type"] = detect_sequence_type(series)
    
    # Detect string patterns
    elif pd.api.types.is_string_dtype(series):
        pattern["string_pattern"] = detect_string_pattern(series)
        if pattern["unique_count"] / pattern["total_count"] < 0.1:
            pattern["categorical_distribution"] = series.value_counts(normalize=True).to_dict()
    
    # Detect datetime patterns
    elif pd.api.types.is_datetime64_any_dtype(series):
        pattern.update({
            "min": series.min().isoformat() if not pd.isna(series.min()) else None,
            "max": series.max().isoformat() if not pd.isna(series.max()) else None,
            "temporal_pattern": detect_temporal_pattern(series)
        })
    
    return pattern

def detect_distribution(series: pd.Series) -> str:
    """Detect the statistical distribution of numerical data."""
    if series.empty:
        return "unknown"
    
    # Calculate basic statistics
    skew = series.skew()
    kurtosis = series.kurtosis()
    
    if abs(skew) < 0.5 and abs(kurtosis) < 0.5:
        return "normal"
    elif skew > 1:
        return "right_skewed"
    elif skew < -1:
        return "left_skewed"
    elif kurtosis > 1:
        return "heavy_tailed"
    else:
        return "unknown"

def is_sequential(series: pd.Series) -> bool:
    """Check if the series follows a sequential pattern."""
    if series.empty or len(series) < 2:
        return False
    
    diffs = series.diff().dropna()
    unique_diffs = diffs.unique()
    return len(unique_diffs) == 1

def detect_sequence_type(series: pd.Series) -> str:
    """Detect the type of sequence (linear, exponential, etc.)."""
    if not is_sequential(series):
        return "none"
    
    # Check for linear sequence
    diffs = series.diff().dropna()
    if len(diffs.unique()) == 1:
        return "linear"
    
    # Check for exponential sequence
    ratios = (series / series.shift(1)).dropna()
    if len(ratios.unique()) == 1:
        return "exponential"
    
    return "unknown"

def detect_string_pattern(series: pd.Series) -> str:
    """Detect common string patterns."""
    if series.empty:
        return "unknown"
    
    # Sample non-null values
    sample = series.dropna().iloc[0] if not series.empty else ""
    
    if series.str.contains(r'@').any():
        return "email"
    elif series.str.contains(r'\d{3}[-.]?\d{3}[-.]?\d{4}').any():
        return "phone"
    elif series.str.contains(r'\d{4}-\d{2}-\d{2}').any():
        return "date"
    elif series.str.match(r'^[A-Z0-9]{8,}$').all():
        return "id"
    elif series.str.contains(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*$').any():
        return "name"
    else:
        return "text"

def detect_temporal_pattern(series: pd.Series) -> Dict[str, Any]:
    """Detect patterns in temporal data."""
    if series.empty:
        return {"type": "unknown"}
    
    # Calculate time differences
    diffs = series.diff().dropna()
    
    # Check for regular intervals
    unique_diffs = diffs.unique()
    if len(unique_diffs) == 1:
        return {
            "type": "regular_interval",
            "interval": str(unique_diffs[0])
        }
    
    # Check for daily pattern
    hour_counts = series.dt.hour.value_counts()
    if len(hour_counts) < 24 and hour_counts.max() / hour_counts.sum() > 0.2:
        return {
            "type": "daily_pattern",
            "peak_hours": hour_counts.nlargest(3).index.tolist()
        }
    
    # Check for weekly pattern
    day_counts = series.dt.dayofweek.value_counts()
    if day_counts.max() / day_counts.sum() > 0.2:
        return {
            "type": "weekly_pattern",
            "peak_days": day_counts.nlargest(3).index.tolist()
        }
    
    return {"type": "irregular"}

def generate_column_data(
    pattern: Dict[str, Any],
    size: int,
    seed: Optional[int] = None
) -> pd.Series:
    """
    Generate column data based on detected pattern.
    
    Args:
        pattern: Pattern dictionary from analyze_column_pattern
        size: Number of rows to generate
        seed: Random seed for reproducibility
        
    Returns:
        Generated pandas Series
    """
    if seed is not None:
        np.random.seed(seed)
        fake.seed_instance(seed)
    
    # Generate numerical data
    if pattern["type"].startswith(('int', 'float')):
        if pattern.get("sequence_type") == "linear":
            start = pattern["min"]
            step = (pattern["max"] - pattern["min"]) / (size - 1)
            data = np.arange(start, pattern["max"] + step, step)[:size]
        elif pattern.get("sequence_type") == "exponential":
            base = (pattern["max"] / pattern["min"]) ** (1 / (size - 1))
            data = pattern["min"] * np.power(base, np.arange(size))
        else:
            if pattern["distribution"] == "normal":
                data = np.random.normal(pattern["mean"], pattern["std"], size)
            elif pattern["distribution"] in ["right_skewed", "heavy_tailed"]:
                data = np.random.exponential(pattern["mean"], size)
            else:
                data = np.random.uniform(pattern["min"], pattern["max"], size)
    
    # Generate string data
    elif pattern["type"] == "object":
        if pattern.get("string_pattern") == "email":
            data = [fake.email() for _ in range(size)]
        elif pattern.get("string_pattern") == "phone":
            data = [fake.phone_number() for _ in range(size)]
        elif pattern.get("string_pattern") == "name":
            data = [fake.name() for _ in range(size)]
        elif pattern.get("string_pattern") == "id":
            data = [fake.uuid4() for _ in range(size)]
        elif pattern.get("categorical_distribution"):
            categories = list(pattern["categorical_distribution"].keys())
            weights = list(pattern["categorical_distribution"].values())
            data = np.random.choice(categories, size=size, p=weights)
        else:
            data = [fake.text(max_nb_chars=50) for _ in range(size)]
    
    # Generate datetime data
    elif pattern["type"].startswith('datetime'):
        if pattern.get("temporal_pattern", {}).get("type") == "regular_interval":
            interval = pd.Timedelta(pattern["temporal_pattern"]["interval"])
            start = pd.to_datetime(pattern["min"])
            data = [start + interval * i for i in range(size)]
        else:
            start = pd.to_datetime(pattern["min"])
            end = pd.to_datetime(pattern["max"])
            data = [fake.date_time_between(start_date=start, end_date=end) for _ in range(size)]
    
    # Convert to series and apply null ratio if specified
    series = pd.Series(data)
    if pattern["null_count"] > 0:
        null_ratio = pattern["null_count"] / pattern["total_count"]
        null_mask = np.random.random(size) < null_ratio
        series[null_mask] = None
    
    return series

def generate_related_data(
    base_data: pd.DataFrame,
    relationship: Dict[str, Any],
    size: int,
    seed: Optional[int] = None
) -> pd.Series:
    """
    Generate related column data based on base data and relationship.
    
    Args:
        base_data: DataFrame containing the base data
        relationship: Dictionary describing the relationship
        size: Number of rows to generate
        seed: Random seed for reproducibility
        
    Returns:
        Generated pandas Series with related data
    """
    if seed is not None:
        np.random.seed(seed)
    
    base_values = base_data[relationship["base_column"]].unique()
    
    if relationship["type"] == "one_to_one":
        return pd.Series(np.random.permutation(base_values)[:size])
    
    elif relationship["type"] == "one_to_many":
        min_refs = relationship.get("min_refs", 1)
        max_refs = relationship.get("max_refs", 5)
        refs_per_row = np.random.randint(min_refs, max_refs + 1, size)
        return pd.Series(np.repeat(base_values, refs_per_row)[:size])
    
    elif relationship["type"] == "many_to_many":
        return pd.Series(np.random.choice(base_values, size=size))
    
    else:
        raise ValueError(f"Unsupported relationship type: {relationship['type']}") 