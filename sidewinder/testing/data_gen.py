"""
Data generation utilities for creating representative test data.
"""

from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import faker
from pathlib import Path

from sidewinder.core.config import Source, Target
from sidewinder.agents.analyzer import DataAnalyzer
from sidewinder.testing.data_patterns import (
    Pattern,
    Relationship,
    RelationshipHandler,
    apply_pattern
)


class ColumnProfile(BaseModel):
    """Profile of a data column."""
    name: str
    data_type: str
    nullable: bool = False
    unique: bool = False
    min_value: Optional[Union[int, float, str]] = None
    max_value: Optional[Union[int, float, str]] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    categories: Optional[List[Any]] = None
    pattern: Optional[str] = None
    distribution: str = "uniform"  # uniform, normal, categorical
    missing_rate: float = 0.0
    data_pattern: Optional[Pattern] = None


class DataGenerationConfig(BaseModel):
    """Configuration for data generation."""
    num_rows: int
    columns: List[ColumnProfile]
    seed: Optional[int] = None
    batch_size: int = 10000
    output_format: str = "parquet"
    relationships: List[Relationship] = []


class DataGenerator:
    """Generates representative test data based on profiles."""
    
    def __init__(self, config: DataGenerationConfig):
        self.config = config
        self.fake = faker.Faker()
        if config.seed is not None:
            np.random.seed(config.seed)
            self.fake.seed_instance(config.seed)
        
        self.relationship_handler = (
            RelationshipHandler(config.relationships)
            if config.relationships
            else None
        )
    
    def _generate_numeric(
        self,
        profile: ColumnProfile,
        size: int
    ) -> np.ndarray:
        """Generate numeric data based on profile."""
        if profile.distribution == "normal" and profile.mean is not None and profile.std is not None:
            data = np.random.normal(profile.mean, profile.std, size)
        else:
            min_val = profile.min_value or 0
            max_val = profile.max_value or 100
            data = np.random.uniform(min_val, max_val, size)
        
        if profile.data_type == "int":
            data = data.astype(int)
        
        return data
    
    def _generate_categorical(
        self,
        profile: ColumnProfile,
        size: int
    ) -> np.ndarray:
        """Generate categorical data based on profile."""
        if not profile.categories:
            raise ValueError(f"Categories not provided for column {profile.name}")
        
        if profile.distribution == "uniform":
            probabilities = None
        elif profile.distribution == "normal":
            # Create normal-like distribution over categories
            x = np.linspace(-2, 2, len(profile.categories))
            probabilities = np.exp(-x**2)
            probabilities /= probabilities.sum()
        else:
            probabilities = None
        
        return np.random.choice(profile.categories, size=size, p=probabilities)
    
    def _generate_datetime(
        self,
        profile: ColumnProfile,
        size: int
    ) -> np.ndarray:
        """Generate datetime data based on profile."""
        if not profile.min_value or not profile.max_value:
            end = datetime.now()
            start = end - timedelta(days=365)
        else:
            start = pd.to_datetime(profile.min_value)
            end = pd.to_datetime(profile.max_value)
        
        time_range = (end - start).total_seconds()
        
        if profile.distribution == "uniform":
            timestamps = np.random.uniform(0, time_range, size)
        elif profile.distribution == "normal":
            mean = time_range / 2
            std = time_range / 6
            timestamps = np.clip(
                np.random.normal(mean, std, size),
                0,
                time_range
            )
        else:
            timestamps = np.random.uniform(0, time_range, size)
        
        return pd.to_datetime(start) + pd.to_timedelta(timestamps, unit='s')
    
    def _generate_string(
        self,
        profile: ColumnProfile,
        size: int
    ) -> np.ndarray:
        """Generate string data based on profile."""
        if profile.pattern:
            # Use faker for common patterns
            if profile.pattern == "email":
                return np.array([self.fake.email() for _ in range(size)])
            elif profile.pattern == "phone":
                return np.array([self.fake.phone_number() for _ in range(size)])
            elif profile.pattern == "name":
                return np.array([self.fake.name() for _ in range(size)])
            elif profile.pattern == "address":
                return np.array([self.fake.address() for _ in range(size)])
            elif profile.pattern == "company":
                return np.array([self.fake.company() for _ in range(size)])
            elif profile.pattern == "job":
                return np.array([self.fake.job() for _ in range(size)])
            elif profile.pattern == "url":
                return np.array([self.fake.url() for _ in range(size)])
            elif profile.pattern == "ipv4":
                return np.array([self.fake.ipv4() for _ in range(size)])
            elif profile.pattern == "username":
                return np.array([self.fake.user_name() for _ in range(size)])
            else:
                # TODO: Support custom regex patterns
                pass
        
        # Default to random strings
        length = 10
        chars = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
        return np.array([''.join(np.random.choice(chars, length)) for _ in range(size)])
    
    def _apply_nulls(
        self,
        data: np.ndarray,
        profile: ColumnProfile
    ) -> np.ndarray:
        """Apply null values based on profile."""
        if profile.nullable and profile.missing_rate > 0:
            mask = np.random.random(len(data)) < profile.missing_rate
            data = data.astype('object')
            data[mask] = None
        return data
    
    def _apply_uniqueness(
        self,
        data: np.ndarray,
        profile: ColumnProfile
    ) -> np.ndarray:
        """Ensure uniqueness if required."""
        if profile.unique:
            if len(data) > len(set(data)):
                raise ValueError(
                    f"Cannot generate {len(data)} unique values for column {profile.name}"
                )
            return np.random.permutation(data)
        return data
    
    def _apply_pattern(
        self,
        data: pd.DataFrame,
        profile: ColumnProfile
    ) -> np.ndarray:
        """Apply data pattern if specified."""
        if profile.data_pattern:
            return apply_pattern(profile.data_pattern, data)
        return data[profile.name].values
    
    def generate_column(
        self,
        profile: ColumnProfile,
        size: int,
        data: Optional[pd.DataFrame] = None
    ) -> np.ndarray:
        """Generate data for a single column."""
        if profile.data_type in ["int", "float"]:
            column_data = self._generate_numeric(profile, size)
        elif profile.data_type == "category":
            column_data = self._generate_categorical(profile, size)
        elif profile.data_type == "datetime":
            column_data = self._generate_datetime(profile, size)
        elif profile.data_type == "string":
            column_data = self._generate_string(profile, size)
        else:
            raise ValueError(f"Unsupported data type: {profile.data_type}")
        
        # Apply pattern if exists and data is available
        if data is not None and profile.data_pattern:
            column_data = self._apply_pattern(data, profile)
        
        column_data = self._apply_nulls(column_data, profile)
        column_data = self._apply_uniqueness(column_data, profile)
        
        return column_data
    
    def generate(self, output_path: str):
        """Generate test data based on configuration."""
        remaining_rows = self.config.num_rows
        
        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        while remaining_rows > 0:
            batch_size = min(self.config.batch_size, remaining_rows)
            
            # Generate data for each column
            data = {}
            df = pd.DataFrame()  # Initialize empty DataFrame for patterns
            
            # First pass: generate basic data
            for profile in self.config.columns:
                data[profile.name] = self.generate_column(profile, batch_size)
                df[profile.name] = data[profile.name]
            
            # Second pass: apply patterns that depend on other columns
            for profile in self.config.columns:
                if profile.data_pattern:
                    data[profile.name] = self.generate_column(profile, batch_size, df)
            
            # Create final DataFrame
            df = pd.DataFrame(data)
            
            # Apply relationships if any
            if self.relationship_handler:
                tables = {output_path: df}  # TODO: Support multiple tables
                tables = self.relationship_handler.apply_relationships(tables)
                df = tables[output_path]
            
            # Save batch
            if self.config.output_format == "parquet":
                if remaining_rows == self.config.num_rows:
                    # First batch
                    df.to_parquet(output_path, index=False)
                else:
                    # Append mode
                    df.to_parquet(output_path, index=False, append=True)
            elif self.config.output_format == "csv":
                mode = "w" if remaining_rows == self.config.num_rows else "a"
                header = remaining_rows == self.config.num_rows
                df.to_csv(output_path, index=False, mode=mode, header=header)
            
            remaining_rows -= batch_size


class ProfileGenerator:
    """Generates data profiles from source data analysis."""
    
    def __init__(self, analyzer: DataAnalyzer):
        self.analyzer = analyzer
    
    def _detect_pattern(
        self,
        stats: Any,
        column_name: str
    ) -> Optional[Pattern]:
        """Detect data pattern from statistics."""
        if not stats or not stats.pattern_stats:
            return None
        
        pattern_type = stats.pattern_stats.get("type")
        if not pattern_type:
            return None
        
        parameters = stats.pattern_stats.get("parameters", {})
        
        return Pattern(
            name=f"{column_name}_pattern",
            type=pattern_type,
            parameters=parameters
        )
    
    def generate_profile(
        self,
        source: Source,
        sample_size: Optional[int] = None
    ) -> DataGenerationConfig:
        """Generate data profile from source analysis."""
        # Analyze source data
        analysis = self.analyzer.analyze(source)
        
        # Extract column profiles
        columns = []
        for col_name, stats in analysis.column_stats.items():
            profile = ColumnProfile(
                name=col_name,
                data_type=stats.data_type,
                nullable=stats.null_count > 0,
                unique=stats.unique_count == stats.total_count,
                min_value=stats.min_value,
                max_value=stats.max_value,
                mean=stats.mean if stats.data_type in ["int", "float"] else None,
                std=stats.std if stats.data_type in ["int", "float"] else None,
                categories=stats.categories if stats.data_type == "category" else None,
                pattern=stats.pattern if stats.data_type == "string" else None,
                distribution=stats.distribution or "uniform",
                missing_rate=stats.null_count / stats.total_count if stats.total_count > 0 else 0,
                data_pattern=self._detect_pattern(stats, col_name)
            )
            columns.append(profile)
        
        # Extract relationships
        relationships = [
            Relationship(**rel) for rel in analysis.relationships
        ] if analysis.relationships else []
        
        # Create configuration
        return DataGenerationConfig(
            num_rows=sample_size or analysis.total_rows,
            columns=columns,
            relationships=relationships
        )


def generate_test_data(
    source: Source,
    output_path: str,
    sample_size: Optional[int] = None,
    seed: Optional[int] = None
) -> str:
    """
    Generate test data based on source analysis.
    
    Args:
        source: Source configuration
        output_path: Path to save generated data
        sample_size: Number of rows to generate (default: same as source)
        seed: Random seed for reproducibility
        
    Returns:
        Path to generated test data
    """
    # Create profile generator
    analyzer = DataAnalyzer()
    profiler = ProfileGenerator(analyzer)
    
    # Generate profile from source
    config = profiler.generate_profile(source, sample_size)
    if seed is not None:
        config.seed = seed
    
    # Generate data
    generator = DataGenerator(config)
    generator.generate(output_path)
    
    return output_path 