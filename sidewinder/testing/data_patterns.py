"""
Data patterns and relationships for test data generation.
"""

from typing import Dict, Any, List, Optional, Callable
from pydantic import BaseModel
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import networkx as nx

from sidewinder.testing.data_gen import ColumnProfile


class Relationship(BaseModel):
    """Defines a relationship between tables/columns."""
    from_table: str
    to_table: str
    type: str = "one_to_many"  # one_to_one, one_to_many, many_to_many
    keys: List[str]
    cardinality: Optional[Dict[str, float]] = None  # e.g., {"min": 1, "max": 10}
    distribution: str = "uniform"  # uniform, normal, zipf


class Pattern(BaseModel):
    """Defines a data pattern."""
    name: str
    type: str  # sequence, cyclic, trend, correlation
    parameters: Dict[str, Any]


class DataPattern:
    """Base class for data patterns."""
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply pattern to data."""
        raise NotImplementedError


class SequencePattern(DataPattern):
    """Generates sequential patterns."""
    
    def __init__(
        self,
        start: Any,
        step: Any,
        noise_factor: float = 0.0
    ):
        self.start = start
        self.step = step
        self.noise_factor = noise_factor
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        size = len(data)
        sequence = np.arange(size) * self.step + self.start
        
        if self.noise_factor > 0:
            noise = np.random.normal(0, self.noise_factor * self.step, size)
            sequence += noise
        
        return sequence


class CyclicPattern(DataPattern):
    """Generates cyclic patterns."""
    
    def __init__(
        self,
        period: int,
        amplitude: float = 1.0,
        phase: float = 0.0,
        noise_factor: float = 0.0
    ):
        self.period = period
        self.amplitude = amplitude
        self.phase = phase
        self.noise_factor = noise_factor
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        size = len(data)
        t = np.arange(size)
        cycle = self.amplitude * np.sin(2 * np.pi * t / self.period + self.phase)
        
        if self.noise_factor > 0:
            noise = np.random.normal(0, self.noise_factor * self.amplitude, size)
            cycle += noise
        
        return cycle


class TrendPattern(DataPattern):
    """Generates trend patterns."""
    
    def __init__(
        self,
        type: str = "linear",  # linear, exponential, logarithmic
        parameters: Dict[str, float] = {"slope": 1.0, "intercept": 0.0},
        noise_factor: float = 0.0
    ):
        self.type = type
        self.parameters = parameters
        self.noise_factor = noise_factor
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        size = len(data)
        t = np.arange(size)
        
        if self.type == "linear":
            trend = self.parameters["slope"] * t + self.parameters["intercept"]
        elif self.type == "exponential":
            trend = self.parameters["base"] ** t * self.parameters["scale"]
        elif self.type == "logarithmic":
            trend = self.parameters["scale"] * np.log(t + 1) + self.parameters["intercept"]
        else:
            raise ValueError(f"Unsupported trend type: {self.type}")
        
        if self.noise_factor > 0:
            noise = np.random.normal(0, self.noise_factor * np.std(trend), size)
            trend += noise
        
        return trend


class CorrelationPattern(DataPattern):
    """Generates correlated data."""
    
    def __init__(
        self,
        correlation: float,
        base_column: str,
        noise_factor: float = 0.0
    ):
        self.correlation = correlation
        self.base_column = base_column
        self.noise_factor = noise_factor
    
    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.base_column not in data.columns:
            raise ValueError(f"Base column {self.base_column} not found in data")
        
        base = data[self.base_column].values
        size = len(base)
        
        # Generate correlated data
        noise = np.random.normal(0, 1, size)
        correlated = (
            self.correlation * base +
            np.sqrt(1 - self.correlation**2) * noise
        )
        
        if self.noise_factor > 0:
            extra_noise = np.random.normal(0, self.noise_factor * np.std(correlated), size)
            correlated += extra_noise
        
        return correlated


class RelationshipHandler:
    """Handles relationships between tables."""
    
    def __init__(self, relationships: List[Relationship]):
        self.relationships = relationships
        self.graph = self._build_graph()
    
    def _build_graph(self) -> nx.DiGraph:
        """Build relationship graph."""
        graph = nx.DiGraph()
        
        for rel in self.relationships:
            graph.add_edge(
                rel.from_table,
                rel.to_table,
                relationship=rel
            )
        
        return graph
    
    def _generate_foreign_keys(
        self,
        relationship: Relationship,
        primary_keys: np.ndarray,
        size: int
    ) -> np.ndarray:
        """Generate foreign keys based on relationship type."""
        if relationship.type == "one_to_one":
            return np.random.permutation(primary_keys)[:size]
        
        elif relationship.type == "one_to_many":
            min_refs = relationship.cardinality.get("min", 1) if relationship.cardinality else 1
            max_refs = relationship.cardinality.get("max", 5) if relationship.cardinality else 5
            
            if relationship.distribution == "uniform":
                refs_per_row = np.random.randint(min_refs, max_refs + 1, size)
            elif relationship.distribution == "normal":
                mean_refs = (min_refs + max_refs) / 2
                std_refs = (max_refs - min_refs) / 4
                refs_per_row = np.clip(
                    np.random.normal(mean_refs, std_refs, size),
                    min_refs,
                    max_refs
                ).astype(int)
            elif relationship.distribution == "zipf":
                alpha = relationship.cardinality.get("alpha", 2.0) if relationship.cardinality else 2.0
                refs_per_row = np.random.zipf(alpha, size)
                refs_per_row = np.clip(refs_per_row, min_refs, max_refs)
            
            return np.repeat(primary_keys, refs_per_row)
        
        elif relationship.type == "many_to_many":
            # TODO: Implement many-to-many relationships
            pass
        
        raise ValueError(f"Unsupported relationship type: {relationship.type}")
    
    def apply_relationships(
        self,
        tables: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """Apply relationships to generated tables."""
        # Sort tables by dependency order
        table_order = list(nx.topological_sort(self.graph))
        
        # Process tables in order
        for table_name in table_order:
            # Get incoming relationships
            incoming = self.graph.in_edges(table_name, data=True)
            
            for from_table, to_table, data in incoming:
                relationship = data["relationship"]
                
                # Generate foreign keys
                primary_keys = tables[to_table][relationship.keys[0]].values
                size = len(tables[from_table])
                
                foreign_keys = self._generate_foreign_keys(
                    relationship,
                    primary_keys,
                    size
                )
                
                # Update table with foreign keys
                tables[from_table][relationship.keys[0]] = foreign_keys
        
        return tables


def apply_pattern(
    pattern: Pattern,
    data: pd.DataFrame
) -> pd.DataFrame:
    """Apply a pattern to data."""
    if pattern.type == "sequence":
        generator = SequencePattern(**pattern.parameters)
    elif pattern.type == "cyclic":
        generator = CyclicPattern(**pattern.parameters)
    elif pattern.type == "trend":
        generator = TrendPattern(**pattern.parameters)
    elif pattern.type == "correlation":
        generator = CorrelationPattern(**pattern.parameters)
    else:
        raise ValueError(f"Unsupported pattern type: {pattern.type}")
    
    return generator.apply(data) 