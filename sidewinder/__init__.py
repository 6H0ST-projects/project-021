"""
Sidewinder: An intelligent data engineering agent for building ETL pipelines.
"""

__version__ = "0.0.1"

from sidewinder.core.pipeline import Pipeline
from sidewinder.core.config import Environment, Source, Target

__all__ = ["Pipeline", "Environment", "Source", "Target"] 