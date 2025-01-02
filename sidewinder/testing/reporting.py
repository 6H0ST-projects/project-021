"""
Test reporting functionality.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd


class TestResult(BaseModel):
    """Result of a single test."""
    name: str
    type: str
    status: str
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None


class TestSuiteResult(BaseModel):
    """Results of a test suite."""
    name: str
    start_time: datetime
    end_time: datetime
    tests: List[TestResult]
    success_rate: float
    total_duration: float
    metrics: Optional[Dict[str, float]] = None


class TestReport:
    """Test report generator."""
    
    def __init__(self, results: List[TestSuiteResult]):
        self.results = results
    
    def generate_html_report(self, output_dir: str):
        """Generate HTML report with test results."""
        # Create test results table
        results_df = pd.DataFrame([
            {
                "Suite": result.name,
                "Success Rate": f"{result.success_rate * 100:.1f}%",
                "Duration": f"{result.total_duration:.2f}s",
                "Tests": len(result.tests),
                "Failed": len([t for t in result.tests if t.status == "failed"]),
                "Start Time": result.start_time.strftime("%Y-%m-%d %H:%M:%S")
            }
            for result in self.results
        ])
        
        # Create test results figure
        fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=list(results_df.columns),
                    fill_color="paleturquoise",
                    align="left"
                ),
                cells=dict(
                    values=[results_df[col] for col in results_df.columns],
                    fill_color="lavender",
                    align="left"
                )
            )
        ])
        
        fig.update_layout(
            title="Test Suite Results",
            margin=dict(t=30, b=10)
        )
        
        # Save HTML report
        fig.write_html(f"{output_dir}/test_report.html")
    
    def generate_performance_report(
        self,
        performance_metrics: List[Dict[str, Any]],
        output_dir: str
    ):
        """Generate performance metrics report."""
        # Create performance metrics figure
        metrics_df = pd.DataFrame(performance_metrics)
        
        fig = go.Figure()
        
        # Add execution time trace
        fig.add_trace(go.Scatter(
            x=list(range(len(metrics_df))),
            y=metrics_df["execution_time"],
            name="Execution Time (s)",
            mode="lines+markers"
        ))
        
        # Add memory usage trace
        fig.add_trace(go.Scatter(
            x=list(range(len(metrics_df))),
            y=metrics_df["memory_usage_mb"],
            name="Memory Usage (MB)",
            mode="lines+markers",
            yaxis="y2"
        ))
        
        fig.update_layout(
            title="Performance Metrics",
            xaxis_title="Test Run",
            yaxis_title="Execution Time (s)",
            yaxis2=dict(
                title="Memory Usage (MB)",
                overlaying="y",
                side="right"
            ),
            showlegend=True
        )
        
        # Save performance report
        fig.write_html(f"{output_dir}/performance_report.html") 