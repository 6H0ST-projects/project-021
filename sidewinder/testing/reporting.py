"""
Test reporting functionality.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
import json
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd


class RequirementStatus(BaseModel):
    """Status of a single requirement."""
    requirement: str
    status: str  # "passed", "failed", "not_implemented"
    details: Optional[str] = None
    code_snippet: Optional[str] = None


class TestResult(BaseModel):
    """Result of a single test."""
    name: str
    type: str
    status: str
    duration: float
    error: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    metrics: Optional[Dict[str, float]] = None
    requirements: List[RequirementStatus] = Field(default_factory=list)
    generated_code: Optional[str] = None


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
        
        # Generate detailed requirements report
        requirements_data = []
        for suite in self.results:
            for test in suite.tests:
                for req in test.requirements:
                    requirements_data.append({
                        "Suite": suite.name,
                        "Test": test.name,
                        "Requirement": req.requirement,
                        "Status": req.status,
                        "Details": req.details or ""
                    })
        
        if requirements_data:
            req_df = pd.DataFrame(requirements_data)
            req_fig = go.Figure(data=[
                go.Table(
                    header=dict(
                        values=list(req_df.columns),
                        fill_color="paleturquoise",
                        align="left"
                    ),
                    cells=dict(
                        values=[req_df[col] for col in req_df.columns],
                        fill_color=[
                            ["lightgreen" if status == "passed"
                             else "lightpink" if status == "failed"
                             else "lightyellow"
                             for status in req_df["Status"]]
                            for _ in range(len(req_df.columns))
                        ],
                        align="left"
                    )
                )
            ])
            
            req_fig.update_layout(
                title="Requirements Status",
                margin=dict(t=30, b=10)
            )
            
            # Save requirements report
            req_fig.write_html(f"{output_dir}/requirements_report.html")
        
        # Save main report
        fig.write_html(f"{output_dir}/test_report.html")
        
        # Generate code report
        code_report = []
        for suite in self.results:
            for test in suite.tests:
                if test.generated_code:
                    code_report.append(
                        f"<h2>{suite.name} - {test.name}</h2>\n"
                        f"<h3>Status: {test.status}</h3>\n"
                        f"<h4>Requirements:</h4>\n"
                        f"<ul>\n"
                        + "\n".join(
                            f"<li>{req.requirement}: <b>{req.status}</b>"
                            + (f" - {req.details}" if req.details else "")
                            + "</li>"
                            for req in test.requirements
                        )
                        + "\n</ul>\n"
                        f"<h4>Generated Code:</h4>\n"
                        f"<pre><code class='python'>{test.generated_code}</code></pre>\n"
                    )
        
        if code_report:
            with open(f"{output_dir}/code_report.html", "w") as f:
                f.write(
                    "<html>\n<head>\n"
                    "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css'>\n"
                    "<script src='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js'></script>\n"
                    "<script>hljs.highlightAll();</script>\n"
                    "<style>pre { background-color: #f6f8fa; padding: 16px; border-radius: 6px; }</style>\n"
                    "</head>\n<body>\n"
                    + "\n".join(code_report)
                    + "\n</body>\n</html>"
                )
    
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