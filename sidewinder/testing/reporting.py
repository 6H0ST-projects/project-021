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
    generated_code: Optional[Dict[str, str]] = Field(default_factory=dict)  # Layer -> Code mapping
    test_results: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)  # Test name -> Results


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
        
        # Generate scalability report if available
        for suite in self.results:
            for test in suite.tests:
                if test.details and "scalability_results" in test.details:
                    self._generate_scalability_report(
                        test.details["scalability_results"],
                        output_dir
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
        
        # Generate detailed code and test report
        code_report = []
        for suite in self.results:
            for test in suite.tests:
                # Add test overview
                code_report.append(
                    f"<h2>{suite.name} - {test.name}</h2>\n"
                    f"<h3>Status: <span style='color: {'green' if test.status == 'passed' else 'red'}'>{test.status}</span></h3>\n"
                )
                
                # Add error if present
                if test.error:
                    code_report.append(
                        f"<div class='error-box'>\n"
                        f"<h4>Error:</h4>\n"
                        f"<pre>{test.error}</pre>\n"
                        f"</div>\n"
                    )
                
                # Add requirements section
                code_report.append(
                    f"<h4>Requirements:</h4>\n"
                    f"<ul>\n"
                    + "\n".join(
                        f"<li>{req.requirement}: "
                        f"<b style='color: {'green' if req.status == 'passed' else 'red' if req.status == 'failed' else 'orange'}'>"
                        f"{req.status}</b>"
                        + (f" - {req.details}" if req.details else "")
                        + "</li>"
                        for req in test.requirements
                    )
                    + "\n</ul>\n"
                )
                
                # Add generated code for each layer
                if test.generated_code:
                    code_report.append("<h4>Generated Code:</h4>")
                    for layer in ["bronze", "silver", "gold"]:
                        if layer in test.generated_code:
                            code_report.append(
                                f"<h5>{layer.title()} Layer:</h5>\n"
                                f"<pre><code class='python'>{test.generated_code[layer]}</code></pre>\n"
                            )
                
                # Add test results
                if test.test_results:
                    code_report.append(
                        f"<h4>Test Results:</h4>\n"
                        f"<ul>\n"
                    )
                    for test_name, result in test.test_results.items():
                        code_report.append(
                            f"<li><b>{test_name}</b>: "
                            f"<span style='color: {'green' if result.get('passed', False) else 'red'}'>"
                            f"{'Passed' if result.get('passed', False) else 'Failed'}</span>"
                            + (f"<br><pre>{result.get('error', '')}</pre>" if result.get('error') else "")
                            + (f"<br>Duration: {result.get('duration', 0):.2f}s" if 'duration' in result else "")
                            + "</li>"
                        )
                    code_report.append("</ul>\n")
                
                # Add metrics if present
                if test.metrics:
                    code_report.append(
                        f"<h4>Performance Metrics:</h4>\n"
                        f"<ul>\n"
                        + "\n".join(
                            f"<li><b>{k}:</b> {v}</li>"
                            for k, v in test.metrics.items()
                        )
                        + "\n</ul>\n"
                    )
        
        if code_report:
            with open(f"{output_dir}/code_report.html", "w") as f:
                f.write(
                    "<html>\n<head>\n"
                    "<link rel='stylesheet' href='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/styles/default.min.css'>\n"
                    "<script src='https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.7.0/highlight.min.js'></script>\n"
                    "<script>hljs.highlightAll();</script>\n"
                    "<style>\n"
                    "body { font-family: Arial, sans-serif; margin: 20px; }\n"
                    "pre { background-color: #f6f8fa; padding: 16px; border-radius: 6px; }\n"
                    ".error-box { background-color: #fff0f0; padding: 16px; border-radius: 6px; margin: 10px 0; }\n"
                    "h2 { border-bottom: 2px solid #eaecef; padding-bottom: 8px; }\n"
                    "h3, h4, h5 { margin-top: 20px; }\n"
                    "</style>\n"
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
    
    def _generate_scalability_report(
        self,
        scalability_results: Dict[int, Dict[str, Any]],
        output_dir: str
    ):
        """Generate scalability test results report."""
        # Create DataFrame from results
        scales = list(scalability_results.keys())
        metrics = ["duration", "peak_memory_gb", "avg_cpu_percent"]
        
        # Create figure with secondary y-axis
        fig = go.Figure()
        
        # Add execution time trace
        fig.add_trace(go.Scatter(
            x=scales,
            y=[results["duration"] for results in scalability_results.values()],
            name="Execution Time (s)",
            mode="lines+markers"
        ))
        
        # Add memory usage trace
        fig.add_trace(go.Scatter(
            x=scales,
            y=[results["peak_memory_gb"] for results in scalability_results.values()],
            name="Peak Memory (GB)",
            mode="lines+markers",
            yaxis="y2"
        ))
        
        # Add CPU usage trace
        fig.add_trace(go.Scatter(
            x=scales,
            y=[results["avg_cpu_percent"] for results in scalability_results.values()],
            name="Avg CPU (%)",
            mode="lines+markers",
            yaxis="y3"
        ))
        
        # Update layout with multiple y-axes
        fig.update_layout(
            title="Scalability Test Results",
            xaxis=dict(
                title="Scale Factor",
                type="log"
            ),
            yaxis=dict(
                title="Execution Time (s)",
                titlefont=dict(color="#1f77b4"),
                tickfont=dict(color="#1f77b4")
            ),
            yaxis2=dict(
                title="Peak Memory (GB)",
                titlefont=dict(color="#ff7f0e"),
                tickfont=dict(color="#ff7f0e"),
                anchor="free",
                overlaying="y",
                side="right",
                position=0.85
            ),
            yaxis3=dict(
                title="Avg CPU (%)",
                titlefont=dict(color="#2ca02c"),
                tickfont=dict(color="#2ca02c"),
                anchor="free",
                overlaying="y",
                side="right",
                position=1.0
            ),
            showlegend=True,
            legend=dict(
                x=1.1,
                y=1.0
            )
        )
        
        # Save scalability report
        fig.write_html(f"{output_dir}/scalability_report.html")
        
        # Generate detailed metrics table
        metrics_df = pd.DataFrame([
            {
                "Scale": scale,
                "Success Rate": f"{results['success_rate'] * 100:.1f}%",
                "Duration (s)": f"{results['duration']:.2f}",
                "Peak Memory (GB)": f"{results['peak_memory_gb']:.2f}",
                "Avg CPU (%)": f"{results['avg_cpu_percent']:.1f}",
                "Join Performance": json.dumps(results['join_metrics'], indent=2) if results.get('join_metrics') else "N/A"
            }
            for scale, results in scalability_results.items()
        ])
        
        metrics_fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=list(metrics_df.columns),
                    fill_color="paleturquoise",
                    align="left"
                ),
                cells=dict(
                    values=[metrics_df[col] for col in metrics_df.columns],
                    fill_color="lavender",
                    align="left"
                )
            )
        ])
        
        metrics_fig.update_layout(
            title="Detailed Scalability Metrics",
            margin=dict(t=30, b=10)
        )
        
        # Save metrics table
        metrics_fig.write_html(f"{output_dir}/scalability_metrics.html") 