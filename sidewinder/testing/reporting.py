"""
Test reporting functionality for Sidewinder.
"""

from typing import Dict, Any, List
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


class TestReport:
    """Generate test execution reports."""
    
    def __init__(self, results: List[Dict[str, Any]]):
        self.results = results
    
    def generate_html_report(self, output_dir: str):
        """Generate HTML test report."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create report sections
        sections = [
            self._generate_summary_section(),
            self._generate_test_details_section(),
            self._generate_error_section()
        ]
        
        # Combine sections into full report
        report_html = """
        <html>
        <head>
            <title>Sidewinder Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .success { color: green; }
                .failure { color: red; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f5f5f5; }
            </style>
        </head>
        <body>
            <h1>Sidewinder Test Report</h1>
            {}
        </body>
        </html>
        """.format("\n".join(sections))
        
        # Write report
        with open(f"{output_dir}/test_report.html", "w") as f:
            f.write(report_html)
    
    def generate_performance_report(
        self,
        performance_metrics: List[Dict[str, Any]],
        output_dir: str
    ):
        """Generate performance metrics report."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create performance visualizations
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                "LLM Response Times",
                "Component Execution Times",
                "Resource Usage",
                "Success Metrics"
            )
        )
        
        # LLM response times
        llm_times = pd.DataFrame([
            {
                "Component": component,
                "Time (s)": metrics["llm"][component]["time"]
            }
            for metrics in performance_metrics
            for component in metrics["llm"]
        ])
        
        fig.add_trace(
            go.Bar(
                x=llm_times["Component"],
                y=llm_times["Time (s)"],
                name="LLM Response Time"
            ),
            row=1, col=1
        )
        
        # Component execution times
        exec_times = pd.DataFrame([
            {
                "Component": "Analyzer",
                "Time (s)": metrics["execution"]["analyzer_time"]
            }
            for metrics in performance_metrics
        ] + [
            {
                "Component": "Transformer",
                "Time (s)": metrics["execution"]["transformer_time"]
            }
            for metrics in performance_metrics
        ] + [
            {
                "Component": "Tests",
                "Time (s)": metrics["execution"]["test_time"]
            }
            for metrics in performance_metrics
        ])
        
        fig.add_trace(
            go.Bar(
                x=exec_times["Component"],
                y=exec_times["Time (s)"],
                name="Execution Time"
            ),
            row=1, col=2
        )
        
        # Resource usage
        resource_usage = pd.DataFrame([
            {
                "Metric": "Memory (MB)",
                "Value": metrics["execution"]["memory_usage"]
            }
            for metrics in performance_metrics
        ] + [
            {
                "Metric": "CPU (%)",
                "Value": metrics["execution"]["cpu_usage"]
            }
            for metrics in performance_metrics
        ])
        
        fig.add_trace(
            go.Bar(
                x=resource_usage["Metric"],
                y=resource_usage["Value"],
                name="Resource Usage"
            ),
            row=2, col=1
        )
        
        # Success metrics
        success_metrics = pd.DataFrame([
            {
                "Metric": "Success Rate",
                "Value": metrics["execution"]["success_rate"] * 100
            }
            for metrics in performance_metrics
        ] + [
            {
                "Metric": "Error Count",
                "Value": metrics["execution"]["error_count"]
            }
            for metrics in performance_metrics
        ])
        
        fig.add_trace(
            go.Bar(
                x=success_metrics["Metric"],
                y=success_metrics["Value"],
                name="Success Metrics"
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            width=1200,
            title_text="Performance Metrics",
            showlegend=False
        )
        
        # Save plot
        fig.write_html(f"{output_dir}/performance_report.html")
        
        # Save raw metrics
        with open(f"{output_dir}/performance_metrics.json", "w") as f:
            json.dump(performance_metrics, f, indent=2)
    
    def _generate_summary_section(self) -> str:
        """Generate summary section of the report."""
        total_tests = sum(len(r.get("tests", [])) for r in self.results)
        passed_tests = sum(
            len([t for t in r.get("tests", []) if t["status"] == "passed"])
            for r in self.results
        )
        
        return f"""
        <div class="section">
            <h2>Summary</h2>
            <p>Total Tests: {total_tests}</p>
            <p>Passed Tests: {passed_tests}</p>
            <p>Success Rate: {(passed_tests/total_tests*100):.2f}%</p>
        </div>
        """
    
    def _generate_test_details_section(self) -> str:
        """Generate test details section of the report."""
        details = []
        
        for result in self.results:
            scenario = result.get("scenario", "Unknown")
            tests = result.get("tests", [])
            
            details.append(f"""
            <div class="section">
                <h3>Scenario: {scenario}</h3>
                <table>
                    <tr>
                        <th>Test</th>
                        <th>Status</th>
                        <th>Duration</th>
                        <th>Details</th>
                    </tr>
                    {"".join(
                        f'''
                        <tr>
                            <td>{test["name"]}</td>
                            <td class="{test["status"]}">{test["status"]}</td>
                            <td>{test.get("duration", "N/A")}s</td>
                            <td>{test.get("details", "")}</td>
                        </tr>
                        '''
                        for test in tests
                    )}
                </table>
            </div>
            """)
        
        return "\n".join(details)
    
    def _generate_error_section(self) -> str:
        """Generate error section of the report."""
        errors = []
        
        for result in self.results:
            scenario = result.get("scenario", "Unknown")
            if "error" in result:
                errors.append(f"""
                <div class="failure">
                    <h4>Scenario: {scenario}</h4>
                    <pre>{result["error"]}</pre>
                </div>
                """)
        
        if not errors:
            return ""
        
        return f"""
        <div class="section">
            <h2>Errors</h2>
            {"".join(errors)}
        </div>
        """ 