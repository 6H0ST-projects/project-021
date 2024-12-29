"""
Test reporting module for visualizing and analyzing test results.
"""

from typing import List, Dict, Any
import json
from datetime import datetime
import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sidewinder.testing.environment import TestResult, TestSuiteResult


class TestReport:
    """Generates reports from test results."""
    
    def __init__(self, results: List[TestSuiteResult]):
        self.results = results
        self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert test results to dictionary format."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "suites": [
                {
                    "name": suite.suite_name,
                    "total_duration_seconds": suite.total_duration_seconds,
                    "peak_memory_gb": suite.peak_memory_gb,
                    "avg_cpu_percent": suite.avg_cpu_percent,
                    "success_rate": suite.success_rate,
                    "tests": [
                        {
                            "name": test.test_name,
                            "success": test.success,
                            "duration_seconds": test.duration_seconds,
                            "memory_usage_gb": test.memory_usage_gb,
                            "cpu_percent": test.cpu_percent,
                            "error_message": test.error_message,
                            "metrics": test.metrics
                        }
                        for test in suite.results
                    ]
                }
                for suite in self.results
            ]
        }
    
    def save_json(self, output_dir: str):
        """Save test results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{timestamp_str}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def generate_html_report(self, output_dir: str):
        """Generate HTML report with interactive visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create DataFrame for analysis
        records = []
        for suite in self.results:
            for test in suite.results:
                records.append({
                    "suite_name": suite.suite_name,
                    "test_name": test.test_name,
                    "success": test.success,
                    "duration_seconds": test.duration_seconds,
                    "memory_usage_gb": test.memory_usage_gb,
                    "cpu_percent": test.cpu_percent
                })
        
        df = pd.DataFrame(records)
        
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Test Duration by Suite",
                "Memory Usage by Suite",
                "CPU Usage by Suite",
                "Success Rate by Suite"
            )
        )
        
        # Duration boxplot
        fig.add_trace(
            go.Box(
                x=df["suite_name"],
                y=df["duration_seconds"],
                name="Duration",
                boxmean=True
            ),
            row=1,
            col=1
        )
        
        # Memory usage boxplot
        fig.add_trace(
            go.Box(
                x=df["suite_name"],
                y=df["memory_usage_gb"],
                name="Memory",
                boxmean=True
            ),
            row=1,
            col=2
        )
        
        # CPU usage boxplot
        fig.add_trace(
            go.Box(
                x=df["suite_name"],
                y=df["cpu_percent"],
                name="CPU",
                boxmean=True
            ),
            row=2,
            col=1
        )
        
        # Success rate bar chart
        success_rate = df.groupby("suite_name")["success"].mean()
        fig.add_trace(
            go.Bar(
                x=success_rate.index,
                y=success_rate.values,
                name="Success Rate"
            ),
            row=2,
            col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Test Suite Performance Metrics",
            showlegend=False
        )
        
        # Save HTML report
        timestamp_str = self.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"test_report_{timestamp_str}.html"
        filepath = os.path.join(output_dir, filename)
        
        fig.write_html(filepath)
        
        # Generate summary table
        summary_df = pd.DataFrame([
            {
                "suite_name": suite.suite_name,
                "total_duration": f"{suite.total_duration_seconds:.2f}s",
                "peak_memory": f"{suite.peak_memory_gb:.2f}GB",
                "avg_cpu": f"{suite.avg_cpu_percent:.1f}%",
                "success_rate": f"{suite.success_rate * 100:.1f}%",
                "total_tests": len(suite.results),
                "failed_tests": len([t for t in suite.results if not t.success])
            }
            for suite in self.results
        ])
        
        # Save summary table
        summary_filename = f"test_summary_{timestamp_str}.html"
        summary_filepath = os.path.join(output_dir, summary_filename)
        
        with open(summary_filepath, "w") as f:
            f.write("<html><head><style>")
            f.write("table { border-collapse: collapse; width: 100%; }")
            f.write("th, td { padding: 8px; text-align: left; border: 1px solid #ddd; }")
            f.write("th { background-color: #f2f2f2; }")
            f.write("tr:nth-child(even) { background-color: #f9f9f9; }")
            f.write("</style></head><body>")
            f.write("<h2>Test Suite Summary</h2>")
            f.write(summary_df.to_html(index=False))
            f.write("</body></html>")
    
    def print_summary(self):
        """Print summary of test results to console."""
        total_tests = sum(len(suite.results) for suite in self.results)
        total_success = sum(
            len([t for t in suite.results if t.success])
            for suite in self.results
        )
        
        print("\n=== Test Execution Summary ===")
        print(f"Timestamp: {self.timestamp}")
        print(f"Total Suites: {len(self.results)}")
        print(f"Total Tests: {total_tests}")
        print(f"Overall Success Rate: {(total_success / total_tests) * 100:.1f}%")
        print("\nSuite Details:")
        
        for suite in self.results:
            print(f"\n{suite.suite_name}:")
            print(f"  Duration: {suite.total_duration_seconds:.2f}s")
            print(f"  Peak Memory: {suite.peak_memory_gb:.2f}GB")
            print(f"  Avg CPU: {suite.avg_cpu_percent:.1f}%")
            print(f"  Success Rate: {suite.success_rate * 100:.1f}%")
            
            failed_tests = [t for t in suite.results if not t.success]
            if failed_tests:
                print("\n  Failed Tests:")
                for test in failed_tests:
                    print(f"    - {test.test_name}")
                    if test.error_message:
                        print(f"      Error: {test.error_message}")


def load_report(json_file: str) -> TestReport:
    """Load test report from JSON file."""
    with open(json_file, "r") as f:
        data = json.load(f)
    
    # Convert JSON data back to TestSuiteResult objects
    suites = []
    for suite_data in data["suites"]:
        results = []
        for test_data in suite_data["tests"]:
            results.append(
                TestResult(
                    success=test_data["success"],
                    test_name=test_data["name"],
                    duration_seconds=test_data["duration_seconds"],
                    memory_usage_gb=test_data["memory_usage_gb"],
                    cpu_percent=test_data["cpu_percent"],
                    error_message=test_data.get("error_message"),
                    metrics=test_data.get("metrics", {})
                )
            )
        
        suites.append(
            TestSuiteResult(
                suite_name=suite_data["name"],
                results=results,
                total_duration_seconds=suite_data["total_duration_seconds"],
                peak_memory_gb=suite_data["peak_memory_gb"],
                avg_cpu_percent=suite_data["avg_cpu_percent"],
                success_rate=suite_data["success_rate"]
            )
        )
    
    report = TestReport(suites)
    report.timestamp = datetime.fromisoformat(data["timestamp"])
    return report 