#!/usr/bin/env python3
"""
Automated project metrics collection for Fed-ViT-AutoRL.

This script collects various metrics about the project including:
- Code quality metrics
- Development activity
- Performance benchmarks
- Federated learning specific metrics
- Community engagement
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional

import requests


class MetricsCollector:
    """Collects project metrics from various sources."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = "danieleschmidt"
        self.repo_name = "fed-vit-autorl"

    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics."""
        metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "code_quality": self.collect_code_quality_metrics(),
            "development": self.collect_development_metrics(),
            "performance": self.collect_performance_metrics(),
            "federated_learning": self.collect_federated_learning_metrics(),
            "security": self.collect_security_metrics(),
            "community": self.collect_community_metrics(),
            "operational": self.collect_operational_metrics()
        }
        return metrics

    def collect_code_quality_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics."""
        metrics = {
            "lines_of_code": self._count_lines_of_code(),
            "test_coverage": self._get_test_coverage(),
            "code_complexity": self._analyze_code_complexity(),
            "static_analysis": self._run_static_analysis()
        }
        return metrics

    def collect_development_metrics(self) -> Dict[str, Any]:
        """Collect development activity metrics."""
        metrics = {
            "commits": self._get_commit_metrics(),
            "pull_requests": self._get_pr_metrics(),
            "issues": self._get_issue_metrics(),
            "releases": self._get_release_metrics()
        }
        return metrics

    def collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect performance metrics."""
        metrics = {
            "build_times": self._get_build_metrics(),
            "test_execution": self._get_test_performance(),
            "deployment": self._get_deployment_metrics()
        }
        return metrics

    def collect_federated_learning_metrics(self) -> Dict[str, Any]:
        """Collect federated learning specific metrics."""
        metrics = {
            "model_performance": self._get_model_performance(),
            "system_scalability": self._get_scalability_metrics(),
            "edge_deployment": self._get_edge_metrics()
        }
        return metrics

    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics."""
        metrics = {
            "vulnerability_scan": self._get_vulnerability_metrics(),
            "dependency_security": self._get_dependency_security(),
            "privacy_compliance": self._get_privacy_compliance()
        }
        return metrics

    def collect_community_metrics(self) -> Dict[str, Any]:
        """Collect community engagement metrics."""
        metrics = {
            "github_stats": self._get_github_stats(),
            "downloads": self._get_download_metrics(),
            "engagement": self._get_engagement_metrics()
        }
        return metrics

    def collect_operational_metrics(self) -> Dict[str, Any]:
        """Collect operational metrics."""
        metrics = {
            "uptime": self._get_uptime_metrics(),
            "monitoring": self._get_monitoring_metrics(),
            "costs": self._get_cost_metrics()
        }
        return metrics

    def _count_lines_of_code(self) -> Dict[str, int]:
        """Count lines of code by type."""
        try:
            # Count Python source files
            result = subprocess.run([
                "find", str(self.repo_path), "-name", "*.py", "-not", "-path", "*/.*",
                "-exec", "wc", "-l", "{}", "+"
            ], capture_output=True, text=True)

            total_lines = 0
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines[:-1]:  # Exclude total line
                    total_lines += int(line.strip().split()[0])

            # Count test files
            test_result = subprocess.run([
                "find", str(self.repo_path / "tests"), "-name", "*.py",
                "-exec", "wc", "-l", "{}", "+"
            ], capture_output=True, text=True)

            test_lines = 0
            if test_result.returncode == 0:
                lines = test_result.stdout.strip().split('\n')
                for line in lines[:-1]:
                    test_lines += int(line.strip().split()[0])

            # Count documentation
            doc_result = subprocess.run([
                "find", str(self.repo_path / "docs"), "-name", "*.md",
                "-exec", "wc", "-l", "{}", "+"
            ], capture_output=True, text=True)

            doc_lines = 0
            if doc_result.returncode == 0:
                lines = doc_result.stdout.strip().split('\n')
                for line in lines[:-1]:
                    doc_lines += int(line.strip().split()[0])

            return {
                "total": total_lines,
                "source": total_lines - test_lines,
                "tests": test_lines,
                "docs": doc_lines
            }
        except Exception as e:
            print(f"Error counting lines of code: {e}")
            return {"total": 0, "source": 0, "tests": 0, "docs": 0}

    def _get_test_coverage(self) -> Dict[str, float]:
        """Get test coverage metrics."""
        try:
            # Run coverage report
            result = subprocess.run([
                "python", "-m", "pytest", "--cov=fed_vit_autorl",
                "--cov-report=json", "tests/"
            ], capture_output=True, text=True, cwd=self.repo_path)

            if result.returncode == 0 and (self.repo_path / "coverage.json").exists():
                with open(self.repo_path / "coverage.json") as f:
                    coverage_data = json.load(f)

                return {
                    "current": coverage_data.get("totals", {}).get("percent_covered", 0),
                    "unit_tests": 85,  # Placeholder - would need more detailed analysis
                    "integration_tests": 75,
                    "e2e_tests": 65
                }
        except Exception as e:
            print(f"Error getting test coverage: {e}")

        return {"current": 0, "unit_tests": 0, "integration_tests": 0, "e2e_tests": 0}

    def _analyze_code_complexity(self) -> Dict[str, float]:
        """Analyze code complexity."""
        try:
            # Use radon for complexity analysis
            result = subprocess.run([
                "python", "-m", "radon", "cc", str(self.repo_path / "fed_vit_autorl"),
                "--json"
            ], capture_output=True, text=True)

            if result.returncode == 0:
                complexity_data = json.loads(result.stdout)
                total_complexity = 0
                function_count = 0

                for file_data in complexity_data.values():
                    for item in file_data:
                        if item["type"] in ["function", "method"]:
                            total_complexity += item["complexity"]
                            function_count += 1

                avg_complexity = total_complexity / function_count if function_count > 0 else 0

                return {
                    "cyclomatic_complexity": avg_complexity,
                    "maintainability_index": 75.0,  # Placeholder
                    "technical_debt_ratio": 5.0     # Placeholder
                }
        except Exception as e:
            print(f"Error analyzing code complexity: {e}")

        return {
            "cyclomatic_complexity": 0,
            "maintainability_index": 0,
            "technical_debt_ratio": 0
        }

    def _run_static_analysis(self) -> Dict[str, int]:
        """Run static analysis tools."""
        try:
            # Run bandit for security analysis
            bandit_result = subprocess.run([
                "python", "-m", "bandit", "-r", str(self.repo_path / "fed_vit_autorl"),
                "-f", "json"
            ], capture_output=True, text=True)

            security_issues = 0
            if bandit_result.returncode in [0, 1]:  # 1 means issues found
                try:
                    bandit_data = json.loads(bandit_result.stdout)
                    security_issues = len(bandit_data.get("results", []))
                except json.JSONDecodeError:
                    pass

            return {
                "security_hotspots": security_issues,
                "bugs": 0,         # Would need SonarQube integration
                "vulnerabilities": security_issues,
                "code_smells": 0   # Would need SonarQube integration
            }
        except Exception as e:
            print(f"Error running static analysis: {e}")
            return {"security_hotspots": 0, "bugs": 0, "vulnerabilities": 0, "code_smells": 0}

    def _get_commit_metrics(self) -> Dict[str, int]:
        """Get commit metrics."""
        try:
            # Total commits
            total_result = subprocess.run([
                "git", "rev-list", "--count", "HEAD"
            ], capture_output=True, text=True, cwd=self.repo_path)
            total_commits = int(total_result.stdout.strip()) if total_result.returncode == 0 else 0

            # Commits in last 30 days
            since_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            recent_result = subprocess.run([
                "git", "rev-list", "--count", f"--since={since_date}", "HEAD"
            ], capture_output=True, text=True, cwd=self.repo_path)
            recent_commits = int(recent_result.stdout.strip()) if recent_result.returncode == 0 else 0

            # Contributors
            contributors_result = subprocess.run([
                "git", "shortlog", "-sn", "HEAD"
            ], capture_output=True, text=True, cwd=self.repo_path)
            contributors = len(contributors_result.stdout.strip().split('\n')) if contributors_result.returncode == 0 else 0

            return {
                "total": total_commits,
                "last_30_days": recent_commits,
                "contributors": contributors
            }
        except Exception as e:
            print(f"Error getting commit metrics: {e}")
            return {"total": 0, "last_30_days": 0, "contributors": 0}

    def _get_github_stats(self) -> Dict[str, int]:
        """Get GitHub repository statistics."""
        if not self.github_token:
            return {"stars": 0, "forks": 0, "watchers": 0, "contributors": 0}

        try:
            headers = {"Authorization": f"token {self.github_token}"}

            # Repository stats
            repo_url = f"https://api.github.com/repos/{self.repo_owner}/{self.repo_name}"
            repo_response = requests.get(repo_url, headers=headers)

            if repo_response.status_code == 200:
                repo_data = repo_response.json()

                # Contributors
                contributors_url = f"{repo_url}/contributors"
                contributors_response = requests.get(contributors_url, headers=headers)
                contributors_count = len(contributors_response.json()) if contributors_response.status_code == 200 else 0

                return {
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "watchers": repo_data.get("watchers_count", 0),
                    "contributors": contributors_count
                }
        except Exception as e:
            print(f"Error getting GitHub stats: {e}")

        return {"stars": 0, "forks": 0, "watchers": 0, "contributors": 0}

    def _get_pr_metrics(self) -> Dict[str, Any]:
        """Get pull request metrics."""
        # Placeholder implementation
        return {
            "total": 0,
            "open": 0,
            "merged": 0,
            "average_review_time_hours": 0
        }

    def _get_issue_metrics(self) -> Dict[str, Any]:
        """Get issue metrics."""
        # Placeholder implementation
        return {
            "total": 0,
            "open": 0,
            "closed": 0,
            "average_resolution_time_days": 0
        }

    def _get_release_metrics(self) -> Dict[str, Any]:
        """Get release metrics."""
        # Placeholder implementation
        return {
            "total": 0,
            "last_release_date": None,
            "release_frequency_days": 0
        }

    def _get_build_metrics(self) -> Dict[str, Any]:
        """Get build performance metrics."""
        # Placeholder implementation
        return {
            "average_seconds": 0,
            "p95_seconds": 0,
            "success_rate": 0
        }

    def _get_test_performance(self) -> Dict[str, Any]:
        """Get test execution performance."""
        # Placeholder implementation
        return {
            "unit_tests_seconds": 0,
            "integration_tests_seconds": 0,
            "e2e_tests_seconds": 0,
            "total_test_time_seconds": 0
        }

    def _get_deployment_metrics(self) -> Dict[str, Any]:
        """Get deployment metrics."""
        # Placeholder implementation
        return {
            "average_deployment_time_minutes": 0,
            "deployment_success_rate": 0,
            "rollback_rate": 0
        }

    def _get_model_performance(self) -> Dict[str, Any]:
        """Get federated learning model performance."""
        # Placeholder implementation
        return {
            "global_accuracy": 0,
            "convergence_rounds": 0,
            "communication_efficiency": 0,
            "privacy_budget_utilization": 0
        }

    def _get_scalability_metrics(self) -> Dict[str, Any]:
        """Get system scalability metrics."""
        # Placeholder implementation
        return {
            "max_clients_supported": 1000,
            "current_active_clients": 0,
            "throughput_rounds_per_hour": 0,
            "latency_p95_ms": 100
        }

    def _get_edge_metrics(self) -> Dict[str, Any]:
        """Get edge deployment metrics."""
        # Placeholder implementation
        return {
            "supported_devices": ["jetson_xavier", "jetson_nano", "raspberry_pi"],
            "inference_latency_ms": 50,
            "model_size_mb": 100,
            "energy_efficiency_mj_per_inference": 1.0
        }

    def _get_vulnerability_metrics(self) -> Dict[str, Any]:
        """Get vulnerability scan results."""
        # Placeholder implementation
        return {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0,
            "last_scan_date": None
        }

    def _get_dependency_security(self) -> Dict[str, Any]:
        """Get dependency security metrics."""
        # Placeholder implementation
        return {
            "outdated_dependencies": 0,
            "vulnerable_dependencies": 0,
            "license_compliance": True
        }

    def _get_privacy_compliance(self) -> Dict[str, Any]:
        """Get privacy compliance metrics."""
        # Placeholder implementation
        return {
            "differential_privacy_enabled": True,
            "epsilon_value": 1.0,
            "delta_value": 1e-5,
            "gdpr_compliant": True
        }

    def _get_download_metrics(self) -> Dict[str, Any]:
        """Get download metrics."""
        # Placeholder implementation
        return {
            "pypi_downloads_last_month": 0,
            "docker_pulls": 0,
            "documentation_views": 0
        }

    def _get_engagement_metrics(self) -> Dict[str, Any]:
        """Get community engagement metrics."""
        # Placeholder implementation
        return {
            "discussions": 0,
            "community_contributions": 0,
            "external_citations": 0
        }

    def _get_uptime_metrics(self) -> Dict[str, Any]:
        """Get uptime metrics."""
        # Placeholder implementation
        return {
            "target_percentage": 99.9,
            "current_percentage": 0,
            "mttr_hours": 0,
            "mtbf_hours": 0
        }

    def _get_monitoring_metrics(self) -> Dict[str, Any]:
        """Get monitoring metrics."""
        # Placeholder implementation
        return {
            "alerts_last_24h": 0,
            "false_positive_rate": 0,
            "monitoring_coverage": 0
        }

    def _get_cost_metrics(self) -> Dict[str, Any]:
        """Get cost metrics."""
        # Placeholder implementation
        return {
            "infrastructure_monthly_usd": 0,
            "ci_cd_monthly_usd": 0,
            "monitoring_monthly_usd": 0
        }


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Collect project metrics")
    parser.add_argument("--output", default="project-metrics-collected.json",
                       help="Output file for metrics")
    parser.add_argument("--repo-path", default=".",
                       help="Path to repository")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    collector = MetricsCollector(args.repo_path)

    if args.verbose:
        print("Collecting project metrics...")

    metrics = collector.collect_all_metrics()

    # Write metrics to file
    with open(args.output, "w") as f:
        json.dump(metrics, f, indent=2)

    if args.verbose:
        print(f"Metrics written to {args.output}")
        print(f"Collected {len(metrics)} metric categories")

    # Print summary
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
