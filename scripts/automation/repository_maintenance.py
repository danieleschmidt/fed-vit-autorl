#!/usr/bin/env python3
"""
Automated repository maintenance for Fed-ViT-AutoRL.

This script performs regular repository maintenance tasks including:
- Code quality monitoring
- Repository health checks
- Automated cleanup tasks
- Documentation updates
- Integration health monitoring
"""

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import requests


class RepositoryMaintenance:
    """Performs automated repository maintenance tasks."""
    
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_owner = "danieleschmidt"
        self.repo_name = "fed-vit-autorl"
        
    def run_maintenance_cycle(self) -> Dict[str, Any]:
        """Run complete maintenance cycle."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "tasks_completed": [],
            "tasks_failed": [],
            "recommendations": [],
            "health_score": 0
        }
        
        maintenance_tasks = [
            ("cleanup_temporary_files", self.cleanup_temporary_files),
            ("update_documentation", self.update_documentation),
            ("check_code_quality", self.check_code_quality),
            ("validate_configurations", self.validate_configurations),
            ("check_dependencies", self.check_dependencies),
            ("validate_docker_setup", self.validate_docker_setup),
            ("check_security_settings", self.check_security_settings),
            ("monitor_repository_health", self.monitor_repository_health),
            ("cleanup_old_artifacts", self.cleanup_old_artifacts),
            ("validate_ci_cd", self.validate_ci_cd)
        ]
        
        for task_name, task_func in maintenance_tasks:
            try:
                print(f"Running {task_name}...")
                task_result = task_func()
                
                if task_result.get("success", True):
                    report["tasks_completed"].append({
                        "task": task_name,
                        "result": task_result
                    })
                else:
                    report["tasks_failed"].append({
                        "task": task_name,
                        "error": task_result.get("error", "Unknown error")
                    })
                    
                # Collect recommendations
                if "recommendations" in task_result:
                    report["recommendations"].extend(task_result["recommendations"])
                    
            except Exception as e:
                print(f"Error in {task_name}: {e}")
                report["tasks_failed"].append({
                    "task": task_name,
                    "error": str(e)
                })
        
        # Calculate health score
        total_tasks = len(maintenance_tasks)
        completed_tasks = len(report["tasks_completed"])
        report["health_score"] = (completed_tasks / total_tasks) * 100
        
        # Generate maintenance summary
        self.generate_maintenance_report(report)
        
        return report
    
    def cleanup_temporary_files(self) -> Dict[str, Any]:
        """Clean up temporary files and caches."""
        result = {"success": True, "cleaned_files": [], "freed_space_mb": 0}
        
        # Patterns for files to clean
        cleanup_patterns = [
            "**/__pycache__",
            "**/*.pyc", 
            "**/*.pyo",
            "**/.pytest_cache",
            "**/.mypy_cache",
            "**/.ruff_cache",
            "**/node_modules",
            "**/.DS_Store",
            "**/Thumbs.db",
            "**/*.tmp",
            "**/*.log",
            "coverage.xml",
            ".coverage",
            "htmlcov/",
            "dist/",
            "build/",
            "*.egg-info/"
        ]
        
        initial_size = self._get_directory_size(self.repo_path)
        
        for pattern in cleanup_patterns:
            try:
                for path in self.repo_path.glob(pattern):
                    if path.exists():
                        if path.is_file():
                            path.unlink()
                            result["cleaned_files"].append(str(path))
                        elif path.is_dir():
                            shutil.rmtree(path)
                            result["cleaned_files"].append(str(path))
            except Exception as e:
                print(f"Error cleaning {pattern}: {e}")
        
        final_size = self._get_directory_size(self.repo_path)
        result["freed_space_mb"] = (initial_size - final_size) / (1024 * 1024)
        
        print(f"Cleaned {len(result['cleaned_files'])} files/directories, "
              f"freed {result['freed_space_mb']:.1f} MB")
        
        return result
    
    def update_documentation(self) -> Dict[str, Any]:
        """Update auto-generated documentation."""
        result = {"success": True, "updates": [], "recommendations": []}
        
        try:
            # Update API documentation
            if (self.repo_path / "docs").exists():
                # Generate API docs if sphinx is available
                try:
                    subprocess.run([
                        "python", "-m", "sphinx.apidoc", "-f", "-o", "docs/api", "fed_vit_autorl"
                    ], check=True, cwd=self.repo_path)
                    result["updates"].append("API documentation regenerated")
                except subprocess.CalledProcessError:
                    result["recommendations"].append("Consider setting up Sphinx for API documentation")
            
            # Update README with current metrics if needed
            readme_path = self.repo_path / "README.md"
            if readme_path.exists():
                # Check if README needs updates
                mtime = datetime.fromtimestamp(readme_path.stat().st_mtime)
                if datetime.now() - mtime > timedelta(days=30):
                    result["recommendations"].append("README.md hasn't been updated in 30+ days")
            
            # Check for outdated documentation links
            doc_files = list(self.repo_path.glob("**/*.md"))
            broken_links = self._check_documentation_links(doc_files)
            if broken_links:
                result["recommendations"].append(f"Found {len(broken_links)} potentially broken links in documentation")
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def check_code_quality(self) -> Dict[str, Any]:
        """Check code quality metrics."""
        result = {"success": True, "metrics": {}, "recommendations": []}
        
        try:
            # Run code quality tools
            quality_checks = [
                ("ruff", ["python", "-m", "ruff", "check", ".", "--output-format=json"]),
                ("mypy", ["python", "-m", "mypy", "fed_vit_autorl", "--json-report", "/tmp/mypy-report.json"]),
                ("bandit", ["python", "-m", "bandit", "-r", "fed_vit_autorl", "-f", "json"])
            ]
            
            for tool, command in quality_checks:
                try:
                    proc_result = subprocess.run(
                        command, capture_output=True, text=True, cwd=self.repo_path
                    )
                    
                    if tool == "ruff" and proc_result.stdout:
                        try:
                            ruff_data = json.loads(proc_result.stdout)
                            result["metrics"]["ruff_issues"] = len(ruff_data)
                        except json.JSONDecodeError:
                            result["metrics"]["ruff_issues"] = 0
                    
                    elif tool == "bandit" and proc_result.stdout:
                        try:
                            bandit_data = json.loads(proc_result.stdout)
                            result["metrics"]["security_issues"] = len(bandit_data.get("results", []))
                        except json.JSONDecodeError:
                            result["metrics"]["security_issues"] = 0
                    
                except subprocess.CalledProcessError:
                    result["recommendations"].append(f"Failed to run {tool} - ensure it's installed")
            
            # Check test coverage
            try:
                cov_result = subprocess.run([
                    "python", "-m", "pytest", "--cov=fed_vit_autorl", "--cov-report=json", "tests/"
                ], capture_output=True, text=True, cwd=self.repo_path)
                
                coverage_file = self.repo_path / "coverage.json"
                if coverage_file.exists():
                    with open(coverage_file) as f:
                        coverage_data = json.load(f)
                    result["metrics"]["test_coverage"] = coverage_data.get("totals", {}).get("percent_covered", 0)
                
            except subprocess.CalledProcessError:
                result["recommendations"].append("Unable to measure test coverage")
            
            # Recommendations based on metrics
            if result["metrics"].get("ruff_issues", 0) > 50:
                result["recommendations"].append("High number of linting issues detected")
            
            if result["metrics"].get("security_issues", 0) > 0:
                result["recommendations"].append("Security issues detected - review with bandit")
            
            if result["metrics"].get("test_coverage", 0) < 80:
                result["recommendations"].append("Test coverage below 80% - consider adding more tests")
        
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def validate_configurations(self) -> Dict[str, Any]:
        """Validate configuration files."""
        result = {"success": True, "validated_files": [], "issues": [], "recommendations": []}
        
        config_files = [
            ("pyproject.toml", self._validate_pyproject_toml),
            (".pre-commit-config.yaml", self._validate_precommit_config),
            ("docker-compose.yml", self._validate_docker_compose),
            (".github/dependabot.yml", self._validate_dependabot_config),
            ("monitoring/prometheus.yml", self._validate_prometheus_config)
        ]
        
        for config_file, validator in config_files:
            config_path = self.repo_path / config_file
            if config_path.exists():
                try:
                    validation_result = validator(config_path)
                    if validation_result["valid"]:
                        result["validated_files"].append(config_file)
                    else:
                        result["issues"].extend(validation_result.get("issues", []))
                except Exception as e:
                    result["issues"].append(f"Error validating {config_file}: {e}")
            else:
                result["recommendations"].append(f"Consider adding {config_file}")
        
        return result
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check dependency health."""
        result = {"success": True, "dependency_info": {}, "recommendations": []}
        
        try:
            # Check for outdated dependencies
            outdated_result = subprocess.run([
                "python", "-m", "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if outdated_result.returncode == 0:
                outdated_packages = json.loads(outdated_result.stdout)
                result["dependency_info"]["outdated_count"] = len(outdated_packages)
                
                if len(outdated_packages) > 10:
                    result["recommendations"].append("Many outdated dependencies - consider updating")
                
                # Check for security-critical packages
                critical_packages = ["cryptography", "requests", "urllib3"]
                outdated_critical = [p for p in outdated_packages if p["name"] in critical_packages]
                if outdated_critical:
                    result["recommendations"].append("Security-critical packages need updates")
            
            # Check dependency tree for conflicts
            try:
                pipdeptree_result = subprocess.run([
                    "python", "-m", "pipdeptree", "--json"
                ], capture_output=True, text=True, cwd=self.repo_path)
                
                if pipdeptree_result.returncode == 0:
                    dep_tree = json.loads(pipdeptree_result.stdout)
                    result["dependency_info"]["total_packages"] = len(dep_tree)
            except subprocess.CalledProcessError:
                result["recommendations"].append("Install pipdeptree for better dependency analysis")
        
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def validate_docker_setup(self) -> Dict[str, Any]:
        """Validate Docker configuration."""
        result = {"success": True, "docker_files": [], "recommendations": []}
        
        docker_files = ["Dockerfile", "docker-compose.yml", ".dockerignore"]
        
        for docker_file in docker_files:
            file_path = self.repo_path / docker_file
            if file_path.exists():
                result["docker_files"].append(docker_file)
                
                # Validate Dockerfile
                if docker_file == "Dockerfile":
                    with open(file_path) as f:
                        dockerfile_content = f.read()
                    
                    # Check for best practices
                    if "COPY . ." in dockerfile_content:
                        result["recommendations"].append("Consider using .dockerignore to reduce image size")
                    
                    if "pip install" in dockerfile_content and "--no-cache-dir" not in dockerfile_content:
                        result["recommendations"].append("Use --no-cache-dir with pip install in Dockerfile")
                    
                    if "USER root" in dockerfile_content or "USER 0" in dockerfile_content:
                        result["recommendations"].append("Avoid running containers as root user")
        
        # Test if Docker is available
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            result["docker_available"] = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            result["docker_available"] = False
            result["recommendations"].append("Docker not available - install for containerization support")
        
        return result
    
    def check_security_settings(self) -> Dict[str, Any]:
        """Check security configuration."""
        result = {"success": True, "security_checks": [], "recommendations": []}
        
        # Check for sensitive files
        sensitive_patterns = [
            "**/*.pem", "**/*.key", "**/.env", "**/secrets.json",
            "**/*password*", "**/*secret*", "**/*token*"
        ]
        
        found_sensitive = []
        for pattern in sensitive_patterns:
            for path in self.repo_path.glob(pattern):
                if path.is_file() and not self._is_ignored_by_git(path):
                    found_sensitive.append(str(path))
        
        if found_sensitive:
            result["recommendations"].append("Potentially sensitive files found - ensure they're in .gitignore")
        
        # Check .gitignore exists and is comprehensive
        gitignore_path = self.repo_path / ".gitignore"
        if gitignore_path.exists():
            with open(gitignore_path) as f:
                gitignore_content = f.read()
            
            required_patterns = ["*.env", "__pycache__", "*.pyc", ".DS_Store", "secrets/"]
            missing_patterns = [p for p in required_patterns if p not in gitignore_content]
            
            if missing_patterns:
                result["recommendations"].append(f"Add to .gitignore: {', '.join(missing_patterns)}")
        else:
            result["recommendations"].append("Add .gitignore file for better security")
        
        result["security_checks"].append("Sensitive file check completed")
        
        return result
    
    def monitor_repository_health(self) -> Dict[str, Any]:
        """Monitor overall repository health."""
        result = {"success": True, "health_metrics": {}, "recommendations": []}
        
        try:
            # Git repository stats
            git_stats = self._get_git_stats()
            result["health_metrics"]["git"] = git_stats
            
            # File system health
            total_size = self._get_directory_size(self.repo_path)
            result["health_metrics"]["repo_size_mb"] = total_size / (1024 * 1024)
            
            # Check for large files
            large_files = self._find_large_files(self.repo_path, size_mb=10)
            if large_files:
                result["recommendations"].append(f"Large files detected: {len(large_files)} files > 10MB")
            
            # Check commit frequency
            if git_stats["days_since_last_commit"] > 7:
                result["recommendations"].append("No commits in the last 7 days")
            
            # Check branch health
            if git_stats["branch_count"] > 20:
                result["recommendations"].append("Many branches exist - consider cleanup")
        
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def cleanup_old_artifacts(self) -> Dict[str, Any]:
        """Clean up old build artifacts and logs."""
        result = {"success": True, "cleaned_artifacts": [], "recommendations": []}
        
        # Directories to clean
        artifact_dirs = ["dist/", "build/", ".pytest_cache/", "htmlcov/", "logs/"]
        
        for artifact_dir in artifact_dirs:
            dir_path = self.repo_path / artifact_dir
            if dir_path.exists():
                # Remove files older than 7 days
                cutoff_time = datetime.now() - timedelta(days=7)
                
                for file_path in dir_path.rglob("*"):
                    if file_path.is_file():
                        file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if file_mtime < cutoff_time:
                            try:
                                file_path.unlink()
                                result["cleaned_artifacts"].append(str(file_path))
                            except Exception as e:
                                print(f"Error removing {file_path}: {e}")
        
        return result
    
    def validate_ci_cd(self) -> Dict[str, Any]:
        """Validate CI/CD configuration."""
        result = {"success": True, "workflow_files": [], "recommendations": []}
        
        workflows_dir = self.repo_path / ".github" / "workflows"
        if workflows_dir.exists():
            workflow_files = list(workflows_dir.glob("*.yml")) + list(workflows_dir.glob("*.yaml"))
            result["workflow_files"] = [str(f.relative_to(self.repo_path)) for f in workflow_files]
            
            if not workflow_files:
                result["recommendations"].append("No GitHub Actions workflows found")
            
            # Check for essential workflows
            essential_workflows = ["ci.yml", "security-scan.yml"]
            existing_names = [f.name for f in workflow_files]
            
            for essential in essential_workflows:
                if essential not in existing_names:
                    result["recommendations"].append(f"Consider adding {essential} workflow")
        
        else:
            result["recommendations"].append("No .github/workflows directory found")
        
        return result
    
    def generate_maintenance_report(self, report: Dict[str, Any]) -> None:
        """Generate maintenance report."""
        report_path = self.repo_path / "maintenance-report.md"
        
        with open(report_path, "w") as f:
            f.write("# Repository Maintenance Report\n\n")
            f.write(f"**Generated**: {report['timestamp']}\n")
            f.write(f"**Health Score**: {report['health_score']:.1f}%\n\n")
            
            f.write("## Tasks Completed\n\n")
            for task in report["tasks_completed"]:
                f.write(f"- ✅ {task['task']}\n")
            
            if report["tasks_failed"]:
                f.write("\n## Tasks Failed\n\n")
                for task in report["tasks_failed"]:
                    f.write(f"- ❌ {task['task']}: {task['error']}\n")
            
            if report["recommendations"]:
                f.write("\n## Recommendations\n\n")
                for rec in report["recommendations"]:
                    f.write(f"- 💡 {rec}\n")
        
        print(f"Maintenance report saved to {report_path}")
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory."""
        total_size = 0
        for file_path in path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _find_large_files(self, path: Path, size_mb: int = 10) -> List[Path]:
        """Find files larger than specified size."""
        large_files = []
        size_bytes = size_mb * 1024 * 1024
        
        for file_path in path.rglob("*"):
            if file_path.is_file() and file_path.stat().st_size > size_bytes:
                large_files.append(file_path)
        
        return large_files
    
    def _get_git_stats(self) -> Dict[str, Any]:
        """Get Git repository statistics."""
        stats = {}
        
        try:
            # Last commit date
            last_commit_result = subprocess.run([
                "git", "log", "-1", "--format=%ct"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if last_commit_result.returncode == 0:
                last_commit_time = int(last_commit_result.stdout.strip())
                last_commit_date = datetime.fromtimestamp(last_commit_time)
                stats["days_since_last_commit"] = (datetime.now() - last_commit_date).days
            
            # Branch count
            branch_result = subprocess.run([
                "git", "branch", "-a"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if branch_result.returncode == 0:
                stats["branch_count"] = len(branch_result.stdout.strip().split('\n'))
            
            # Commit count
            commit_result = subprocess.run([
                "git", "rev-list", "--count", "HEAD"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if commit_result.returncode == 0:
                stats["total_commits"] = int(commit_result.stdout.strip())
                
        except Exception as e:
            print(f"Error getting git stats: {e}")
        
        return stats
    
    def _is_ignored_by_git(self, file_path: Path) -> bool:
        """Check if file is ignored by git."""
        try:
            result = subprocess.run([
                "git", "check-ignore", str(file_path)
            ], capture_output=True, cwd=self.repo_path)
            return result.returncode == 0
        except:
            return False
    
    def _check_documentation_links(self, doc_files: List[Path]) -> List[str]:
        """Check for broken links in documentation."""
        broken_links = []
        
        # Simple implementation - would need more sophisticated link checking
        for doc_file in doc_files:
            try:
                with open(doc_file) as f:
                    content = f.read()
                
                # Look for obvious broken patterns
                if "](http://localhost" in content:
                    broken_links.append(f"{doc_file}: localhost link found")
                
                if "](TODO" in content or "](FIXME" in content:
                    broken_links.append(f"{doc_file}: TODO/FIXME link found")
                    
            except Exception:
                continue
        
        return broken_links
    
    def _validate_pyproject_toml(self, file_path: Path) -> Dict[str, Any]:
        """Validate pyproject.toml."""
        try:
            import toml
            with open(file_path) as f:
                data = toml.load(f)
            
            issues = []
            
            # Check required fields
            if "project" not in data:
                issues.append("Missing [project] section")
            elif "name" not in data["project"]:
                issues.append("Missing project name")
            
            return {"valid": len(issues) == 0, "issues": issues}
        except Exception as e:
            return {"valid": False, "issues": [str(e)]}
    
    def _validate_precommit_config(self, file_path: Path) -> Dict[str, Any]:
        """Validate pre-commit configuration."""
        try:
            import yaml
            with open(file_path) as f:
                data = yaml.safe_load(f)
            
            issues = []
            
            if "repos" not in data:
                issues.append("Missing repos section")
            
            return {"valid": len(issues) == 0, "issues": issues}
        except Exception as e:
            return {"valid": False, "issues": [str(e)]}
    
    def _validate_docker_compose(self, file_path: Path) -> Dict[str, Any]:
        """Validate docker-compose.yml."""
        try:
            import yaml
            with open(file_path) as f:
                data = yaml.safe_load(f)
            
            issues = []
            
            if "services" not in data:
                issues.append("Missing services section")
            
            return {"valid": len(issues) == 0, "issues": issues}
        except Exception as e:
            return {"valid": False, "issues": [str(e)]}
    
    def _validate_dependabot_config(self, file_path: Path) -> Dict[str, Any]:
        """Validate dependabot configuration."""
        try:
            import yaml
            with open(file_path) as f:
                data = yaml.safe_load(f)
            
            issues = []
            
            if "updates" not in data:
                issues.append("Missing updates section")
            
            return {"valid": len(issues) == 0, "issues": issues}
        except Exception as e:
            return {"valid": False, "issues": [str(e)]}
    
    def _validate_prometheus_config(self, file_path: Path) -> Dict[str, Any]:
        """Validate Prometheus configuration."""
        try:
            import yaml
            with open(file_path) as f:
                data = yaml.safe_load(f)
            
            issues = []
            
            if "scrape_configs" not in data:
                issues.append("Missing scrape_configs section")
            
            return {"valid": len(issues) == 0, "issues": issues}
        except Exception as e:
            return {"valid": False, "issues": [str(e)]}


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository maintenance automation")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--output", help="Output file for maintenance report")
    parser.add_argument("--task", help="Run specific maintenance task")
    
    args = parser.parse_args()
    
    maintenance = RepositoryMaintenance(args.repo_path)
    
    if args.task:
        # Run specific task
        task_method = getattr(maintenance, args.task, None)
        if task_method:
            result = task_method()
            print(json.dumps(result, indent=2))
        else:
            print(f"Unknown task: {args.task}")
            sys.exit(1)
    else:
        # Run full maintenance cycle
        report = maintenance.run_maintenance_cycle()
        
        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Maintenance report saved to {args.output}")
        
        print(f"\nMaintenance completed with health score: {report['health_score']:.1f}%")
        
        if report["recommendations"]:
            print("\nRecommendations:")
            for rec in report["recommendations"]:
                print(f"  - {rec}")


if __name__ == "__main__":
    main()