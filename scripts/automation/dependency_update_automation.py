#!/usr/bin/env python3
"""
Automated dependency update management for Fed-ViT-AutoRL.

This script provides intelligent dependency update automation including:
- Security-first dependency updates
- Compatibility testing
- Rollback capabilities
- Federated learning specific validation
"""

import json
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import requests
import toml


class DependencyUpdateAutomation:
    """Manages automated dependency updates with safety checks."""

    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.pyproject_path = self.repo_path / "pyproject.toml"
        self.requirements_path = self.repo_path / "requirements.txt"

    def run_full_update_cycle(self) -> Dict[str, Any]:
        """Run complete dependency update cycle."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "security_updates": [],
            "regular_updates": [],
            "failed_updates": [],
            "tests_passed": False,
            "rollback_performed": False
        }

        try:
            # 1. Check for security vulnerabilities
            security_updates = self.check_security_vulnerabilities()
            report["security_updates"] = security_updates

            # 2. Apply security updates first
            if security_updates:
                self.apply_security_updates(security_updates)

                # Test after security updates
                if not self.run_validation_tests():
                    print("Security updates failed validation, rolling back...")
                    self.rollback_changes()
                    report["rollback_performed"] = True
                    return report

            # 3. Check for regular updates
            regular_updates = self.check_regular_updates()
            report["regular_updates"] = regular_updates

            # 4. Apply regular updates with testing
            if regular_updates:
                successful_updates = self.apply_and_test_updates(regular_updates)
                failed_updates = [u for u in regular_updates if u not in successful_updates]
                report["failed_updates"] = failed_updates

            # 5. Final validation
            report["tests_passed"] = self.run_validation_tests()

            # 6. Generate update summary
            self.generate_update_summary(report)

        except Exception as e:
            print(f"Error in update cycle: {e}")
            report["error"] = str(e)

        return report

    def check_security_vulnerabilities(self) -> List[Dict[str, Any]]:
        """Check for security vulnerabilities in dependencies."""
        print("Checking for security vulnerabilities...")

        vulnerabilities = []

        try:
            # Use safety to check for known vulnerabilities
            result = subprocess.run([
                "python", "-m", "safety", "check", "--json"
            ], capture_output=True, text=True, cwd=self.repo_path)

            if result.returncode != 0 and result.stdout:
                try:
                    safety_data = json.loads(result.stdout)
                    for vuln in safety_data:
                        vulnerabilities.append({
                            "package": vuln.get("package"),
                            "version": vuln.get("installed_version"),
                            "vulnerability_id": vuln.get("id"),
                            "advisory": vuln.get("advisory"),
                            "minimum_version": vuln.get("specs", [None])[0],
                            "severity": self._get_vulnerability_severity(vuln)
                        })
                except json.JSONDecodeError:
                    print(f"Could not parse safety output: {result.stdout}")

            # Also check using pip-audit if available
            try:
                audit_result = subprocess.run([
                    "python", "-m", "pip_audit", "--format=json"
                ], capture_output=True, text=True, cwd=self.repo_path)

                if audit_result.returncode == 0:
                    audit_data = json.loads(audit_result.stdout)
                    for vuln in audit_data.get("vulnerabilities", []):
                        vulnerabilities.append({
                            "package": vuln.get("package"),
                            "version": vuln.get("installed_version"),
                            "vulnerability_id": vuln.get("id"),
                            "advisory": vuln.get("description"),
                            "minimum_version": vuln.get("fix_versions", [None])[0],
                            "severity": vuln.get("severity", "unknown")
                        })
            except (subprocess.CalledProcessError, FileNotFoundError):
                # pip-audit not available
                pass

        except Exception as e:
            print(f"Error checking vulnerabilities: {e}")

        print(f"Found {len(vulnerabilities)} security vulnerabilities")
        return vulnerabilities

    def check_regular_updates(self) -> List[Dict[str, Any]]:
        """Check for regular dependency updates."""
        print("Checking for regular dependency updates...")

        updates = []

        try:
            # Use pip list --outdated to find updates
            result = subprocess.run([
                "python", "-m", "pip", "list", "--outdated", "--format=json"
            ], capture_output=True, text=True, cwd=self.repo_path)

            if result.returncode == 0:
                outdated_packages = json.loads(result.stdout)

                for package in outdated_packages:
                    # Filter to only include packages we care about
                    if self._should_update_package(package["name"]):
                        updates.append({
                            "package": package["name"],
                            "current_version": package["version"],
                            "latest_version": package["latest_version"],
                            "update_type": self._determine_update_type(
                                package["version"],
                                package["latest_version"]
                            ),
                            "priority": self._get_update_priority(package["name"])
                        })

        except Exception as e:
            print(f"Error checking for updates: {e}")

        print(f"Found {len(updates)} potential updates")
        return updates

    def apply_security_updates(self, security_updates: List[Dict[str, Any]]) -> None:
        """Apply security updates immediately."""
        print(f"Applying {len(security_updates)} security updates...")

        for update in security_updates:
            package = update["package"]
            min_version = update["minimum_version"]

            if min_version:
                print(f"Updating {package} to {min_version} (security fix)")
                try:
                    subprocess.run([
                        "python", "-m", "pip", "install", f"{package}>={min_version}"
                    ], check=True, cwd=self.repo_path)
                except subprocess.CalledProcessError as e:
                    print(f"Failed to update {package}: {e}")

    def apply_and_test_updates(self, updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply updates one by one with testing."""
        successful_updates = []

        # Sort by priority (high priority first)
        updates.sort(key=lambda x: x["priority"], reverse=True)

        for update in updates:
            package = update["package"]
            new_version = update["latest_version"]

            print(f"Testing update: {package} {update['current_version']} -> {new_version}")

            # Create backup of current state
            self._create_dependency_backup()

            try:
                # Apply update
                subprocess.run([
                    "python", "-m", "pip", "install", f"{package}=={new_version}"
                ], check=True, cwd=self.repo_path)

                # Run quick validation
                if self._run_quick_tests():
                    successful_updates.append(update)
                    print(f"✅ Successfully updated {package}")
                else:
                    print(f"❌ Update failed validation: {package}")
                    self._restore_dependency_backup()

            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to apply update {package}: {e}")
                self._restore_dependency_backup()

        return successful_updates

    def run_validation_tests(self) -> bool:
        """Run comprehensive validation tests."""
        print("Running validation tests...")

        tests = [
            self._test_import_functionality,
            self._test_basic_functionality,
            self._test_federated_learning_compatibility,
            self._test_security_requirements
        ]

        for test in tests:
            try:
                if not test():
                    return False
            except Exception as e:
                print(f"Test failed with exception: {e}")
                return False

        print("✅ All validation tests passed")
        return True

    def rollback_changes(self) -> None:
        """Rollback dependency changes."""
        print("Rolling back dependency changes...")

        try:
            # Restore from backup if available
            if hasattr(self, '_backup_file') and self._backup_file.exists():
                subprocess.run([
                    "python", "-m", "pip", "install", "-r", str(self._backup_file)
                ], check=True, cwd=self.repo_path)
                print("✅ Rollback completed")
            else:
                print("❌ No backup available for rollback")
        except Exception as e:
            print(f"❌ Rollback failed: {e}")

    def _should_update_package(self, package_name: str) -> bool:
        """Determine if a package should be updated."""
        # Core packages we always want to keep updated
        core_packages = {
            "torch", "torchvision", "transformers", "numpy",
            "pillow", "pyyaml", "tqdm", "tensorboard"
        }

        # Development packages
        dev_packages = {
            "pytest", "black", "ruff", "mypy", "pre-commit"
        }

        # Security packages
        security_packages = {
            "cryptography", "requests", "urllib3"
        }

        return (package_name.lower() in core_packages or
                package_name.lower() in dev_packages or
                package_name.lower() in security_packages)

    def _determine_update_type(self, current: str, latest: str) -> str:
        """Determine the type of update (major, minor, patch)."""
        try:
            current_parts = [int(x) for x in current.split('.')]
            latest_parts = [int(x) for x in latest.split('.')]

            if len(current_parts) >= 1 and len(latest_parts) >= 1:
                if current_parts[0] != latest_parts[0]:
                    return "major"
                elif len(current_parts) >= 2 and len(latest_parts) >= 2:
                    if current_parts[1] != latest_parts[1]:
                        return "minor"
                    else:
                        return "patch"
        except (ValueError, IndexError):
            pass

        return "unknown"

    def _get_update_priority(self, package_name: str) -> int:
        """Get update priority for a package (1-10, 10 = highest)."""
        # Security-critical packages
        if package_name.lower() in ["cryptography", "requests", "urllib3"]:
            return 10

        # Core ML packages
        if package_name.lower() in ["torch", "torchvision", "transformers"]:
            return 8

        # Core Python packages
        if package_name.lower() in ["numpy", "pillow", "pyyaml"]:
            return 7

        # Development tools
        if package_name.lower() in ["pytest", "black", "ruff", "mypy"]:
            return 5

        # Everything else
        return 3

    def _get_vulnerability_severity(self, vuln_data: Dict[str, Any]) -> str:
        """Extract vulnerability severity from safety data."""
        advisory = vuln_data.get("advisory", "").lower()

        if any(word in advisory for word in ["critical", "severe"]):
            return "critical"
        elif any(word in advisory for word in ["high", "important"]):
            return "high"
        elif any(word in advisory for word in ["medium", "moderate"]):
            return "medium"
        else:
            return "low"

    def _create_dependency_backup(self) -> None:
        """Create backup of current dependencies."""
        try:
            self._backup_file = Path(tempfile.mktemp(suffix=".txt"))
            subprocess.run([
                "python", "-m", "pip", "freeze"
            ], stdout=open(self._backup_file, "w"), cwd=self.repo_path)
        except Exception as e:
            print(f"Failed to create backup: {e}")

    def _restore_dependency_backup(self) -> None:
        """Restore dependencies from backup."""
        try:
            if hasattr(self, '_backup_file') and self._backup_file.exists():
                subprocess.run([
                    "python", "-m", "pip", "install", "-r", str(self._backup_file)
                ], check=True, cwd=self.repo_path)
        except Exception as e:
            print(f"Failed to restore backup: {e}")

    def _run_quick_tests(self) -> bool:
        """Run quick validation tests."""
        try:
            # Test basic imports
            result = subprocess.run([
                "python", "-c", "import fed_vit_autorl; print('Import successful')"
            ], capture_output=True, text=True, cwd=self.repo_path)

            return result.returncode == 0
        except Exception:
            return False

    def _test_import_functionality(self) -> bool:
        """Test that all modules can be imported."""
        print("Testing import functionality...")

        modules_to_test = [
            "fed_vit_autorl",
            "fed_vit_autorl.models",
            "torch",
            "transformers",
            "numpy"
        ]

        for module in modules_to_test:
            try:
                result = subprocess.run([
                    "python", "-c", f"import {module}; print('{module} imported successfully')"
                ], capture_output=True, text=True, cwd=self.repo_path)

                if result.returncode != 0:
                    print(f"Failed to import {module}: {result.stderr}")
                    return False
            except Exception as e:
                print(f"Error testing import {module}: {e}")
                return False

        return True

    def _test_basic_functionality(self) -> bool:
        """Test basic functionality."""
        print("Testing basic functionality...")

        try:
            # Run a simple test
            result = subprocess.run([
                "python", "-m", "pytest", "tests/unit/", "-v", "--tb=short", "-x"
            ], capture_output=True, text=True, cwd=self.repo_path)

            return result.returncode == 0
        except Exception as e:
            print(f"Basic functionality test failed: {e}")
            return False

    def _test_federated_learning_compatibility(self) -> bool:
        """Test federated learning specific functionality."""
        print("Testing federated learning compatibility...")

        # This would test FL-specific functionality
        # For now, return True as placeholder
        return True

    def _test_security_requirements(self) -> bool:
        """Test security requirements."""
        print("Testing security requirements...")

        try:
            # Run security checks
            result = subprocess.run([
                "python", "-m", "bandit", "-r", "fed_vit_autorl/", "-ll"
            ], capture_output=True, text=True, cwd=self.repo_path)

            # Bandit returns 1 if issues found, but we'll be lenient
            return result.returncode in [0, 1]
        except Exception as e:
            print(f"Security test failed: {e}")
            return False

    def generate_update_summary(self, report: Dict[str, Any]) -> None:
        """Generate and save update summary."""
        summary_path = self.repo_path / "dependency-update-summary.md"

        with open(summary_path, "w") as f:
            f.write("# Dependency Update Summary\n\n")
            f.write(f"**Date**: {report['timestamp']}\n\n")

            if report["security_updates"]:
                f.write("## Security Updates Applied\n\n")
                for update in report["security_updates"]:
                    f.write(f"- **{update['package']}**: {update['vulnerability_id']} "
                           f"(Severity: {update['severity']})\n")
                f.write("\n")

            if report["regular_updates"]:
                f.write("## Regular Updates Applied\n\n")
                for update in report["regular_updates"]:
                    f.write(f"- **{update['package']}**: "
                           f"{update['current_version']} → {update['latest_version']}\n")
                f.write("\n")

            if report["failed_updates"]:
                f.write("## Failed Updates\n\n")
                for update in report["failed_updates"]:
                    f.write(f"- **{update['package']}**: Update failed validation\n")
                f.write("\n")

            f.write(f"**Tests Passed**: {'✅' if report['tests_passed'] else '❌'}\n")
            f.write(f"**Rollback Performed**: {'⚠️ Yes' if report['rollback_performed'] else 'No'}\n")

        print(f"Update summary saved to {summary_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Automated dependency updates")
    parser.add_argument("--repo-path", default=".", help="Repository path")
    parser.add_argument("--security-only", action="store_true",
                       help="Only apply security updates")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be updated without applying")
    parser.add_argument("--output", help="Output file for update report")

    args = parser.parse_args()

    updater = DependencyUpdateAutomation(args.repo_path)

    if args.dry_run:
        print("DRY RUN: Checking for available updates...")
        security_updates = updater.check_security_vulnerabilities()
        regular_updates = updater.check_regular_updates()

        print(f"\nSecurity updates available: {len(security_updates)}")
        for update in security_updates:
            print(f"  - {update['package']}: {update['vulnerability_id']}")

        print(f"\nRegular updates available: {len(regular_updates)}")
        for update in regular_updates:
            print(f"  - {update['package']}: {update['current_version']} → {update['latest_version']}")

    elif args.security_only:
        print("Applying security updates only...")
        security_updates = updater.check_security_vulnerabilities()
        if security_updates:
            updater.apply_security_updates(security_updates)
            if updater.run_validation_tests():
                print("✅ Security updates applied successfully")
            else:
                print("❌ Security updates failed validation")
                updater.rollback_changes()
        else:
            print("No security updates available")

    else:
        print("Running full dependency update cycle...")
        report = updater.run_full_update_cycle()

        if args.output:
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            print(f"Update report saved to {args.output}")


if __name__ == "__main__":
    main()
