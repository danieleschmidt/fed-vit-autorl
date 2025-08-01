#!/usr/bin/env python3
"""Continuous value discovery engine for Fed-ViT-AutoRL."""

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import yaml


class ValueDiscoveryEngine:
    """Discovers and prioritizes high-value work items."""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = repo_path
        self.config = self._load_config()
        self.metrics_file = repo_path / ".terragon" / "value-metrics.json"
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from .terragon/config.yaml."""
        config_path = self.repo_path / ".terragon" / "config.yaml"
        with open(config_path) as f:
            return yaml.safe_load(f)
    
    def discover_work_items(self) -> List[Dict[str, Any]]:
        """Discover potential work items from various sources."""
        items = []
        
        # 1. Git history analysis
        items.extend(self._analyze_git_history())
        
        # 2. Static analysis
        items.extend(self._run_static_analysis())
        
        # 3. TODO/FIXME extraction
        items.extend(self._extract_code_comments())
        
        # 4. Dependency analysis
        items.extend(self._analyze_dependencies())
        
        # 5. Test coverage gaps
        items.extend(self._analyze_test_coverage())
        
        return items
    
    def _analyze_git_history(self) -> List[Dict[str, Any]]:
        """Analyze git commits for patterns indicating technical debt."""
        items = []
        
        try:
            # Get recent commits with "fix", "hack", "todo" etc.
            result = subprocess.run([
                "git", "log", "--oneline", "-n", "50", "--grep=fix",
                "--grep=hack", "--grep=todo", "--grep=temporary", "-i"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            commits = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            for commit in commits:
                if commit:
                    items.append({
                        'id': f'git-{hash(commit) % 10000}',
                        'title': f'Review and refactor: {commit[:50]}',
                        'category': 'technical-debt',
                        'source': 'git-history',
                        'description': f'Commit suggests quick fix or technical debt: {commit}',
                        'estimatedEffort': 2,
                        'priority': 'medium',
                        'discoveredAt': datetime.now().isoformat()
                    })
                    
        except Exception as e:
            print(f"Git analysis failed: {e}")
            
        return items
    
    def _run_static_analysis(self) -> List[Dict[str, Any]]:
        """Run static analysis tools to find issues."""
        items = []
        
        # Check if we have Python files
        python_files = list(self.repo_path.glob("**/*.py"))
        if not python_files:
            return items
            
        # Run ruff for linting
        try:
            result = subprocess.run([
                "ruff", "check", ".", "--output-format=json"
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            if result.stdout:
                issues = json.loads(result.stdout)
                for issue in issues[:10]:  # Limit to top 10
                    items.append({
                        'id': f'ruff-{hash(str(issue)) % 10000}',
                        'title': f'Fix linting issue: {issue.get("code", "unknown")}',
                        'category': 'code-quality',
                        'source': 'static-analysis',
                        'description': f'{issue.get("message", "")}, {issue.get("filename", "")}:{issue.get("location", {}).get("row", 0)}',
                        'estimatedEffort': 0.5,
                        'priority': 'low' if issue.get("code", "").startswith("E") else 'medium',
                        'discoveredAt': datetime.now().isoformat()
                    })
                    
        except Exception as e:
            print(f"Ruff analysis failed: {e}")
            
        return items
    
    def _extract_code_comments(self) -> List[Dict[str, Any]]:
        """Extract TODO, FIXME, HACK comments from code."""
        items = []
        
        try:
            result = subprocess.run([
                "grep", "-r", "-n", "-i", 
                "--include=*.py", "--include=*.js", "--include=*.ts",
                r"TODO\|FIXME\|HACK\|BUG", "."
            ], capture_output=True, text=True, cwd=self.repo_path)
            
            lines = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            for line in lines[:20]:  # Limit to 20 items
                if line and ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) >= 3:
                        filename, line_num, comment = parts
                        items.append({
                            'id': f'comment-{hash(line) % 10000}',
                            'title': f'Address comment in {filename}:{line_num}',
                            'category': 'technical-debt',
                            'source': 'code-comments',
                            'description': comment.strip(),
                            'estimatedEffort': 1,
                            'priority': 'high' if 'FIXME' in comment.upper() else 'medium',
                            'discoveredAt': datetime.now().isoformat()
                        })
                        
        except Exception:
            pass  # grep might not find anything
            
        return items
    
    def _analyze_dependencies(self) -> List[Dict[str, Any]]:
        """Analyze for outdated or vulnerable dependencies."""
        items = []
        
        # Check for Python dependencies
        if (self.repo_path / "pyproject.toml").exists():
            try:
                result = subprocess.run([
                    "pip", "list", "--outdated", "--format=json"
                ], capture_output=True, text=True)
                
                if result.stdout:
                    outdated = json.loads(result.stdout)
                    for pkg in outdated[:5]:  # Top 5 outdated packages
                        items.append({
                            'id': f'dep-{pkg["name"]}',
                            'title': f'Update {pkg["name"]} from {pkg["version"]} to {pkg["latest_version"]}',
                            'category': 'dependency-update',
                            'source': 'dependency-analysis',
                            'description': f'Package {pkg["name"]} is outdated',
                            'estimatedEffort': 0.5,
                            'priority': 'low',
                            'discoveredAt': datetime.now().isoformat()
                        })
                        
            except Exception as e:
                print(f"Dependency analysis failed: {e}")
                
        return items
    
    def _analyze_test_coverage(self) -> List[Dict[str, Any]]:
        """Identify areas with low test coverage."""
        items = []
        
        # Check if tests directory exists
        if not (self.repo_path / "tests").exists():
            items.append({
                'id': 'test-setup',
                'title': 'Set up comprehensive test suite',
                'category': 'testing',
                'source': 'coverage-analysis',
                'description': 'Repository needs comprehensive test coverage',
                'estimatedEffort': 8,
                'priority': 'high',
                'discoveredAt': datetime.now().isoformat()
            })
        
        return items
    
    def calculate_value_scores(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate WSJF, ICE, and composite scores for items."""
        maturity = self.config['repository']['maturityLevel']
        weights = self.config['scoring']['weights'][maturity]
        
        for item in items:
            # WSJF Components
            user_value = self._score_user_value(item)
            time_criticality = self._score_time_criticality(item)
            risk_reduction = self._score_risk_reduction(item)
            opportunity = self._score_opportunity(item)
            
            cost_of_delay = user_value + time_criticality + risk_reduction + opportunity
            job_size = item.get('estimatedEffort', 1)
            wsjf = cost_of_delay / max(job_size, 0.1)
            
            # ICE Components
            impact = self._score_impact(item)
            confidence = self._score_confidence(item)
            ease = self._score_ease(item)
            ice = impact * confidence * ease
            
            # Technical Debt Score
            debt_score = self._score_technical_debt(item)
            
            # Composite Score
            composite = (
                weights['wsjf'] * min(wsjf / 10, 10) +
                weights['ice'] * min(ice / 100, 10) +
                weights['technicalDebt'] * min(debt_score / 10, 10)
            )
            
            # Apply boosts
            if item['category'] == 'security':
                composite *= self.config['scoring']['thresholds']['securityBoost']
            
            item.update({
                'scores': {
                    'wsjf': round(wsjf, 2),
                    'ice': round(ice, 2),
                    'technicalDebt': round(debt_score, 2),
                    'composite': round(composite, 2)
                }
            })
            
        return sorted(items, key=lambda x: x['scores']['composite'], reverse=True)
    
    def _score_user_value(self, item: Dict) -> float:
        """Score user/business value (1-10)."""
        category = item.get('category', '')
        mapping = {
            'security': 9,
            'performance': 8,
            'testing': 7,
            'technical-debt': 6,
            'code-quality': 5,
            'dependency-update': 4,
            'documentation': 3
        }
        return mapping.get(category, 5)
    
    def _score_time_criticality(self, item: Dict) -> float:
        """Score time criticality (1-10)."""
        priority = item.get('priority', 'medium')
        mapping = {'high': 8, 'medium': 5, 'low': 2}
        return mapping.get(priority, 5)
    
    def _score_risk_reduction(self, item: Dict) -> float:
        """Score risk reduction (1-10)."""
        category = item.get('category', '')
        if 'security' in category:
            return 9
        elif 'testing' in category:
            return 7
        elif 'technical-debt' in category:
            return 6
        return 4
    
    def _score_opportunity(self, item: Dict) -> float:
        """Score opportunity enablement (1-10)."""
        category = item.get('category', '')
        if category in ['testing', 'performance']:
            return 7
        return 4
    
    def _score_impact(self, item: Dict) -> float:
        """Score impact (1-10)."""
        return self._score_user_value(item)
    
    def _score_confidence(self, item: Dict) -> float:
        """Score confidence in execution (1-10)."""
        effort = item.get('estimatedEffort', 1)
        if effort <= 1:
            return 9
        elif effort <= 4:
            return 7
        elif effort <= 8:
            return 5
        return 3
    
    def _score_ease(self, item: Dict) -> float:
        """Score ease of implementation (1-10)."""
        effort = item.get('estimatedEffort', 1)
        return max(10 - effort, 1)
    
    def _score_technical_debt(self, item: Dict) -> float:
        """Score technical debt impact (1-10)."""
        category = item.get('category', '')
        if 'technical-debt' in category:
            return 8
        elif 'code-quality' in category:
            return 6
        elif 'testing' in category:
            return 7
        return 4
    
    def update_metrics(self, items: List[Dict[str, Any]]) -> None:
        """Update value metrics file."""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                metrics = json.load(f)
        else:
            metrics = {}
        
        metrics.update({
            'continuousDiscovery': {
                'lastScan': datetime.now().isoformat(),
                'discoveredItems': len(items),
                'prioritizedItems': len([i for i in items if i['scores']['composite'] > 10]),
                'executedItems': metrics.get('continuousDiscovery', {}).get('executedItems', 0)
            },
            'discoveredItems': items[:20]  # Store top 20 items
        })
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def run(self) -> List[Dict[str, Any]]:
        """Run the complete value discovery process."""
        print(f"🔍 Running value discovery at {datetime.now()}")
        
        # Discover work items
        items = self.discover_work_items()
        print(f"📋 Discovered {len(items)} work items")
        
        # Calculate scores
        scored_items = self.calculate_value_scores(items)
        print(f"📊 Scored and prioritized items")
        
        # Update metrics
        self.update_metrics(scored_items)
        print(f"💾 Updated metrics")
        
        # Show top items
        print("\n🎯 Top 5 Value Items:")
        for i, item in enumerate(scored_items[:5], 1):
            print(f"{i}. [{item['scores']['composite']:.1f}] {item['title']}")
        
        return scored_items


if __name__ == "__main__":
    engine = ValueDiscoveryEngine()
    items = engine.run()