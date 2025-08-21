"""Experimental Framework for Federated Learning Research.

This module provides a comprehensive framework for conducting reproducible
federated learning experiments with statistical validation and publication-ready results.
"""

import os
import json
import pickle
import hashlib
import time
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for federated learning experiments."""

    # Experiment metadata
    experiment_name: str
    description: str
    tags: List[str]

    # Federated learning setup
    num_clients: int = 100
    clients_per_round: int = 10
    num_rounds: int = 100
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001

    # Data distribution
    data_distribution: str = "iid"  # "iid", "non-iid", "pathological"
    alpha: float = 0.5  # Dirichlet concentration parameter for non-IID

    # Model configuration
    model_name: str = "vit_base"
    aggregation_method: str = "fedavg"

    # Privacy settings
    use_differential_privacy: bool = False
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5

    # Evaluation settings
    eval_frequency: int = 5
    metrics: List[str] = None

    # Reproducibility
    random_seed: int = 42
    deterministic: bool = True

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ["accuracy", "loss", "f1_score"]


@dataclass
class ExperimentResult:
    """Results from a federated learning experiment."""

    config: ExperimentConfig
    metrics: Dict[str, List[float]]
    final_metrics: Dict[str, float]
    round_times: List[float]
    communication_costs: List[float]
    client_participation: List[List[int]]
    model_checkpoints: List[str]
    statistical_tests: Dict[str, Any]

    # Reproducibility information
    git_commit: Optional[str] = None
    environment_hash: Optional[str] = None
    timestamp: Optional[str] = None


class StatisticalValidator:
    """Statistical validation for experiment results."""

    def __init__(self, significance_level: float = 0.05):
        """Initialize statistical validator.

        Args:
            significance_level: P-value threshold for statistical significance
        """
        self.significance_level = significance_level

    def compare_algorithms(
        self,
        results_a: List[float],
        results_b: List[float],
        test_type: str = "paired_ttest",
    ) -> Dict[str, Any]:
        """Compare two algorithms statistically.

        Args:
            results_a: Results from algorithm A
            results_b: Results from algorithm B
            test_type: Type of statistical test

        Returns:
            Statistical test results
        """
        if test_type == "paired_ttest":
            statistic, p_value = stats.ttest_rel(results_a, results_b)
            test_name = "Paired t-test"
        elif test_type == "wilcoxon":
            statistic, p_value = stats.wilcoxon(results_a, results_b)
            test_name = "Wilcoxon signed-rank test"
        elif test_type == "mannwhitney":
            statistic, p_value = stats.mannwhitneyu(results_a, results_b)
            test_name = "Mann-Whitney U test"
        else:
            raise ValueError(f"Unknown test type: {test_type}")

        # Effect size (Cohen's d for t-test)
        effect_size = None
        if test_type == "paired_ttest":
            pooled_std = np.sqrt((np.var(results_a) + np.var(results_b)) / 2)
            effect_size = (np.mean(results_a) - np.mean(results_b)) / pooled_std

        # Confidence interval
        if test_type == "paired_ttest":
            diff = np.array(results_a) - np.array(results_b)
            ci_low, ci_high = stats.t.interval(
                1 - self.significance_level,
                len(diff) - 1,
                loc=np.mean(diff),
                scale=stats.sem(diff)
            )
        else:
            ci_low, ci_high = None, None

        return {
            "test_name": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "is_significant": p_value < self.significance_level,
            "effect_size": effect_size,
            "confidence_interval": (ci_low, ci_high) if ci_low is not None else None,
            "mean_a": np.mean(results_a),
            "mean_b": np.mean(results_b),
            "std_a": np.std(results_a),
            "std_b": np.std(results_b),
        }

    def power_analysis(
        self,
        effect_size: float,
        sample_size: int,
        alpha: float = 0.05,
    ) -> float:
        """Compute statistical power of experiment.

        Args:
            effect_size: Expected effect size (Cohen's d)
            sample_size: Sample size (number of runs)
            alpha: Type I error rate

        Returns:
            Statistical power (1 - β)
        """
        from scipy.stats import norm

        # Critical value for two-tailed test
        critical_value = norm.ppf(1 - alpha / 2)

        # Power calculation
        beta = norm.cdf(critical_value - effect_size * np.sqrt(sample_size))
        power = 1 - beta

        return power

    def minimum_sample_size(
        self,
        effect_size: float,
        power: float = 0.8,
        alpha: float = 0.05,
    ) -> int:
        """Compute minimum sample size for desired power.

        Args:
            effect_size: Expected effect size
            power: Desired statistical power
            alpha: Type I error rate

        Returns:
            Minimum sample size
        """
        from scipy.stats import norm

        # Critical values
        z_alpha = norm.ppf(1 - alpha / 2)
        z_beta = norm.ppf(power)

        # Sample size calculation
        n = ((z_alpha + z_beta) / effect_size) ** 2

        return int(np.ceil(n))


class ExperimentRunner:
    """Main experiment runner for federated learning research."""

    def __init__(
        self,
        base_dir: str = "./experiments",
        use_wandb: bool = True,
        wandb_project: str = "fed-vit-autorl",
    ):
        """Initialize experiment runner.

        Args:
            base_dir: Base directory for storing results
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: W&B project name
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project

        self.validator = StatisticalValidator()

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.base_dir / "experiments.log"),
                logging.StreamHandler()
            ]
        )

    def _set_seed(self, seed: int, deterministic: bool = True):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _get_environment_hash(self) -> str:
        """Get hash of current environment for reproducibility."""
        import pkg_resources
        import platform

        # Get installed packages
        installed_packages = [str(d) for d in pkg_resources.working_set]
        installed_packages.sort()

        # Environment info
        env_info = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "packages": installed_packages,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }

        # Create hash
        env_str = json.dumps(env_info, sort_keys=True)
        return hashlib.sha256(env_str.encode()).hexdigest()[:16]

    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            return commit
        except:
            return None

    def run_single_experiment(
        self,
        config: ExperimentConfig,
        train_fn: Callable,
        eval_fn: Callable,
        model_factory: Callable,
        data_loader_factory: Callable,
    ) -> ExperimentResult:
        """Run a single experiment.

        Args:
            config: Experiment configuration
            train_fn: Training function
            eval_fn: Evaluation function
            model_factory: Function to create model
            data_loader_factory: Function to create data loaders

        Returns:
            Experiment results
        """
        # Set up reproducibility
        self._set_seed(config.random_seed, config.deterministic)

        # Initialize W&B if enabled
        if self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=config.experiment_name,
                config=asdict(config),
                tags=config.tags,
            )

        # Create experiment directory
        exp_dir = self.base_dir / config.experiment_name
        exp_dir.mkdir(exist_ok=True)

        logger.info(f"Starting experiment: {config.experiment_name}")

        # Initialize tracking variables
        metrics = {metric: [] for metric in config.metrics}
        round_times = []
        communication_costs = []
        client_participation = []
        model_checkpoints = []

        # Create model and data
        model = model_factory(config)
        train_loaders, test_loader = data_loader_factory(config)

        # Training loop
        start_time = time.time()

        for round_idx in tqdm(range(config.num_rounds), desc="Training Rounds"):
            round_start = time.time()

            # Simulate client selection and participation
            available_clients = list(range(config.num_clients))
            participating_clients = np.random.choice(
                available_clients,
                size=min(config.clients_per_round, len(available_clients)),
                replace=False
            ).tolist()

            client_participation.append(participating_clients)

            # Federated training round
            round_metrics = train_fn(
                model=model,
                client_loaders=[train_loaders[i] for i in participating_clients],
                config=config,
                round_idx=round_idx,
            )

            round_time = time.time() - round_start
            round_times.append(round_time)

            # Estimate communication cost (simplified)
            num_params = sum(p.numel() for p in model.parameters())
            comm_cost = len(participating_clients) * num_params * 4  # 4 bytes per float32
            communication_costs.append(comm_cost)

            # Evaluation
            if round_idx % config.eval_frequency == 0:
                eval_metrics = eval_fn(model, test_loader, config)

                # Store metrics
                for metric_name in config.metrics:
                    if metric_name in eval_metrics:
                        metrics[metric_name].append(eval_metrics[metric_name])

                # Log to W&B
                if self.use_wandb:
                    log_dict = {f"eval/{k}": v for k, v in eval_metrics.items()}
                    log_dict.update({
                        "round": round_idx,
                        "round_time": round_time,
                        "communication_cost": comm_cost,
                    })
                    wandb.log(log_dict)

                # Save model checkpoint
                checkpoint_path = exp_dir / f"checkpoint_round_{round_idx}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                model_checkpoints.append(str(checkpoint_path))

                logger.info(
                    f"Round {round_idx}: "
                    + " | ".join([f"{k}={v:.4f}" for k, v in eval_metrics.items()])
                )

        total_time = time.time() - start_time
        logger.info(f"Experiment completed in {total_time:.2f}s")

        # Final evaluation
        final_metrics = eval_fn(model, test_loader, config)

        # Statistical analysis
        statistical_tests = {}
        if len(metrics[config.metrics[0]]) >= 5:  # Minimum for statistical tests
            # Test for convergence
            recent_values = metrics[config.metrics[0]][-5:]
            early_values = metrics[config.metrics[0]][:5]

            if len(early_values) >= 5:
                convergence_test = self.validator.compare_algorithms(
                    early_values, recent_values, test_type="paired_ttest"
                )
                statistical_tests["convergence"] = convergence_test

        # Create result object
        result = ExperimentResult(
            config=config,
            metrics=metrics,
            final_metrics=final_metrics,
            round_times=round_times,
            communication_costs=communication_costs,
            client_participation=client_participation,
            model_checkpoints=model_checkpoints,
            statistical_tests=statistical_tests,
            git_commit=self._get_git_commit(),
            environment_hash=self._get_environment_hash(),
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

        # Save results
        result_path = exp_dir / "results.json"
        with open(result_path, 'w') as f:
            # Convert result to JSON-serializable format
            result_dict = asdict(result)
            json.dump(result_dict, f, indent=2, default=str)

        # Save detailed results as pickle
        pickle_path = exp_dir / "results.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(result, f)

        if self.use_wandb:
            wandb.save(str(result_path))
            wandb.save(str(pickle_path))
            wandb.finish()

        return result

    def compare_algorithms(
        self,
        experiment_results: Dict[str, List[ExperimentResult]],
        metric: str = "accuracy",
        save_plots: bool = True,
    ) -> Dict[str, Any]:
        """Compare multiple algorithms statistically.

        Args:
            experiment_results: Dictionary mapping algorithm names to result lists
            metric: Metric to compare
            save_plots: Whether to save comparison plots

        Returns:
            Comparison results with statistical tests
        """
        comparison_results = {}
        algorithm_names = list(experiment_results.keys())

        # Extract final metric values for each algorithm
        algorithm_values = {}
        for alg_name, results in experiment_results.items():
            values = [r.final_metrics.get(metric, 0.0) for r in results]
            algorithm_values[alg_name] = values

        # Pairwise comparisons
        pairwise_tests = {}
        for i, alg_a in enumerate(algorithm_names):
            for j, alg_b in enumerate(algorithm_names[i+1:], i+1):
                test_key = f"{alg_a}_vs_{alg_b}"

                test_result = self.validator.compare_algorithms(
                    algorithm_values[alg_a],
                    algorithm_values[alg_b],
                    test_type="paired_ttest"
                )

                pairwise_tests[test_key] = test_result

        # ANOVA test for multiple groups
        if len(algorithm_names) > 2:
            values_list = [algorithm_values[alg] for alg in algorithm_names]
            f_stat, p_value = stats.f_oneway(*values_list)

            anova_result = {
                "f_statistic": f_stat,
                "p_value": p_value,
                "is_significant": p_value < self.validator.significance_level,
            }
        else:
            anova_result = None

        comparison_results = {
            "pairwise_tests": pairwise_tests,
            "anova": anova_result,
            "algorithm_statistics": {
                alg: {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
                    "min": np.min(values),
                    "max": np.max(values),
                }
                for alg, values in algorithm_values.items()
            }
        }

        # Generate plots
        if save_plots:
            self._plot_algorithm_comparison(
                algorithm_values,
                metric,
                comparison_results
            )

        return comparison_results

    def _plot_algorithm_comparison(
        self,
        algorithm_values: Dict[str, List[float]],
        metric: str,
        comparison_results: Dict[str, Any],
    ):
        """Generate comparison plots."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Box plot
        ax1 = axes[0, 0]
        data_for_boxplot = list(algorithm_values.values())
        labels = list(algorithm_values.keys())

        bp = ax1.boxplot(data_for_boxplot, labels=labels, patch_artist=True)
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax1.set_title(f'Algorithm Comparison - {metric}')
        ax1.set_ylabel(metric.capitalize())
        ax1.tick_params(axis='x', rotation=45)

        # Violin plot
        ax2 = axes[0, 1]
        positions = range(1, len(algorithm_values) + 1)
        parts = ax2.violinplot(data_for_boxplot, positions=positions)

        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)

        ax2.set_xticks(positions)
        ax2.set_xticklabels(labels, rotation=45)
        ax2.set_title(f'Distribution Comparison - {metric}')
        ax2.set_ylabel(metric.capitalize())

        # Bar plot with error bars
        ax3 = axes[1, 0]
        means = [comparison_results["algorithm_statistics"][alg]["mean"]
                for alg in labels]
        stds = [comparison_results["algorithm_statistics"][alg]["std"]
               for alg in labels]

        bars = ax3.bar(labels, means, yerr=stds, capsize=5, color=colors)
        ax3.set_title(f'Mean {metric} with Error Bars')
        ax3.set_ylabel(metric.capitalize())
        ax3.tick_params(axis='x', rotation=45)

        # Significance matrix
        ax4 = axes[1, 1]
        n_algs = len(labels)
        sig_matrix = np.ones((n_algs, n_algs))

        for i, alg_a in enumerate(labels):
            for j, alg_b in enumerate(labels):
                if i != j:
                    test_key = f"{alg_a}_vs_{alg_b}" if i < j else f"{alg_b}_vs_{alg_a}"
                    if test_key in comparison_results["pairwise_tests"]:
                        p_val = comparison_results["pairwise_tests"][test_key]["p_value"]
                        sig_matrix[i, j] = p_val

        im = ax4.imshow(sig_matrix, cmap='RdYlGn_r', vmin=0, vmax=0.05)
        ax4.set_xticks(range(n_algs))
        ax4.set_yticks(range(n_algs))
        ax4.set_xticklabels(labels, rotation=45)
        ax4.set_yticklabels(labels)
        ax4.set_title('P-values Matrix (Darker = More Significant)')

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax4)
        cbar.set_label('P-value')

        plt.tight_layout()

        # Save plot
        plot_path = self.base_dir / f"algorithm_comparison_{metric}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Saved comparison plot to {plot_path}")

    def generate_report(
        self,
        experiment_results: Dict[str, List[ExperimentResult]],
        output_path: Optional[str] = None,
    ) -> str:
        """Generate a comprehensive experiment report.

        Args:
            experiment_results: Dictionary of experiment results
            output_path: Optional path to save report

        Returns:
            Report content as string
        """
        report_lines = []

        # Header
        report_lines.extend([
            "# Federated Learning Experiment Report",
            "",
            f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total algorithms tested: {len(experiment_results)}",
            "",
        ])

        # Summary statistics
        report_lines.append("## Summary Statistics")
        report_lines.append("")

        for alg_name, results in experiment_results.items():
            if results:
                report_lines.extend([
                    f"### {alg_name}",
                    f"- Number of runs: {len(results)}",
                    f"- Configuration: {results[0].config.description}",
                    ""
                ])

        # Detailed comparison for main metrics
        main_metrics = ["accuracy", "f1_score", "loss"]
        for metric in main_metrics:
            if all(metric in r.final_metrics for results in experiment_results.values()
                   for r in results):

                report_lines.extend([
                    f"## {metric.capitalize()} Comparison",
                    ""
                ])

                comparison = self.compare_algorithms(experiment_results, metric, save_plots=False)

                # Algorithm statistics
                report_lines.append("### Algorithm Performance")
                for alg, stats in comparison["algorithm_statistics"].items():
                    report_lines.append(
                        f"- **{alg}**: {stats['mean']:.4f} ± {stats['std']:.4f} "
                        f"(median: {stats['median']:.4f})"
                    )

                report_lines.append("")

                # Significant differences
                report_lines.append("### Statistical Significance")
                significant_pairs = []
                for test_name, test_result in comparison["pairwise_tests"].items():
                    if test_result["is_significant"]:
                        alg_a, alg_b = test_name.split("_vs_")
                        p_val = test_result["p_value"]
                        effect_size = test_result.get("effect_size", "N/A")
                        significant_pairs.append((alg_a, alg_b, p_val, effect_size))

                if significant_pairs:
                    for alg_a, alg_b, p_val, effect_size in significant_pairs:
                        report_lines.append(
                            f"- **{alg_a} vs {alg_b}**: p={p_val:.6f}, "
                            f"effect size={effect_size:.3f if isinstance(effect_size, float) else effect_size}"
                        )
                else:
                    report_lines.append("- No statistically significant differences found")

                report_lines.append("")

        # Reproducibility information
        report_lines.extend([
            "## Reproducibility",
            "",
        ])

        # Get unique environment hashes and git commits
        env_hashes = set()
        git_commits = set()

        for results in experiment_results.values():
            for result in results:
                if result.environment_hash:
                    env_hashes.add(result.environment_hash)
                if result.git_commit:
                    git_commits.add(result.git_commit)

        report_lines.extend([
            f"- Environment hashes: {', '.join(env_hashes) if env_hashes else 'None'}",
            f"- Git commits: {', '.join(git_commits) if git_commits else 'None'}",
            "",
        ])

        # Recommendations
        report_lines.extend([
            "## Recommendations",
            "",
            "Based on the statistical analysis:",
            "",
        ])

        # Find best performing algorithm for main metric
        if "accuracy" in main_metrics:
            best_alg = None
            best_score = -float('inf')

            for alg_name, results in experiment_results.items():
                if results and "accuracy" in results[0].final_metrics:
                    avg_score = np.mean([r.final_metrics["accuracy"] for r in results])
                    if avg_score > best_score:
                        best_score = avg_score
                        best_alg = alg_name

            if best_alg:
                report_lines.extend([
                    f"1. **{best_alg}** shows the highest average accuracy ({best_score:.4f})",
                    "",
                ])

        report_lines.extend([
            "2. Ensure sufficient statistical power for future experiments",
            "3. Consider effect sizes alongside p-values for practical significance",
            "4. Validate findings with different datasets and scenarios",
            "",
        ])

        # Join all lines
        report_content = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
            logger.info(f"Saved experiment report to {output_path}")

        return report_content
