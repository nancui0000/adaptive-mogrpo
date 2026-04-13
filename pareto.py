"""
Pareto Frontier Analysis for MO-GRPO experiments.
Reads training logs from TRL's trainer_state.json files.

Usage:
    python pareto.py --results_dir ./outputs --output_dir ./figures
"""
import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from pathlib import Path


def load_eval_results(results_dir: str) -> list[dict]:
    """Load evaluation results from all experiment runs."""
    results = []
    for run_dir in Path(results_dir).iterdir():
        if not run_dir.is_dir():
            continue

        eval_file = run_dir / "eval_metrics.json"
        config_file = run_dir / "mo_grpo_config.json"

        if not eval_file.exists():
            continue

        with open(eval_file) as f:
            metrics = json.load(f)
        config = {}
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)

        results.append({
            "run_name": run_dir.name,
            "alpha": config.get("alpha", 1.0),
            "beta": config.get("beta", 0.0),
            "gamma": config.get("gamma", 0.0),
            "preset": config.get("preset", "unknown"),
            "adaptive": config.get("adaptive", False),
            "log_dir": str(run_dir),
            **metrics,
        })

    return results


def load_training_logs(results_dir: str) -> dict:
    """Load training logs from trainer_state.json files."""
    logs = {}
    for run_dir in Path(results_dir).iterdir():
        if not run_dir.is_dir():
            continue

        # Look for trainer_state.json in root or latest checkpoint
        state_file = run_dir / "trainer_state.json"
        if not state_file.exists():
            # Try to find in checkpoints
            ckpts = sorted(
                [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
                key=lambda x: int(x.name.split("-")[1]) if x.name.split("-")[1].isdigit() else 0
            )
            if ckpts:
                state_file = ckpts[-1] / "trainer_state.json"

        if state_file.exists():
            with open(state_file) as f:
                state = json.load(f)
            log_history = state.get("log_history", [])
            if log_history:
                logs[run_dir.name] = log_history

    return logs


def plot_pareto_2d(results: list[dict], output_dir: str):
    """Plot 2D Pareto frontiers."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    colors = sns.color_palette("husl", len(results))

    # --- Correctness vs Efficiency ---
    ax = axes[0]
    for i, r in enumerate(results):
        label = r.get("preset", r["run_name"])
        marker = '★' if r.get("adaptive") else '●'
        size = 200 if r.get("adaptive") else 150
        edgecolor = 'red' if r.get("adaptive") else 'black'
        ax.scatter(
            r.get("pass_at_1", 0), r.get("avg_efficiency", 0),
            c=[colors[i]], s=size, zorder=5, edgecolors=edgecolor, linewidth=1.5
        )
        ax.annotate(
            f'{label}\n(β={r["beta"]})',
            (r.get("pass_at_1", 0), r.get("avg_efficiency", 0)),
            textcoords="offset points", xytext=(10, 5), fontsize=9,
        )
    ax.set_xlabel("Pass@1 (Correctness) →", fontsize=12)
    ax.set_ylabel("Avg Efficiency Score →", fontsize=12)
    ax.set_title("Correctness vs Efficiency Tradeoff", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # --- Correctness vs Brevity ---
    ax = axes[1]
    for i, r in enumerate(results):
        label = r.get("preset", r["run_name"])
        size = 200 if r.get("adaptive") else 150
        edgecolor = 'red' if r.get("adaptive") else 'black'
        ax.scatter(
            r.get("pass_at_1", 0), r.get("avg_brevity", 0),
            c=[colors[i]], s=size, zorder=5, edgecolors=edgecolor, linewidth=1.5
        )
        ax.annotate(
            f'{label}\n(γ={r["gamma"]})',
            (r.get("pass_at_1", 0), r.get("avg_brevity", 0)),
            textcoords="offset points", xytext=(10, 5), fontsize=9,
        )
    ax.set_xlabel("Pass@1 (Correctness) →", fontsize=12)
    ax.set_ylabel("Avg Brevity Score →", fontsize=12)
    ax.set_title("Correctness vs Brevity Tradeoff", fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "pareto_2d.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 2D Pareto plot: {path}")


def plot_pareto_3d(results: list[dict], output_dir: str):
    """3D scatter plot."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    colors = sns.color_palette("husl", len(results))

    for i, r in enumerate(results):
        ax.scatter(
            r.get("pass_at_1", 0), r.get("avg_efficiency", 0), r.get("avg_brevity", 0),
            c=[colors[i]], s=200, edgecolors='black', linewidth=1,
            label=r.get("preset", r["run_name"]),
        )
    ax.set_xlabel("Pass@1 →")
    ax.set_ylabel("Efficiency →")
    ax.set_zlabel("Brevity →")
    ax.set_title("3D Pareto Space", fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    path = os.path.join(output_dir, "pareto_3d.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved 3D Pareto plot: {path}")


def plot_training_dynamics(training_logs: dict, output_dir: str):
    """Plot training dynamics from trainer_state.json log_history."""
    if not training_logs:
        print("No training logs found, skipping training dynamics plot")
        return

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    metrics_to_plot = [
        ("reward", "Reward (mean)"),
        ("loss", "Loss"),
        ("grad_norm", "Gradient Norm"),
        ("entropy", "Entropy"),
        ("frac_reward_zero_std", "Frac Zero-Std Reward"),
        ("completions/mean_length", "Avg Completion Length"),
    ]

    colors = sns.color_palette("husl", len(training_logs))

    for ax, (metric, title) in zip(axes.flatten(), metrics_to_plot):
        for idx, (run_name, log_entries) in enumerate(training_logs.items()):
            steps = []
            values = []
            for entry in log_entries:
                if metric in entry and "step" in entry:
                    steps.append(entry["step"])
                    values.append(entry[metric])

            if steps:
                # Smooth with moving average for readability
                if len(values) > 20:
                    window = min(20, len(values) // 5)
                    smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                    smooth_steps = steps[window-1:]
                    ax.plot(smooth_steps, smoothed,
                            label=run_name.split("_a")[0], linewidth=2,
                            color=colors[idx], alpha=0.8)
                    ax.plot(steps, values, color=colors[idx], alpha=0.15, linewidth=0.5)
                else:
                    ax.plot(steps, values,
                            label=run_name.split("_a")[0], linewidth=2,
                            color=colors[idx])

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Value")
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "training_dynamics.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved training dynamics: {path}")


def plot_comparison_table(results: list[dict], output_dir: str):
    """Generate a summary comparison table as an image."""
    fig, ax = plt.subplots(figsize=(14, max(3, len(results) + 1)))
    ax.axis('off')

    headers = ["Config", "α", "β", "γ", "Adaptive", "Pass@1", "MBPP", "HumanEval",
               "Avg Tokens", "Brevity"]

    table_data = []
    for r in results:
        table_data.append([
            r.get("preset", r["run_name"]),
            f'{r["alpha"]:.1f}',
            f'{r["beta"]:.1f}',
            f'{r["gamma"]:.1f}',
            "✓" if r.get("adaptive") else "✗",
            f'{r.get("pass_at_1", 0):.1%}',
            f'{r.get("mbpp_pass_at_1", 0):.1%}',
            f'{r.get("humaneval_pass_at_1", 0):.1%}',
            f'{r.get("avg_tokens", 0):.0f}',
            f'{r.get("avg_brevity", 0):.3f}',
        ])

    table = ax.table(
        cellText=table_data, colLabels=headers,
        loc='center', cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    for j in range(len(headers)):
        table[0, j].set_facecolor('#4472C4')
        table[0, j].set_text_props(color='white', fontweight='bold')
    for i in range(len(table_data)):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[i+1, j].set_facecolor('#D9E2F3')

    plt.title("MO-GRPO Experiment Results", fontsize=14, fontweight='bold', pad=20)
    path = os.path.join(output_dir, "results_table.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved results table: {path}")


def plot_adaptive_vs_fixed(training_logs: dict, output_dir: str):
    """
    Special plot comparing adaptive vs fixed weight training.
    Shows how reward components evolve differently.
    """
    adaptive_logs = {k: v for k, v in training_logs.items() if "adaptive" in k.lower()}
    fixed_logs = {k: v for k, v in training_logs.items() if "adaptive" not in k.lower() and k != "base_model"}

    if not adaptive_logs:
        print("No adaptive logs found, skipping adaptive vs fixed plot")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, (metric, title) in zip(axes, [
        ("reward", "Total Reward"),
        ("grad_norm", "Gradient Norm"),
        ("loss", "Training Loss"),
    ]):
        # Plot fixed weight runs
        for run_name, entries in fixed_logs.items():
            steps = [e["step"] for e in entries if metric in e]
            values = [e[metric] for e in entries if metric in e]
            if steps and len(values) > 10:
                window = min(15, len(values) // 5)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], smoothed,
                        label=f"Fixed: {run_name.split('_a')[0]}",
                        linewidth=1.5, alpha=0.6, linestyle='--')

        # Plot adaptive runs
        for run_name, entries in adaptive_logs.items():
            steps = [e["step"] for e in entries if metric in e]
            values = [e[metric] for e in entries if metric in e]
            if steps and len(values) > 10:
                window = min(15, len(values) // 5)
                smoothed = np.convolve(values, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], smoothed,
                        label=f"Adaptive: {run_name.split('_a')[0]}",
                        linewidth=2.5, color='red')

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel("Training Step")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Adaptive vs Fixed Weight Scheduling", fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "adaptive_vs_fixed.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved adaptive vs fixed plot: {path}")


def generate_all_plots(results_dir: str, output_dir: str):
    """Generate all analysis plots."""
    os.makedirs(output_dir, exist_ok=True)

    results = load_eval_results(results_dir)
    if not results:
        print(f"No results found in {results_dir}")
        return

    print(f"Found {len(results)} experiment runs:")
    for r in results:
        adaptive_tag = " [ADAPTIVE]" if r.get("adaptive") else ""
        print(f"  - {r.get('preset', r['run_name'])}: "
              f"α={r['alpha']}, β={r['beta']}, γ={r['gamma']}{adaptive_tag}")

    plot_pareto_2d(results, output_dir)
    plot_pareto_3d(results, output_dir)
    plot_comparison_table(results, output_dir)

    # Training dynamics from trainer_state.json
    training_logs = load_training_logs(results_dir)
    print(f"Found training logs for {len(training_logs)} runs: {list(training_logs.keys())}")
    plot_training_dynamics(training_logs, output_dir)
    plot_adaptive_vs_fixed(training_logs, output_dir)

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="./outputs")
    parser.add_argument("--output_dir", type=str, default="./figures")
    args = parser.parse_args()

    generate_all_plots(args.results_dir, args.output_dir)