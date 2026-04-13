"""
Multi-Objective Reward Functions for GRPO Code Generation.

This is the core innovation: extending GRPO's reward beyond binary correctness
to jointly optimize correctness, runtime efficiency, and code brevity.

Composite reward: R = α·R_correct + β·R_efficiency + γ·R_brevity

Where:
- R_correct ∈ {0, 1}: binary test pass/fail (or partial: num_passed/num_total)
- R_efficiency ∈ [0, 1]: 1 - (runtime / baseline), clamped to [0, 1]
- R_brevity ∈ [0, 1]: 1 - (num_tokens / baseline), clamped to [0, 1]
"""
import re
import numpy as np
from dataclasses import dataclass
from typing import Optional
from sandbox import ExecutionResult


@dataclass
class RewardBreakdown:
    """Detailed reward breakdown for logging and analysis."""
    total: float
    correctness: float
    efficiency: float
    brevity: float
    # Raw metrics (before normalization)
    passed: bool
    pass_rate: float      # num_passed / num_total
    runtime_sec: float
    num_tokens: int


def compute_reward(
    execution_result: ExecutionResult,
    generated_code: str,
    tokenizer=None,
    # Weights
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.1,
    # Baselines for normalization
    efficiency_baseline: float = 5.0,
    brevity_baseline: int = 512,
    # Options
    partial_credit: bool = True,  # Give credit for passing some tests
    efficiency_only_if_correct: bool = True,  # Only reward efficiency for passing code
) -> RewardBreakdown:
    """
    Compute multi-objective reward for a single code generation.

    Args:
        execution_result: Result from sandbox execution
        generated_code: The raw generated code string
        tokenizer: Optional tokenizer for accurate token counting
        alpha/beta/gamma: Reward component weights
        efficiency_baseline: Runtime (sec) that maps to R_efficiency=0
        brevity_baseline: Token count that maps to R_brevity=0
        partial_credit: If True, R_correct = num_passed/num_total instead of binary
        efficiency_only_if_correct: If True, R_efficiency=0 for failing code

    Returns:
        RewardBreakdown with total reward and all components
    """
    # ===== 1. Correctness reward =====
    if partial_credit and execution_result.num_total > 0:
        r_correct = execution_result.num_passed / execution_result.num_total
    else:
        r_correct = 1.0 if execution_result.passed else 0.0

    # ===== 2. Efficiency reward =====
    runtime = execution_result.runtime_seconds

    if efficiency_only_if_correct and not execution_result.passed:
        # Don't reward efficiency for incorrect code
        r_efficiency = 0.0
    elif execution_result.error_type == "timeout":
        r_efficiency = 0.0
    else:
        # Linear: faster → higher reward, clamped to [0, 1]
        r_efficiency = max(0.0, 1.0 - (runtime / efficiency_baseline))

    # ===== 3. Brevity reward =====
    if tokenizer is not None:
        num_tokens = len(tokenizer.encode(generated_code))
    else:
        # Rough approximation: ~3.5 chars per token for code
        num_tokens = len(generated_code) // 4

    r_brevity = max(0.0, 1.0 - (num_tokens / brevity_baseline))

    # ===== Composite reward =====
    total = alpha * r_correct + beta * r_efficiency + gamma * r_brevity

    return RewardBreakdown(
        total=total,
        correctness=r_correct,
        efficiency=r_efficiency,
        brevity=r_brevity,
        passed=execution_result.passed,
        pass_rate=execution_result.num_passed / max(execution_result.num_total, 1),
        runtime_sec=runtime,
        num_tokens=num_tokens,
    )


def compute_batch_rewards(
    execution_results: list[ExecutionResult],
    generated_codes: list[str],
    tokenizer=None,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.1,
    **kwargs,
) -> tuple[list[float], list[RewardBreakdown]]:
    """
    Compute rewards for a batch of generations.
    Returns (reward_values, breakdowns) for logging.
    """
    rewards = []
    breakdowns = []

    for exec_result, code in zip(execution_results, generated_codes):
        breakdown = compute_reward(
            execution_result=exec_result,
            generated_code=code,
            tokenizer=tokenizer,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            **kwargs,
        )
        rewards.append(breakdown.total)
        breakdowns.append(breakdown)

    return rewards, breakdowns


def log_reward_stats(breakdowns: list[RewardBreakdown], step: int, wandb_run=None):
    """Log detailed reward statistics to wandb and/or console."""
    if not breakdowns:
        return

    stats = {
        "reward/total_mean": np.mean([b.total for b in breakdowns]),
        "reward/total_std": np.std([b.total for b in breakdowns]),
        "reward/correctness_mean": np.mean([b.correctness for b in breakdowns]),
        "reward/efficiency_mean": np.mean([b.efficiency for b in breakdowns]),
        "reward/brevity_mean": np.mean([b.brevity for b in breakdowns]),
        "reward/pass_rate": np.mean([b.pass_rate for b in breakdowns]),
        "reward/full_pass_rate": np.mean([b.passed for b in breakdowns]),
        "metrics/runtime_mean": np.mean([b.runtime_sec for b in breakdowns]),
        "metrics/runtime_p50": np.median([b.runtime_sec for b in breakdowns]),
        "metrics/num_tokens_mean": np.mean([b.num_tokens for b in breakdowns]),
        "metrics/num_tokens_p50": np.median([b.num_tokens for b in breakdowns]),
    }

    print(f"[Step {step}] "
          f"R={stats['reward/total_mean']:.3f}±{stats['reward/total_std']:.3f} | "
          f"pass={stats['reward/full_pass_rate']:.1%} | "
          f"eff={stats['reward/efficiency_mean']:.3f} | "
          f"brev={stats['reward/brevity_mean']:.3f} | "
          f"runtime={stats['metrics/runtime_mean']:.3f}s | "
          f"tokens={stats['metrics/num_tokens_mean']:.0f}")

    if wandb_run is not None:
        wandb_run.log(stats, step=step)

    return stats


# ===== Reward function factory for TRL GRPOTrainer =====

def make_reward_function(
    test_cases_lookup: dict,  # Maps prompt_hash -> list of test cases
    tokenizer=None,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.1,
    efficiency_baseline: float = 5.0,
    brevity_baseline: int = 512,
    timeout: float = 10.0,
):
    """
    Create a reward function compatible with TRL GRPOTrainer.

    TRL's GRPOTrainer expects a function:
        reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]

    Args:
        test_cases_lookup: Dict mapping problem identifier to test cases.
        Other args: reward configuration.

    Returns:
        Callable compatible with GRPOTrainer's reward_func parameter.
    """
    from sandbox import execute_code_with_tests

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        rewards = []

        for completion, prompt in zip(completions, prompts):
            # Extract problem ID from prompt to get test cases
            problem_id = _extract_problem_id(prompt)
            tests = test_cases_lookup.get(problem_id, [])

            if not tests:
                rewards.append(0.0)
                continue

            # Extract just the code from the completion
            code = extract_code_from_completion(completion)

            # Execute
            exec_result = execute_code_with_tests(
                code=code,
                test_cases=tests,
                timeout=timeout,
            )

            # Compute multi-objective reward
            breakdown = compute_reward(
                execution_result=exec_result,
                generated_code=code,
                tokenizer=tokenizer,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                efficiency_baseline=efficiency_baseline,
                brevity_baseline=brevity_baseline,
            )
            rewards.append(breakdown.total)

        return rewards

    return reward_fn


def extract_code_from_completion(completion: str) -> str:
    """Extract Python code from a model completion, handling markdown blocks."""
    # Try to extract from ```python ... ``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # Try generic code blocks
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, completion, re.DOTALL)
    if matches:
        return matches[-1].strip()

    # If no code blocks, return the whole completion
    # (strip common non-code prefixes)
    lines = completion.strip().split('\n')
    code_lines = []
    for line in lines:
        # Skip lines that look like natural language explanations
        if line.strip() and not line.strip().startswith('#') and any(
            kw in line for kw in ['def ', 'class ', 'import ', 'from ', 'if ', 'for ',
                                   'while ', 'return ', 'print(', '=', '(', ')']
        ):
            code_lines.append(line)
        elif code_lines:  # Already in code mode, keep going
            code_lines.append(line)

    return '\n'.join(code_lines) if code_lines else completion.strip()


def _extract_problem_id(prompt: str) -> str:
    """Extract a unique problem identifier from the prompt for test case lookup."""
    # Use hash of the problem description part
    # Strip the instruction prefix to get just the problem
    import hashlib
    return hashlib.md5(prompt.encode()).hexdigest()[:16]
