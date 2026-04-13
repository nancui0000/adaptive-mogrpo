"""
MO-GRPO Configuration
All hyperparameters in one place for easy ablation.
Optimized for A100 80GB.
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ModelConfig:
    model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    torch_dtype: str = "bfloat16"

    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"]
    )


@dataclass
class GRPOConfig:
    """GRPO training hyperparameters (DAPO variant). Optimized for A100 80GB."""
    # Core GRPO
    num_generations: int = 16          # More generations = better advantage signal
    max_completion_length: int = 1024
    temperature: float = 1.0

    # Optimization
    learning_rate: float = 2e-6
    num_train_epochs: int = 1
    max_steps: int = 2000
    per_device_train_batch_size: int = 2   # Can be higher on 80GB
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 0.1
    warmup_ratio: float = 0.05

    # vLLM - enabled on 80GB
    use_vllm: bool = False  # Keep false for stability with TRL v1.0 + vllm 0.19

    # Logging
    logging_steps: int = 5
    save_steps: int = 200
    report_to: str = "wandb"
    run_name: str = "mogrpo"


@dataclass
class RewardConfig:
    """Multi-objective reward weights and settings."""

    # ===== Reward weights =====
    alpha: float = 1.0   # Correctness weight
    beta: float = 0.3    # Efficiency weight
    gamma: float = 0.1   # Brevity weight

    # ===== Predefined weight configurations for experiments =====
    PRESETS = {
        "correctness_only":  {"alpha": 1.0, "beta": 0.0, "gamma": 0.0},
        "correctness_heavy": {"alpha": 1.0, "beta": 0.2, "gamma": 0.1},
        "balanced":          {"alpha": 1.0, "beta": 0.3, "gamma": 0.15},
        "efficiency_heavy":  {"alpha": 1.0, "beta": 0.5, "gamma": 0.1},
        # Adaptive scheduling presets - final target weights
        "adaptive_balanced":      {"alpha": 1.0, "beta": 0.3, "gamma": 0.15},
        "adaptive_eff_heavy":     {"alpha": 1.0, "beta": 0.5, "gamma": 0.1},
    }

    # ===== Execution sandbox settings =====
    execution_timeout: float = 10.0
    memory_limit_mb: int = 256

    # ===== Efficiency reward =====
    efficiency_baseline: float = 0.5

    # ===== Brevity reward =====
    brevity_baseline: int = 200
    brevity_min: float = 0.0


@dataclass
class DataConfig:
    """Dataset configuration."""
    dataset_name: str = "codeparrot/apps"
    dataset_split: str = "train"
    difficulty_filter: list = field(
        default_factory=lambda: ["introductory", "interview"]
    )
    max_samples: int = 2000
    min_tests: int = 3
    test_split_ratio: float = 0.1
    use_mbpp: bool = False
    mbpp_dataset: str = "google-research-datasets/mbpp"


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    eval_humaneval: bool = True
    eval_mbpp: bool = True
    num_samples_per_problem: int = 5
    eval_temperature: float = 0.2
    eval_max_tokens: int = 1024


def get_preset_reward_config(preset_name: str) -> RewardConfig:
    """Load a predefined reward weight configuration."""
    if preset_name not in RewardConfig.PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. "
                         f"Available: {list(RewardConfig.PRESETS.keys())}")
    cfg = RewardConfig()
    weights = RewardConfig.PRESETS[preset_name]
    cfg.alpha = weights["alpha"]
    cfg.beta = weights["beta"]
    cfg.gamma = weights["gamma"]
    return cfg