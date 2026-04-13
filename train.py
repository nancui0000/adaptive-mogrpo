"""
MO-GRPO: Multi-Objective GRPO Training for Code Generation
with Adaptive Weight Scheduling.

Adaptive scheduling gradually introduces multi-objective rewards:
  - Phase 1 (0% - warmup_frac): correctness only (α=1, β=0, γ=0)
  - Phase 2 (warmup_frac - 100%): linearly ramp β and γ to target values

This prevents reward hacking by ensuring the model first learns correctness
before being optimized for efficiency and brevity.

Usage:
    # Fixed weights (same as before)
    python train.py --preset correctness_only --run_name baseline --max_steps 2000

    # Adaptive scheduling (NEW)
    python train.py --preset adaptive_balanced --run_name adaptive_bal --max_steps 2000 --adaptive
    python train.py --preset adaptive_eff_heavy --run_name adaptive_eff --max_steps 2000 --adaptive

    # Custom adaptive with warmup fraction
    python train.py --alpha 1.0 --beta 0.3 --gamma 0.15 --adaptive --warmup_frac 0.4 --run_name custom_adaptive --max_steps 2000
"""
import argparse
import hashlib
import json
import os
import tempfile
import subprocess
import torch
import wandb
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from peft import LoraConfig, TaskType
from trl import GRPOConfig, GRPOTrainer

from config import ModelConfig, RewardConfig, GRPOConfig as MyGRPOConfig, DataConfig
from config import get_preset_reward_config
from rewards import extract_code_from_completion


def parse_args():
    parser = argparse.ArgumentParser(description="MO-GRPO Training")
    parser.add_argument("--preset", type=str, default=None,
                        choices=list(RewardConfig.PRESETS.keys()),
                        help="Use a predefined reward weight configuration")
    parser.add_argument("--alpha", type=float, default=None, help="Correctness weight")
    parser.add_argument("--beta", type=float, default=None, help="Efficiency weight")
    parser.add_argument("--gamma", type=float, default=None, help="Brevity weight")
    parser.add_argument("--run_name", type=str, default="mogrpo", help="W&B run name")
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--dataset", type=str, default="mbpp",
                        choices=["mbpp", "apps"], help="Which dataset to use")
    parser.add_argument("--no_wandb", action="store_true")
    # Adaptive scheduling args
    parser.add_argument("--adaptive", action="store_true",
                        help="Enable adaptive weight scheduling")
    parser.add_argument("--warmup_frac", type=float, default=0.3,
                        help="Fraction of training for correctness-only phase (default: 0.3 = first 30%%)")
    return parser.parse_args()


class AdaptiveWeightScheduler:
    """
    Adaptive Weight Scheduling for Multi-Objective GRPO.

    Instead of using fixed reward weights throughout training, we gradually
    introduce multi-objective rewards:

    Phase 1 (step 0 to warmup_steps):
        Only correctness reward (α=target_α, β=0, γ=0)
        Model learns to generate correct code first.

    Phase 2 (warmup_steps to max_steps):
        Linearly ramp β from 0 → target_β, γ from 0 → target_γ
        Gradually introduce efficiency and brevity pressure.

    This is analogous to curriculum learning: easy objective first,
    then progressively harder multi-objective optimization.
    """

    def __init__(self, target_alpha, target_beta, target_gamma,
                 max_steps, warmup_frac=0.3):
        self.target_alpha = target_alpha
        self.target_beta = target_beta
        self.target_gamma = target_gamma
        self.max_steps = max_steps
        self.warmup_steps = int(max_steps * warmup_frac)
        self.current_step = 0

    def get_weights(self, step=None):
        """Get current reward weights based on training step."""
        if step is not None:
            self.current_step = step

        if self.current_step <= self.warmup_steps:
            # Phase 1: correctness only
            return self.target_alpha, 0.0, 0.0
        else:
            # Phase 2: linear ramp
            progress = (self.current_step - self.warmup_steps) / max(self.max_steps - self.warmup_steps, 1)
            progress = min(progress, 1.0)
            beta = self.target_beta * progress
            gamma = self.target_gamma * progress
            return self.target_alpha, beta, gamma

    def step(self):
        self.current_step += 1
        return self.get_weights()


def prepare_mbpp_dataset(tokenizer, max_samples=2000):
    """Prepare MBPP dataset for GRPO training."""
    print("Loading MBPP dataset...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="train")

    test_cases_lookup = {}
    formatted_data = []

    for i, example in enumerate(ds):
        if i >= max_samples:
            break

        prompt = format_code_prompt(example["prompt"])
        tests = parse_mbpp_tests(example.get("test_list", []))
        if len(tests) < 1:
            continue

        problem_id = hashlib.md5(prompt.encode()).hexdigest()[:16]
        test_cases_lookup[problem_id] = tests
        formatted_data.append({"prompt": prompt, "problem_id": problem_id})

    dataset = Dataset.from_list(formatted_data)
    print(f"Prepared {len(dataset)} problems with test cases")
    return dataset, test_cases_lookup


def prepare_apps_dataset(tokenizer, max_samples=2000):
    """Prepare APPS dataset for GRPO training."""
    print("Loading APPS dataset...")
    ds = load_dataset("codeparrot/apps", split="train", trust_remote_code=True)
    ds = ds.filter(lambda x: x.get("difficulty", "") in ["introductory", "interview"])

    test_cases_lookup = {}
    formatted_data = []

    for i, example in enumerate(ds):
        if i >= max_samples:
            break
        try:
            io_pairs = json.loads(example.get("input_output", "{}"))
            inputs = io_pairs.get("inputs", [])
            outputs = io_pairs.get("outputs", [])
            if len(inputs) < 2:
                continue
            tests = [
                {"input": str(inp).strip(), "expected_output": str(out).strip()}
                for inp, out in zip(inputs[:5], outputs[:5])
            ]
        except (json.JSONDecodeError, KeyError):
            continue

        prompt = format_code_prompt(example["question"])
        problem_id = hashlib.md5(prompt.encode()).hexdigest()[:16]
        test_cases_lookup[problem_id] = tests
        formatted_data.append({"prompt": prompt, "problem_id": problem_id})

    dataset = Dataset.from_list(formatted_data)
    print(f"Prepared {len(dataset)} problems with test cases")
    return dataset, test_cases_lookup


def format_code_prompt(problem_description: str) -> str:
    return (
        "Solve the following programming problem. "
        "Write clean, efficient Python code. "
        "Return ONLY the code inside a ```python``` block.\n\n"
        f"Problem:\n{problem_description.strip()}\n\n"
        "Solution:"
    )


def parse_mbpp_tests(test_list: list) -> list:
    tests = []
    for test_str in test_list:
        test_str = test_str.strip()
        if test_str.startswith("assert"):
            tests.append({
                "input": "",
                "expected_output": "PASS",
                "_assertion": test_str,
            })
    return tests


def make_mbpp_reward_function(
    test_cases_lookup: dict,
    tokenizer=None,
    alpha: float = 1.0,
    beta: float = 0.3,
    gamma: float = 0.1,
    efficiency_baseline: float = 0.5,
    brevity_baseline: int = 200,
    timeout: float = 10.0,
    scheduler: AdaptiveWeightScheduler = None,
):
    """
    MBPP reward function with dense signals and optional adaptive scheduling.

    If scheduler is provided, weights are dynamically adjusted each call
    based on the current training step.
    """
    import time as time_module
    import ast

    # Track call count for adaptive scheduling
    call_state = {"calls": 0, "current_alpha": alpha, "current_beta": beta, "current_gamma": gamma}

    def reward_fn(completions: list[str], prompts: list[str], **kwargs) -> list[float]:
        # Update weights if using adaptive scheduling
        if scheduler is not None:
            a, b, g = scheduler.step()
            call_state["current_alpha"] = a
            call_state["current_beta"] = b
            call_state["current_gamma"] = g
        else:
            a = alpha
            b = beta
            g = gamma

        call_state["calls"] += 1

        # Log weight changes periodically
        if call_state["calls"] % 50 == 0:
            print(f"  [Reward weights @ call {call_state['calls']}] "
                  f"α={a:.3f}, β={b:.3f}, γ={g:.3f}")

        rewards = []

        for completion, prompt in zip(completions, prompts):
            problem_id = hashlib.md5(prompt.encode()).hexdigest()[:16]
            tests = test_cases_lookup.get(problem_id, [])
            code = extract_code_from_completion(completion)

            # --- Format rewards ---
            format_reward = 0.0
            try:
                ast.parse(code)
                format_reward += 0.1
            except SyntaxError:
                rewards.append(a * 0.0 + b * 0.0 + g * 0.0)
                continue

            if "def " in code:
                format_reward += 0.1

            if not tests:
                rewards.append(format_reward)
                continue

            # --- Test passing (partial credit) ---
            num_passed = 0
            total_runtime = 0.0
            num_total = len([t for t in tests if t.get("_assertion")])

            for test in tests:
                assertion = test.get("_assertion", "")
                if not assertion:
                    continue

                test_script = f"{code}\n\n{assertion}\nprint('PASS')"

                try:
                    with tempfile.NamedTemporaryFile(
                        mode='w', suffix='.py', delete=False
                    ) as f:
                        f.write(test_script)
                        path = f.name

                    start = time_module.monotonic()
                    proc = subprocess.run(
                        ["python3", path],
                        capture_output=True, text=True,
                        timeout=timeout / max(num_total, 1),
                    )
                    elapsed = time_module.monotonic() - start
                    total_runtime += elapsed

                    if proc.returncode == 0 and "PASS" in proc.stdout:
                        num_passed += 1

                    os.unlink(path)
                except subprocess.TimeoutExpired:
                    total_runtime += timeout / max(num_total, 1)
                    try: os.unlink(path)
                    except: pass
                except Exception:
                    try: os.unlink(path)
                    except: pass

            if num_total == 0:
                rewards.append(format_reward)
                continue

            r_correct = num_passed / num_total
            r_efficiency = 0.0
            if num_passed > 0:
                r_efficiency = max(0.0, 1.0 - total_runtime / efficiency_baseline)

            if tokenizer is not None:
                num_tokens = len(tokenizer.encode(code))
            else:
                num_tokens = len(code) // 4
            r_brevity = max(0.0, 1.0 - num_tokens / brevity_baseline)

            total = a * (format_reward + r_correct) + b * r_efficiency + g * r_brevity
            rewards.append(total)

        return rewards

    return reward_fn


def main():
    args = parse_args()

    # ===== Resolve reward weights =====
    if args.preset:
        reward_cfg = get_preset_reward_config(args.preset)
        print(f"Using preset '{args.preset}': α={reward_cfg.alpha}, β={reward_cfg.beta}, γ={reward_cfg.gamma}")
    else:
        reward_cfg = RewardConfig()
        if args.alpha is not None: reward_cfg.alpha = args.alpha
        if args.beta is not None: reward_cfg.beta = args.beta
        if args.gamma is not None: reward_cfg.gamma = args.gamma
        print(f"Using custom weights: α={reward_cfg.alpha}, β={reward_cfg.beta}, γ={reward_cfg.gamma}")

    # ===== Setup adaptive scheduler =====
    scheduler = None
    if args.adaptive:
        scheduler = AdaptiveWeightScheduler(
            target_alpha=reward_cfg.alpha,
            target_beta=reward_cfg.beta,
            target_gamma=reward_cfg.gamma,
            max_steps=args.max_steps,
            warmup_frac=args.warmup_frac,
        )
        print(f"Adaptive scheduling ENABLED:")
        print(f"  Phase 1 (steps 0-{scheduler.warmup_steps}): correctness only")
        print(f"  Phase 2 (steps {scheduler.warmup_steps}-{args.max_steps}): "
              f"ramp β→{reward_cfg.beta}, γ→{reward_cfg.gamma}")
    else:
        print(f"Fixed weights (no adaptive scheduling)")

    # ===== Initialize W&B =====
    if not args.no_wandb:
        wandb.init(
            project="mo-grpo-code",
            name=f"{args.run_name}_a{reward_cfg.alpha}_b{reward_cfg.beta}_g{reward_cfg.gamma}",
            config={
                "alpha": reward_cfg.alpha,
                "beta": reward_cfg.beta,
                "gamma": reward_cfg.gamma,
                "preset": args.preset,
                "dataset": args.dataset,
                "max_steps": args.max_steps,
                "adaptive": args.adaptive,
                "warmup_frac": args.warmup_frac if args.adaptive else None,
            },
        )

    # ===== Load tokenizer =====
    model_cfg = ModelConfig()
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ===== Prepare dataset =====
    if args.dataset == "mbpp":
        dataset, test_cases_lookup = prepare_mbpp_dataset(tokenizer)
    else:
        dataset, test_cases_lookup = prepare_apps_dataset(tokenizer)

    # ===== Build reward function =====
    reward_fn = make_mbpp_reward_function(
        test_cases_lookup=test_cases_lookup,
        tokenizer=tokenizer,
        alpha=reward_cfg.alpha,
        beta=reward_cfg.beta,
        gamma=reward_cfg.gamma,
        efficiency_baseline=reward_cfg.efficiency_baseline,
        brevity_baseline=reward_cfg.brevity_baseline,
        timeout=reward_cfg.execution_timeout,
        scheduler=scheduler,
    )

    # ===== LoRA config =====
    peft_config = LoraConfig(
        r=model_cfg.lora_r,
        lora_alpha=model_cfg.lora_alpha,
        lora_dropout=model_cfg.lora_dropout,
        target_modules=model_cfg.lora_target_modules,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )

    # ===== GRPO Training config =====
    grpo_cfg = MyGRPOConfig()
    scheduling_tag = "_adaptive" if args.adaptive else ""
    output_dir = os.path.join(
        args.output_dir,
        f"{args.run_name}_a{reward_cfg.alpha}_b{reward_cfg.beta}_g{reward_cfg.gamma}{scheduling_tag}"
    )

    training_args = GRPOConfig(
        output_dir=output_dir,
        run_name=args.run_name,

        # GRPO specific
        num_generations=grpo_cfg.num_generations,
        max_completion_length=grpo_cfg.max_completion_length,
        temperature=grpo_cfg.temperature,
        generation_batch_size=grpo_cfg.num_generations,

        # Optimization
        learning_rate=grpo_cfg.learning_rate,
        num_train_epochs=grpo_cfg.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=grpo_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=grpo_cfg.gradient_accumulation_steps,
        max_grad_norm=grpo_cfg.max_grad_norm,
        warmup_ratio=grpo_cfg.warmup_ratio,

        # vLLM
        use_vllm=grpo_cfg.use_vllm,

        # Logging
        logging_steps=grpo_cfg.logging_steps,
        save_steps=grpo_cfg.save_steps,
        report_to="wandb" if not args.no_wandb else "none",

        # Mixed precision
        bf16=True,

        # Gradient checkpointing
        gradient_checkpointing=True,
    )

    # ===== Initialize trainer =====
    print(f"\n{'='*60}")
    print(f"Starting MO-GRPO Training")
    print(f"Model: {model_cfg.model_name}")
    print(f"Target weights: α={reward_cfg.alpha}, β={reward_cfg.beta}, γ={reward_cfg.gamma}")
    print(f"Adaptive: {args.adaptive}")
    if args.adaptive:
        print(f"Warmup: {args.warmup_frac*100:.0f}% ({scheduler.warmup_steps} steps)")
    print(f"Num generations: {grpo_cfg.num_generations}")
    print(f"Batch size: {grpo_cfg.per_device_train_batch_size}")
    print(f"Max steps: {args.max_steps}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")

    trainer = GRPOTrainer(
        model=model_cfg.model_name,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        reward_funcs=reward_fn,
    )

    # ===== Train =====
    trainer.train()

    # ===== Save =====
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config for reproducibility
    with open(os.path.join(output_dir, "mo_grpo_config.json"), "w") as f:
        json.dump({
            "alpha": reward_cfg.alpha,
            "beta": reward_cfg.beta,
            "gamma": reward_cfg.gamma,
            "preset": args.preset,
            "dataset": args.dataset,
            "max_steps": args.max_steps,
            "model": model_cfg.model_name,
            "adaptive": args.adaptive,
            "warmup_frac": args.warmup_frac if args.adaptive else None,
        }, f, indent=2)

    # ===== Save training log from last checkpoint =====
    last_ckpt = os.path.join(output_dir, f"checkpoint-{args.max_steps}")
    if not os.path.exists(last_ckpt):
        # Find the latest checkpoint
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                       key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0)
        if ckpts:
            last_ckpt = os.path.join(output_dir, ckpts[-1])

    trainer_state_path = os.path.join(last_ckpt, "trainer_state.json")
    if os.path.exists(trainer_state_path):
        # Copy trainer_state.json to output root for easy access
        import shutil
        shutil.copy(trainer_state_path, os.path.join(output_dir, "trainer_state.json"))
        print(f"Copied training log to {output_dir}/trainer_state.json")

    print(f"\nTraining complete! Model saved to {output_dir}")

    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()