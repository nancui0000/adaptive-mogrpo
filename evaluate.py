"""
Evaluate MO-GRPO trained models on HumanEval+ and MBPP+.
Measures correctness, runtime efficiency, and code brevity.

Usage:
    python evaluate.py --model_dir ./outputs/balanced_a1.0_b0.3_g0.15
    python evaluate.py --model_dir ./outputs/baseline_a1.0_b0.0_g0.0

    # Evaluate base model (no fine-tuning)
    python evaluate.py --base_model
"""
import argparse
import hashlib
import json
import os
import time
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default=None,
                        help="Path to trained model (LoRA adapter)")
    parser.add_argument("--base_model", action="store_true",
                        help="Evaluate base model without LoRA")
    parser.add_argument("--base_model_name", type=str,
                        default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Samples per problem for pass@k")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_mbpp", action="store_true", default=True)
    parser.add_argument("--eval_humaneval", action="store_true", default=True)
    return parser.parse_args()


def load_model(args):
    """Load base model + optional LoRA adapter."""
    print(f"Loading model: {args.base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
            args.base_model_name,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    if args.model_dir and not args.base_model:
        print(f"Loading LoRA adapter: {args.model_dir}")
        model = PeftModel.from_pretrained(model, args.model_dir)
        model = model.merge_and_unload()

    model.eval()
    return model, tokenizer


def generate_solutions(
    model, tokenizer, prompt: str, n: int = 5,
    temperature: float = 0.2, max_tokens: int = 1024,
) -> list[str]:
    """Generate n solutions for a given prompt."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    solutions = []
    for _ in range(n):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
            )
        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        solutions.append(completion)

    return solutions


def extract_code(completion: str) -> str:
    """Extract code from model completion."""
    import re
    # Try ```python blocks
    matches = re.findall(r"```python\s*\n(.*?)```", completion, re.DOTALL)
    if matches:
        return matches[-1].strip()
    matches = re.findall(r"```\s*\n(.*?)```", completion, re.DOTALL)
    if matches:
        return matches[-1].strip()
    return completion.strip()


def execute_with_assertions(code: str, assertions: list[str], timeout: float = 10.0):
    """Execute code with assertion-style tests. Returns (passed, runtime)."""
    test_script = code + "\n\n" + "\n".join(assertions) + "\nprint('ALL_PASS')"

    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_script)
            path = f.name

        start = time.monotonic()
        proc = subprocess.run(
            ["python3", path],
            capture_output=True, text=True,
            timeout=timeout,
        )
        elapsed = time.monotonic() - start
        os.unlink(path)

        passed = proc.returncode == 0 and "ALL_PASS" in proc.stdout
        return passed, elapsed

    except subprocess.TimeoutExpired:
        try:
            os.unlink(path)
        except:
            pass
        return False, timeout
    except Exception:
        try:
            os.unlink(path)
        except:
            pass
        return False, 0.0


def evaluate_mbpp(model, tokenizer, args) -> dict:
    """Evaluate on MBPP sanitized test split."""
    print("\n--- Evaluating on MBPP (sanitized test) ---")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")

    results = []

    for i, example in enumerate(ds):
        prompt = (
            "Solve the following programming problem. "
            "Write clean, efficient Python code. "
            "Return ONLY the code inside a ```python``` block.\n\n"
            f"Problem:\n{example['prompt']}\n\n"
            "Solution:"
        )
        assertions = example.get("test_list", [])
        if not assertions:
            continue

        solutions = generate_solutions(
            model, tokenizer, prompt,
            n=args.num_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        problem_results = []
        for sol in solutions:
            code = extract_code(sol)
            passed, runtime = execute_with_assertions(code, assertions)
            num_tokens = len(tokenizer.encode(code))
            problem_results.append({
                "passed": passed,
                "runtime": runtime,
                "num_tokens": num_tokens,
                "code": code,
            })

        results.append({
            "task_id": example.get("task_id", i),
            "solutions": problem_results,
        })

        # Progress
        if (i + 1) % 20 == 0:
            pass_1 = np.mean([
                any(s["passed"] for s in r["solutions"])
                for r in results
            ])
            print(f"  [{i+1}/{len(ds)}] Running pass@1: {pass_1:.1%}")

    return _compute_metrics(results, "mbpp")


def evaluate_humaneval(model, tokenizer, args) -> dict:
    """Evaluate on HumanEval."""
    print("\n--- Evaluating on HumanEval ---")
    ds = load_dataset("openai/openai_humaneval", split="test")

    results = []

    for i, example in enumerate(ds):
        prompt = (
            "Complete the following Python function. "
            "Write clean, efficient code. "
            "Return ONLY the code inside a ```python``` block.\n\n"
            f"{example['prompt']}\n"
        )

        test_code = example.get("test", "")
        entry_point = example.get("entry_point", "")

        solutions = generate_solutions(
            model, tokenizer, prompt,
            n=args.num_samples,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        problem_results = []
        for sol in solutions:
            code = extract_code(sol)
            # HumanEval: prepend the prompt (function signature) if not present
            if entry_point and f"def {entry_point}" not in code:
                code = example["prompt"] + code

            full_test = code + "\n\n" + test_code + f"\ncheck({entry_point})\nprint('ALL_PASS')"

            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                    f.write(full_test)
                    path = f.name
                start = time.monotonic()
                proc = subprocess.run(
                    ["python3", path], capture_output=True, text=True, timeout=10.0,
                )
                elapsed = time.monotonic() - start
                os.unlink(path)
                passed = proc.returncode == 0 and "ALL_PASS" in proc.stdout
            except subprocess.TimeoutExpired:
                passed, elapsed = False, 10.0
                try: os.unlink(path)
                except: pass
            except:
                passed, elapsed = False, 0.0
                try: os.unlink(path)
                except: pass

            num_tokens = len(tokenizer.encode(code))
            problem_results.append({
                "passed": passed,
                "runtime": elapsed,
                "num_tokens": num_tokens,
            })

        results.append({
            "task_id": example["task_id"],
            "solutions": problem_results,
        })

        if (i + 1) % 20 == 0:
            pass_1 = np.mean([
                any(s["passed"] for s in r["solutions"])
                for r in results
            ])
            print(f"  [{i+1}/{len(ds)}] Running pass@1: {pass_1:.1%}")

    return _compute_metrics(results, "humaneval")


def _compute_metrics(results: list[dict], benchmark: str) -> dict:
    """Compute pass@k, efficiency, and brevity metrics."""
    # pass@1: fraction of problems where at least one sample passes
    pass_at_1_per_problem = []
    all_runtimes = []
    all_tokens = []
    all_passing_runtimes = []
    all_passing_tokens = []

    for r in results:
        solutions = r["solutions"]
        any_pass = any(s["passed"] for s in solutions)
        pass_at_1_per_problem.append(float(any_pass))

        for s in solutions:
            all_runtimes.append(s["runtime"])
            all_tokens.append(s["num_tokens"])
            if s["passed"]:
                all_passing_runtimes.append(s["runtime"])
                all_passing_tokens.append(s["num_tokens"])

    metrics = {
        f"{benchmark}_pass_at_1": np.mean(pass_at_1_per_problem),
        f"{benchmark}_num_problems": len(results),
        f"{benchmark}_avg_runtime": np.mean(all_runtimes) if all_runtimes else 0,
        f"{benchmark}_avg_tokens": np.mean(all_tokens) if all_tokens else 0,
    }

    if all_passing_runtimes:
        metrics[f"{benchmark}_passing_avg_runtime"] = np.mean(all_passing_runtimes)
        metrics[f"{benchmark}_passing_avg_tokens"] = np.mean(all_passing_tokens)
        metrics[f"{benchmark}_passing_median_runtime"] = np.median(all_passing_runtimes)

    # Compute efficiency and brevity scores (same formula as training)
    if all_passing_runtimes:
        eff_scores = [max(0, 1 - rt / 5.0) for rt in all_passing_runtimes]
        brev_scores = [max(0, 1 - nt / 512) for nt in all_passing_tokens]
        metrics[f"{benchmark}_avg_efficiency"] = np.mean(eff_scores)
        metrics[f"{benchmark}_avg_brevity"] = np.mean(brev_scores)
    else:
        metrics[f"{benchmark}_avg_efficiency"] = 0.0
        metrics[f"{benchmark}_avg_brevity"] = 0.0

    print(f"\n{benchmark.upper()} Results:")
    print(f"  Pass@1: {metrics[f'{benchmark}_pass_at_1']:.1%}")
    print(f"  Avg Runtime (passing): {metrics.get(f'{benchmark}_passing_avg_runtime', 0):.3f}s")
    print(f"  Avg Tokens (passing): {metrics.get(f'{benchmark}_passing_avg_tokens', 0):.0f}")
    print(f"  Efficiency Score: {metrics[f'{benchmark}_avg_efficiency']:.3f}")
    print(f"  Brevity Score: {metrics[f'{benchmark}_avg_brevity']:.3f}")

    return metrics


def main():
    args = parse_args()
    model, tokenizer = load_model(args)

    all_metrics = {}

    if args.eval_mbpp:
        mbpp_metrics = evaluate_mbpp(model, tokenizer, args)
        all_metrics.update(mbpp_metrics)

    if args.eval_humaneval:
        he_metrics = evaluate_humaneval(model, tokenizer, args)
        all_metrics.update(he_metrics)

    # Aggregate metrics for Pareto analysis
    pass_rates = [v for k, v in all_metrics.items() if "pass_at_1" in k]
    eff_scores = [v for k, v in all_metrics.items() if "avg_efficiency" in k]
    brev_scores = [v for k, v in all_metrics.items() if "avg_brevity" in k]
    runtime_vals = [v for k, v in all_metrics.items() if "passing_avg_runtime" in k]
    token_vals = [v for k, v in all_metrics.items() if "passing_avg_tokens" in k]

    all_metrics["pass_at_1"] = np.mean(pass_rates) if pass_rates else 0
    all_metrics["avg_efficiency"] = np.mean(eff_scores) if eff_scores else 0
    all_metrics["avg_brevity"] = np.mean(brev_scores) if brev_scores else 0
    all_metrics["avg_runtime"] = np.mean(runtime_vals) if runtime_vals else 0
    all_metrics["avg_tokens"] = np.mean(token_vals) if token_vals else 0

    # Save results
    output_dir = args.output_dir or args.model_dir or "./eval_results/base_model"
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "eval_metrics.json"), "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\n{'='*50}")
    print("FINAL AGGREGATED METRICS")
    print(f"  Pass@1:     {all_metrics['pass_at_1']:.1%}")
    print(f"  Efficiency: {all_metrics['avg_efficiency']:.3f}")
    print(f"  Brevity:    {all_metrics['avg_brevity']:.3f}")
    print(f"  Avg Runtime: {all_metrics['avg_runtime']:.3f}s")
    print(f"  Avg Tokens:  {all_metrics['avg_tokens']:.0f}")
    print(f"Results saved to {output_dir}/eval_metrics.json")


if __name__ == "__main__":
    main()
