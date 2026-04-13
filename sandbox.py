"""
Sandboxed code execution with timing measurement.
Executes generated code against test cases safely with resource limits.
"""
import subprocess
import tempfile
import time
import os
import json
import signal
from dataclasses import dataclass
from typing import Optional


@dataclass
class ExecutionResult:
    """Result of executing code against test cases."""
    passed: bool           # Did all tests pass?
    num_passed: int        # How many tests passed
    num_total: int         # Total number of tests
    runtime_seconds: float # Wall-clock time for execution
    error_type: Optional[str] = None  # "syntax", "runtime", "timeout", "wrong_answer"
    error_message: Optional[str] = None
    stdout: Optional[str] = None


def execute_code_with_tests(
    code: str,
    test_cases: list[dict],
    timeout: float = 10.0,
    memory_limit_mb: int = 256,
) -> ExecutionResult:
    """
    Execute generated code against test cases in a subprocess sandbox.

    Args:
        code: The generated Python code (function or full program)
        test_cases: List of {"input": str, "expected_output": str}
        timeout: Max execution time in seconds
        memory_limit_mb: Memory limit in MB

    Returns:
        ExecutionResult with pass/fail, timing, and error info
    """
    num_passed = 0
    num_total = len(test_cases)
    total_runtime = 0.0
    last_error_type = None
    last_error_msg = None

    for test in test_cases:
        result = _run_single_test(
            code=code,
            test_input=test.get("input", ""),
            expected_output=test.get("expected_output", "").strip(),
            timeout=timeout / max(num_total, 1),  # Divide timeout across tests
            memory_limit_mb=memory_limit_mb,
        )
        total_runtime += result["runtime"]

        if result["passed"]:
            num_passed += 1
        else:
            last_error_type = result.get("error_type")
            last_error_msg = result.get("error_message", "")

    return ExecutionResult(
        passed=(num_passed == num_total),
        num_passed=num_passed,
        num_total=num_total,
        runtime_seconds=total_runtime,
        error_type=last_error_type,
        error_message=last_error_msg,
    )


def _run_single_test(
    code: str,
    test_input: str,
    expected_output: str,
    timeout: float,
    memory_limit_mb: int,
) -> dict:
    """Run code with a single test case in an isolated subprocess."""

    # Build the execution script
    exec_script = f"""
import sys
import resource
import time

# Set memory limit
memory_bytes = {memory_limit_mb} * 1024 * 1024
resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

# Capture start time
_start = time.monotonic()

try:
{_indent(code, 4)}
except Exception as e:
    print(f"RUNTIME_ERROR: {{type(e).__name__}}: {{e}}", file=sys.stderr)
    sys.exit(1)
finally:
    _elapsed = time.monotonic() - _start
    print(f"__RUNTIME__:{{_elapsed:.6f}}", file=sys.stderr)
"""

    try:
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False, dir='/tmp'
        ) as f:
            f.write(exec_script)
            script_path = f.name

        start = time.monotonic()
        proc = subprocess.run(
            ["python3", script_path],
            input=test_input,
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        wall_time = time.monotonic() - start

        # Extract precise runtime from stderr
        runtime = wall_time
        stderr_lines = proc.stderr.strip().split('\n') if proc.stderr else []
        for line in stderr_lines:
            if line.startswith("__RUNTIME__:"):
                runtime = float(line.split(":")[1])
                break

        # Check for errors
        if proc.returncode != 0:
            error_lines = [l for l in stderr_lines if not l.startswith("__RUNTIME__")]
            error_msg = "\n".join(error_lines[-3:])  # Last 3 lines

            if "SyntaxError" in error_msg:
                error_type = "syntax"
            elif "MemoryError" in error_msg or "RLIMIT" in error_msg:
                error_type = "memory"
            else:
                error_type = "runtime"

            return {
                "passed": False,
                "runtime": runtime,
                "error_type": error_type,
                "error_message": error_msg[:500],
            }

        # Compare output
        actual = proc.stdout.strip()
        if actual == expected_output:
            return {"passed": True, "runtime": runtime}
        else:
            return {
                "passed": False,
                "runtime": runtime,
                "error_type": "wrong_answer",
                "error_message": f"Expected: {expected_output[:100]}, Got: {actual[:100]}",
            }

    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "runtime": timeout,
            "error_type": "timeout",
            "error_message": f"Exceeded {timeout:.1f}s time limit",
        }
    except Exception as e:
        return {
            "passed": False,
            "runtime": 0.0,
            "error_type": "sandbox_error",
            "error_message": str(e)[:500],
        }
    finally:
        try:
            os.unlink(script_path)
        except:
            pass


def _indent(code: str, spaces: int) -> str:
    """Indent all lines of code by given number of spaces."""
    prefix = " " * spaces
    return "\n".join(prefix + line for line in code.split("\n"))


# ===== Batch execution for GRPO rollouts =====

def batch_execute(
    codes: list[str],
    test_cases_per_problem: list[list[dict]],
    timeout: float = 10.0,
    memory_limit_mb: int = 256,
) -> list[ExecutionResult]:
    """Execute a batch of generated codes against their respective test cases."""
    results = []
    for code, tests in zip(codes, test_cases_per_problem):
        result = execute_code_with_tests(
            code=code,
            test_cases=tests,
            timeout=timeout,
            memory_limit_mb=memory_limit_mb,
        )
        results.append(result)
    return results


if __name__ == "__main__":
    # Quick sanity check
    code = """
def add(a, b):
    return a + b

import sys
data = sys.stdin.read().split()
print(add(int(data[0]), int(data[1])))
"""
    tests = [
        {"input": "2 3", "expected_output": "5"},
        {"input": "0 0", "expected_output": "0"},
        {"input": "-1 1", "expected_output": "0"},
    ]
    result = execute_code_with_tests(code, tests)
    print(f"Passed: {result.passed} ({result.num_passed}/{result.num_total})")
    print(f"Runtime: {result.runtime_seconds:.4f}s")
    print(f"Error: {result.error_type}: {result.error_message}")
