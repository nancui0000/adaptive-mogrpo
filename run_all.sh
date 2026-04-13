#!/bin/bash
# =============================================================
# MO-GRPO: Full Experiment Pipeline
# Run this on your rented A100 machine
# =============================================================

set -e

echo "============================================"
echo "MO-GRPO: Multi-Objective GRPO for Code Gen"
echo "============================================"

# ===== Step 0: Environment Setup =====
echo ""
echo "[Step 0] Setting up environment..."
pip install -r requirements.txt 2>&1 | tail -5

# Verify GPU
python3 -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB')"

# ===== Step 1: Evaluate Base Model (Day 1) =====
echo ""
echo "[Step 1] Evaluating base model (zero-shot baseline)..."
python3 evaluate.py \
    --base_model \
    --num_samples 3 \
    --output_dir ./outputs/base_model

# ===== Step 2: Baseline - Correctness Only (Day 2) =====
echo ""
echo "[Step 2] Training baseline: correctness only..."
python3 train.py \
    --preset correctness_only \
    --run_name baseline \
    --max_steps 800 \
    --dataset mbpp

echo "[Step 2] Evaluating baseline..."
python3 evaluate.py \
    --model_dir ./outputs/baseline_a1.0_b0.0_g0.0 \
    --num_samples 3

# ===== Step 3: Multi-Objective Experiments (Day 3-4) =====
echo ""
echo "[Step 3] Training MO-GRPO: correctness-heavy..."
python3 train.py \
    --preset correctness_heavy \
    --run_name corr_heavy \
    --max_steps 800 \
    --dataset mbpp

echo "[Step 3] Training MO-GRPO: balanced..."
python3 train.py \
    --preset balanced \
    --run_name balanced \
    --max_steps 800 \
    --dataset mbpp

echo "[Step 3] Training MO-GRPO: efficiency-heavy..."
python3 train.py \
    --preset efficiency_heavy \
    --run_name eff_heavy \
    --max_steps 800 \
    --dataset mbpp

# ===== Step 4: Evaluate All Models (Day 5) =====
echo ""
echo "[Step 4] Evaluating all trained models..."

for dir in ./outputs/*/; do
    if [ -d "$dir" ] && [ "$dir" != "./outputs/base_model/" ]; then
        echo "  Evaluating: $dir"
        python3 evaluate.py \
            --model_dir "$dir" \
            --num_samples 3
    fi
done

# ===== Step 5: Generate Pareto Analysis (Day 6) =====
echo ""
echo "[Step 5] Generating Pareto analysis and plots..."
python3 pareto.py --results_dir ./outputs --output_dir ./figures

echo ""
echo "============================================"
echo "DONE! Check ./figures/ for Pareto plots"
echo "and ./outputs/*/eval_metrics.json for results"
echo "============================================"
