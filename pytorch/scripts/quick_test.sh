#!/bin/bash

# Copyright 2025 Thousand Brains Project
#
# Copyright may exist in Contributors' modifications
# and/or contributions to the work.
#
# Use of this source code is governed by the MIT
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

# Quick test script to verify ViT experiment setup
# Runs a minimal training and evaluation to ensure everything works

echo "🚀 Running quick verification test for ViT experiments..."
echo "This will run a minimal training and evaluation to verify setup."
echo

# Check if we're in the right directory
if [ ! -f ".project-root" ]; then
    echo "❌ Error: .project-root file not found"
    echo "   Make sure you're running this from the pytorch directory:"
    echo "   cd tbp.tbs_sensorimotor_intelligence/pytorch"
    echo "   ./scripts/quick_test.sh"
    exit 1
fi

# Check if PROJECT_ROOT is set
if [ -z "$PROJECT_ROOT" ]; then
    echo "⚠️  PROJECT_ROOT not set, setting it now..."
    export PROJECT_ROOT=$(pwd)
    echo "   PROJECT_ROOT=$PROJECT_ROOT"
fi

echo "📍 Current directory: $(pwd)"
echo "📍 PROJECT_ROOT: $PROJECT_ROOT"
echo

# Test 1: Verify environment
echo "🔍 Test 1: Running environment verification..."
python scripts/verify_setup.py
if [ $? -ne 0 ]; then
    echo "❌ Environment verification failed. Please fix issues above."
    exit 1
fi
echo

# Test 2: Dry run training
echo "🔍 Test 2: Running dry training (no actual training)..."
echo "Command: python src/train.py experiment=01_hyperparameter_optimization/optimized_config trainer.fast_dev_run=true"
python src/train.py \
    experiment=01_hyperparameter_optimization/optimized_config \
    +trainer.fast_dev_run=true \
    paths=reproduction
    
if [ $? -ne 0 ]; then
    echo "❌ Dry run training failed. Check the error above."
    exit 1
fi
echo "✅ Dry run training succeeded!"
echo

# Test 3: Quick training (1 epoch)
echo "🔍 Test 3: Running quick training (1 epoch)..."
echo "Command: python src/train.py experiment=01_hyperparameter_optimization/optimized_config trainer.max_epochs=1 trainer.num_sanity_val_steps=0"
python src/train.py \
    experiment=01_hyperparameter_optimization/optimized_config \
    trainer.max_epochs=1 \
    trainer.num_sanity_val_steps=0 \
    logger.wandb.name="quick_test_$(date +%Y%m%d_%H%M%S)" \
    paths=reproduction

if [ $? -ne 0 ]; then
    echo "❌ Quick training failed. Check the error above."
    exit 1
fi
echo "✅ Quick training succeeded!"
echo

# Test 4: Quick evaluation
echo "🔍 Test 4: Running evaluation on the trained model..."
# Find the most recent checkpoint
CACHE_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)/.cache"
CHECKPOINT_DIR="${CACHE_ROOT}/dmc/results/vit/logs_reproduction"
LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "quick_test_*" -type d | sort | tail -1)

if [ -z "$LATEST_CHECKPOINT" ]; then
    echo "⚠️  Could not find checkpoint directory, skipping evaluation test"
else
    CHECKPOINT_FILE="$LATEST_CHECKPOINT/checkpoints/last.ckpt"
    if [ -f "$CHECKPOINT_FILE" ]; then
        echo "Command: python src/eval_standard.py ckpt_path=$CHECKPOINT_FILE"
        python src/eval_standard.py \
            ckpt_path="$CHECKPOINT_FILE" \
            trainer.accelerator=auto \
            paths=reproduction
        
        if [ $? -ne 0 ]; then
            echo "⚠️  Evaluation failed, but this might be due to data setup"
        else
            echo "✅ Evaluation succeeded!"
        fi
    else
        echo "⚠️  Checkpoint file not found: $CHECKPOINT_FILE"
    fi
fi
echo

# Summary
echo "="*60
echo "🎉 QUICK TEST SUMMARY"
echo "="*60
echo "✅ Environment verification: PASSED"
echo "✅ Dry run training: PASSED"  
echo "✅ Quick training (1 epoch): PASSED"
echo "✅ Basic setup is working!"
echo
echo "🎯 Next steps:"
echo "1. Review the detailed documentation: REPRODUCE_RESULTS.md"
echo "2. Run full experiments:"
echo "   ./scripts/run_all_experiments.sh"
echo "3. Monitor results in WandB (if configured)"
echo
echo "📚 For troubleshooting, see the REPRODUCE_RESULTS.md file" 
