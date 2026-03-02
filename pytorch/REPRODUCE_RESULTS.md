# Reproducing Vision Transformer (ViT) Results

This guide provides step-by-step instructions to reproduce the Vision Transformer experiments for object classification and pose estimation using RGB-D inputs.

## 📋 Prerequisites

- Conda/Miniconda installed
- Access to the YCB dataset (RGB-D images from Habitat simulator)
- WandB account for experiment tracking (optional but recommended)

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)

**One-command setup and verification:**
```bash
# Clone the repository (if not already done)
cd tbp.tbs_sensorimotor_intelligence/pytorch

# Run automated setup and verification (5-15 minutes)
./scripts/setup_and_test.sh
```

This will automatically:
- ✅ Check conda installation
- ✅ Create/activate `vit` environment  
- ✅ Verify all dependencies
- ✅ Run end-to-end tests
- ✅ Show next steps for running experiments

**Run ALL experiments:**
```bash
# Complete reproduction suite with limited continual learning
./scripts/run_all_experiments.sh
```

### Option 2: Manual Setup

```bash
# Clone the repository (if not already done)
cd tbp.tbs_sensorimotor_intelligence/pytorch

# Create and activate conda environment
conda env create -f environment.yaml
conda activate vit

# Set PROJECT_ROOT environment variable (critical for proper paths)
export PROJECT_ROOT=$(pwd)

# Verify setup
python scripts/verify_setup.py

# Run quick test (optional but recommended)
./scripts/quick_test.sh
```

**Important**: Always run experiments from the `tbp.tbs_sensorimotor_intelligence/pytorch` directory where the `.project-root` file is located.

### 2. Data Setup

The experiments expect data in the following structure:
```
<repo_root>/.cache/dmc/view_finder_images/
├── view_finder_32/
│   └── view_finder_rgbd/
├── view_finder_base/
│   └── view_finder_rgbd/
└── view_finder_randrot/
    └── view_finder_rgbd/
```

## 🧪 Running Experiments


### Batch Experiments

We provide comprehensive scripts for reproducing all results:

#### All Experiments (Complete Reproduction)
```bash
# Run ALL experiments with setup verification 
./scripts/run_all_experiments.sh
```

This comprehensive script runs:
- **Figure 7a**: Rapid learning experiments (all rotation counts)
- **Figure 7b**: Continual learning (first 5 tasks only for demo)
- **Figure 8b**: FLOP analysis (all ViT architectures)

#### Individual Experiment Suites

We also provide individual experiment scripts corresponding to different research figures:

#### Figure 7a: Rapid Learning Experiments

```bash
chmod +x scripts/fig7a_rapid_learning.sh
./scripts/fig7a_rapid_learning.sh
```

This runs:
- **Pretrained models**: 25 epochs with 1, 2, 4, 8, 16, 32 rotations
- **Random init models**: 75 epochs and 1 epoch with same rotation counts
- **Evaluation**: All trained models on test set

#### Figure 7b: Continual Learning Experiments
Tests continual learning across sequential tasks:

```bash
chmod +x scripts/fig7b_continual_learning.sh
./scripts/fig7b_continual_learning.sh
```

#### Figure 8b: FLOP Analysis Experiments
Compares different ViT architectures (B/16, B/32, L/16, L/32, H/14):

```bash
chmod +x scripts/fig8b_flops.sh
./scripts/fig8b_flops.sh
```

This runs:
- **Training**: All 5 ViT variants with both pretrained and random initialization
- **Evaluation**: All trained models with FLOP counting

## 📊 Analyzing Results

### WandB Dashboard

1. **Access your project**: Go to https://wandb.ai and navigate to project `benchmark_vit`

2. **Compare runs**: 
   - Use the "Table" view for side-by-side comparison
   - Sort by "Name" to group related experiments
   - Filter by tags (e.g., `fig7a`, `fig8b`) to focus on specific experiment types

### Local Results Analysis

Training and evaluation outputs are saved to:
```
<repo_root>/.cache/dmc/results/vit/logs/<experiment_name>/
├── checkpoints/          # Model checkpoints
├── inference/            # Evaluation results
└── <experiment_name>.log # Training logs
```
