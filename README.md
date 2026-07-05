# SGA: Self-Guided Alignment

Official implementation for the paper **"Self-Guided Alignment: Adaptive Preference Sensing for Multi-Objective Generation"**.

## Overview

Existing prompt-conditioned alignment approaches face a critical **training-inference discrepancy**: they rely on ground-truth preference scores during training while requiring manual user-specification at inference. This gap imposes a cognitive burden on users and leads to suboptimal alignment.

**Self-Guided Alignment (SGA)** bridges this gap by transforming passive reward dependency into an intrinsic adaptive sensing capability. SGA employs a **dual-head architecture** on a shared backbone to unify preference internalization with conditional generation:

- **Regression Head** — Learns to predict latent preference scores from prompts alone, internalizing the preference space via MSE loss.
- **LM Head** — Learns preference-conditioned generation via DPO or SFT objectives.

At inference, SGA autonomously predicts the preference profile from the raw prompt and uses it to self-guide generation, eliminating the need for manual score specification.

### Key Results

Extensive experiments across six models (Llama-3.2-1B, Llama-3.2-3B, Llama-3.1-8B, Qwen3-4B, Qwen2.5-7B, Qwen3-8B) on three datasets demonstrate that SGA achieves an **average 9% improvement** over all baselines in multi-objective win rates.

## Project Structure

```
SGA/
├── sga/                            # Core source code
│   ├── run_sga_train.py            # Training entry point
│   ├── sga_trainer.py              # SGATrainer with DPO+BT and DPO+MSE modes
│   ├── sga_utils.py                # Model architecture (ValueHead, AutoModelForCausalLMWithValueHead)
│   ├── base_trainer.py             # Base reward trainer
│   ├── reward_trainer.py           # Reward training utilities
│   ├── load_datasets.py            # Dataset loaders (HelpSteer2, UltraFeedback, HH-RLHF)
│   ├── utils.py                    # General utilities and metrics
│   ├── infer/                      # Inference pipeline
│   │   ├── infer_preference.py     # Stage-1: Preference score prediction
│   │   ├── vllm_sga.py             # Stage-2: vLLM score-guided generation
│   │   ├── batch_inference_preference_guided.py  # Batch inference pipeline
│   │   ├── preference_injection.py # Score injection into prompts
│   │   ├── normalization.py        # Score normalization strategies
│   │   ├── load_datasets.py        # Inference dataset loaders
│   │   └── sga_utils.py            # Inference model utilities
│   └── eval/                       # Evaluation scripts
│       ├── score.py                # Reward model scoring
│       ├── armorm_score.py         # ArmoRM scoring
│       ├── compare_scores3.py      # Win/Loss/Tie rate calculation
│       ├── multi_reward_models.py  # Multi-reward model ensemble
│       ├── test3.py                # Evaluation test scripts
│       ├── utils.py                # Evaluation utilities
│       ├── score.sh                # Batch scoring script
│       └── compare.sh              # Batch comparison script
├── scripts/                        # Shell scripts for batch experiments
│   ├── train_sga_helpsteer2_all_models.sh
│   ├── train_sga_hh_rlhf_all_models.sh
│   ├── train_sga_ultrafeedback_all_models.sh
│   ├── infer-helpsteer2-all.sh
│   ├── infer-hhf_allmodels.sh
│   └── infer-ultrafeedback-all.sh
├── requirements.txt
├── environment.yml
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+
- Linux (recommended)

### Option 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate sga
pip install -r requirements.txt
```

### Option 2: pip

```bash
conda create -n sga python=3.10 -y
conda activate sga
pip install -r requirements.txt
```

### Key Dependencies

| Package        | Version  |
|----------------|----------|
| PyTorch        | 2.2.0    |
| Transformers   | 4.51.0   |
| PEFT           | 0.10.0   |
| TRL            | 0.8.0    |
| Accelerate     | 0.34.0   |
| DeepSpeed      | 0.11.2   |
| Flash Attention| 2.6.3    |
| vLLM           | (required for Stage-2 inference) |

## Usage

### Training

- **`dpo_mse`** — DPO loss + MSE regression loss (primary mode used in the paper)

#### Single Model Training

```bash
cd sga

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --num_processes 4 \
    --main_process_port 9956 \
    run_sga_train.py \
    --dataset <path_to_dataset> \
    --base_model <path_to_base_model> \
    --training_mode dpo_mse \
    --dpo_weight 1 \
    --mse_weight 1000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --num_train_epochs 2 \
    --learning_rate 1e-5 \
    --max_length 1024 \
    --output_dir ./outputs \
    --use_lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_dropout 0.05 \
    --beta 0.1 \
    --report_to wandb
```

#### Batch Training (All Models on a Dataset)

```bash
# HelpSteer2
bash scripts/train_sga_helpsteer2_all_models.sh llama1b

# HH-RLHF
bash scripts/train_sga_hh_rlhf_all_models.sh llama1b

# UltraFeedback
bash scripts/train_sga_ultrafeedback_all_models.sh llama1b
```

Supported model keys: `llama1b`, `llama3b`, `llama8b`, `qwen4b`, `qwen7b`, `qwen8b`.

> **Note:** Before running batch scripts, update the paths in the script files (`MODEL_CONFIGS`, `DATASET`, etc.) to match your local environment.

### Inference

#### Batch Inference (All Models)

```bash
# HelpSteer2
bash scripts/infer-helpsteer2-all.sh llama1b

# HH-RLHF
bash scripts/infer-hhf_allmodels.sh llama1b

# UltraFeedback
bash scripts/infer-ultrafeedback-all.sh llama1b
```

> **Note:** Update model paths and checkpoint paths in the inference scripts before running.

### Evaluation

#### Step 1: Score Responses with Reward Models

```bash
cd sga/eval

python score.py \
    --csv_path <path_to_inference_results.csv> \
    --gpu_id 0 \
    --batch_size 16 \
    --prompt_column prompt \
    --response_column generated_response \
    --reward_names harmless,helpful \
    --clean_prompt
```

#### Step 2: Calculate Win/Loss/Tie Rates

```bash
python compare_scores3.py \
    --model_file <path_to_scored_results.csv> \
    --gt_file <path_to_ground_truth.csv> \
    --score_columns harmless,helpful \
    --output_csv results.csv
```

## Datasets

SGA is evaluated on three multi-objective alignment datasets:

| Dataset | Objectives | Dimensions |
|---------|-----------|------------|
| [HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) | General utility & safety | Helpful, Harmless |
| [UltraFeedback](https://huggingface.co/datasets/openbmb/UltraFeedback) | Core quality axes | Instruction-Following, Truthfulness, Honesty, Helpfulness |
| [HelpSteer2](https://huggingface.co/datasets/nvidia/HelpSteer2) | Steerability | Helpfulness, Correctness, Coherence, Complexity, Verbosity |

## Supported Models

| Model Family | Models |
|-------------|--------|
| Llama | Llama-3.2-1B-Instruct, Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct |
| Qwen | Qwen3-4B, Qwen2.5-7B-Instruct, Qwen3-8B |

and so on.
## Acknowledgements

This codebase is built upon [Generalizable-Reward-Model (GRM)](https://github.com/YangRui2015/Generalizable-Reward-Model). We thank the authors for their excellent open-source work, which provided the foundational architecture and training pipeline that made this project possible.

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{wang-etal-2026-self-guided,
    title = "Self-Guided Alignment: Adaptive Preference {S}ensing for Multi-Objective Generation",
    author = "Wang, Ning  and
      Liu, Zhanyang  and
      Zhou, Taotao  and
      Zhang, Xinrui  and
      Shao, Zongru  and
      Zhou, Haojie",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.2184/",
    doi = "10.18653/v1/2026.acl-long.2184",
    pages = "47202--47220",
    ISBN = "979-8-89176-390-6",
    abstract = "Aligning Large Language Models (LLMs) with diverse and potentially conflicting human values necessitates navigating complex multi-objective landscapes. However, existing prompt-conditioned approaches face a critical training-inference discrepancy: they rely on ground-truth scores during training while requiring manual user-specification at inference. We introduce prediction of implicit preferences to bridge this gap while reducing user burden. To this end, we propose Self-Guided Alignment (SGA), a framework that transforms passive reward dependency into an intrinsic adaptive sensing capability. It employs a dual-head architecture to unify preference internalization with conditional generation, enabling the model to learn a latent mapping between raw prompts and preference profiles. Through adaptive preference sensing, the model autonomously predicts the latent preference score to self-guide the generation, thereby eliminating the need for manual specification at inference. Extensive experiments across diverse model scales demonstrate that SGA often outperforms state-of-the-art baselines, achieving superior multi-objective trade-offs and improved preference alignment. Code is available at https://github.com/python-yyds/SGA."
}
```
