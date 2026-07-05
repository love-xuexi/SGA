#!/bin/bash

# Usage: time bash ../scripts/train_sga_helpsteer2_all_models.sh llama1b
# This script trains SGA (SGA) on HelpSteer2 dataset with multiple models

# Default: train all models
TARGET_MODEL="${1:-all}"

# Model configuration
# Format: name|devices|n_gpu|main_process_port|base_model|base_output_dir|log_dir|batch_size|grad_accum|gradient_checkpointing
declare -A MODEL_CONFIGS=(
    [llama1b]="llama1b|2,3,4,5,6,7|6|9956|path/to/models/Llama-3.2-1B-Instruct|path/to/SGA/outputs/helpsteer2/llama-1b|path/to/SGA/logs/helpsteer2/llama-1b|2|8|False"
    [llama8b]="llama8b|2,3,4,5,6,7|6|9928|path/to/models/Llama-3.1-8B-Instruct|path/to/SGA/outputs/helpsteer2/llama-8b|path/to/SGA/logs/helpsteer2/llama-8b|1|16|True"
    [qwen7b]="qwen7b|2,3,4,5,6,7|6|9988|path/to/models/qwen-2.5-7b-instruct|path/to/SGA/outputs/helpsteer2/qwen-7b|path/to/SGA/logs/helpsteer2/qwen-7b|1|16|True"
    [llama3b]="llama3b|2,3,4,5,6,7|6|4127|path/to/models/Llama-3.2-3B-Instruct|path/to/SGA/outputs/helpsteer2/llama-3b|path/to/SGA/logs/helpsteer2/llama-3b|1|16|False"
    [qwen8b]="qwen8b|2,3,4,5,6,7|6|7218|path/to/models/Qwen3-8B|path/to/SGA/outputs/helpsteer2/qwen-8b|path/to/SGA/logs/helpsteer2/qwen-8b|1|16|True"
    [qwen4b]="qwen4b|2,3,4,5,6,7|6|9688|path/to/models/Qwen3-4B|path/to/SGA/outputs/helpsteer2/qwen-4b|path/to/SGA/logs/helpsteer2/qwen-4b|1|16|True"
)

# Dataset configuration
DATASET="path/to/exp/datasets/HelpSteer2"
report_to="wandb"
max_length=1024
num_train_epochs=2
learning_rate=1e-5
wandb_name_prefix="sga_helpsteer2"

# Training modes to test
declare -a training_modes=("dpo_mse")

# Weight combinations for different modes
# For dpo_bt mode: weight_ratio
declare -a dpo_bt_weight_ratios=(
    "0.01"
    "0.1"
)

# For dpo_mse mode: dpo_weight mse_weight
declare -a dpo_mse_weight_combinations=(
    # "1 1"   # dpo_weight=0.1, mse_weight=0.9
    "1 10"
    # "1 100"   # dpo_weight=1, mse_weight=1
    "1 1000"   # dpo_weight=1, mse_weight=1000
)

train_model() {
    local model_name=$1
    local config="${MODEL_CONFIGS[$model_name]}"

    if [ -z "$config" ]; then
        echo "✗ Unknown model: $model_name"
        return 1
    fi

    # Parse model configuration
    IFS='|' read -r name devices n_gpu main_process_port base_model base_output_dir log_dir batch_size grad_accum gradient_checkpointing <<< "$config"

    # Ensure directories exist
    mkdir -p "${base_output_dir}"
    mkdir -p "${log_dir}"

    # Loop through training modes
    for mode in "${training_modes[@]}"
    do
        if [ "$mode" == "dpo_bt" ]; then
            # DPO+BT mode: loop through weight_ratio values
            for weight_ratio in "${dpo_bt_weight_ratios[@]}"
            do
                # Build a unique experiment identifier
                exp_id="${mode}_wr${weight_ratio}"

                # Set a dedicated output directory for this experiment
                CURRENT_OUTPUT_DIR="${base_output_dir}/${exp_id}"
                echo "Starting HelpSteer2 SGA Training Task: [$model_name]"
                echo "  Training Mode:     ${mode}"
                echo "  Weight Ratio:      ${weight_ratio}"
                echo "  Batch Size:        ${batch_size}  Gradient Accumulation Steps: ${grad_accum}"
                echo "  Gradient Checkpointing: ${gradient_checkpointing}"
                echo "  Output Dir:        ${CURRENT_OUTPUT_DIR}"
                CUDA_VISIBLE_DEVICES=${devices} accelerate launch \
                    --num_processes ${n_gpu} \
                    --main_process_port ${main_process_port} \
                    run_sga_reward_train.py \
    

                # Check training result
                if [ $? -eq 0 ]; then
                    echo "✓ Training [${model_name}/${exp_id}] completed successfully"
                else
                    echo "✗ Training [${model_name}/${exp_id}] failed! Check log: ${log_dir}/${model_name}_${exp_id}.log"
                    return 1
                fi

                # Cooldown time for GPU memory
                echo "Waiting 30 seconds for GPU cooldown..."
                sleep 30
            done

        elif [ "$mode" == "dpo_mse" ]; then
            # DPO+MSE mode: loop through weight combinations
            for weight_pair in "${dpo_mse_weight_combinations[@]}"
            do
                set -- $weight_pair
                dpo_w=$1
                mse_w=$2

                # Build a unique experiment identifier
                exp_id="${mode}_dpo${dpo_w}_mse${mse_w}"

                # Set a dedicated output directory for this experiment
                CURRENT_OUTPUT_DIR="${base_output_dir}/${exp_id}"
                echo "Starting HelpSteer2 SGA Training Task: [$model_name]"
                echo "  Training Mode:     ${mode}"
                echo "  DPO Weight:        ${dpo_w}   MSE Weight: ${mse_w}"
                echo "  Batch Size:        ${batch_size}  Gradient Accumulation Steps: ${grad_accum}"
                echo "  Gradient Checkpointing: ${gradient_checkpointing}"
                echo "  Output Dir:        ${CURRENT_OUTPUT_DIR}"
                CUDA_VISIBLE_DEVICES=${devices} accelerate launch \
                    --num_processes ${n_gpu} \
                    --main_process_port ${main_process_port} \
                    run_sga_reward_train.py \
                    --dataset ${DATASET} \
                    --base_model ${base_model} \
                    --training_mode ${mode} \
                    --dpo_weight ${dpo_w} \
                    --mse_weight ${mse_w} \
                    --per_device_train_batch_size ${batch_size} \
                    --gradient_accumulation_steps ${grad_accum} \
                    --gradient_checkpointing ${gradient_checkpointing} \
                    --num_train_epochs ${num_train_epochs} \
                    --learning_rate ${learning_rate} \
                    --max_length ${max_length} \
                    --output_dir ${CURRENT_OUTPUT_DIR} \
                    --wandb_name "${wandb_name_prefix}_${model_name}_${exp_id}" \
                    --report_to ${report_to} \
                    --save_strategy "epoch" \
                    --save_steps 500 \
                    --eval_steps 100 \
                    --use_lora \
                    --lora_r 32 \
                    --lora_alpha 64 \
                    --lora_dropout 0.05 \
                    --beta 0.1 \
                    > ${log_dir}/${model_name}_${exp_id}.log 2>&1

                # Check training result
                if [ $? -eq 0 ]; then
                    echo "✓ Training [${model_name}/${exp_id}] completed successfully"
                else
                    echo "✗ Training [${model_name}/${exp_id}] failed! Check log: ${log_dir}/${model_name}_${exp_id}.log"
                    return 1
                fi

                # Cooldown time for GPU memory
                echo "Waiting 30 seconds for GPU cooldown..."
                sleep 10
            done
        fi
    done

    return 0
}

main() {
    if [ "$TARGET_MODEL" == "all" ]; then
        # Train all models (from smaller to larger)
        for model in llama1b llama8b qwen7b llama3b qwen8b qwen4b; do
            train_model "$model"
            if [ $? -ne 0 ]; then
                echo "✗ Model $model training failed, aborting."
                exit 1
            fi
        done

        echo ""
        echo "✓ All HelpSteer2 SGA models have been trained successfully!"
    else
        # Train the specified model
        train_model "$TARGET_MODEL"
        if [ $? -ne 0 ]; then
            echo "✗ Model $TARGET_MODEL training failed."
            exit 1
        fi

        echo ""
        echo "✓ Model $TARGET_MODEL has been trained successfully!"
    fi
}

# Show usage
if [ "$TARGET_MODEL" == "-h" ] || [ "$TARGET_MODEL" == "--help" ]; then
    echo "Usage: bash train_sga_helpsteer2_all_models.sh [model_name]"
    echo ""
    echo "Supported models:"
    echo "  llama1b   - Llama 3.2 1B Instruct"
    echo "  llama8b   - Llama 3.1 8B Instruct"
    echo "  qwen7b    - Qwen 2.5 7B Instruct"
    echo "  llama3b   - Llama 3.2 3B Instruct"
    echo "  qwen8b    - Qwen 3 8B"
    echo "  qwen4b    - Qwen 3 4B"
    echo "  all       - Train all models (default)"
    echo ""
    echo "Training Modes:"
    echo "  dpo_bt    - DPO + Bradley-Terry loss (original SGA mode)"
    echo "  dpo_mse   - DPO + MSE loss (new regression mode)"
    echo ""
    echo "Examples:"
    echo "  bash train_sga_helpsteer2_all_models.sh llama1b      # Train only Llama 1B"
    echo "  bash train_sga_helpsteer2_all_models.sh all          # Train all models"
    echo "  bash train_sga_helpsteer2_all_models.sh              # Train all models (default)"
    exit 0
fi

# Execute main program
main
