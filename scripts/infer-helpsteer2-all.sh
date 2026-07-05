#!/bin/bash

# Usage example:
#   time bash ../scripts/infer-helpsteer2-all.sh llama1b      # Only run llama1b
#   bash scripts/infer-helpsteer2-all.sh             # Run all model types sequentially

MODEL_TYPE="$1"

# If MODEL_TYPE is not provided, run all supported model types sequentially
if [ -z "${MODEL_TYPE}" ]; then
    echo "No MODEL_TYPE provided, will run all: llama7b -> llama1b -> llama8b -> qwen7b"
    for mt in llama1b llama3b llama8b qwen4b qwen7b qwen8b; do
        echo "[MASTER] Starting MODEL_TYPE: ${mt}"
        bash "$0" "${mt}"
    done
    echo "All MODEL_TYPE finished."
    exit 0
fi

case "${MODEL_TYPE}" in
    llama3b)
        LOG_DIR="./inference_results/GAM/helpsteer2/llama3.2-3b-helpsteer2/logs"
        LOG_FILE_NAME="llama3_2_3b_helpsteer2.log"
        BASE_MODEL="path/to/models/Llama-3.2-3B-Instruct"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/llama3.2-3b-helpsteer2"
        DEVICE_ID=0
        MODEL_PATHS=(
            "path/to/SGA/outputs/helpsteer2/llama-3b/dpo_mse_dpo1_mse1000/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_20_56/checkpoint-90"
            "path/to/SGA/outputs/helpsteer2/llama-3b/dpo_mse_dpo1_mse1000/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_20_56/checkpoint-178"
        )
        ;;
    qwen4b)
        LOG_DIR="./inference_results/GAM/helpsteer2/qwen3-4b-helpsteer2/logs"
        LOG_FILE_NAME="qwen3_4b_helpsteer2.log"
        BASE_MODEL="path/to/models/Qwen3-4B"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/qwen3-4b-helpsteer2"
        DEVICE_ID=1
        MODEL_PATHS=(
            "path/to/SGA/outputs/helpsteer2/qwen-4b/dpo_mse_dpo1_mse10/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_20_27/checkpoint-90"
            "path/to/SGA/outputs/helpsteer2/qwen-4b/dpo_mse_dpo1_mse10/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_20_27/checkpoint-178"
            "path/to/SGA/outputs/helpsteer2/qwen-4b/dpo_mse_dpo1_mse1000/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_21_22/checkpoint-90"
            "path/to/SGA/outputs/helpsteer2/qwen-4b/dpo_mse_dpo1_mse1000/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_21_22/checkpoint-178"
        )
        ;;
    qwen8b)
        LOG_DIR="./inference_results/GAM/helpsteer2/qwen3-8b-helpsteer2/logs"
        LOG_FILE_NAME="qwen3_8b_helpsteer2.log"
        BASE_MODEL="path/to/models/Qwen3-8B"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/qwen3-8b-helpsteer2"
        DEVICE_ID=0
        MODEL_PATHS=(
            "path/to/SGA/outputs/helpsteer2/qwen-8b/dpo_mse_dpo1_mse10/Qwen3-8B_dpo_len1024_lora32_1e-05_02_13_07_00/checkpoint-90"
            "path/to/SGA/outputs/helpsteer2/qwen-8b/dpo_mse_dpo1_mse10/Qwen3-8B_dpo_len1024_lora32_1e-05_02_13_07_00/checkpoint-178"
            "path/to/SGA/outputs/helpsteer2/qwen-8b/dpo_mse_dpo1_mse1000/Qwen3-8B_dpo_len1024_lora32_1e-05_02_13_08_07/checkpoint-90"
            "path/to/SGA/outputs/helpsteer2/qwen-8b/dpo_mse_dpo1_mse1000/Qwen3-8B_dpo_len1024_lora32_1e-05_02_13_08_07/checkpoint-178"
        )
        ;;
    llama1b)
        LOG_DIR="./inference_results/GAM/helpsteer2/llama3.2-1b-helpsteer2/logs"
        LOG_FILE_NAME="llama3_2_1b_helpsteer2.log"
        BASE_MODEL="path/to/models/Llama-3.2-1B-Instruct"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/llama3.2-1b-helpsteer2"
        DEVICE_ID=0
        MODEL_PATHS=(
            "path/to/SGA/outputs/helpsteer2/llama-1b/dpo_mse_dpo1_mse1000/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_11_18_44/checkpoint-90"
            "path/to/SGA/outputs/helpsteer2/llama-1b/dpo_mse_dpo1_mse1000/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_11_18_44/checkpoint-178"

        )
        ;;
    llama7b)
        LOG_DIR="./inference_results/GAM/helpsteer2/llama2-7b-helpsteer2/logs"
        LOG_FILE_NAME="llama2_7b_helpsteer2.log"
        BASE_MODEL="path/to/models/Llama-2-7b-hf"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/llama2-7b-helpsteer2"
        DEVICE_ID=0
        MODEL_PATHS=(
            # "path/to/exp/NO.1/GAM/llama-7b-helpsteer2/plain_L1.0_P1000.0_debug/Llama-2-7b-hf_parallel_debug_helpsteer2_llama7b_plain_L1.0_P1000.0_debug_len1500_lora32_1e-05_pref5_12_11_21_21/checkpoint-284"
        )
        ;;
    llama8b)
        LOG_DIR="./inference_results/GAM/helpsteer2/llama3.1-8b-helpsteer2/logs"
        LOG_FILE_NAME="llama3_1_8b_helpsteer2.log"
        BASE_MODEL="path/to/models/Llama-3.1-8B-Instruct"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/llama3.1-8b-helpsteer2"
        DEVICE_ID=0
        MODEL_PATHS=(
            "path/to/SGA/outputs/helpsteer2/llama-8b/dpo_mse_dpo1_mse1000/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_11_18_57/checkpoint-90"
            "path/to/SGA/outputs/helpsteer2/llama-8b/dpo_mse_dpo1_mse1000/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_11_18_57/checkpoint-178"
        )
        ;;
    qwen7b)
        LOG_DIR="./inference_results/GAM/helpsteer2/qwen2.5-7b-helpsteer2/logs"
        LOG_FILE_NAME="qwen2.5_7b_helpsteer2.log"
        BASE_MODEL="path/to/models/qwen-2.5-7b-instruct"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/qwen2.5-7b-helpsteer2"
        DEVICE_ID=0
        MODEL_PATHS=(
            "path/to/SGA/outputs/helpsteer2/qwen-7b/dpo_mse_dpo1_mse1000/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_11_19_58/checkpoint-90"
            "path/to/SGA/outputs/helpsteer2/qwen-7b/dpo_mse_dpo1_mse1000/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_11_19_58/checkpoint-178"
        )
        ;;
    *)
        echo "Unknown MODEL_TYPE: ${MODEL_TYPE}"
        echo "  MODEL_TYPE in {llama7b, llama1b, llama8b, qwen7b}"
        exit 1
        ;;
 esac

mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/${LOG_FILE_NAME}"

exec > >(tee "${LOG_FILE}") 2>&1
echo "Logging to: ${LOG_FILE}"

DATASET_PATH="path/to/exp/datasets/HelpSteer2_processed/"
NORMALIZATION_DATASET_PATH="path/to/exp/datasets/HelpSteer2_processed"
DATASET_SPLIT="validation"
dataset_type="helpsteer2"

BATCH_SIZE=2
MAX_LENGTH=1500
MAX_SAMPLES=''
NUM_PREFERENCES=5
PREFERENCE_NAMES="helpfulness correctness coherence complexity verbosity"
LAYER_TYPE="mlp"
NUM_NEURONS=1024
NUM_LAYERS=3

TEMPERATURE=0.6
TOP_P=0.9
MAX_TOKENS=1024
SEED=3407

eval "$(/opt/miniconda3/bin/conda shell.bash hook)"

run_inference() {
    local MODEL_PATH=$1

    echo "Processing Model: ${MODEL_PATH}"

    MODEL_BASENAME=$(basename "${MODEL_PATH}")
    if [[ "${MODEL_BASENAME}" == checkpoint-* ]]; then
        CKPT="${MODEL_BASENAME}"
        MODEL_NAME=$(basename "$(dirname "${MODEL_PATH}")")
    else
        MODEL_NAME="${MODEL_BASENAME}"
        CKPT="xx"
    fi
    FINAL_OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${MODEL_NAME}_${CKPT}"
    mkdir -p "${FINAL_OUTPUT_DIR}"
    echo "Output Directory: ${FINAL_OUTPUT_DIR}"

    echo "[Stage-1] Preference Score Prediction..."
    conda activate ric_qwen3

    python infer/infer_preference.py \
        --model_path "${MODEL_PATH}" \
        --base_model "${BASE_MODEL}" \
        --device "cuda:${DEVICE_ID}" \
        --num_preferences ${NUM_PREFERENCES} \
        --preference_names ${PREFERENCE_NAMES} \
        --layer_type "${LAYER_TYPE}" \
        --num_neurons ${NUM_NEURONS} \
        --num_layers ${NUM_LAYERS} \
        --dataset_path "${DATASET_PATH}" \
        --dataset_split "${DATASET_SPLIT}" \
        --dataset_type "${dataset_type}" \
        $([ -n "${MAX_SAMPLES}" ] && echo "--max_samples ${MAX_SAMPLES}") \
        --batch_size ${BATCH_SIZE} \
        --max_length ${MAX_LENGTH} \
        --output_dir "${FINAL_OUTPUT_DIR}" \
        --normalization_method "sigmoid"

    STAGE1_OUTPUT_JSON="${FINAL_OUTPUT_DIR}/stage1_preference_scores.json"
    if [ ! -f "${STAGE1_OUTPUT_JSON}" ]; then
        echo "❌ Stage-1 failed for model: ${MODEL_PATH}"
        return 1
    fi
    SAMPLE_COUNT=$(python -c "import json; print(len(json.load(open('${STAGE1_OUTPUT_JSON}'))))")
    echo "✓ Stage-1 done: ${SAMPLE_COUNT} samples"

    echo "[Stage-2] vLLM GAM Generation..."
    conda activate vllm

    CUDA_VISIBLE_DEVICES=${DEVICE_ID} python infer/vllm_gam.py \
        --dataset_path "${STAGE1_OUTPUT_JSON}" \
        --base_model "${BASE_MODEL}" \
        --lora_path "${MODEL_PATH}" \
        --output_dir "${OUTPUT_ROOT_DIR}" \
        --dataset_type "${dataset_type}" \
        $([ -n "${MAX_SAMPLES}" ] && echo "--max_samples ${MAX_SAMPLES}") \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.5 \
        --temperature ${TEMPERATURE} \
        --top_p ${TOP_P} \
        --max_tokens ${MAX_TOKENS} \
        --seed ${SEED} \
        --max_lora_rank 128

    echo "✓ Stage-2 done"
    echo "✓ Model ${MODEL_NAME}_${CKPT} completed successfully!"

    echo "Waiting 10 seconds before next model..."
    sleep 10
}
echo "Multi-Model Inference Pipeline (Unified)"
echo "MODEL_TYPE: ${MODEL_TYPE}"
echo "Total Models: ${#MODEL_PATHS[@]}"
SUCCESS_COUNT=0
FAILED_MODELS=()

for i in "${!MODEL_PATHS[@]}"; do
    MODEL_PATH="${MODEL_PATHS[$i]}"
    echo "Processing model $((i+1))/${#MODEL_PATHS[@]}"

    if run_inference "${MODEL_PATH}"; then
        ((SUCCESS_COUNT++))
    else
        FAILED_MODELS+=("${MODEL_PATH}")
    fi
done

echo ""
echo "Inference Complete!"
echo "Total Models: ${#MODEL_PATHS[@]}"
echo "Successful: ${SUCCESS_COUNT}"
echo "Failed: ${#FAILED_MODELS[@]}"

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "Failed Models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  - ${model}"
    done
    exit 1
fi

echo ""
echo "✓ All models processed successfully!"
echo ""
echo "Finished at: $(date)"
echo "Log saved to: ${LOG_FILE}"
