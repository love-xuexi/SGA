#!/bin/bash

# Usage example:
#   time bash ../scripts/infer-ultrafeedback-all.sh llama1b      # Only run llama1b ultrafeedback
#   bash infer-ultrafeedback-all.sh             # Run all model types sequentially (llama1b -> llama8b -> qwen7b)

MODEL_TYPE="$1"

# If MODEL_TYPE is not provided, run all supported model types sequentially
if [ -z "${MODEL_TYPE}" ]; then
    echo "No MODEL_TYPE provided, will run all: llama3b -> qwen4b -> qwen8b"
    for mt in llama3b qwen4b qwen7b qwen8b; do
        echo "[MASTER] Starting MODEL_TYPE: ${mt}"
        bash "$0" "${mt}"
    done
    echo "All MODEL_TYPE finished."
    exit 0
fi

case "${MODEL_TYPE}" in
    llama1b)
        LOG_DIR="./inference_results/GAM/ultrafeedback/llama3.2-1b-ultrafeedback/logs"
        LOG_FILE_NAME="llama3_2_1b_ultrafeedback.log"
        BASE_MODEL="path/to/models/Llama-3.2-1B-Instruct"
        DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed/"
        NORMALIZATION_DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed"
        DATASET_SPLIT="test"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/llama3.2-1b-ultrafeedback"
        dataset_type="ultrafeedback"
        DEVICE_ID=0
        MODEL_PATHS=(
    "path/to/SGA/outputs/ultrafeedback/llama-1b/dpo_mse_dpo1_mse10/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_10_01_19/checkpoint-624"
    "path/to/SGA/outputs/ultrafeedback/llama-1b/dpo_mse_dpo1_mse10/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_10_01_19/checkpoint-1248"
    "path/to/SGA/outputs/ultrafeedback/llama-1b/dpo_mse_dpo1_mse1000/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_10_02_33/checkpoint-624"
    "path/to/SGA/outputs/ultrafeedback/llama-1b/dpo_mse_dpo1_mse1000/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_10_02_33/checkpoint-1248"
        )
        ;;
    llama7b)
        LOG_DIR="./inference_results/GAM/ultrafeedback/llama2-7b-ultrafeedback/logs"
        LOG_FILE_NAME="llama2_7b_ultrafeedback.log"
        BASE_MODEL="path/to/models/Llama-2-7b-hf"
        DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed/"
        NORMALIZATION_DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed"
        DATASET_SPLIT="test"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/llama2-7b-ultrafeedback"
        dataset_type="ultrafeedback"
        DEVICE_ID=1
        MODEL_PATHS=(
            "path/to/exp/NO.1/GAM/llama-7b-ultrafeedback/plain_L1.0_P1000.0_debug/Llama-2-7b-hf_parallel_debug_ultrafeedback_llama7b_plain_L1.0_P1000.0_debug_len1500_lora32_1e-05_pref4_12_11_16_40/checkpoint-1656"
        )
        ;;
    llama8b)
        LOG_DIR="./inference_results/GAM/ultrafeedback/llama3.1-8b-ultrafeedback/logs"
        LOG_FILE_NAME="llama3_1_8b_ultrafeedback.log"
        BASE_MODEL="path/to/models/Llama-3.1-8B-Instruct"
        DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed/"
        NORMALIZATION_DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed"
        DATASET_SPLIT="test"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/llama3.1-8b-ultrafeedback"
        dataset_type="ultrafeedback"
        DEVICE_ID=3
        MODEL_PATHS=(
            # "path/to/SGA/outputs/ultrafeedback/llama-8b/dpo_mse_dpo1_mse10/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_10_03_47/checkpoint-624"
            # "path/to/SGA/outputs/ultrafeedback/llama-8b/dpo_mse_dpo1_mse10/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_10_03_47/checkpoint-1248"
            # "path/to/SGA/outputs/ultrafeedback/llama-8b/dpo_mse_dpo1_mse1000/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_10_10_36/checkpoint-624"
            # "path/to/SGA/outputs/ultrafeedback/llama-8b/dpo_mse_dpo1_mse1000/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_10_10_36/checkpoint-1248"
            "path/to/user/Llama-3.1-8B-Instruct/OpenBMB/UltraFeedback/modpo/lm/(0.25)helpfulness+(1-0.25)instruction_following/checkpoint-2544"
        )
        ;;
    qwen7b)
        LOG_DIR="./inference_results/GAM/ultrafeedback/qwen2.5-7b-ultrafeedback/logs"
        LOG_FILE_NAME="qwen2_5_7b_ultrafeedback.log"
        BASE_MODEL="path/to/models/qwen-2.5-7b-instruct"
        DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed/"
        NORMALIZATION_DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed"
        DATASET_SPLIT="test"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/qwen2.5-7b-ultrafeedback"
        dataset_type="ultrafeedback"
        DEVICE_ID=1
        MODEL_PATHS=(
            # "path/to/SGA/outputs/ultrafeedback/qwen-7b/dpo_mse_dpo1_mse10/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_10_17_23/checkpoint-624"
            # "path/to/SGA/outputs/ultrafeedback/qwen-7b/dpo_mse_dpo1_mse10/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_10_17_23/checkpoint-1248"
            # "path/to/SGA/outputs/ultrafeedback/qwen-7b/dpo_mse_dpo1_mse1000/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_10_23_52/checkpoint-624"
            "path/to/SGA/outputs/ultrafeedback/qwen-7b/dpo_mse_dpo1_mse1000/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_10_23_52/checkpoint-1248"
        )
        ;;
    llama3b)
        LOG_DIR="./inference_results/GAM/ultrafeedback/llama3.2-3b-ultrafeedback/logs"
        LOG_FILE_NAME="llama3_2_3b_ultrafeedback.log"
        BASE_MODEL="path/to/models/Llama-3.2-3B-Instruct"
        DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed/"
        NORMALIZATION_DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed"
        DATASET_SPLIT="test"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/llama3.2-3b-ultrafeedback"
        dataset_type="ultrafeedback"
        DEVICE_ID=1
        MODEL_PATHS=(
            # "path/to/SGA/outputs/ultrafeedback/llama-3b/dpo_mse_dpo1_mse10/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_06_14/checkpoint-624"
            # "path/to/SGA/outputs/ultrafeedback/llama-3b/dpo_mse_dpo1_mse10/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_06_14/checkpoint-1248"
            # "path/to/SGA/outputs/ultrafeedback/llama-3b/dpo_mse_dpo1_mse1000/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_09_05/checkpoint-624"
            "path/to/SGA/outputs/ultrafeedback/llama-3b/dpo_mse_dpo1_mse1000/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_09_05/checkpoint-1248"
        )
        ;;
    qwen4b)
        LOG_DIR="./inference_results/GAM/ultrafeedback/qwen3-4b-ultrafeedback/logs"
        LOG_FILE_NAME="qwen3_4b_ultrafeedback.log"
        BASE_MODEL="path/to/models/Qwen3-4B"
        DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed/"
        NORMALIZATION_DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed"
        DATASET_SPLIT="test"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/qwen3-4b-ultrafeedback"
        dataset_type="ultrafeedback"
        DEVICE_ID=0
        MODEL_PATHS=(
            "path/to/SGA/outputs/ultrafeedback/qwen-4b/dpo_mse_dpo1_mse10/Qwen3-4B_dpo_len1024_lora32_1e-05_02_14_08_38/checkpoint-624"
            "path/to/SGA/outputs/ultrafeedback/qwen-4b/dpo_mse_dpo1_mse10/Qwen3-4B_dpo_len1024_lora32_1e-05_02_14_08_38/checkpoint-1248"
            # "path/to/SGA/outputs/ultrafeedback/qwen-4b/dpo_mse_dpo1_mse1000/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_22_16/checkpoint-624"
            # "path/to/SGA/outputs/ultrafeedback/qwen-4b/dpo_mse_dpo1_mse1000/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_22_16/checkpoint-1248"
        )
        ;;
    qwen8b)
        LOG_DIR="./inference_results/GAM/ultrafeedback/qwen3-8b-ultrafeedback/logs"
        LOG_FILE_NAME="qwen3_8b_ultrafeedback.log"
        BASE_MODEL="path/to/models/Qwen3-8B"
        DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed/"
        NORMALIZATION_DATASET_PATH="path/to/exp/datasets/ultrafeedback-binarized-preferences-cleaned-kto/processed"
        DATASET_SPLIT="test"
        OUTPUT_ROOT_DIR="path/to/exp/inference_results/DPO/qwen3-8b-ultrafeedback"
        dataset_type="ultrafeedback"
        DEVICE_ID=1
        MODEL_PATHS=(
            "path/to/SGA/outputs/ultrafeedback/qwen-8b/dpo_mse_dpo1_mse10/Qwen3-8B_dpo_len1024_lora32_1e-05_02_11_11_53/checkpoint-624"
            "path/to/SGA/outputs/ultrafeedback/qwen-8b/dpo_mse_dpo1_mse10/Qwen3-8B_dpo_len1024_lora32_1e-05_02_11_11_53/checkpoint-1248"
            "path/to/SGA/outputs/ultrafeedback/qwen-8b/dpo_mse_dpo1_mse1000/Qwen3-8B_dpo_len1024_lora32_1e-05_02_12_23_25/checkpoint-624"
            "path/to/SGA/outputs/ultrafeedback/qwen-8b/dpo_mse_dpo1_mse1000/Qwen3-8B_dpo_len1024_lora32_1e-05_02_12_23_25/checkpoint-1248"
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

BATCH_SIZE=2
MAX_LENGTH=1500
MAX_SAMPLES=''
NUM_PREFERENCES=4
PREFERENCE_NAMES="helpfulness honesty instruction_following truthfulness"
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
echo "Multi-Model Inference Pipeline (UltraFeedback, Unified)"
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
