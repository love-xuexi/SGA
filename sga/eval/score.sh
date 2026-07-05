#!/bin/bash

# ================= 配置区域 =================
# 在这里列出所有需要处理的 CSV 文件路径
# 只需要修改这个列表，想跑几个就写几个
CSV_LIST=(
    # dpo
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-1b/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_04_14_30_checkpoint-1501/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-1b/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_04_17_30_checkpoint-1501/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-1b/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_04_18_33_checkpoint-1501/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-1b/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_04_16_29_checkpoint-1501/vllm_gam_results.csv"
    
    # here
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-1b/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_05_23_48_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-1b/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_05_23_48_checkpoint-3002/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-1b/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_06_01_58_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-1b/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_06_01_58_checkpoint-3002/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.1-8b/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_06_04_07_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.1-8b/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_06_04_07_checkpoint-3002/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.1-8b/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_06_15_59_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.1-8b/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_06_15_59_checkpoint-3002/vllm_gam_results.csv"

    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-3b/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_08_01_48_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-3b/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_08_01_48_checkpoint-3002/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-3b/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_08_07_20_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/llama3.2-3b/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_08_07_20_checkpoint-3002/vllm_gam_results.csv"

    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen2.5-7b/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_07_04_03_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen2.5-7b/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_07_04_03_checkpoint-3002/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen2.5-7b/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_07_14_54_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen2.5-7b/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_07_14_54_checkpoint-3002/vllm_gam_results.csv"

    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen3_4b/Qwen3-4B_dpo_len1024_lora32_1e-05_02_09_09_58_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen3_4b/Qwen3-4B_dpo_len1024_lora32_1e-05_02_09_09_58_checkpoint-3002/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen3_4b/Qwen3-4B_dpo_len1024_lora32_1e-05_02_09_17_41_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen3_4b/Qwen3-4B_dpo_len1024_lora32_1e-05_02_09_17_41_checkpoint-3002/vllm_gam_results.csv"

    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen3_8b/Qwen3-8B_dpo_len1024_lora32_1e-05_02_08_12_47_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen3_8b/Qwen3-8B_dpo_len1024_lora32_1e-05_02_08_12_47_checkpoint-3002/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen3_8b/Qwen3-8B_dpo_len1024_lora32_1e-05_02_08_23_26_checkpoint-1502/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO/qwen3_8b/Qwen3-8B_dpo_len1024_lora32_1e-05_02_08_23_26_checkpoint-3002/vllm_gam_results.csv"

    # "/mnt/ai4s/lzy_exp/inference_results/modpo/llama3.2-1b/hh_rlhf/hh_llama1b-checkpoint-1237/llama3.2-1b.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/modpo/llama3.2-3b/hh_rlhf/hh_llama3b-checkpoint-618/llama3.2-3b.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/modpo/llama3.1-8b/hh_rlhf/hh_llama8b-checkpoint-1237/llama3.1-8b.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/modpo/qwen3-4b/hh_rlhf/hh_qwen4b-checkpoint-1237/qwen3-4b.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/modpo/qwen2.5-7b/hh_rlhf/hh_qwen7b-checkpoint-1236/qwen2.5-7b.csv"
    "/mnt/ai4s/lzy_exp/inference_results/modpo/qwen3-8b/hh_rlhf/hh_qwen8b-checkpoint-1236/qwen3-8b.csv"
)

# 固定参数设置
GPU_ID=0
BATCH_SIZE=16
PROMPT_COL="prompt"
# RESPONSE_COL="ground_truth_response"
RESPONSE_COL="generated_response"
REWARD_NAMES="harmless,helpful"

# ================= 执行区域 =================
# 循环读取数组中的每一个路径
for FILE_PATH in "${CSV_LIST[@]}"; do
    echo "--------------------------------------------------"
    echo "正在处理文件: $FILE_PATH"
    echo "--------------------------------------------------"

    python score.py \
        --csv_path "$FILE_PATH" \
        --gpu_id $GPU_ID \
        --batch_size $BATCH_SIZE \
        --prompt_column $PROMPT_COL \
        --response_column $RESPONSE_COL \
        --reward_names $REWARD_NAMES \
        --clean_prompt

    echo "完成: $FILE_PATH"
    echo ""
done

echo "所有任务已完成！"