# #!/bin/bash

# # ================= 配置区域 =================

# # 1. 你的模型 CSV 路径 (只需要改这里)
# MODEL_CSV="/mnt/ai4s/zhouhaojie/liuzhanyang/21/nice/GAM/reward_models/infer/score/Llama-3.1-8B-Instruct_parallel_pref_guided_plain_L1.0_P1000.0_len1024_lora32_1e-05_pref2_11_30_16_00_checkpoint-2256.csv"

# # 2. Ground Truth CSV 路径 (之前代码里硬编码的路径)
# GT_CSV="/mnt/ai4s/zhouhaojie/liuzhanyang/21/nice/GAM/reward_models/infer/score/Llama-3.1-8B-Instruct_parallel_pref_guided_plain_L1.0_P1000.0_len1024_lora32_1e-05_pref2_11_30_16_00_checkpoint-2256_truth.csv"

# # SCORE_COLUMNS="hs_helpfulness,hs_correctness,hs_coherence,hs_complexity,hs_verbosity"
# SCORE_COLUMNS="harmless_score,helpful_score"
# # ================= 变量设置 =================
# # 4. 最终输出的统计结果 CSV 路径
# OUTPUT_CSV="compare.csv"

# # ================= 执行 =================

# echo "开始计算胜率和平均分..."

# python compare_scores3.py \
#     --model_file "$MODEL_CSV" \
#     --gt_file "$GT_CSV" \
#     --score_columns "$SCORE_COLUMNS" \
#     --output_csv "$OUTPUT_CSV"


#!/bin/bash

# ================= 配置区域 =================

# 1. Ground Truth CSV 路径 (固定的真实标签文件)
#help
# GT_CSV="/mnt/ai4s/lzy_exp/inference_results/truth_scored_results/GAM/llama3.1-8b-helpsteer2/Llama-3.1-8B-Instruct_parallel_debug_helpsteer2_llama8b_plain_L1.0_P1000.0_debug_len1500_lora32_1e-05_pref5_12_07_15_11_checkpoint-284/vllm_gam_results.csv"

#ultra
GT_CSV="/mnt/ai4s/lzy_exp/inference_results/truth_scored_results/GAM/llama3.1-8b-ultrafeedback/Llama-3.1-8B-Instruct_parallel_debug_ultrafeedback_llama8b_plain_L1.0_P1000.0_debug_len1500_lora32_1e-05_pref4_12_06_17_41_checkpoint-1676/vllm_gam_results.csv"

#hh
# GT_CSV="/mnt/ai4s/zhouhaojie/liuzhanyang/21/nice/GAM/reward_models/infer/score/Llama-3.1-8B-Instruct_parallel_pref_guided_plain_L1.0_P1000.0_len1024_lora32_1e-05_pref2_11_30_16_00_checkpoint-2256_truth.csv"

# 2. 评分列名配置
# SCORE_COLUMNS="hs_helpfulness,hs_correctness,hs_coherence,hs_complexity,hs_verbosity"
# SCORE_COLUMNS="helpful_score,harmless_score"
SCORE_COLUMNS="uf_helpfulness,uf_honesty,uf_instruction_following,uf_truthfulness"
# 3. 输出文件名 (注意：Python脚本会将此文件保存在对应的模型文件夹下)
OUTPUT_FILENAME="summary_stats.csv"

# 4. 模型 CSV 路径列表 (在此处添加任意多个模型文件路径)
# 格式：在括号内，用空格或换行分隔，建议使用双引号包裹路径
MODEL_LIST=(
    # hh-rlhf
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_06_04_07_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_06_04_07_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_06_15_59_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_06_15_59_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_05_23_48_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_05_23_48_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_06_01_58_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_06_01_58_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_08_01_48_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_08_01_48_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_08_07_20_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_08_07_20_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_07_04_03_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_07_04_03_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_07_14_54_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_07_14_54_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Qwen3-4B_dpo_len1024_lora32_1e-05_02_09_09_58_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Qwen3-4B_dpo_len1024_lora32_1e-05_02_09_09_58_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Qwen3-4B_dpo_len1024_lora32_1e-05_02_09_17_41_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Qwen3-4B_dpo_len1024_lora32_1e-05_02_09_17_41_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Qwen3-8B_dpo_len1024_lora32_1e-05_02_08_12_47_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Qwen3-8B_dpo_len1024_lora32_1e-05_02_08_12_47_checkpoint-3002.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Qwen3-8B_dpo_len1024_lora32_1e-05_02_08_23_26_checkpoint-1502.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/Qwen3-8B_dpo_len1024_lora32_1e-05_02_08_23_26_checkpoint-3002.csv"

    # # helpsteer2
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.1-8b-helpsteer2/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_11_18_57_checkpoint-90/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.1-8b-helpsteer2/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_11_18_57_checkpoint-178/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-1b-helpsteer2/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_11_18_44_checkpoint-90/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-1b-helpsteer2/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_11_18_44_checkpoint-178/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-3b-helpsteer2/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_20_56_checkpoint-90/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-3b-helpsteer2/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_20_56_checkpoint-178/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen2.5-7b-helpsteer2/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_11_19_58_checkpoint-90/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen2.5-7b-helpsteer2/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_11_19_58_checkpoint-178/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-4b-helpsteer2/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_20_27_checkpoint-90/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-4b-helpsteer2/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_20_27_checkpoint-178/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-4b-helpsteer2/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_21_22_checkpoint-90/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-4b-helpsteer2/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_21_22_checkpoint-178/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-8b-helpsteer2/Qwen3-8B_dpo_len1024_lora32_1e-05_02_13_07_00_checkpoint-90/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-8b-helpsteer2/Qwen3-8B_dpo_len1024_lora32_1e-05_02_13_07_00_checkpoint-178/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-8b-helpsteer2/Qwen3-8B_dpo_len1024_lora32_1e-05_02_13_08_07_checkpoint-90/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-8b-helpsteer2/Qwen3-8B_dpo_len1024_lora32_1e-05_02_13_08_07_checkpoint-178/vllm_gam_results.csv"

    # ultrafeedback
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-1b-ultrafeedback/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_10_01_19_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-1b-ultrafeedback/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_10_01_19_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-1b-ultrafeedback/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_10_02_33_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-1b-ultrafeedback/Llama-3.2-1B-Instruct_dpo_len1024_lora32_1e-05_02_10_02_33_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.1-8b-ultrafeedback/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_10_03_47_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.1-8b-ultrafeedback/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_10_03_47_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.1-8b-ultrafeedback/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_10_10_36_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.1-8b-ultrafeedback/Llama-3.1-8B-Instruct_dpo_len1024_lora32_1e-05_02_10_10_36_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-3b-ultrafeedback/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_06_14_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-3b-ultrafeedback/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_06_14_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-3b-ultrafeedback/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_09_05_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.2-3b-ultrafeedback/Llama-3.2-3B-Instruct_dpo_len1024_lora32_1e-05_02_11_09_05_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen2.5-7b-ultrafeedback/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_10_17_23_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen2.5-7b-ultrafeedback/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_10_17_23_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen2.5-7b-ultrafeedback/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_10_23_52_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen2.5-7b-ultrafeedback/qwen-2.5-7b-instruct_dpo_len1024_lora32_1e-05_02_10_23_52_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-4b-ultrafeedback/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_22_16_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-4b-ultrafeedback/Qwen3-4B_dpo_len1024_lora32_1e-05_02_13_22_16_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-4b-ultrafeedback/Qwen3-4B_dpo_len1024_lora32_1e-05_02_14_08_38_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-4b-ultrafeedback/Qwen3-4B_dpo_len1024_lora32_1e-05_02_14_08_38_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-8b-ultrafeedback/Qwen3-8B_dpo_len1024_lora32_1e-05_02_11_11_53_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-8b-ultrafeedback/Qwen3-8B_dpo_len1024_lora32_1e-05_02_11_11_53_checkpoint-1248/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-8b-ultrafeedback/Qwen3-8B_dpo_len1024_lora32_1e-05_02_12_23_25_checkpoint-624/vllm_gam_results.csv"
    # "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/qwen3-8b-ultrafeedback/Qwen3-8B_dpo_len1024_lora32_1e-05_02_12_23_25_checkpoint-1248/vllm_gam_results.csv"
    "/mnt/ai4s/lzy_exp/inference_results/DPO_score/scored_results/DPO/llama3.1-8b-ultrafeedback/(0.25)helpfulness+(1-0.25)instruction_following_checkpoint-2544/vllm_gam_results.csv"
)  

# ================= 执行区域 =================

echo "========================================"
echo "开始批量评估任务"
echo "共计 ${#MODEL_LIST[@]} 个模型文件待处理"
echo "Ground Truth: $(basename "$GT_CSV")"
echo "========================================"

# 遍历数组中的每一个模型路径
for model_path in "${MODEL_LIST[@]}"; do
    
    # 检查文件是否存在，防止路径写错报错
    if [ ! -f "$model_path" ]; then
        echo "❌ 错误: 找不到文件 -> $model_path"
        echo "----------------------------------------"
        continue
    fi

    echo "正在处理模型: $(basename "$model_path") ..."
    
    # 调用 Python 脚本
    # 注意：output_csv 只需要传文件名，Python 脚本会自动拼接到 model_path 的目录
    python compare_scores3.py \
        --model_file "$model_path" \
        --gt_file "$GT_CSV" \
        --score_columns "$SCORE_COLUMNS" \
        --output_csv "$OUTPUT_FILENAME"

    # 检查上一步命令是否成功
    if [ $? -eq 0 ]; then
        echo "✅ 完成."
    else
        echo "⚠️  脚本执行出现错误."
    fi
    echo "----------------------------------------"

done

echo "所有任务执行完毕。"