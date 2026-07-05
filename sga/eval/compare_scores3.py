import argparse
import pandas as pd
import numpy as np
import os


def parse_args():
    parser = argparse.ArgumentParser(description='Calculate Win/Loss/Tie Rates with Auto-Directory Generation')
    parser.add_argument('--model_file', type=str, required=True, help='Model results CSV path')
    parser.add_argument('--gt_file', type=str, required=True, help='Ground Truth CSV path')
    parser.add_argument('--score_columns', type=str, required=True, 
                        help='Comma-separated exact column names (e.g. accuracy,fluency)')
    parser.add_argument('--output_csv', type=str, required=True, 
                        help='Filename for the output CSV (saved inside the generated folder)')
    return parser.parse_args()

def main():
    args = parse_args()
    score_cols = [c.strip() for c in args.score_columns.split(',')]
    
    # Specify the key field for alignment
    JOIN_KEY = 'ground_truth_response'
    # JOIN_KEY = 'reference_response'
    # ================= Core modification: path processing logic =================
    # 1. Get the absolute path of the model file
    abs_model_path = os.path.abspath(args.model_file)
    
    # 2. Remove the .csv suffix to get the directory path
    # Example: /home/data/result.csv -> /home/data/result
    output_dir = os.path.splitext(abs_model_path)[0]
    
    # 3. Automatically create this directory (if it doesn't exist)
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created new directory: {output_dir}")
        except Exception as e:
            print(f"Error creating directory {output_dir}: {e}")
            return

    # 4. Get the user-specified output filename (to prevent passing paths)
    output_filename = os.path.basename(args.output_csv)
    
    # 5. Concatenate the final save path
    final_output_path = os.path.join(output_dir, output_filename)
    # =======================================================

    print(f"Loading Model: {args.model_file}")
    print(f"Loading GT   : {args.gt_file}")
    print(f"Output will be saved to: {final_output_path}") 
    
    try:
        df_model = pd.read_csv(args.model_file)
        df_gt = pd.read_csv(args.gt_file)
    except Exception as e:
        print(f"Error reading CSVs: {e}")
        return

    # 1. Check if key fields exist
    if JOIN_KEY not in df_model.columns:
        print(f"Error: Column '{JOIN_KEY}' not found in Model file.")
        return
    if JOIN_KEY not in df_gt.columns:
        print(f"Error: Column '{JOIN_KEY}' not found in GT file.")
        return

    print(f"\nAligning data based on content of: '{JOIN_KEY}'...")

    # 2. Data cleaning
    df_model[JOIN_KEY] = df_model[JOIN_KEY].astype(str).str.strip()
    df_gt[JOIN_KEY] = df_gt[JOIN_KEY].astype(str).str.strip()
    print(f"Original Model samples: {len(df_model)}")
    print(f"Original GT samples   : {len(df_gt)}")
    # 3. Duplicate value check
    if df_model[JOIN_KEY].duplicated().any():
        dup_count = df_model[JOIN_KEY].duplicated().sum()
        print(f"Warning: Found {dup_count} duplicate '{JOIN_KEY}' in Model file. Keeping first occurrences.")
        df_model = df_model.drop_duplicates(subset=[JOIN_KEY])
        
    if df_gt[JOIN_KEY].duplicated().any():
        dup_count = df_gt[JOIN_KEY].duplicated().sum()
        print(f"Warning: Found {dup_count} duplicate '{JOIN_KEY}' in GT file. Keeping first occurrences.")
        df_gt = df_gt.drop_duplicates(subset=[JOIN_KEY])

    # 4. Execute merge (Inner Join)
    df_merged = pd.merge(
        df_model, 
        df_gt, 
        on=JOIN_KEY, 
        how='inner', 
        suffixes=('_model', '_gt')
    )
    # print(df_merged.head(5))
    print(f"Original Model samples: {len(df_model)}")
    print(f"Original GT samples   : {len(df_gt)}")
    print(f"Matched samples       : {len(df_merged)}")
    
    if len(df_merged) == 0:
        print("Error: No matching data found.")
        return

    summary_data = []

    print("\n" + "="*100)
    print(f"{'Dimension':<20} | {'Win%':<8} | {'Loss%':<8} | {'Tie%':<8} | {'Model Avg':<10} | {'GT Avg':<10}")
    print("-" * 100)

    for col in score_cols:
        col_model_name = f"{col}_model"
        col_gt_name = f"{col}_gt"

        if col_model_name in df_merged.columns:
            scores_model = df_merged[col_model_name].values
        elif col in df_merged.columns:
            print(f"Warning: Column conflict logic unexpected for '{col}'. Skipping.")
            continue
        else:
            print(f"Warning: Score column '{col}' not found in Model file.")
            continue

        if col_gt_name in df_merged.columns:
            scores_gt = df_merged[col_gt_name].values
        else:
            print(f"Warning: Score column '{col}' not found in GT file.")
            continue

        try:
            scores_model = pd.to_numeric(scores_model, errors='coerce')
            scores_gt = pd.to_numeric(scores_gt, errors='coerce')
            
            valid_mask = ~np.isnan(scores_model) & ~np.isnan(scores_gt)
            scores_model = scores_model[valid_mask]
            scores_gt = scores_gt[valid_mask]
        except Exception as e:
            print(f"Error converting column '{col}' to numbers: {e}")
            continue

        wins_model = np.sum(scores_model > scores_gt)
        loss_model = np.sum(scores_gt > scores_model)
        ties = np.sum(scores_model == scores_gt)
        total = len(scores_model)
        
        if total > 0:
            win_rate = (wins_model / total) * 100
            loss_rate = (loss_model / total) * 100
            tie_rate = (ties / total) * 100
            avg_model = np.mean(scores_model)
            avg_gt = np.mean(scores_gt)
        else:
            win_rate, loss_rate, tie_rate, avg_model, avg_gt = 0.0, 0.0, 0.0, 0.0, 0.0

        print(f"{col:<20} | {win_rate:6.2f}%  | {loss_rate:6.2f}%  | {tie_rate:6.2f}%  | {avg_model:8.4f}   | {avg_gt:8.4f}")

        summary_data.append({
            'Dimension': col,
            'Model_Win_Rate(%)': round(win_rate, 2),
            'Model_Loss_Rate(%)': round(loss_rate, 2),
            'Tie_Rate(%)': round(tie_rate, 2),
            'Model_Avg_Score': round(avg_model, 4),
            'GT_Avg_Score': round(avg_gt, 4),
            'Avg_Diff': round(avg_model - avg_gt, 4),
            'Model_Wins': wins_model,
            'Model_Losses': loss_model,
            'Ties': ties,
            'Total_Samples': total
        })

    print("="*100)

    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        cols_order = [
            'Dimension', 'Model_Win_Rate(%)', 'Model_Loss_Rate(%)', 'Tie_Rate(%)',
            'Model_Avg_Score', 'GT_Avg_Score', 'Avg_Diff', 
            'Model_Wins', 'Model_Losses', 'Ties', 'Total_Samples'
        ]
        df_summary = df_summary[cols_order]
        
        df_summary.to_csv(final_output_path, index=False)
        print(f"\nResult successfully saved to: {final_output_path}")
    else:
        print("No valid score columns processed.")

if __name__ == "__main__":
    main()