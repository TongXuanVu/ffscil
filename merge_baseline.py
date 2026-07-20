import pandas as pd
import glob
import os

def process_df(df):
    # Rename columns to match requested format
    df = df.rename(columns={
        'round': 'round_in_task',
        'acc1': 'acc',
        'p_micro': 'prec_mic',
        'p_macro': 'prec_mac',
        'p_weighted': 'prec_wei',
        'r_micro': 'rec_mic',
        'r_macro': 'rec_mac',
        'r_weighted': 'rec_wei',
        'f1_micro': 'f1_mic',
        'f1_macro': 'f1_mac',
        'f1_weighted': 'f1_wei'
    })
    
    # Calculate global_round
    df['global_round'] = (df['task_id'] - 1) * 30 + df['round_in_task']
    
    # Reorder columns
    cols = ['task_id', 'round_in_task', 'global_round', 'acc', 'prec_mic', 'prec_mac', 'prec_wei', 'rec_mic', 'rec_mac', 'rec_wei', 'f1_mic', 'f1_mac', 'f1_wei', 'loss']
    df = df[cols]
    
    # Multiply decimal metrics by 100 to make them percentages like 'acc'
    pct_cols = ['prec_mic', 'prec_mac', 'prec_wei', 'rec_mic', 'rec_mac', 'rec_wei', 'f1_mic', 'f1_mac', 'f1_wei']
    df[pct_cols] = df[pct_cols] * 100
    
    # Round metrics to 2 decimal places (excluding loss and index columns)
    round_cols = ['acc'] + pct_cols
    df[round_cols] = df[round_cols].round(2)
    
    # Round loss to 6 decimal places
    df['loss'] = df['loss'].round(6)
    
    return df

base_dir = 'C:\\FederatedLearning\\FFSCIL\\output_iot_cnn1d'
files = glob.glob(os.path.join(base_dir, 'round_metrics*.csv'))

dfs = []
for f in files:
    dfs.append(pd.read_csv(f))

if dfs:
    df_combined = pd.concat(dfs, ignore_index=True)
    df_combined = df_combined.drop_duplicates(subset=['task_id', 'round'], keep='last')
    df_combined = df_combined.sort_values(by=['task_id', 'round']).reset_index(drop=True)
    
    df_final = process_df(df_combined)
    out_path = os.path.join(base_dir, 'aggregated_metrics.csv')
    df_final.to_csv(out_path, index=False)
    print(f"Saved aggregated baseline metrics with {len(df_final)} rows.")

    # Check for completeness
    full_rounds = pd.DataFrame([(t, r) for t in range(1, 7) for r in range(1, 31)], columns=['task_id', 'round_in_task'])
    merged = pd.merge(full_rounds, df_final, on=['task_id', 'round_in_task'], how='left')
    missing = merged[merged['acc'].isna()]
    if not missing.empty:
        print(f"Warning: Baseline is still missing {len(missing)} rounds:")
        for _, row in missing.iterrows():
            print(f"  Task {int(row['task_id'])} Round {int(row['round_in_task'])}")
    else:
        print("The baseline is perfectly complete (180 rounds)!")
