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

# 1. Process 10-shot
df_10 = pd.read_csv('output/10shot/output_iot_real/round_metrics.csv')
df_10_final = process_df(df_10)
df_10_final.to_csv('10shot_metrics_final_v2.csv', index=False)
print("Saved 10shot_metrics_final_v2.csv with {} rows".format(len(df_10_final)))

# 2. Process 1%
fs_files = [
    'output/1%/round_metrics 1.csv',
    'output/1%/round_metrics 2.csv',
    'output/1%/round_metrics.csv',
    'output/1%/output_iot_real/round_metrics.csv'
]
dfs = []
for f in fs_files:
    if os.path.exists(f):
        dfs.append(pd.read_csv(f))

if dfs:
    df_1p = pd.concat(dfs, ignore_index=True)
    df_1p = df_1p.drop_duplicates(subset=['task_id', 'round'], keep='last')
    df_1p = df_1p.sort_values(by=['task_id', 'round']).reset_index(drop=True)
    df_1p_final = process_df(df_1p)
    df_1p_final.to_csv('1percent_metrics_final_v2.csv', index=False)
    print("Saved 1percent_metrics_final_v2.csv with {} rows".format(len(df_1p_final)))

    # Print missing rounds if any
    full_rounds = pd.DataFrame([(t, r) for t in range(1, 7) for r in range(1, 31)], columns=['task_id', 'round_in_task'])
    merged = pd.merge(full_rounds, df_1p_final, on=['task_id', 'round_in_task'], how='left')
    missing = merged[merged['acc'].isna()]
    if not missing.empty:
        print("Warning: 1% metrics is missing {} rounds:".format(len(missing)))
        for _, row in missing.iterrows():
            print("  Task {} Round {}".format(int(row['task_id']), int(row['round_in_task'])))
    else:
        print("1% metrics is perfectly complete (180 rounds)!")
