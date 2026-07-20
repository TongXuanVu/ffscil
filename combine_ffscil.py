import pandas as pd
import os

files = {
    'standard': 'C:\\FederatedLearning\\FFSCIL\\output_iot_cnn1d\\aggregated_metrics.csv',
    'fewshot_1percent': 'C:\\FederatedLearning\\FFSCIL\\1percent_metrics_final_v3.csv',
    '10shot': 'C:\\FederatedLearning\\FFSCIL\\10shot_metrics_final_v2.csv'
}

dfs = []
for scenario, path in files.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Insert 'scenario' column at the first position
        df.insert(0, 'scenario', scenario)
        dfs.append(df)
        print(f"Loaded {scenario} with {len(df)} rows.")
    else:
        print(f"File not found for {scenario}: {path}")

if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    out_path = 'C:\\FederatedLearning\\FFSCIL\\ffscil_combined_metrics.csv'
    combined_df.to_csv(out_path, index=False)
    print(f"\nSuccessfully saved combined metrics to: {out_path}")
    print(f"Total rows: {len(combined_df)}")
