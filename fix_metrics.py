import os
import subprocess
import pandas as pd

# Run combination
subprocess.run(['python', 'combine_ffscil.py'], check=True)

# Round loss
f_comb = 'ffscil_combined_metrics.csv'
df = pd.read_csv(f_comb)
df['loss'] = df['loss'].round(4)
df.to_csv(f_comb, index=False)

f_agg = 'output_iot_cnn1d\\aggregated_metrics.csv'
df2 = pd.read_csv(f_agg)
df2['loss'] = df2['loss'].round(4)
df2.to_csv(f_agg, index=False)

# Recreate task summaries
last_rounds = df.sort_values(by=['scenario', 'task_id', 'round_in_task']).groupby(['scenario', 'task_id']).tail(1)
scenario_order = {'standard': 1, 'fewshot_1percent': 2, '10shot': 3}
last_rounds['sort_order'] = last_rounds['scenario'].map(scenario_order)
last_rounds = last_rounds.sort_values(by=['task_id', 'sort_order'])
last_rounds = last_rounds.drop(columns=['sort_order'])

cols = ['scenario', 'task_id', 'round_in_task', 'acc', 'f1_mic', 'f1_mac', 'f1_wei', 'prec_mic', 'prec_mac', 'prec_wei', 'rec_mic', 'rec_mac', 'rec_wei', 'loss']
last_rounds = last_rounds[cols]

f_sum = 'ffscil_task_summaries_v2.csv'
last_rounds.to_csv(f_sum, index=False)

# Generate Markdown artifact
scenario_map = {'standard': 'Baseline', 'fewshot_1percent': '1% Few-Shot', '10shot': '10-Shot'}
md_lines = [
    '# Bảng Tổng Hợp Metrics (Round Cuối của Từng Task)',
    '',
    'Bảng này chứa dữ liệu các thông số quan trọng nhất của từng Task (lấy Round cuối cùng đại diện) cho cả 3 kịch bản: **Baseline (Standard), 1% Few-Shot, và 10-Shot**.',
    '',
    '| Task | Kịch Bản | Accuracy | F1 Micro | F1 Macro | F1 Weight | Prec Micro | Prec Macro | Prec Weight | Rec Micro | Rec Macro | Rec Weight | Loss |',
    '|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|'
]

for t in range(1, 7):
    task_df = last_rounds[last_rounds['task_id'] == t]
    first_row = True
    for _, r in task_df.iterrows():
        sc_name = scenario_map[r['scenario']]
        task_col = f'**Task {t}**' if first_row else ''
        first_row = False
        md_lines.append(f"| {task_col} | {sc_name} | {r['acc']:.2f} | {r['f1_mic']:.2f} | {r['f1_mac']:.2f} | {r['f1_wei']:.2f} | {r['prec_mic']:.2f} | {r['prec_mac']:.2f} | {r['prec_wei']:.2f} | {r['rec_mic']:.2f} | {r['rec_mac']:.2f} | {r['rec_wei']:.2f} | {r['loss']:.4f} |")

with open('C:\\Users\\xuan vu\\.gemini\\antigravity\\brain\\7f75ac45-6bb9-4755-8b56-db6853c5c1da\\artifacts\\ffscil_summary_table.md', 'w', encoding='utf-8') as f:
    f.write('\n'.join(md_lines))

print('Successfully processed everything and generated v2 summaries!')
