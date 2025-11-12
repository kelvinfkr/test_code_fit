#!/usr/bin/env python3
"""
数据对齐脚本：按行号匹配 input 和 output，生成新的对齐文件
"""

import pandas as pd
import numpy as np

print("=" * 70)
print("数据对齐处理")
print("=" * 70)

# 读取原始文件
print("\n【Step 1】加载原始数据...")
df_input = pd.read_excel('input/input_data-10k.xlsx')
df_output = pd.read_excel('input/output_data-10k.xlsx')

print(f"  Input 原始形状: {df_input.shape}")
print(f"  Output 原始形状: {df_output.shape}")

# 需要的 17 列变量（按 MATLAB 代码中的顺序）
required_columns = [
    'mass_kg',           # 第1列
    'CG_x',              # 第2列
    'Altitude_IC_ft',    # 第3列
    'CAS',               # 第4列
    'KCL',               # 第5列
    'KCD',               # 第6列
    'KCY',               # 第7列
    'KCI',               # 第8列
    'KCM',               # 第9列
    'KCN',               # 第10列
    'pressure_hPa',      # 第11列
    'temperature_oC',    # 第12列
    'Mean_wind_vel',     # 第13列
    'Mean_wind_dir',     # 第14列
    'cross_ang_deg',     # 第15列
    'distance_AB_nm',    # 第16列
    'distance_AD_nm'     # 第17列
]

# 【Step 2】按行号匹配
print("\n【Step 2】按行号匹配数据...")
print(f"  Input 行号范围: {df_input['No.'].min()} - {df_input['No.'].max()}")
print(f"  Output 行号范围: {df_output['n'].min()} - {df_output['n'].max()}")

# 使用 output 的 'n' 列作为基准，匹配 input 的 'No.' 列
df_merged = pd.merge(
    df_output,
    df_input,
    left_on='n',
    right_on='No.',
    how='inner'  # 只保留两边都有的行
)

print(f"  匹配后行数: {len(df_merged)}")

# 【Step 3】提取需要的列
print("\n【Step 3】提取所需列...")

# 新的 input 数据：只包含 17 列变量
new_input_df = df_merged[required_columns].copy()
print(f"  新 Input 形状: {new_input_df.shape}")

# 新的 output 数据：去掉行号列 'n'
output_data_columns = [col for col in df_output.columns if col != 'n']
new_output_df = df_merged[output_data_columns].copy()
print(f"  新 Output 形状: {new_output_df.shape}")

# 【Step 4】保存对齐后的文件
print("\n【Step 4】保存新文件...")

# 保存到 input 目录
new_input_file = 'input/input_aligned.xlsx'
new_output_file = 'input/output_aligned.xlsx'

new_input_df.to_excel(new_input_file, index=False)
print(f"  ✓ 保存: {new_input_file}")

new_output_df.to_excel(new_output_file, index=False)
print(f"  ✓ 保存: {new_output_file}")

# 【Step 5】验证结果
print("\n【Step 5】验证结果...")
print(f"\n新 Input 文件:")
print(f"  形状: {new_input_df.shape}")
print(f"  列名: {new_input_df.columns.tolist()}")
print(f"\n前 3 行数据:")
print(new_input_df.head(3))

print(f"\n新 Output 文件:")
print(f"  形状: {new_output_df.shape}")
print(f"  列名: {new_output_df.columns.tolist()}")
print(f"\n前 3 行数据:")
print(new_output_df.head(3))

# 【Step 6】生成数据统计信息
print("\n" + "=" * 70)
print("数据统计信息")
print("=" * 70)

stats_df = pd.DataFrame({
    'Column': new_input_df.columns,
    'Mean': new_input_df.mean(),
    'Std': new_input_df.std(),
    'Min': new_input_df.min(),
    'Max': new_input_df.max()
})

stats_file = 'out/input_statistics.xlsx'
stats_df.to_excel(stats_file, index=False)
print(f"\n✓ 输入特征统计信息保存到: {stats_file}")

print("\n✅ 数据对齐完成！")
print(f"\n使用对齐后的文件:")
print(f"  - {new_input_file} ({new_input_df.shape[0]} 行 × {new_input_df.shape[1]} 列)")
print(f"  - {new_output_file} ({new_output_df.shape[0]} 行 × {new_output_df.shape[1]} 列)")
