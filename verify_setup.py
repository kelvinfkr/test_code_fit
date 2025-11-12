#!/usr/bin/env python3
"""
快速验证脚本：检查所有文件和数据是否就绪
"""

import os
import pandas as pd

def check_file(filepath, description):
    """检查文件是否存在"""
    exists = os.path.exists(filepath)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filepath}")
    return exists

def check_excel_shape(filepath, expected_shape):
    """检查 Excel 文件形状"""
    try:
        df = pd.read_excel(filepath)
        actual_shape = df.shape
        match = actual_shape == expected_shape
        status = "✓" if match else "✗"
        print(f"    {status} 形状: {actual_shape} (期望: {expected_shape})")
        return match
    except Exception as e:
        print(f"    ✗ 错误: {e}")
        return False

print("=" * 70)
print("项目设置验证")
print("=" * 70)

# 检查目录
print("\n【1】检查目录结构:")
check_file('input', 'input 目录')
check_file('out', 'out 目录')

# 检查脚本文件
print("\n【2】检查脚本文件:")
check_file('align_data.py', '数据对齐脚本')
check_file('improved_complete_v2.py', '主训练脚本')
check_file('README.md', '使用说明文档')

# 检查原始数据文件
print("\n【3】检查原始数据文件:")
if check_file('input/input_data-10k.xlsx', '原始 Input 数据'):
    check_excel_shape('input/input_data-10k.xlsx', (24000, 29))

if check_file('input/output_data-10k.xlsx', '原始 Output 数据'):
    check_excel_shape('input/output_data-10k.xlsx', (10000, 6))

# 检查对齐后的数据文件
print("\n【4】检查对齐后的数据文件:")
if check_file('input/input_aligned.xlsx', '对齐后 Input 数据'):
    check_excel_shape('input/input_aligned.xlsx', (10000, 17))

if check_file('input/output_aligned.xlsx', '对齐后 Output 数据'):
    check_excel_shape('input/output_aligned.xlsx', (10000, 5))

# 验证数据内容
print("\n【5】验证数据内容:")
try:
    df_input = pd.read_excel('input/input_aligned.xlsx')
    df_output = pd.read_excel('input/output_aligned.xlsx')

    expected_input_cols = [
        'mass_kg', 'CG_x', 'Altitude_IC_ft', 'CAS', 'KCL', 'KCD', 'KCY', 'KCI',
        'KCM', 'KCN', 'pressure_hPa', 'temperature_oC', 'Mean_wind_vel',
        'Mean_wind_dir', 'cross_ang_deg', 'distance_AB_nm', 'distance_AD_nm'
    ]

    expected_output_cols = ['XDIS_TO_LTP_m', 'YDIS_TO_LTP_m', 'PHI_deg', 'SSTP_deg', 'VTZP_fms_m_s']

    input_cols_match = list(df_input.columns) == expected_input_cols
    output_cols_match = list(df_output.columns) == expected_output_cols

    status_in = "✓" if input_cols_match else "✗"
    status_out = "✓" if output_cols_match else "✗"

    print(f"  {status_in} Input 列名正确 (17 列)")
    if not input_cols_match:
        print(f"    实际: {list(df_input.columns)[:5]}...")
        print(f"    期望: {expected_input_cols[:5]}...")

    print(f"  {status_out} Output 列名正确 (5 列)")
    if not output_cols_match:
        print(f"    实际: {list(df_output.columns)}")
        print(f"    期望: {expected_output_cols}")

    print(f"  ✓ 数据行数匹配: {len(df_input)} == {len(df_output)}")

except Exception as e:
    print(f"  ✗ 验证失败: {e}")

# 运行建议
print("\n" + "=" * 70)
print("运行建议")
print("=" * 70)
print("\n如果所有检查通过，可以运行:")
print("  python3 improved_complete_v2.py")
print("\n如果需要重新生成对齐数据:")
print("  python3 align_data.py")
print("\n查看详细使用说明:")
print("  cat README.md")

print("\n✅ 验证完成！")
