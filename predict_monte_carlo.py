#!/usr/bin/env python3
"""
使用训练好的模型对MonteCarlodata_input_0708.mat进行推理
找出预测值最大的300个样本（降落最晚）和最小的1700个样本（降落最早）
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.io import loadmat, savemat
import os
import pickle

# 定义模型结构（需要与训练时一致）
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + residual
        out = self.relu(out)
        return out


class ImprovedComplexNet(nn.Module):
    """改进版网络：线性捷径 + 分位辅助头"""
    def __init__(self, input_dim):
        super(ImprovedComplexNet, self).__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )

        self.res_blocks_1 = nn.Sequential(
            ResidualBlock(256),
            ResidualBlock(256),
        )

        self.transition_1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        self.res_blocks_2 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
        )

        self.transition_2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # 点估计输出
        self.output_point = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 1)
        )

        # 线性捷径
        self.linear_shortcut = nn.Linear(input_dim, 1)

        # 分位数辅助头
        self.output_quant = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        x_orig = x

        x = self.input_proj(x)
        x = self.res_blocks_1(x)
        x = self.transition_1(x)
        x = self.res_blocks_2(x)
        x = self.transition_2(x)

        # 点估计 + 线性捷径
        point = self.output_point(x)
        shortcut = self.linear_shortcut(x_orig)
        point = point + 0.1 * shortcut

        # 分位数
        quant = self.output_quant(x)

        return point, quant


def load_monte_carlo_data(mat_file):
    """加载MonteCarlodata_input_0708.mat文件"""
    print(f"Loading Monte Carlo data from: {mat_file}")

    if not os.path.exists(mat_file):
        raise FileNotFoundError(f"File not found: {mat_file}")

    mat_data = loadmat(mat_file)

    # 查看mat文件中的变量
    print(f"Variables in MAT file: {list(mat_data.keys())}")

    # 获取MonteCarlodata_Input
    if 'MonteCarlodata_Input' in mat_data:
        mc_data = mat_data['MonteCarlodata_Input']
        print(f"MonteCarlodata_Input shape: {mc_data.shape}")
        print(f"MonteCarlodata_Input dtype: {mc_data.dtype}")

        # 构建MonteCarloMatrix（按照MATLAB代码的逻辑）
        # 假设mc_data是一个结构体数组，需要提取各个字段
        if hasattr(mc_data[0, 0], '_fieldnames'):
            field_names = mc_data[0, 0]._fieldnames
            print(f"Fields: {field_names}")

            # 按照MATLAB代码中的顺序提取字段
            fields_order = [
                'mass_kg', 'CG_x', 'Altitude_IC_ft', 'CAS',
                'KCL', 'KCD', 'KCY', 'KCI', 'KCM', 'KCN',
                'pressure_hPa', 'temperature_oC',
                'Mean_wind_vel', 'Mean_wind_dir',
                'cross_ang_deg', 'distance_AB_nm', 'distance_AD_nm'
            ]

            matrix_list = []
            for field in fields_order:
                if field in field_names:
                    field_data = mc_data[field][0, 0].flatten()
                    matrix_list.append(field_data)
                else:
                    print(f"Warning: Field '{field}' not found in data")

            MonteCarloMatrix = np.column_stack(matrix_list)
            print(f"Constructed MonteCarloMatrix shape: {MonteCarloMatrix.shape}")

            return MonteCarloMatrix
        else:
            # 如果不是结构体，直接返回数据
            print("MonteCarlodata_Input is not a struct, using as is")
            return mc_data
    else:
        raise ValueError("MonteCarlodata_Input not found in MAT file")


def predict_with_ensemble(models, X_input, X_mean, X_std, y_mean, y_std, iso_model, device):
    """使用集成模型进行预测"""
    print(f"\nPredicting with {len(models)} ensemble models...")

    # 标准化输入
    X_input_norm = (X_input - X_mean) / X_std
    X_tensor = torch.FloatTensor(X_input_norm).to(device)

    # 集成预测
    all_predictions = []

    for i, model in enumerate(models):
        model.eval()
        with torch.no_grad():
            # 批量预测
            batch_size = 10000
            n_samples = X_tensor.shape[0]
            predictions = []

            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch = X_tensor[start_idx:end_idx]

                point_pred, _ = model(batch)
                predictions.append(point_pred.cpu().numpy())

            predictions = np.concatenate(predictions, axis=0)
            all_predictions.append(predictions)

        print(f"  Model {i+1}/{len(models)} done")

    # 平均集成预测
    y_pred_norm = np.mean(all_predictions, axis=0).squeeze()

    # 反标准化
    y_pred = y_pred_norm * y_std + y_mean

    # 应用Isotonic校准
    if iso_model is not None:
        print("Applying Isotonic calibration...")
        y_pred = iso_model.predict(y_pred)

    print(f"Prediction done! Shape: {y_pred.shape}")
    print(f"Prediction range: [{y_pred.min():.2f}, {y_pred.max():.2f}]")
    print(f"Prediction mean: {y_pred.mean():.2f}, std: {y_pred.std():.2f}")

    return y_pred


def main():
    print("="*70)
    print("Monte Carlo Data Prediction")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # 1. 检查是否存在训练好的模型
    model_dir = 'out'
    model_files = [f'{model_dir}/model_{i}.pth' for i in range(5)]
    normalization_file = f'{model_dir}/normalization_params.pkl'
    iso_file = f'{model_dir}/isotonic_model.pkl'

    if not all(os.path.exists(f) for f in model_files):
        print("Error: Trained models not found!")
        print("Please run 'python improved_complete_v2.py' first to train the models.")
        print("\nExpected files:")
        for f in model_files:
            status = "✓" if os.path.exists(f) else "✗"
            print(f"  {status} {f}")
        return

    print("✓ All model files found!\n")

    # 2. 加载标准化参数
    print("Loading normalization parameters...")
    with open(normalization_file, 'rb') as f:
        norm_params = pickle.load(f)

    X_mean = norm_params['X_mean']
    X_std = norm_params['X_std']
    y_mean = norm_params['y_mean']
    y_std = norm_params['y_std']
    input_dim = norm_params['input_dim']

    print(f"  Input dimension: {input_dim}")
    print(f"  y_mean: {y_mean:.2f}, y_std: {y_std:.2f}\n")

    # 3. 加载模型
    print("Loading ensemble models...")
    models = []
    for i, model_file in enumerate(model_files):
        model = ImprovedComplexNet(input_dim).to(device)
        model.load_state_dict(torch.load(model_file, map_location=device))
        model.eval()
        models.append(model)
        print(f"  Model {i+1}/5 loaded")

    # 4. 加载Isotonic校准模型
    iso_model = None
    if os.path.exists(iso_file):
        print("\nLoading Isotonic calibration model...")
        with open(iso_file, 'rb') as f:
            iso_model = pickle.load(f)
        print("  ✓ Isotonic model loaded")
    else:
        print("\nWarning: Isotonic calibration model not found, skipping calibration")

    # 5. 加载MonteCarlodata_input_0708.mat
    mat_file = 'MonteCarlodata_input_0708.mat'
    try:
        MonteCarloMatrix = load_monte_carlo_data(mat_file)
    except Exception as e:
        print(f"\nError loading Monte Carlo data: {e}")
        return

    # 检查输入维度是否匹配
    if MonteCarloMatrix.shape[1] != input_dim:
        print(f"\nError: Input dimension mismatch!")
        print(f"  Expected: {input_dim}, Got: {MonteCarloMatrix.shape[1]}")
        return

    print(f"\n✓ Input dimension matches: {input_dim}")
    print(f"✓ Number of samples: {MonteCarloMatrix.shape[0]}\n")

    # 6. 进行预测
    y_pred = predict_with_ensemble(
        models, MonteCarloMatrix,
        X_mean, X_std, y_mean, y_std,
        iso_model, device
    )

    # 7. 找出预测值最大的300个样本（降落最晚）
    print("\n" + "="*70)
    print("Finding extreme samples...")
    print("="*70)

    # 排序获取索引
    sorted_indices = np.argsort(y_pred)

    # 最小的1700个（降落最早）
    earliest_indices = sorted_indices[:1700]
    earliest_inputs = MonteCarloMatrix[earliest_indices]
    earliest_outputs = y_pred[earliest_indices]

    print(f"\n【降落最早的1700个样本】")
    print(f"  预测值范围: [{earliest_outputs.min():.2f}, {earliest_outputs.max():.2f}]")
    print(f"  预测值平均: {earliest_outputs.mean():.2f}")

    # 最大的300个（降落最晚）
    latest_indices = sorted_indices[-300:]
    latest_inputs = MonteCarloMatrix[latest_indices]
    latest_outputs = y_pred[latest_indices]

    print(f"\n【降落最晚的300个样本】")
    print(f"  预测值范围: [{latest_outputs.min():.2f}, {latest_outputs.max():.2f}]")
    print(f"  预测值平均: {latest_outputs.mean():.2f}")

    # 8. 保存结果到MAT文件
    print("\n" + "="*70)
    print("Saving results...")
    print("="*70)

    # 保存降落最早的样本
    earliest_data = {
        'inputs': earliest_inputs,
        'predicted_outputs': earliest_outputs.reshape(-1, 1),
        'indices': earliest_indices.reshape(-1, 1),
        'description': '降落最早的1700个样本 (预测值最小)'
    }

    earliest_file = 'out/monte_carlo_earliest_1700.mat'
    savemat(earliest_file, earliest_data)
    print(f"✓ Saved earliest 1700 samples to: {earliest_file}")

    # 保存降落最晚的样本
    latest_data = {
        'inputs': latest_inputs,
        'predicted_outputs': latest_outputs.reshape(-1, 1),
        'indices': latest_indices.reshape(-1, 1),
        'description': '降落最晚的300个样本 (预测值最大)'
    }

    latest_file = 'out/monte_carlo_latest_300.mat'
    savemat(latest_file, latest_data)
    print(f"✓ Saved latest 300 samples to: {latest_file}")

    # 9. 保存完整的预测结果（可选）
    all_predictions_file = 'out/monte_carlo_all_predictions.mat'
    all_data = {
        'inputs': MonteCarloMatrix,
        'predicted_outputs': y_pred.reshape(-1, 1),
        'description': '所有样本的预测结果'
    }
    savemat(all_predictions_file, all_data)
    print(f"✓ Saved all predictions to: {all_predictions_file}")

    print("\n" + "="*70)
    print("✅ Done!")
    print("="*70)
    print("\n输出文件:")
    print(f"  1. {earliest_file} - 降落最早的1700个样本")
    print(f"  2. {latest_file} - 降落最晚的300个样本")
    print(f"  3. {all_predictions_file} - 所有样本的预测结果")


if __name__ == '__main__':
    main()
