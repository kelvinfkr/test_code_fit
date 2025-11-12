# Monte Carlo 数据预测使用指南

## 概述

本指南说明如何使用训练好的模型对 `MonteCarlodata_input_0708.mat` 进行预测，并提取降落最早和最晚的样本。

## 文件说明

### 新增文件

1. **`predict_monte_carlo.py`** - Monte Carlo数据预测脚本
   - 加载训练好的模型
   - 对MonteCarlodata进行批量预测
   - 提取极端样本（最早/最晚）
   - 保存结果到MAT文件

2. **`requirements.txt`** - Python依赖包列表

3. **`improved_complete_v2.py`** (已修改)
   - 新增：保存训练好的模型到 `out/model_*.pth`
   - 新增：保存标准化参数到 `out/normalization_params.pkl`
   - 新增：保存Isotonic校准模型到 `out/isotonic_model.pkl`

## 使用步骤

### 步骤 1: 安装依赖

```bash
pip install -r requirements.txt
```

依赖包包括：
- numpy
- pandas
- matplotlib
- torch (PyTorch)
- scikit-learn
- scipy
- tqdm
- openpyxl

### 步骤 2: 训练模型（如果还没有训练）

```bash
python improved_complete_v2.py
```

这个脚本会：
- 训练5个集成模型
- 应用Isotonic校准
- 保存所有模型和参数到 `out/` 目录
- 生成可视化图表和指标报告

**预期输出文件：**
- `out/model_0.pth` ~ `out/model_4.pth` - 5个训练好的模型
- `out/normalization_params.pkl` - 数据标准化参数
- `out/isotonic_model.pkl` - Isotonic校准模型
- `out/improved_resnet_complete.png` - 可视化图表
- `out/predictions.xlsx` - 训练数据的预测结果
- `out/metrics_summary.xlsx` - 性能指标摘要

### 步骤 3: 准备Monte Carlo数据

确保 `MonteCarlodata_input_0708.mat` 文件在项目根目录下。

**数据格式要求：**
- MAT文件中应包含变量 `MonteCarlodata_Input`
- 数据应为结构体，包含以下字段（按顺序）：
  1. mass_kg
  2. CG_x
  3. Altitude_IC_ft
  4. CAS
  5. KCL
  6. KCD
  7. KCY
  8. KCI
  9. KCM
  10. KCN
  11. pressure_hPa
  12. temperature_oC
  13. Mean_wind_vel
  14. Mean_wind_dir
  15. cross_ang_deg
  16. distance_AB_nm
  17. distance_AD_nm

**总计：17个输入特征**

### 步骤 4: 运行预测

```bash
python predict_monte_carlo.py
```

这个脚本会：
1. 加载训练好的5个模型
2. 加载标准化参数和Isotonic校准模型
3. 读取 `MonteCarlodata_input_0708.mat`
4. 进行批量预测（支持大规模数据）
5. 找出预测值最小的1700个样本（降落最早）
6. 找出预测值最大的300个样本（降落最晚）
7. 保存结果到MAT文件

**预期输出文件：**
- `out/monte_carlo_earliest_1700.mat` - 降落最早的1700个样本
- `out/monte_carlo_latest_300.mat` - 降落最晚的300个样本
- `out/monte_carlo_all_predictions.mat` - 所有样本的预测结果

### 输出MAT文件内容

每个输出MAT文件包含以下变量：

```matlab
% monte_carlo_earliest_1700.mat
inputs               % (1700 × 17) 输入特征矩阵
predicted_outputs    % (1700 × 1) 预测的降落时间
indices              % (1700 × 1) 在原始数据中的索引
description          % 描述字符串

% monte_carlo_latest_300.mat
inputs               % (300 × 17) 输入特征矩阵
predicted_outputs    % (300 × 1) 预测的降落时间
indices              % (300 × 1) 在原始数据中的索引
description          % 描述字符串

% monte_carlo_all_predictions.mat
inputs               % (N × 17) 所有样本的输入特征
predicted_outputs    % (N × 1) 所有样本的预测结果
description          % 描述字符串
```

## 在MATLAB中使用结果

```matlab
% 加载降落最早的样本
load('out/monte_carlo_earliest_1700.mat')
fprintf('降落最早的1700个样本:\n')
fprintf('  预测值范围: [%.2f, %.2f]\n', min(predicted_outputs), max(predicted_outputs))

% 加载降落最晚的样本
load('out/monte_carlo_latest_300.mat')
fprintf('降落最晚的300个样本:\n')
fprintf('  预测值范围: [%.2f, %.2f]\n', min(predicted_outputs), max(predicted_outputs))

% 查看输入特征
fprintf('输入特征维度: %d\n', size(inputs, 2))
```

## 性能特点

### 模型特性
- **集成学习**: 5个独立训练的模型进行平均，提高鲁棒性
- **ResNet架构**: 深度残差网络，支持梯度流动
- **线性捷径**: 直接从输入到输出的线性路径
- **分位数辅助**: 帮助模型学习极端值
- **Focal-MSE损失**: 平衡大小误差的处理
- **Isotonic校准**: 修正系统性偏差

### 批量预测
- 支持大规模数据（百万级样本）
- 自动批处理（batch_size=10000）
- GPU加速（如果可用）
- 内存高效

## 故障排除

### 问题：ModuleNotFoundError

**解决方案**：安装缺失的包
```bash
pip install <package_name>
```

### 问题：CUDA out of memory

**解决方案**：减小批处理大小
在 `predict_monte_carlo.py` 中修改：
```python
batch_size = 5000  # 从10000减少到5000
```

### 问题：输入维度不匹配

**解决方案**：检查MonteCarlodata格式
确保数据包含正确的17个特征，顺序与训练数据一致。

### 问题：MAT文件格式错误

**解决方案**：确保使用MATLAB v7.3或更早的格式
```matlab
% 在MATLAB中保存
save('MonteCarlodata_input_0708.mat', 'MonteCarlodata_Input', '-v7.3')
```

## 技术细节

### 数据标准化
所有输入数据使用训练时计算的均值和标准差进行标准化：
```
X_normalized = (X - X_mean) / X_std
```

### 预测流程
1. 加载原始数据
2. 应用标准化变换
3. 通过5个模型分别预测
4. 计算平均预测值
5. 反标准化到原始尺度
6. 应用Isotonic校准

### 集成策略
使用简单平均集成：
```
y_pred = mean([model_1(X), model_2(X), ..., model_5(X)])
```

## 联系与支持

如有问题或需要帮助，请查阅：
- `improved_complete_v2.py` - 训练代码
- `predict_monte_carlo.py` - 预测代码
- `out/metrics_summary.xlsx` - 模型性能指标

## 版本历史

### v1.0 (2025-11-12)
- 初始版本
- 支持Monte Carlo数据预测
- 提取极端样本（最早/最晚）
- 保存结果到MAT格式
