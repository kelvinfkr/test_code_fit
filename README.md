# ResNet 飞行数据预测模型

基于 ResNet + Focal-MSE + 线性捷径 + 分位辅助头 + Isotonic 校准的飞行数据预测系统。

## 项目结构

```
test_code_fit/
├── input/                          # 输入数据目录
│   ├── input_data-10k.xlsx        # 原始输入数据 (24000×29)
│   ├── output_data-10k.xlsx       # 原始输出数据 (10000×6)
│   ├── input_aligned.xlsx         # 对齐后输入数据 (10000×17) ✓
│   └── output_aligned.xlsx        # 对齐后输出数据 (10000×5) ✓
│
├── out/                            # 输出结果目录（运行后生成）
│   ├── predictions.xlsx           # 预测结果（3个sheet）
│   ├── metrics_summary.xlsx       # 性能指标摘要
│   ├── input_statistics.xlsx      # 输入特征统计信息
│   └── improved_resnet_complete.png  # 可视化图表
│
├── align_data.py                   # 数据对齐脚本
├── improved_complete_v2.py         # 主训练脚本
└── README.md                       # 本文件
```

## 使用流程

### 1. 数据准备（首次运行或数据更新时）

如果你有新的原始数据文件，需要先进行数据对齐：

```bash
python3 align_data.py
```

此脚本会：
- 从 `input/input_data-10k.xlsx` 和 `input/output_data-10k.xlsx` 读取原始数据
- 按照 output 的行号（`n` 列）匹配 input 的行号（`No.` 列）
- 从 input 中提取 17 个指定特征（按 MATLAB 代码顺序）
- 生成对齐后的文件：`input_aligned.xlsx` 和 `output_aligned.xlsx`

### 2. 训练模型

```bash
python3 improved_complete_v2.py
```

此脚本会：
- 加载对齐后的数据
- 训练 5 个集成模型（ResNet 架构）
- 应用 Isotonic 校准
- 生成可视化图表和预测结果

## 输入特征说明

模型使用 **17 个输入特征**（按原始 MATLAB MonteCarloMatrix 顺序）：

| 序号 | 特征名           | 说明                  |
|------|------------------|-----------------------|
| 1    | mass_kg          | 飞机质量（千克）      |
| 2    | CG_x             | 重心 X 坐标           |
| 3    | Altitude_IC_ft   | 初始高度（英尺）      |
| 4    | CAS              | 校准空速              |
| 5    | KCL              | 升力系数 K            |
| 6    | KCD              | 阻力系数 K            |
| 7    | KCY              | 侧向力系数 K          |
| 8    | KCI              | 滚转力矩系数 K        |
| 9    | KCM              | 俯仰力矩系数 K        |
| 10   | KCN              | 偏航力矩系数 K        |
| 11   | pressure_hPa     | 大气压强（百帕）      |
| 12   | temperature_oC   | 温度（摄氏度）        |
| 13   | Mean_wind_vel    | 平均风速              |
| 14   | Mean_wind_dir    | 平均风向              |
| 15   | cross_ang_deg    | 侧风角度（度）        |
| 16   | distance_AB_nm   | AB 距离（海里）       |
| 17   | distance_AD_nm   | AD 距离（海里）       |

## 输出变量说明

模型预测 **XDIS_TO_LTP_m**（第一列）作为目标变量，其他输出变量保留在数据文件中：

| 序号 | 变量名           | 说明                           | 是否作为目标 |
|------|------------------|--------------------------------|--------------|
| 1    | XDIS_TO_LTP_m    | 到着陆触地点的纵向距离（米）   | ✓ 是         |
| 2    | YDIS_TO_LTP_m    | 到着陆触地点的横向距离（米）   | 否           |
| 3    | PHI_deg          | 滚转角（度）                   | 否           |
| 4    | SSTP_deg         | 侧滑角（度）                   | 否           |
| 5    | VTZP_fms_m_s     | 垂直速度（米/秒）              | 否           |

## 输出文件说明

### 1. predictions.xlsx（预测结果）

包含 3 个 sheet：

- **Predictions**: 预测结果主表
  - Index: 原始数据索引
  - True_Value: 真实值
  - Predicted_Before_Calibration: 校准前预测值
  - Predicted_After_Calibration: 校准后预测值
  - Residual_Before: 校准前残差
  - Residual_After: 校准后残差
  - Dataset: Train/Val 标识

- **Input_Features**: 输入特征列名列表

- **Output_Columns**: 输出变量列名及是否作为目标

### 2. metrics_summary.xlsx（性能指标）

包含以下指标的校准前后对比：
- RMSE: 均方根误差
- MAE: 平均绝对误差
- R²: 决定系数
- Tail_MAE: 尾部平均绝对误差（5-95 百分位）
- Tail_RMSE: 尾部均方根误差
- Train_RMSE: 训练集 RMSE
- Val/Train_Ratio: 过拟合比率

### 3. improved_resnet_complete.png（可视化图表）

包含 6 张子图：
- Before/After Calibration: 预测值 vs 真实值散点图
- Isotonic Mapping: 校准映射曲线
- Residuals Before/After: 残差分布图
- Performance Metrics: 性能指标对比

## 模型特点

1. **架构**: ResNet + 线性捷径（直接从输入到输出）
2. **损失函数**: Focal-MSE（gamma=1.2）+ Pinball 辅助损失
3. **采样策略**: 加权随机采样（尾部权重增强）
4. **校准方法**: Isotonic Regression
5. **集成学习**: 5 个模型集成
6. **正则化**: 轻度 Dropout（≤0.1）+ weight_decay（1e-6）

## 依赖环境

```bash
pip install numpy pandas openpyxl matplotlib torch scikit-learn tqdm
```

## 常见问题

**Q: 如何使用新的数据？**

A: 将新的数据文件放入 `input/` 目录，命名为 `input_data-10k.xlsx` 和 `output_data-10k.xlsx`，然后先运行 `align_data.py` 进行对齐，再运行 `improved_complete_v2.py`。

**Q: 原始数据行数不匹配怎么办？**

A: `align_data.py` 会自动按行号匹配，只保留两边都有的行。

**Q: 如何更改目标变量？**

A: 修改 `improved_complete_v2.py` 中的 `X_output[:, 0]` 为其他列索引（0-4）。

**Q: 如何调整模型参数？**

A: 主要参数位于 `improved_complete_v2.py` 中：
- `epochs=300`: 训练轮数
- `gamma=1.2`: Focal-MSE 的 gamma 参数
- `n_models=5`: 集成模型数量
- 网络结构参数在 `ImprovedComplexNet` 类中

## 许可证

请根据项目需求添加相应许可证信息。
