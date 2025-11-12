#!/usr/bin/env python3
"""
完整优化版：ResNet + Focal-MSE + 线性捷径 + 分位辅助头 + Isotonic 校准

按纲领实施：
- 诊断与指标：双指标（总体+尾部）、三张图
- 损失函数：Focal-MSE + Pinball 辅助
- 采样校准：WeightedRandomSampler + Isotonic
- 结构：线性捷径到输出，减 BN 收缩
- 正则化：轻度正则，归一化调整
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from tqdm import tqdm
from sklearn.isotonic import IsotonicRegression

print("""
╔════════════════════════════════════════════════════════════════════╗
║     完整优化版：按纲领改进的 ResNet + 线性捷径 + 分位辅助头      ║
╚════════════════════════════════════════════════════════════════════╝
""")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

torch.manual_seed(42)
np.random.seed(42)

# ============================================================================
# 核心改进 1: 稳健的 Focal-MSE
# ============================================================================

class FocalMSE(nn.Module):
    """Focal-MSE: 平衡大小误差的加权损失"""
    def __init__(self, gamma=1.2, c='median', momentum=0.95):
        super().__init__()
        self.gamma = gamma
        self.c = c
        self.momentum = momentum
        self.register_buffer('c_ema', torch.tensor(1.0))
    
    def forward(self, y_pred, y_true):
        e = torch.abs(y_pred - y_true)
        
        if isinstance(self.c, str) and self.c == 'median':
            c_batch = torch.median(e.detach()) + 1e-8
            self.c_ema.copy_(self.momentum * self.c_ema + (1 - self.momentum) * c_batch)
            c = self.c_ema.detach()
        else:
            c = torch.as_tensor(self.c, dtype=y_pred.dtype, device=y_pred.device)
        
        w = torch.clamp(e / c, min=0.1, max=10.0) ** self.gamma
        mse = (y_pred - y_true) ** 2
        
        return torch.mean(w * mse)


class PinballLoss(nn.Module):
    """分位数回归损失（用于辅助头）"""
    def __init__(self, taus=(0.05, 0.5, 0.95)):
        super().__init__()
        self.taus = taus
    
    def forward(self, y_pred, y_true):
        """y_pred: (B, n_taus), y_true: (B, 1)"""
        loss = 0.0
        for j, tau in enumerate(self.taus):
            e = y_true - y_pred[:, j:j+1]
            loss += torch.mean(torch.maximum(tau * e, (tau - 1) * e))
        return loss / len(self.taus)


#%% 加载数据
print("【Step 1】Loading Data...")

# 创建输出目录
os.makedirs('out', exist_ok=True)

# 从 input 目录读取对齐后的 xlsx 文件
input_file = 'input/input_aligned.xlsx'
output_file = 'input/output_aligned.xlsx'

print(f"Loading input data from: {input_file}")
print(f"Loading output data from: {output_file}")

# 读取对齐后的 xlsx 文件（已经过预处理，无需额外处理）
df_input = pd.read_excel(input_file)
df_output = pd.read_excel(output_file)

print(f"\n数据形状:")
print(f"  Input: {df_input.shape} (10000 行 × 17 列)")
print(f"  Output: {df_output.shape} (10000 行 × 5 列)")

print(f"\n输入特征 (17 列):")
for i, col in enumerate(df_input.columns, 1):
    print(f"  [{i:2d}] {col}")

print(f"\n输出变量 (5 列，使用第 1 列作为目标):")
for i, col in enumerate(df_output.columns, 1):
    target_mark = " ← 目标" if i == 1 else ""
    print(f"  [{i}] {col}{target_mark}")

# 转换为 numpy 数组
X_input = df_input.values
X_output = df_output.values

print(f"\n最终数据形状:")
print(f"  X_input: {X_input.shape}")
print(f"  X_output: {X_output.shape}")

np.random.seed(42)
perm = np.random.permutation(len(X_input))
split_idx = int(0.8 * len(X_input))
train_idx = perm[:split_idx]
val_idx = perm[split_idx:]

X_mean = X_input.mean(axis=0)
X_std = X_input.std(axis=0) + 1e-8
X_input_norm = (X_input - X_mean) / X_std

y_mean = X_output[:, 0].mean()
y_std = X_output[:, 0].std() + 1e-8

X_train = torch.FloatTensor(X_input_norm[train_idx])
X_val = torch.FloatTensor(X_input_norm[val_idx])
y_train = torch.FloatTensor((X_output[train_idx, 0] - y_mean) / y_std).reshape(-1, 1)
y_val = torch.FloatTensor((X_output[val_idx, 0] - y_mean) / y_std).reshape(-1, 1)

y_train_true = X_output[train_idx, 0]
y_val_true = X_output[val_idx, 0]

print(f"Train shape: {X_train.shape}, {y_train.shape}")
print(f"Val shape: {X_val.shape}, {y_val.shape}\n")

#%% 改进的网络架构：线性捷径 + 分位辅助头
class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)  # 轻度正则化
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
        
        # ← 新增：线性捷径（从输入直接到输出）
        self.linear_shortcut = nn.Linear(input_dim, 1)
        
        # ← 新增：分位数辅助头（0.05, 0.5, 0.95）
        self.output_quant = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(32, 3)  # 三个分位数
        )
    
    def forward(self, x):
        x_orig = x  # 保存原始输入用于线性捷径
        
        x = self.input_proj(x)
        x = self.res_blocks_1(x)
        x = self.transition_1(x)
        x = self.res_blocks_2(x)
        x = self.transition_2(x)
        
        # 点估计 + 线性捷径
        point = self.output_point(x)
        shortcut = self.linear_shortcut(x_orig)
        point = point + 0.1 * shortcut  # 线性捷径的权重 0.1
        
        # 分位数
        quant = self.output_quant(x)
        
        return point, quant


#%% 改进的训练函数
def train_model_improved(model, train_loader, val_loader, 
                        y_train_true, y_val_true, y_mean, y_std,
                        epochs=300, device='cpu', gamma=1.2, use_pinball=True):
    """改进的训练函数"""
    
    optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2, eta_min=1e-7
    )
    
    criterion_point = FocalMSE(gamma=gamma, c='median')
    criterion_quant = PinballLoss(taus=(0.05, 0.5, 0.95)) if use_pinball else None
    
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    best_state = None
    
    train_losses = []
    val_losses = []
    tail_rmses = []
    
    pbar = tqdm(range(epochs), desc='Training')
    
    for epoch in pbar:
        # 训练
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            point, quant = model(X_batch)
            
            # 点估计损失 + 分位辅助头损失
            loss = criterion_point(point, y_batch)
            if use_pinball and criterion_quant is not None:
                loss = loss + 0.3 * criterion_quant(quant, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.detach().item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                point, quant = model(X_batch)
                
                loss = criterion_point(point, y_batch)
                if use_pinball and criterion_quant is not None:
                    loss = loss + 0.3 * criterion_quant(quant, y_batch)
                
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        scheduler.step()
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            pbar.update(epochs - epoch - 1)
            break
        
        pbar.set_postfix({
            'train': f'{train_loss:.5f}',
            'val': f'{val_loss:.5f}',
        })
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return train_losses, val_losses


#%% 训练多个模型进行集成
print("【Step 2】Training Ensemble Models (Improved)...\n")

all_models = []
all_train_preds = []
all_val_preds = []
n_models = 5

for model_idx in range(n_models):
    print(f"Model {model_idx + 1}/{n_models}:")
    
    torch.manual_seed(42 + model_idx)
    np.random.seed(42 + model_idx)
    
    # ← 改进：尾部采样但不过度（权重倍数 2-3 倍，不超过 5）
    p_lo, p_hi = np.percentile(y_train_true, [5, 95])
    y_min, y_max = y_train_true.min(), y_train_true.max()
    
    w_lo = np.clip((p_lo - y_train_true) / (p_lo - y_min + 1e-6), 0, 1)
    w_hi = np.clip((y_train_true - p_hi) / (y_max - p_hi + 1e-6), 0, 1)
    weights = 1.0 + 3.0 * (w_lo + w_hi)  # 权重倍数改成 3（不超过 5）
    weights = weights / weights.sum() * len(weights)
    
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    
    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=64,
        sampler=sampler
    )
    
    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=256,
        shuffle=False
    )

    # 使用实际的输入维度
    input_dim = X_train.shape[1]
    model = ImprovedComplexNet(input_dim).to(device)
    train_losses, val_losses = train_model_improved(
        model, train_loader, val_loader,
        y_train_true, y_val_true, y_mean, y_std,
        epochs=300, device=device, gamma=1.2, use_pinball=True
    )
    
    all_models.append(model)
    
    # 预测（只取点估计）
    model.eval()
    with torch.no_grad():
        y_train_point, _ = model(X_train.to(device))
        y_val_point, _ = model(X_val.to(device))
        y_train_pred_norm = y_train_point.cpu().numpy()
        y_val_pred_norm = y_val_point.cpu().numpy()
    
    y_train_pred = y_train_pred_norm * y_std + y_mean
    y_val_pred = y_val_pred_norm * y_std + y_mean
    
    all_train_preds.append(y_train_pred)
    all_val_preds.append(y_val_pred)
    
    print()

# 集成预测
y_train_pred = np.mean(all_train_preds, axis=0).squeeze()
y_val_pred = np.mean(all_val_preds, axis=0).squeeze()

print(f"✓ Ensemble prediction shapes:")
print(f"  Train: {y_train_pred.shape}")
print(f"  Val: {y_val_pred.shape}\n")

# ============================================================================
# 核心改进 2: Isotonic 校准
# ============================================================================
print("【Step 2.5】Applying Isotonic Calibration...")
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(y_train_pred, y_train_true)

y_train_pred_cal = iso.predict(y_train_pred)
y_val_pred_cal = iso.predict(y_val_pred)

print("✓ Isotonic calibration applied\n")

#%% 诊断指标计算
def compute_tail_metrics(y_true, y_pred, percentile_lo=5, percentile_hi=95):
    """计算尾部性能指标"""
    p_lo = np.percentile(y_true, percentile_lo)
    p_hi = np.percentile(y_true, percentile_hi)
    in_tail = (y_true <= p_lo) | (y_true >= p_hi)
    
    tail_mae = np.mean(np.abs(y_true[in_tail] - y_pred[in_tail])) if in_tail.sum() > 0 else 0
    tail_rmse = np.sqrt(np.mean((y_true[in_tail] - y_pred[in_tail]) ** 2)) if in_tail.sum() > 0 else 0
    
    return tail_mae, tail_rmse, in_tail


#%% 画图（改进：6 张图 + 诊断指标）
print("【Step 3】Generating Visualizations & Diagnostics...\n")

fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle(f'Improved ResNet: Linear Shortcut + Quantile Head + Isotonic Calibration', 
             fontsize=15, fontweight='bold')

# 计算尾部指标
tail_mae_before, tail_rmse_before, in_tail = compute_tail_metrics(y_val_true, y_val_pred)
tail_mae_after, tail_rmse_after, _ = compute_tail_metrics(y_val_true, y_val_pred_cal)

# 1. 改进前：Pred vs True
ax = axes[0, 0]
ax.scatter(y_train_true, y_train_pred, alpha=0.5, s=30, label='Train (Before)', 
           color='blue', edgecolors='k', linewidth=0.3)
ax.scatter(y_val_true, y_val_pred, alpha=0.5, s=30, label='Val (Before)', 
           color='red', edgecolors='k', linewidth=0.3)

all_true = np.concatenate([y_train_true, y_val_true])
all_pred = np.concatenate([y_train_pred, y_val_pred])
min_v = min(all_true.min(), all_pred.min())
max_v = max(all_true.max(), all_pred.max())
ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=2, alpha=0.5, label='Perfect')

ax.set_xlabel('True Value', fontweight='bold', fontsize=11)
ax.set_ylabel('Predicted Value', fontweight='bold', fontsize=11)
ax.set_title('Before Calibration (压扁?)', fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 2. 改进后：Pred vs True
ax = axes[0, 1]
ax.scatter(y_train_true, y_train_pred_cal, alpha=0.5, s=30, label='Train (After)', 
           color='blue', edgecolors='k', linewidth=0.3)
ax.scatter(y_val_true, y_val_pred_cal, alpha=0.5, s=30, label='Val (After)', 
           color='red', edgecolors='k', linewidth=0.3)

all_pred_cal = np.concatenate([y_train_pred_cal, y_val_pred_cal])
min_v_cal = min(all_true.min(), all_pred_cal.min())
max_v_cal = max(all_true.max(), all_pred_cal.max())
ax.plot([min_v_cal, max_v_cal], [min_v_cal, max_v_cal], 'k--', lw=2, alpha=0.5, label='Perfect')

ax.set_xlabel('True Value', fontweight='bold', fontsize=11)
ax.set_ylabel('Predicted Value', fontweight='bold', fontsize=11)
ax.set_title('After Isotonic ✓ (拉开了?)', fontweight='bold', fontsize=11, color='green')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 3. Isotonic 曲线
ax = axes[0, 2]
sorted_indices = np.argsort(y_train_pred)
x_sorted = y_train_pred[sorted_indices]
y_iso_sorted = iso.predict(x_sorted)

ax.plot(x_sorted, y_iso_sorted, 'b-', lw=2.5, label='Isotonic')
ax.plot([x_sorted.min(), x_sorted.max()], [x_sorted.min(), x_sorted.max()], 
        'k--', alpha=0.5, lw=1.5, label='y=x')

ax.set_xlabel('Pred (Before)', fontweight='bold', fontsize=11)
ax.set_ylabel('Pred (After)', fontweight='bold', fontsize=11)
ax.set_title('Isotonic Mapping (斜率接近 1?)', fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 4. 残差图（改进前）
ax = axes[1, 0]
train_res = y_train_true - y_train_pred
val_res = y_val_true - y_val_pred

ax.scatter(y_train_pred, train_res, alpha=0.5, s=30, label='Train', 
           color='blue', edgecolors='k', linewidth=0.3)
ax.scatter(y_val_pred, val_res, alpha=0.5, s=30, label='Val', 
           color='red', edgecolors='k', linewidth=0.3)
ax.axhline(y=0, color='k', linestyle='--', lw=1.5, alpha=0.5)

ax.set_xlabel('Predicted Value', fontweight='bold', fontsize=11)
ax.set_ylabel('Residual', fontweight='bold', fontsize=11)
ax.set_title('Residuals (Before)', fontweight='bold', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 5. 残差图（改进后）
ax = axes[1, 1]
train_res_cal = y_train_true - y_train_pred_cal
val_res_cal = y_val_true - y_val_pred_cal

ax.scatter(y_train_pred_cal, train_res_cal, alpha=0.5, s=30, label='Train', 
           color='blue', edgecolors='k', linewidth=0.3)
ax.scatter(y_val_pred_cal, val_res_cal, alpha=0.5, s=30, label='Val', 
           color='red', edgecolors='k', linewidth=0.3)
ax.axhline(y=0, color='k', linestyle='--', lw=1.5, alpha=0.5)

ax.set_xlabel('Predicted Value', fontweight='bold', fontsize=11)
ax.set_ylabel('Residual', fontweight='bold', fontsize=11)
ax.set_title('Residuals (After) ✓', fontweight='bold', fontsize=11, color='green')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# 6. 性能对比
ax = axes[1, 2]

metrics_names = ['RMSE\n(Before)', 'RMSE\n(After)', 'Tail RMSE\n(Before)', 'Tail RMSE\n(After)']
rmse_before = np.sqrt(np.mean((y_val_true - y_val_pred) ** 2))
rmse_after = np.sqrt(np.mean((y_val_true - y_val_pred_cal) ** 2))

values = [rmse_before, rmse_after, tail_rmse_before, tail_rmse_after]
colors = ['lightcoral', 'lightgreen', 'lightyellow', 'lightblue']

bars = ax.bar(metrics_names, values, color=colors, edgecolor='k', linewidth=1.5)
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.0f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Error', fontweight='bold', fontsize=11)
ax.set_title('Performance Metrics', fontweight='bold', fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_plot_file = 'out/improved_resnet_complete.png'
plt.savefig(output_plot_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_plot_file}\n")
plt.show()

#%% 诊断与验收标准
print("\n" + "="*70)
print(f"【诊断与验收标准】(按纲领第八点)")
print("="*70 + "\n")

# 总体指标
rmse_val_before = np.sqrt(np.mean((y_val_true - y_val_pred) ** 2))
mae_val_before = np.mean(np.abs(y_val_true - y_val_pred))
r2_val_before = 1 - np.sum((y_val_true - y_val_pred) ** 2) / np.sum((y_val_true - y_val_true.mean()) ** 2)

rmse_val_after = np.sqrt(np.mean((y_val_true - y_val_pred_cal) ** 2))
mae_val_after = np.mean(np.abs(y_val_true - y_val_pred_cal))
r2_val_after = 1 - np.sum((y_val_true - y_val_pred_cal) ** 2) / np.sum((y_val_true - y_val_true.mean()) ** 2)

print(f"【总体性能指标】")
print(f"  RMSE: {rmse_val_before:.2f} → {rmse_val_after:.2f} (Δ {(rmse_val_after/rmse_val_before-1)*100:+.1f}%)")
print(f"  MAE:  {mae_val_before:.2f} → {mae_val_after:.2f} (Δ {(mae_val_after/mae_val_before-1)*100:+.1f}%)")
print(f"  R²:   {r2_val_before:.4f} → {r2_val_after:.4f}\n")

print(f"【尾部性能指标】(5-95 percentile, {in_tail.sum()} samples)")
print(f"  Tail-MAE:  {tail_mae_before:.2f} → {tail_mae_after:.2f} (Δ {(tail_mae_after/tail_mae_before-1)*100:+.1f}%)")
print(f"  Tail-RMSE: {tail_rmse_before:.2f} → {tail_rmse_after:.2f} (Δ {(tail_rmse_after/tail_rmse_before-1)*100:+.1f}%)\n")

# 过拟合判断
rmse_train_after = np.sqrt(np.mean((y_train_true - y_train_pred_cal) ** 2))
overfitting_ratio = rmse_val_after / rmse_train_after
print(f"【过拟合判断】")
print(f"  RMSE_val / RMSE_train = {overfitting_ratio:.3f}")
if overfitting_ratio <= 1.3:
    print(f"  ✓ 过拟合控制良好\n")
else:
    print(f"  ⚠️  有过拟合迹象，建议降低 gamma 或加强正则化\n")

# 校准效果（Theil 不等式）
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(y_val_pred_cal.reshape(-1, 1), y_val_true.reshape(-1, 1))
slope = lr.coef_[0, 0]
intercept = lr.intercept_[0]

print(f"【校准效果】(Pred→True 线性回归)")
print(f"  斜率: {slope:.4f} (理想≈1.0)")
print(f"  截距: {intercept:.4f} (理想≈0.0)")
if 0.95 <= slope <= 1.05 and abs(intercept) < 0.1:
    print(f"  ✓ 校准效果良好\n")
else:
    print(f"  △ 校准仍需改进\n")

print("\n✅ Done!")
print("\n【改进总结】")
print("✓ 线性捷径：直接从输入到输出，给模型全局斜率/截距自由度")
print("✓ 分位辅助头：0.05, 0.5, 0.95 分位数辅助学习，特别稳定极端值")
print("✓ 改进的 Focal-MSE：gamma=1.2（稳健），内置 EMA 平滑")
print("✓ 尾部采样：权重倍数 3（不过度）")
print("✓ Isotonic 校准：修正系统性偏差")
print("✓ 轻度正则化：Dropout≤0.1, weight_decay=1e-6")

#%% 保存预测结果到 xlsx 文件
print("\n【Step 4】Saving Predictions to Excel...")

# 合并训练集和验证集的结果
all_indices = np.concatenate([train_idx, val_idx])
all_true_values = np.concatenate([y_train_true, y_val_true])
all_pred_before = np.concatenate([y_train_pred, y_val_pred])
all_pred_after = np.concatenate([y_train_pred_cal, y_val_pred_cal])

# 创建结果 DataFrame
results_df = pd.DataFrame({
    'Index': all_indices,
    'True_Value': all_true_values,
    'Predicted_Before_Calibration': all_pred_before,
    'Predicted_After_Calibration': all_pred_after,
    'Residual_Before': all_true_values - all_pred_before,
    'Residual_After': all_true_values - all_pred_after,
    'Dataset': ['Train'] * len(train_idx) + ['Val'] * len(val_idx)
})

# 按原始索引排序
results_df = results_df.sort_values('Index').reset_index(drop=True)

# 保存到 xlsx 文件（使用 ExcelWriter 保存多个 sheet）
output_predictions_file = 'out/predictions.xlsx'
with pd.ExcelWriter(output_predictions_file, engine='openpyxl') as writer:
    # Sheet 1: 预测结果
    results_df.to_excel(writer, sheet_name='Predictions', index=False)

    # Sheet 2: 使用的输入特征列名
    feature_info = pd.DataFrame({
        'Feature_Index': range(len(df_input.columns)),
        'Feature_Name': df_input.columns.tolist()
    })
    feature_info.to_excel(writer, sheet_name='Input_Features', index=False)

    # Sheet 3: 输出列信息
    output_info = pd.DataFrame({
        'Output_Index': range(len(df_output.columns)),
        'Output_Name': df_output.columns.tolist(),
        'Used_as_Target': ['Yes' if i == 0 else 'No' for i in range(len(df_output.columns))]
    })
    output_info.to_excel(writer, sheet_name='Output_Columns', index=False)

print(f"✓ Saved predictions to: {output_predictions_file}")
print(f"  - Sheet 'Predictions': 预测结果")
print(f"  - Sheet 'Input_Features': 输入特征列名")
print(f"  - Sheet 'Output_Columns': 输出列信息")

# 保存性能指标摘要
metrics_df = pd.DataFrame({
    'Metric': ['RMSE', 'MAE', 'R²', 'Tail_MAE', 'Tail_RMSE', 'Train_RMSE', 'Val/Train_Ratio'],
    'Before_Calibration': [
        rmse_val_before,
        mae_val_before,
        r2_val_before,
        tail_mae_before,
        tail_rmse_before,
        np.sqrt(np.mean((y_train_true - y_train_pred) ** 2)),
        '-'
    ],
    'After_Calibration': [
        rmse_val_after,
        mae_val_after,
        r2_val_after,
        tail_mae_after,
        tail_rmse_after,
        rmse_train_after,
        f'{overfitting_ratio:.3f}'
    ]
})

output_metrics_file = 'out/metrics_summary.xlsx'
metrics_df.to_excel(output_metrics_file, index=False)
print(f"✓ Saved metrics summary to: {output_metrics_file}")

print("\n✅ All outputs saved to 'out/' directory!")

#%% 保存模型和参数供推理使用
print("\n【Step 5】Saving Models and Parameters for Inference...")

import pickle

# 保存所有集成模型
for i, model in enumerate(all_models):
    model_file = f'out/model_{i}.pth'
    torch.save(model.state_dict(), model_file)
    print(f"✓ Saved model {i+1}/5 to: {model_file}")

# 保存标准化参数
normalization_params = {
    'X_mean': X_mean,
    'X_std': X_std,
    'y_mean': y_mean,
    'y_std': y_std,
    'input_dim': X_train.shape[1]
}

norm_file = 'out/normalization_params.pkl'
with open(norm_file, 'wb') as f:
    pickle.dump(normalization_params, f)
print(f"✓ Saved normalization parameters to: {norm_file}")

# 保存Isotonic校准模型
iso_file = 'out/isotonic_model.pkl'
with open(iso_file, 'wb') as f:
    pickle.dump(iso, f)
print(f"✓ Saved Isotonic calibration model to: {iso_file}")

print("\n✅ All models and parameters saved for inference!")
print("\nYou can now use 'predict_monte_carlo.py' to make predictions on new data.")