import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
import math
import os
import glob
import seaborn as sns

# =========================================================
# 1. 模型定义 (保持不变)
# =========================================================
# ... (SinusoidalPosEmb, Conv1dBlock, ConditionalResidualBlock1D, ConditionalUnet1D, MinMaxNormalizer, Gamma functions)
# 为了节省篇幅，请保留你之前代码里的这一部分，或者直接 import 进来
# 务必确保定义和 inference_v0.py 一致

# ... (在此处粘贴之前的模型类定义，或者保持原样) ...
# 为了方便你直接复制，我把最简版本放这里占位
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1): super().__init__(); self.dim = dim; self.scale = scale 
    def forward(self, x): x=x*self.scale; return torch.cat((x.sin(), x.cos()), dim=-1) # (简化版占位，请用完整版)

class MinMaxNormalizer:
    def __init__(self, data=None): self.min=None; self.max=None; self.scale=None
    def normalize(self, x): return (x - self.min) / self.scale * 2 - 1
sys.modules['dataset'] = sys.modules[__name__]

EPS = 1e-6
def gamma_t_si(t): return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)
def d_gamma_dt_si(t): return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))

# =========================================================
# 2. INFERENCE LOG DATA
# =========================================================
# Inference Input (Step 0)
inf_pose_log = np.array([0.48327076, 0.04026159, 0.38554416, -0.9996992, -0.0049929, -0.00896409, -0.02227603])
inf_grip_log = 3.0 
inf_obj_log = np.array([0.58648731, 0.01768983, 0.09015887]) 

# Full Log Trajectory (For plotting)
log_traj_raw = np.array([
    [0.48327076, 0.04026159, 0.38554416, -0.9996992, -0.0049929, -0.00896409, -0.02227603, 3.0],
    [0.48108962, 0.03178788, 0.40048939, -1.0632324, 0.04099637, 0.03466385, -0.02490192, 4.17051244],
    [0.48139408, 0.02800756, 0.4082067, -1.06066193, 0.05364134, 0.04942676, -0.02263388, 4.42402768],
    [0.48198297, 0.02736998, 0.4137264, -1.04626752, 0.0498152, 0.05163207, -0.0171863, 4.3217926],
    [0.48254276, 0.03018797, 0.41526781, -1.04568126, 0.04596553, 0.05129013, -0.01348264, 4.1696434],
    [0.48362392, 0.03330099, 0.41579423, -1.05292895, 0.04473415, 0.05536716, -0.01125529, 4.11990094],
    [0.48486044, 0.03720651, 0.41566398, -1.05786734, 0.04395545, 0.06098466, -0.00885116, 4.08693123],
    [0.48617911, 0.04103875, 0.41544327, -1.06350067, 0.04380868, 0.06642694, -0.00548643, 4.21265531],
])

# =========================================================
# 3. 功能函数 (generate_chunk, load_all_csvs) 
# =========================================================
# ... 请保留之前的 generate_chunk 和 load_all_csvs 函数 ...
def load_all_csvs(data_dir="processed_data"):
    csv_files = glob.glob(os.path.join(data_dir, "train_data_trial_*.csv"))
    if not csv_files:
        print(f"❌ Error: No CSVs found in {data_dir}")
        return None
    print(f"Loading {len(csv_files)} training files...")
    df_list = []
    for f in csv_files:
        df_list.append(pd.read_csv(f))
    return pd.concat(df_list, ignore_index=True)

# =========================================================
# 4. 主程序 (修改了 OOD 可视化部分)
# =========================================================
def main():
    # ... (模型加载部分保持不变) ...
    # 为了演示 OOD 绘图逻辑，这里假设 df_train 已经加载
    df_train = load_all_csvs()
    if df_train is None: return

    # --- 4. Comprehensive OOD Visualization ---
    print("\n[Plotting] Generating full OOD report...")
    
    # Construct Full Inference Vector (11 dims)
    # [x, y, z, qx, qy, qz, qw, gripper, ox, oy, oz]
    inf_input_vec = np.concatenate([inf_pose_log, [inf_grip_log], inf_obj_log])
    
    # Define all 11 Dimensions to plot
    # Format: (Column Name, Label, Vector Index)
    dims_map = [
        ('ee_x', 'EE X', 0), ('ee_y', 'EE Y', 1), ('ee_z', 'EE Z', 2),
        ('ee_qx', 'Quat X', 3), ('ee_qy', 'Quat Y', 4), ('ee_qz', 'Quat Z', 5), ('ee_qw', 'Quat W', 6),
        ('gripper', 'Gripper', 7),
        ('obj_x', 'Obj X', 8), ('obj_y', 'Obj Y', 9), ('obj_z', 'Obj Z', 10)
    ]
    
    # 3行4列布局 (共12个格，最后一个空着)
    fig, axes = plt.subplots(3, 4, figsize=(24, 15))
    axes = axes.flatten()
    
    for i, (col_name, label, vec_idx) in enumerate(dims_map):
        ax = axes[i]
        
        # 1. Training Distribution (Gray)
        # 使用 KDE (核密度估计) 会更平滑，能看清概率分布
        sns.histplot(df_train[col_name], ax=ax, color='skyblue', alpha=0.5, kde=True, stat='density', label='Train Dist')
        
        # 2. Inference Input Value (Red Line)
        val = inf_input_vec[vec_idx]
        ax.axvline(val, color='red', linewidth=3, linestyle='--', label=f'Inf: {val:.3f}')
        
        # Check OOD visually & Set Title Color
        t_min, t_max = df_train[col_name].min(), df_train[col_name].max()
        # Add 5% tolerance
        range_span = t_max - t_min
        tol = range_span * 0.05 if range_span > 0 else 1e-3
        
        if val < t_min - tol or val > t_max + tol:
            ax.set_title(f"{label} [⚠️ OOD]", color='red', fontweight='bold', fontsize=12)
            # Add text annotation
            ax.text(0.5, 0.9, f"OOD by {min(abs(val-t_min), abs(val-t_max)):.3f}", 
                    transform=ax.transAxes, color='red', ha='center', fontweight='bold')
        else:
            ax.set_title(f"{label} [OK]", color='green', fontweight='bold', fontsize=12)
            
        ax.legend()

    # Hide unused subplot (11th plot is last used, hide 12th)
    for j in range(len(dims_map), len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.savefig("eval_ood_dist_full.png")
    print("  > Saved 'eval_ood_dist_full.png' (Full Input Distribution Check)")
    
    # ... (如果你还需要 Task B 的轨迹对比，保留那部分代码) ...

if __name__ == "__main__":
    main()