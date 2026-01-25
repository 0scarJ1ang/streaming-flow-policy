import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# =========================================================
# 1. 你的 Inference Log 数据 (Step 0 ~ 68)
# =========================================================
# 为了节省篇幅，我只放你 Log 的前 8 步和后 8 步，加上中间一些采样
# 建议你把完整的 Log 贴在这里，或者只用这部分代表性数据
log_data_raw = """
0.48327076 0.04026159 0.38554416 -0.9996992 -0.0049929 -0.00896409 -0.02227603 3.0
0.48108962 0.03178788 0.40048939 -1.0632324 0.04099637 0.03466385 -0.02490192 4.17051244
0.48139408 0.02800756 0.4082067 -1.06066193 0.05364134 0.04942676 -0.02263388 4.42402768
0.48198297 0.02736998 0.4137264 -1.04626752 0.0498152 0.05163207 -0.0171863 4.3217926
0.48254276 0.03018797 0.41526781 -1.04568126 0.04596553 0.05129013 -0.01348264 4.1696434
0.48362392 0.03330099 0.41579423 -1.05292895 0.04473415 0.05536716 -0.01125529 4.11990094
0.48486044 0.03720651 0.41566398 -1.05786734 0.04395545 0.06098466 -0.00885116 4.08693123
0.48617911 0.04103875 0.41544327 -1.06350067 0.04380868 0.06642694 -0.00548643 4.21265531
0.41031644 -0.00442759 0.47454457 -1.03913952 -0.04582414 0.16355316 0.03360422 3.12087822
0.41319008 0.00014869 0.47589304 -1.03666672 -0.04921542 0.16145063 0.03880091 3.16293168
"""

# 解析 Log 数据
log_rows = [list(map(float, line.split())) for line in log_data_raw.strip().split('\n')]
log_arr = np.array(log_rows)
# Shape: (N, 8) -> [x, y, z, qx, qy, qz, qw, gripper]

# =========================================================
# 2. 加载 Training Data
# =========================================================
data_dir = "processed_data"
csv_files = glob.glob(os.path.join(data_dir, "train_data_trial_*.csv"))

if not csv_files:
    print(f"Error: No CSV files found in {data_dir}")
    exit()

print(f"Loading {len(csv_files)} training files...")

train_ee_x = []
train_ee_y = []
train_ee_z = []
train_gripper = []
train_qx = []

for f in csv_files:
    df = pd.read_csv(f)
    train_ee_x.extend(df['ee_x'].values)
    train_ee_y.extend(df['ee_y'].values)
    train_ee_z.extend(df['ee_z'].values)
    train_gripper.extend(df['gripper'].values)
    train_qx.extend(df['ee_qx'].values)

# =========================================================
# 3. 对比分析 (Visualization)
# =========================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# --- Plot 1: X-Y Plane Trajectory ---
ax = axes[0, 0]
# 1. 画出所有训练数据的点（用灰色，高透明度，形成背景热力图效果）
ax.scatter(train_ee_x, train_ee_y, c='gray', alpha=0.05, s=1, label='Training Data')
# 2. 画出 Inference Log 的轨迹（红色，醒目）
ax.plot(log_arr[:, 0], log_arr[:, 1], c='red', linewidth=2, marker='x', label='Inference Log')
ax.set_title("Robot Trajectory (X-Y Plane)")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend()
ax.grid(True)

# --- Plot 2: Z Height Distribution ---
ax = axes[0, 1]
ax.hist(train_ee_z, bins=50, color='gray', alpha=0.5, density=True, label='Training Z Hist')
# 画出 Inference Z 的范围
z_min_log = log_arr[:, 2].min()
z_max_log = log_arr[:, 2].max()
ax.axvline(z_min_log, color='red', linestyle='--', label=f'Log Min Z ({z_min_log:.3f})')
ax.axvline(z_max_log, color='red', linestyle='-', label=f'Log Max Z ({z_max_log:.3f})')
ax.set_title("Z Height Distribution")
ax.legend()

# --- Plot 3: Gripper Value Distribution ---
ax = axes[1, 0]
ax.hist(train_gripper, bins=50, color='gray', alpha=0.5, density=True, label='Training Gripper')
# 画出 Inference Gripper
g_min_log = log_arr[:, 7].min()
g_max_log = log_arr[:, 7].max()
ax.axvline(g_min_log, color='red', linestyle='--', label=f'Log Min G ({g_min_log:.1f})')
ax.axvline(g_max_log, color='red', linestyle='-', label=f'Log Max G ({g_max_log:.1f})')
ax.set_title("Gripper Value Distribution")
ax.legend()

# --- Plot 4: Quaternion Qx Distribution ---
ax = axes[1, 1]
ax.hist(train_qx, bins=50, color='gray', alpha=0.5, density=True, label='Training Qx')
q_min_log = log_arr[:, 3].min()
q_max_log = log_arr[:, 3].max()
ax.axvline(q_min_log, color='red', linestyle='--', label=f'Log Min Qx ({q_min_log:.3f})')
ax.axvline(q_max_log, color='red', linestyle='-', label=f'Log Max Qx ({q_max_log:.3f})')
ax.set_title("Orientation (Qx) Distribution")
ax.legend()

plt.tight_layout()
plt.savefig("compare_result.png")
print("\n✅ 对比完成！请查看 'compare_result.png'")

# =========================================================
# 4. 数值统计检查 (Text Report)
# =========================================================
print("\n=== OOD (Out-Of-Distribution) Check ===")

def check_ood(name, train_data, log_vals):
    t_min, t_max = np.min(train_data), np.max(train_data)
    l_min, l_max = np.min(log_vals), np.max(log_vals)
    
    print(f"[{name}]")
    print(f"  Train Range: [{t_min:.4f}, {t_max:.4f}]")
    print(f"  Log   Range: [{l_min:.4f}, {l_max:.4f}]")
    
    if l_min < t_min - 0.05 or l_max > t_max + 0.05: # 留一点 buffer
        print(f"  ⚠️ ALERT: Potential OOD! Log data is outside training range.")
    else:
        print(f"  OK.")

check_ood("X Position", train_ee_x, log_arr[:, 0])
check_ood("Y Position", train_ee_y, log_arr[:, 1])
check_ood("Z Position", train_ee_z, log_arr[:, 2])
check_ood("Gripper", train_gripper, log_arr[:, 7])
check_ood("Quat X", train_qx, log_arr[:, 3])