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
# 1. MODEL DEFINITION (Must match exactly)
# =========================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1):
        super().__init__(); self.dim = dim; self.scale = scale 
    def forward(self, x):
        x = x * self.scale; device = x.device; half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2), nn.GroupNorm(n_groups, out_channels), nn.Mish())
    def forward(self, x): return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(in_channels, out_channels, kernel_size, n_groups), Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)])
        cond_channels = out_channels * 2; self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1)))
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x, cond):
        out = self.blocks[0](x); embed = self.cond_encoder(cond); embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:,0,...], embed[:,1,...]; out = scale * out + bias
        out = self.blocks[1](out); out = out + self.residual_conv(x); return out

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, updownsample_type='Linear', sin_embedding_scale=100, diffusion_step_embed_dim=256, down_dims=[256,512,1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims); start_dim = down_dims[0]; dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(SinusoidalPosEmb(dsed, scale = sin_embedding_scale), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed))
        cond_dim = dsed + global_cond_dim; in_out = list(zip(all_dims[:-1], all_dims[1:])); mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups), ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups)])
        self.down_modules = nn.ModuleList([]); self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            class LinearDownsample1d(nn.Module): 
                def __init__(self, dim): super().__init__(); self.linear = nn.Linear(dim, dim)
                def forward(self, x): return self.linear(x.transpose(1,2)).transpose(1,2)
            downsample_layer = LinearDownsample1d(dim_out) if not is_last else nn.Identity()
            self.down_modules.append(nn.ModuleList([ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups), ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups), downsample_layer]))
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            class LinearUpsample1d(nn.Module): 
                def __init__(self, dim): super().__init__(); self.linear = nn.Linear(dim, dim)
                def forward(self, x): return self.linear(x.transpose(1,2)).transpose(1,2)
            upsample_layer = LinearUpsample1d(dim_in) if not is_last else nn.Identity()
            self.up_modules.append(nn.ModuleList([ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim, kernel_size, n_groups), ConditionalResidualBlock1D(dim_in, dim_in, cond_dim, kernel_size, n_groups), upsample_layer]))
        self.final_conv = nn.Sequential(Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size), nn.Conv1d(start_dim, input_dim, 1))
    def forward(self, sample, timestep, global_cond=None):
        sample = sample.moveaxis(-1,-2); timesteps = timestep
        if not torch.is_tensor(timesteps): timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0: timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0]); global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None: global_feature = torch.cat([global_feature, global_cond], axis=-1)
        x = sample; h = []
        for resnet, resnet2, downsample in self.down_modules: x = resnet(x, global_feature); x = resnet2(x, global_feature); h.append(x); x = downsample(x)
        for mid_module in self.mid_modules: x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules: x = torch.cat((x, h.pop()), dim=1); x = resnet(x, global_feature); x = resnet2(x, global_feature); x = upsample(x)
        x = self.final_conv(x); x = x.moveaxis(-1,-2); return x

class MinMaxNormalizer:
    def __init__(self, data=None): self.min=None; self.max=None; self.scale=None
    def normalize(self, x): return (x - self.min) / self.scale * 2 - 1
sys.modules['dataset'] = sys.modules[__name__]

EPS = 1e-6
def gamma_t_si(t): return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)
def d_gamma_dt_si(t): return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))

# =========================================================
# 2. DATA LOADING & GENERATION HELPERS
# =========================================================
inf_pose_log = np.array([0.48327076, 0.04026159, 0.38554416, -0.9996992, -0.0049929, -0.00896409, -0.02227603])
inf_grip_log = 3.0 
inf_obj_log = np.array([0.58648731, 0.01768983, 0.09015887]) 

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

def generate_chunk(current_pose, current_gripper, obj_pos, model_components, device, steps=8):
    velocity_net, denoiser_net, normalizer = model_components
    # Construct Obs: [Pose(7), Grip(1), Obj(3)]
    raw_obs = np.concatenate([current_pose, [current_gripper], obj_pos]).astype(np.float32)
    raw_obs = raw_obs.reshape(1, -1) 
    
    nobs_np = normalizer.normalize(raw_obs)
    nobs = torch.from_numpy(nobs_np).float().to(device)
    
    # Warm Start: from current state (first 8 dims)
    na = nobs[:, :8].unsqueeze(1) 
    
    pred_actions = []
    dt_val = 1.0 / (16 - 1)
    
    with torch.no_grad():
        for i in range(steps):
            # Decode
            na_cpu = na.detach().cpu().numpy().squeeze()
            act_min = normalizer.min[:8]; act_scale = normalizer.scale[:8]
            raw_action = ((na_cpu + 1) / 2) * act_scale + act_min
            pred_actions.append(raw_action)
            
            # Integrate
            t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
            t = torch.tensor([t_scalar], device=device, dtype=torch.float32)
            
            v_pred = velocity_net(sample=na, timestep=t, global_cond=nobs)
            eta_pred = denoiser_net(sample=na, timestep=t, global_cond=nobs)
            gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
            b_drift = v_pred + gamma_dot * eta_pred
            na = na + b_drift * dt_val
            
    return np.array(pred_actions)

def load_all_csvs(data_dir="processed_data"):
    csv_files = glob.glob(os.path.join(data_dir, "train_data_trial_*.csv"))
    if not csv_files: return None
    print(f"Loading {len(csv_files)} training files...")
    df_list = []
    for f in csv_files:
        df_list.append(pd.read_csv(f))
    return pd.concat(df_list, ignore_index=True)

# =========================================================
# 4. MAIN VISUALIZATION
# =========================================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 1. Load Model
    ckpt_path = "models/real_robot_epoch_700.ckpt"
    print("Loading checkpoint...")
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    v_net = ConditionalUnet1D(input_dim=8, global_cond_dim=11).to(device)
    d_net = ConditionalUnet1D(input_dim=8, global_cond_dim=11).to(device)
    v_net.load_state_dict(checkpoint['velocity_net']); v_net.eval()
    d_net.load_state_dict(checkpoint['denoiser_net']); d_net.eval()
    normalizer = checkpoint['normalizer']
    components = (v_net, d_net, normalizer)
    
    df_train = load_all_csvs()
    if df_train is None: return

    # 2. Config
    num_train_samples = 4 # How many training samples to visualize
    total_rows = 1 + num_train_samples # 1 Inference + N Training
    
    fig, axes = plt.subplots(total_rows, 4, figsize=(20, 4 * total_rows))
    # Columns: [XY Path, Z-Time, Gripper-Time, QuatX-Time]
    
    print("\n[Processing] Running Comparisons...")
    
    # --- ROW 0: Inference Log Case ---
    print("  > Processing Inference Log Case...")
    pred_log = generate_chunk(inf_pose_log, inf_grip_log, inf_obj_log, components, device)
    
    # Col 0: XY Path
    ax = axes[0, 0]
    ax.plot(log_traj_raw[:, 0], log_traj_raw[:, 1], 'r--x', label='Log (GT-ish)', linewidth=2)
    ax.plot(pred_log[:, 0], pred_log[:, 1], 'b-o', label='Model Repro', linewidth=2)
    ax.set_title("Inference Case: XY Path")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(); ax.grid(True)
    
    # Col 1: Z Profile
    ax = axes[0, 1]
    ax.plot(log_traj_raw[:, 2], 'r--', label='Log Z')
    ax.plot(pred_log[:, 2], 'b-', label='Pred Z')
    ax.set_title("Inference Case: Z vs Time")
    ax.legend(); ax.grid(True)
    
    # Col 2: Gripper Profile
    ax = axes[0, 2]
    ax.plot(log_traj_raw[:, 7], 'r--', label='Log Grip')
    ax.plot(pred_log[:, 7], 'b-', label='Pred Grip')
    ax.set_title("Inference Case: Gripper")
    ax.legend(); ax.grid(True)
    
    # Col 3: Quat X Profile
    ax = axes[0, 3]
    ax.plot(log_traj_raw[:, 3], 'r--', label='Log Qx')
    ax.plot(pred_log[:, 3], 'b-', label='Pred Qx')
    ax.set_title("Inference Case: Quat X")
    ax.legend(); ax.grid(True)
    
    # --- ROW 1 ~ N: Random Training Samples ---
    indices = np.random.choice(len(df_train) - 16, num_train_samples, replace=False)
    
    for i, idx in enumerate(indices):
        row_idx = i + 1
        print(f"  > Processing Training Sample {i+1} (Idx {idx})...")
        
        # Get Input
        row = df_train.iloc[idx]
        t_pose = row[['ee_x','ee_y','ee_z','ee_qx','ee_qy','ee_qz','ee_qw']].values
        t_grip = row['gripper']
        t_obj = row[['obj_x','obj_y','obj_z']].values
        
        # Get GT (8 steps)
        gt_traj = df_train.iloc[idx : idx+8][['ee_x','ee_y','ee_z','ee_qx','ee_qy','ee_qz','ee_qw','gripper']].values
        
        # Model Prediction
        pred_traj = generate_chunk(t_pose, t_grip, t_obj, components, device)
        
        # Col 0: XY Path
        ax = axes[row_idx, 0]
        ax.plot(gt_traj[:, 0], gt_traj[:, 1], 'g--x', label='GT Path', linewidth=2)
        ax.plot(pred_traj[:, 0], pred_traj[:, 1], 'b-o', label='Model Pred', linewidth=2)
        ax.set_title(f"Train Sample {i+1}: XY Path")
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend(); ax.grid(True)
        
        # Col 1: Z Profile
        ax = axes[row_idx, 1]
        ax.plot(gt_traj[:, 2], 'g--', label='GT Z')
        ax.plot(pred_traj[:, 2], 'b-', label='Pred Z')
        ax.set_title(f"Train Sample {i+1}: Z vs Time")
        ax.grid(True)
        
        # Col 2: Gripper Profile
        ax = axes[row_idx, 2]
        ax.plot(gt_traj[:, 7], 'g--', label='GT Grip')
        ax.plot(pred_traj[:, 7], 'b-', label='Pred Grip')
        ax.set_title(f"Train Sample {i+1}: Gripper")
        ax.grid(True)
        
        # Col 3: Quat X Profile
        ax = axes[row_idx, 3]
        ax.plot(gt_traj[:, 3], 'g--', label='GT Qx')
        ax.plot(pred_traj[:, 3], 'b-', label='Pred Qx')
        ax.set_title(f"Train Sample {i+1}: Quat X")
        ax.grid(True)

    plt.tight_layout()
    save_path = "eval_multi_chunk_compare.png"
    plt.savefig(save_path)
    print(f"\n✅ Done! Saved comprehensive comparison to '{save_path}'")

if __name__ == "__main__":
    main()