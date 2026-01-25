import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

# =========================================================
# 1. 粘贴 inference_v0.py 中的模型定义 (必须完全一致)
# =========================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1):
        super().__init__()
        self.dim = dim; self.scale = scale 
    def forward(self, x):
        x = x * self.scale; device = x.device; half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels), nn.Mish(),
        )
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
        out = self.blocks[1](out); out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, updownsample_type='Linear', sin_embedding_scale=100, diffusion_step_embed_dim=256, down_dims=[256,512,1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims); start_dim = down_dims[0]; dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(SinusoidalPosEmb(dsed, scale = sin_embedding_scale), nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed))
        cond_dim = dsed + global_cond_dim; in_out = list(zip(all_dims[:-1], all_dims[1:])); mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups), ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups)])
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1); downsample_layer = nn.Sequential(nn.Linear(dim_out, dim_out)) if not is_last else nn.Identity() # Simplification for brev
            class LinearDownsample1d(nn.Module): # Re-define locally to be safe
                def __init__(self, dim): super().__init__(); self.linear = nn.Linear(dim, dim)
                def forward(self, x): return self.linear(x.transpose(1,2)).transpose(1,2)
            downsample_layer = LinearDownsample1d(dim_out) if not is_last else nn.Identity()
            self.down_modules.append(nn.ModuleList([ConditionalResidualBlock1D(dim_in, dim_out, cond_dim, kernel_size, n_groups), ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups), downsample_layer]))
        self.up_modules = nn.ModuleList([])
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

# Dataset/Pickle Hack
class MinMaxNormalizer:
    def __init__(self, data=None): self.min=None; self.max=None; self.scale=None
    def normalize(self, x): return (x - self.min) / self.scale * 2 - 1
sys.modules['dataset'] = sys.modules[__name__]

EPS = 1e-6
def gamma_t_si(t): return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)
def d_gamma_dt_si(t): return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))

# =========================================================
# 2. LOG DATA (Your Real Test Data)
# =========================================================

# Object Position (From log: [Vision] Object calibrated at: ...)
static_obj_pos = np.array([0.58648731, 0.01768983, 0.09015887])
# static_obj_pos = np.array([0.46, 0.05, 0.08])

# The Logged Actions (Step 0 to 7) - This is what your model PRODUCED in inference
# We will use Step 0 as the "Current State" for input, and see if we can reproduce Step 1-7
# Or better: We assume Step 0 is the current robot state when planning started.
log_traj_raw = np.array([
    [0.48327076, 0.04026159, 0.38554416, -0.9996992, -0.0049929, -0.00896409, -0.02227603, 3.0], # Step 0
    [0.48108962, 0.03178788, 0.40048939, -1.0632324, 0.04099637, 0.03466385, -0.02490192, 4.17051244], # Step 1
    [0.48139408, 0.02800756, 0.4082067, -1.06066193, 0.05364134, 0.04942676, -0.02263388, 4.42402768], # Step 2
    [0.48198297, 0.02736998, 0.4137264, -1.04626752, 0.0498152, 0.05163207, -0.0171863, 4.3217926], # Step 3
    [0.48254276, 0.03018797, 0.41526781, -1.04568126, 0.04596553, 0.05129013, -0.01348264, 4.1696434], # Step 4
    [0.48362392, 0.03330099, 0.41579423, -1.05292895, 0.04473415, 0.05536716, -0.01125529, 4.11990094], # Step 5
    [0.48486044, 0.03720651, 0.41566398, -1.05786734, 0.04395545, 0.06098466, -0.00885116, 4.08693123], # Step 6
    [0.48617911, 0.04103875, 0.41544327, -1.06350067, 0.04380868, 0.06642694, -0.00548643, 4.21265531], # Step 7
])

def run_test():
    device = torch.device('cuda')
    print(f"Device: {device}")
    
    # 1. Load Checkpoint
    ckpt_path = "models/real_robot_epoch_700.ckpt"
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    velocity_net = ConditionalUnet1D(input_dim=8, global_cond_dim=11).to(device)
    denoiser_net = ConditionalUnet1D(input_dim=8, global_cond_dim=11).to(device)
    velocity_net.load_state_dict(checkpoint['velocity_net']); velocity_net.eval()
    denoiser_net.load_state_dict(checkpoint['denoiser_net']); denoiser_net.eval()
    normalizer = checkpoint['normalizer']
    
    # 2. Reconstruct Observation
    # Inference logic: raw_obs = [current_ee_pose(7), gripper(1), obj(3)]
    # We assume Step 0 of the log was the current_ee_pose
    current_ee_pose = log_traj_raw[0, :7] # XYZ + Quat
    current_gripper = np.array([1.0]) # Hardcoded in inference code as 1.0
    
    raw_obs = np.concatenate([current_ee_pose, current_gripper, static_obj_pos]).astype(np.float32)
    raw_obs = raw_obs.reshape(1, -1) # (1, 11)
    
    # Normalize Obs
    nobs_np = normalizer.normalize(raw_obs)
    nobs = torch.from_numpy(nobs_np).float().to(device)
    
    # 3. Warm Start
    # Inference logic: Init from current state (nobs[:8])
    current_state_norm = nobs[:, :8] # (1, 8)
    na = current_state_norm.unsqueeze(1) # (1, 1, 8)
    
    # 4. Generate Chunk (8 steps)
    chunk_size = 8
    dt_val = 1.0 / (16 - 1)
    
    pred_actions_list = []
    
    print("\nStarting Generation...")
    with torch.no_grad():
        for i in range(chunk_size):
            # Decode
            na_cpu = na.detach().cpu().numpy().squeeze()
            action_min = normalizer.min[:8]; action_scale = normalizer.scale[:8]
            action_01 = (na_cpu + 1) / 2
            raw_action = action_01 * action_scale + action_min
            pred_actions_list.append(raw_action)
            
            # Integrate
            t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
            t = torch.tensor([t_scalar], device=device, dtype=torch.float32)
            
            # Drift Calculation
            v_pred = velocity_net(sample=na, timestep=t, global_cond=nobs)
            eta_pred = denoiser_net(sample=na, timestep=t, global_cond=nobs)
            gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
            b_drift = v_pred + gamma_dot * eta_pred
            
            na = na + b_drift * dt_val
            
    pred_actions = np.array(pred_actions_list) # (8, 8)
    
    # 5. Compare & Plot
    print("\n--- Comparison ---")
    print(f"Log Shape: {log_traj_raw.shape}")
    print(f"Pred Shape: {pred_actions.shape}")
    
    mse = np.mean((log_traj_raw - pred_actions)**2)
    print(f"\nMean Squared Error between Log and Reproduction: {mse:.8f}")
    if mse < 1e-6:
        print("✅ SUCCESS: Reproduction matches Log perfectly! Model is deterministic and inputs align.")
    else:
        print("❌ WARNING: Mismatch detected. Inputs or randomness might differ.")

    # Plotting
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # XYZ
    ax = axes[0]
    ax.plot(log_traj_raw[:, 0], label='Log X', linestyle='--', color='r')
    ax.plot(pred_actions[:, 0], label='Pred X', color='r')
    ax.plot(log_traj_raw[:, 1], label='Log Y', linestyle='--', color='g')
    ax.plot(pred_actions[:, 1], label='Pred Y', color='g')
    ax.plot(log_traj_raw[:, 2], label='Log Z', linestyle='--', color='b')
    ax.plot(pred_actions[:, 2], label='Pred Z', color='b')
    ax.set_title("XYZ Position")
    ax.legend()
    
    # Quat
    ax = axes[1]
    ax.plot(log_traj_raw[:, 3], label='Log Qx', linestyle='--', color='c')
    ax.plot(pred_actions[:, 3], label='Pred Qx', color='c')
    ax.set_title("Orientation (Qx)")
    
    # Gripper
    ax = axes[2]
    # Note: Log has Step 0 gripper = 3.0, but we fed 1.0. Let's see what happens.
    ax.plot(log_traj_raw[:, 7], label='Log Gripper', linestyle='--', color='k')
    ax.plot(pred_actions[:, 7], label='Pred Gripper', color='k')
    ax.set_title("Gripper")
    
    plt.tight_layout()
    plt.savefig("eval_log_repro.png")
    print("Saved plot to eval_log_repro.png")

if __name__ == "__main__":
    run_test()