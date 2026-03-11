import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import sys
import os

# --- 1. 引入必要的类 (为了独立运行，我们把定义复制过来，确保一致性) ---
# 必须与 inference_v0.py 中修复后的定义完全一致

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1):
        super().__init__()
        self.dim = dim
        self.scale = scale 
    def forward(self, x):
        x = x * self.scale
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Conv1dBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x): return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim, kernel_size=3, n_groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(), nn.Linear(cond_dim, cond_channels), nn.Unflatten(-1, (-1, 1))
        )
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
        scale, bias = embed[:,0,...], embed[:,1,...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, updownsample_type='Linear', sin_embedding_scale=100, diffusion_step_embed_dim=256, down_dims=[256,512,1024], kernel_size=5, n_groups=8):
        super().__init__()
        # ... (简化代码，结构与 inference_v0 保持一致) ...
        # 为了节省篇幅，这里使用了通用的结构构建逻辑，确保 final_conv 使用 Conv1dBlock
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed, scale = sin_embedding_scale),
            nn.Linear(dsed, dsed * 4), nn.Mish(), nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
            ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
        ])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            # 简化：默认 LinearDownsample
            class LinearDownsample1d(nn.Module):
                def __init__(self, dim): super().__init__(); self.linear = nn.Linear(dim, dim)
                def forward(self, x): return self.linear(x.transpose(1,2)).transpose(1,2)
            
            downsample_layer = LinearDownsample1d(dim_out) if not is_last else nn.Identity()
            self.down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                downsample_layer,
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            class LinearUpsample1d(nn.Module):
                def __init__(self, dim): super().__init__(); self.linear = nn.Linear(dim, dim)
                def forward(self, x): return self.linear(x.transpose(1,2)).transpose(1,2)
            upsample_layer = LinearUpsample1d(dim_in) if not is_last else nn.Identity()
            self.up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                upsample_layer,
            ]))

        # [KEY FIX] 使用 Conv1dBlock
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample, timestep, global_cond=None):
        # Forward 逻辑与 inference_v0 一致
        sample = sample.moveaxis(-1,-2)
        timesteps = timestep
        if not torch.is_tensor(timesteps): timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif len(timesteps.shape) == 0: timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None: global_feature = torch.cat([global_feature, global_cond], axis=-1)
        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules: x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.moveaxis(-1,-2)
        return x

import math
EPS = 1e-6
def gamma_t_si(t): return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)
def d_gamma_dt_si(t): return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))

# --- 2. 引入 Dataset (Hack to fix pickle) ---
# 确保 dataset.py 在同一目录下
try:
    from dataset import RealRobotDataset, MinMaxNormalizer
except ImportError:
    print("错误：请确保 dataset.py 在当前目录下")
    exit()

sys.modules['dataset'] = sys.modules[__name__] # Pickle Hack

# --- 3. 评估主逻辑 ---

def evaluate():
    # 配置
    ckpt_path = "models/real_robot_epoch_700.ckpt" # 或者 "models/real_robot_epoch_700.ckpt"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred_horizon = 16
    obs_horizon = 1
    action_dim = 8
    obs_dim = 11
    
    print(f"Using device: {device}")

    # A. 加载 Dataset
    dataset = RealRobotDataset(dataset_dir="processed_data", pred_horizon=pred_horizon, obs_horizon=obs_horizon)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # 随机取样
    print(f"Dataset size: {len(dataset)}")

    # B. 加载模型
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    
    velocity_net = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim*obs_horizon).to(device)
    denoiser_net = ConditionalUnet1D(input_dim=action_dim, global_cond_dim=obs_dim*obs_horizon).to(device)
    
    velocity_net.load_state_dict(checkpoint['velocity_net'])
    denoiser_net.load_state_dict(checkpoint['denoiser_net'])
    
    velocity_net.eval()
    denoiser_net.eval()
    
    normalizer = checkpoint['normalizer'] # 确保使用训练时的 normalizer

    # C. 抽取样本并预测
    # 我们只看前 5 个样本
    num_viz = 5
    
    # 准备 Plot
    fig, axes = plt.subplots(num_viz, 4, figsize=(20, 3 * num_viz))
    # Columns: XYZ Pos, Orientation(Quaternion), Gripper, 3D Path
    
    iterator = iter(dataloader)
    
    for idx in range(num_viz):
        batch = next(iterator)
        
        # 1. 准备数据
        nobs = batch['obs'].to(device)       # (1, 11)
        naction_gt = batch['action'].to(device) # (1, 16, 8) GT Trajectory
        
        # 2. 生成预测 (Generation Loop)
        # SFP Inference: 从 t=0 积分到 t=1
        
        # 初始化起点: 使用当前的 Observation (Robot State 部分)
        # 对应 inference 代码里的 Warm Start
        current_state_norm = nobs[:, :8] # (1, 8)
        na_curr = current_state_norm.unsqueeze(1) # (1, 1, 8)
        
        na_pred_list = [na_curr] # 记录轨迹
        
        dt = 1.0 / (pred_horizon - 1) # 积分步长
        
        # 模拟生成 16 步 (Open Loop)
        with torch.no_grad():
            for i in range(pred_horizon - 1): # 跑 15 次积分得到 16 个点
                t_val = i * dt
                t_tensor = torch.tensor([t_val], device=device, dtype=torch.float32)
                
                # 计算 Drift
                global_cond = nobs.flatten(start_dim=1).unsqueeze(0) # (1, 11) -> (1, 11) fix shape if needed
                if len(global_cond.shape) == 3: global_cond = global_cond.squeeze(1)

                v_pred = velocity_net(sample=na_curr, timestep=t_tensor, global_cond=nobs)
                eta_pred = denoiser_net(sample=na_curr, timestep=t_tensor, global_cond=nobs)
                
                # Drift Formula (Deterministic sigma=0)
                # b = v + score_coeff * s
                # s = -eta / gamma
                # score_coeff = - gamma * gamma_dot (when sigma=0)
                # b = v + (-gamma * gamma_dot) * (-eta / gamma) = v + gamma_dot * eta
                
                gamma_dot = d_gamma_dt_si(t_tensor).view(-1, 1, 1).to(device)
                b_drift = v_pred + gamma_dot * eta_pred
                
                # Euler Integration
                na_next = na_curr + b_drift * dt
                na_pred_list.append(na_next)
                na_curr = na_next
                
        # 拼接预测轨迹
        na_pred_traj = torch.cat(na_pred_list, dim=1) # (1, 16, 8)
        
        # 3. 反归一化 (Un-normalize) 回到真实物理空间
        # 我们手动反归一化以便画图
        # 需要把 (1, 16, 8) -> numpy -> unnormalize
        
        na_gt_np = naction_gt.cpu().numpy().squeeze(0) # (16, 8)
        na_pred_np = na_pred_traj.cpu().numpy().squeeze(0) # (16, 8)
        
        # 提取 stats
        stats_min = normalizer.min[:8]
        stats_scale = normalizer.scale[:8]
        
        # Unnormalize Func
        def unnorm(x): return ((x + 1) / 2) * stats_scale + stats_min
        
        act_gt = unnorm(na_gt_np)
        act_pred = unnorm(na_pred_np)
        
        # 4. 画图
        
        # Col 1: XYZ Position
        ax = axes[idx, 0]
        ax.plot(act_gt[:, 0], label='GT X', color='r', linestyle='--')
        ax.plot(act_pred[:, 0], label='Pred X', color='r')
        ax.plot(act_gt[:, 1], label='GT Y', color='g', linestyle='--')
        ax.plot(act_pred[:, 1], label='Pred Y', color='g')
        ax.plot(act_gt[:, 2], label='GT Z', color='b', linestyle='--')
        ax.plot(act_pred[:, 2], label='Pred Z', color='b')
        ax.set_title(f"Sample {idx}: XYZ Position")
        if idx==0: ax.legend()
        
        # Col 2: Orientation (Just compare first component qx for clarity)
        ax = axes[idx, 1]
        ax.plot(act_gt[:, 3], label='GT Qx', color='c', linestyle='--')
        ax.plot(act_pred[:, 3], label='Pred Qx', color='c')
        ax.plot(act_gt[:, 6], label='GT Qw', color='m', linestyle='--')
        ax.plot(act_pred[:, 6], label='Pred Qw', color='m')
        ax.set_title("Orientation (Qx, Qw)")
        
        # Col 3: Gripper
        ax = axes[idx, 2]
        ax.plot(act_gt[:, 7], label='GT Gripper', color='k', linestyle='--')
        ax.plot(act_pred[:, 7], label='Pred Gripper', color='k')
        # ax.set_ylim(-0.1, 1.1)
        ax.set_title("Gripper State")
        
        # Col 4: 2D Trajectory (X-Y Plane)
        ax = axes[idx, 3]
        ax.plot(act_gt[:, 0], act_gt[:, 1], label='GT', color='blue', marker='o', markersize=2)
        ax.plot(act_pred[:, 0], act_pred[:, 1], label='Pred', color='red', marker='x', markersize=2)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_aspect('equal')
        ax.set_title("X-Y Path")
        
        # 计算误差
        mse = np.mean((act_gt - act_pred)**2)
        print(f"Sample {idx} MSE: {mse:.6f}")

    plt.tight_layout()
    # 保存图片
    plt.savefig("eval_result.png")
    print("\n✅ 评估完成！结果已保存至 'eval_result.png'")
    print("请查看图片：实线是预测(Pred)，虚线是真值(GT)。两者应该高度重合。")

if __name__ == "__main__":
    evaluate()