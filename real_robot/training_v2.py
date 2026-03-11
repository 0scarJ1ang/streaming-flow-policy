import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import numpy as np
import os
import math
from typing import Union, Literal
from torch import Tensor
from diffusers.optimization import get_scheduler # 需要安装 diffusers: pip install diffusers

# 引入我们写好的 Dataset
from dataset import RealRobotDataset

# =========================================================
# 0. 辅助类与函数 (EMA, Model, Math)
#    (保留你提供的模型定义，为了完整性我把它放在这里)
# =========================================================

class EMAModel:
    """简单的 EMA 实现"""
    def __init__(self, parameters, power=0.75):
        self.power = power
        self.shadow = [p.clone().detach() for p in parameters]

    def step(self, parameters):
        for s, p in zip(self.shadow, parameters):
            if p.requires_grad:
                # 简单的指数滑动平均
                s.data.mul_(self.power).add_(p.data, alpha=1 - self.power)

    def copy_to(self, parameters):
        for s, p in zip(self.shadow, parameters):
            if p.requires_grad:
                p.data.copy_(s.data)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale = 1):
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

class ConvDownsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x): return self.conv(x)

class ConvUpsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)
    def forward(self, x): return self.conv(x)

class LinearDownsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    def forward(self, x: Tensor):
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = x.view(batch_size, channels, seq_len)
        return x

class LinearUpsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
    def forward(self, x: Tensor):
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, -1)
        x = self.linear(x)
        x = x.view(batch_size, channels, seq_len)
        return x

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
    def __init__(self, input_dim, global_cond_dim, updownsample_type: Literal['Conv', 'Linear'], sin_embedding_scale, diffusion_step_embed_dim=256, down_dims=[256,512,1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
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

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            if updownsample_type == 'Linear': downsample_layer = LinearDownsample1d(dim_out) if not is_last else nn.Identity()
            elif updownsample_type == 'Conv': downsample_layer = ConvDownsample1d(dim_out) if not is_last else nn.Identity()
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                downsample_layer,
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            if updownsample_type == 'Linear': upsample_layer = LinearUpsample1d(dim_in) if not is_last else nn.Identity()
            elif updownsample_type == 'Conv': upsample_layer = ConvUpsample1d(dim_in) if not is_last else nn.Identity()
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                upsample_layer,
            ]))

        final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(self, sample: Tensor, timestep: Union[Tensor, float, int], global_cond=None) -> Tensor:
        sample = sample.moveaxis(-1,-2)
        timesteps = timestep
        if not torch.is_tensor(timesteps): timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0: timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.diffusion_step_encoder(timesteps)
        if global_cond is not None: global_feature = torch.cat([global_feature, global_cond], axis=-1)
        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules: x = mid_module(x, global_feature)
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = x.moveaxis(-1,-2)
        return x

# =========================================================
# 1. 核心数学函数 (SFP / SI)
# =========================================================
EPS = 1e-6

def gamma_t_si(t):
    # SI 气泡控制
    return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)

def LinearlyInterpolateTrajectory(ξ, t):
    B, T, A = ξ.shape
    scaled_t = t * (T - 1)
    l = scaled_t.floor().long().clamp(0, T - 2)
    u = (l + 1).clamp(0, T - 1)
    λ = scaled_t - l.float()
    batch_idx = torch.arange(B, device=ξ.device)
    ξl = ξ[batch_idx, l, :]
    ξu = ξ[batch_idx, u, :]
    λ = λ.unsqueeze(-1)
    ξt = ξl + λ * (ξu - ξl)
    dξdt = (ξu - ξl) * (T - 1)
    return ξt, dξdt

# =========================================================
# 2. 训练主流程
# =========================================================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- A. 参数配置 (适配 Real Robot) ---
    pred_horizon = 16
    
    # 【修改点】 真实机器人维度
    action_dim = 10  # 9 (pose) + 1 (gripper)
    obs_dim = 13    # 10 (robot) + 3 (obj)
    obs_horizon = 1 # 我们的 Dataset 目前只返回当前这一帧
    
    num_epochs_si = 2000
    batch_size = 64 # 根据显存调整

    # --- B. 数据加载 ---
    dataset = RealRobotDataset(
        dataset_dir="processed_data", 
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Dataset size: {len(dataset)}")

    # --- C. 初始化模型 ---
    # Velocity Network (确定性场)
    si_velocity_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        updownsample_type='Linear',
        sin_embedding_scale=100,
    ).to(device)

    # Denoiser Network (随机噪声场)
    si_denoiser_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        updownsample_type='Linear',
        sin_embedding_scale=100,
    ).to(device)

    print("Models initialized for Streaming SI Policy (Real Robot).")

    # --- D. 优化器 ---
    optimizer_si = torch.optim.AdamW([
        {'params': si_velocity_net.parameters()},
        {'params': si_denoiser_net.parameters()}
    ], lr=1e-4, weight_decay=1e-6)

    # EMA
    ema_si_v = EMAModel(parameters=si_velocity_net.parameters(), power=0.75)
    ema_si_eta = EMAModel(parameters=si_denoiser_net.parameters(), power=0.75)

    lr_scheduler_si = get_scheduler(
        name='cosine',
        optimizer=optimizer_si,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs_si
    )

    # SFP 超参数
    sigma0 = 0.4
    k = 10.0

    # 创建保存目录
    os.makedirs("models", exist_ok=True)

    print("Starting training...")

    # --- E. 训练循环 ---
    for epoch_idx in range(num_epochs_si):
        epoch_loss = []
        epoch_v_loss = []
        epoch_eta_loss = []
        
        # 进度条
        pbar = tqdm(dataloader, desc=f'Epoch {epoch_idx+1}/{num_epochs_si}', leave=False)
        
        for nbatch in pbar:
            # 1. 获取数据 (已经归一化到 -1~1)
            # nobs: [B, 11]
            nobs = nbatch['obs'].to(device)
            # naction: [B, 16, 8]
            naction = nbatch['action'].to(device)
            
            # 【修改点】 动作处理
            # 我们的 Dataset 已经对齐好了，naction 就是未来的轨迹
            # ξ (xi) 是 Ground Truth Trajectory
            ξ = naction 
            B = ξ.shape[0]
            
            # 随机采样时间 t ~ U[0, 1]
            t = torch.rand(B, device=device)
            
            # 2. 计算 GT 位置和速度 (Linear Interpolation)
            ξt, dξdt = LinearlyInterpolateTrajectory(ξ, t)
            
            # 3. 构建 Flow Matching (FP) 基础目标
            t_expanded = t.view(B, 1)
            sigma_t_fp = sigma0 * torch.exp(-k * t_expanded)
            noise_fp = torch.randn_like(ξt)
            a_t_fp = ξt + sigma_t_fp * noise_fp
            
            # Velocity Target
            v_target = dξdt - k * (a_t_fp - ξt)
            
            # 4. 构建 SI 扰动 (气泡)
            gamma = gamma_t_si(t_expanded)
            z_noise_si = torch.randn_like(ξt)
            x_t_in_s = a_t_fp + gamma * z_noise_si
            
            # 5. 网络前向传播
            # Reshape input: (B, 1, 8) 
            net_input = x_t_in_s.unsqueeze(1)
            # Flatten obs: (B, 11)
            global_cond = nobs.flatten(start_dim=1)
            
            # 预测 v
            v_pred = si_velocity_net(sample=net_input, timestep=t, global_cond=global_cond)
            v_pred = v_pred.squeeze(1)
            
            # 预测 eta (noise)
            eta_pred = si_denoiser_net(sample=net_input, timestep=t, global_cond=global_cond)
            eta_pred = eta_pred.squeeze(1)
            
            # 6. Loss
            loss_v = nn.functional.mse_loss(v_pred, v_target)
            loss_eta = nn.functional.mse_loss(eta_pred, z_noise_si)
            
            loss = loss_v + loss_eta
            
            # 7. Backward
            loss.backward()
            optimizer_si.step()
            optimizer_si.zero_grad()
            lr_scheduler_si.step()
            
            # Update EMA
            ema_si_v.step(si_velocity_net.parameters())
            ema_si_eta.step(si_denoiser_net.parameters())
            
            # Logging
            epoch_loss.append(loss.item())
            epoch_v_loss.append(loss_v.item())
            epoch_eta_loss.append(loss_eta.item())
            
            pbar.set_postfix(loss=loss.item(), v_loss=loss_v.item(), s_loss=loss_eta.item())

        # 每个 Epoch 结束打印信息
        avg_loss = np.mean(epoch_loss)
        print(f"Epoch {epoch_idx+1} | Loss: {avg_loss:.6f}")

        # 定期保存 Checkpoint
        if (epoch_idx + 1) % 100 == 0:
            ckpt_path_epoch = f"models/real_robot_epoch_{epoch_idx+1}.ckpt"
            torch.save({
                'velocity_net': si_velocity_net.state_dict(),
                'denoiser_net': si_denoiser_net.state_dict(),
                'normalizer': dataset.get_normalizer(), # 【重要】保存归一化参数，推理要用！
            }, ckpt_path_epoch)
            print(f" Saved checkpoint to {ckpt_path_epoch}")

    # 保存最终模型
    ckpt_path_final = "models/real_robot_final.ckpt"
    torch.save({
        'velocity_net': si_velocity_net.state_dict(),
        'denoiser_net': si_denoiser_net.state_dict(),
        'normalizer': dataset.get_normalizer(),
    }, ckpt_path_final)
    print("Training Finished!")

if __name__ == "__main__":
    train()