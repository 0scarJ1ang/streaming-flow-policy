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
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

# 引入我们写好的 Dataset
from dataset_dp import RealRobotDataset

# =========================================================
# 0. 辅助类与函数 (EMA, Model, Math)
# =========================================================

class EMAModel:
    """简单的 EMA 实现"""
    def __init__(self, parameters, power=0.75):
        self.power = power
        self.shadow = [p.clone().detach() for p in parameters]

    def step(self, parameters):
        for s, p in zip(self.shadow, parameters):
            if p.requires_grad:
                s.data.mul_(self.power).add_(p.data, alpha=1 - self.power)

    def copy_to(self, parameters):
        for s, p in zip(self.shadow, parameters):
            if p.requires_grad:
                p.data.copy_(s.data)

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
        scale, bias = embed[:, 0, ...], embed[:, 1, ...]
        out = scale * out + bias
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, updownsample_type: Literal['Conv', 'Linear'], sin_embedding_scale, diffusion_step_embed_dim=256, down_dims=[256, 512, 1024], kernel_size=5, n_groups=8):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed, scale=sin_embedding_scale),
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
                ConditionalResidualBlock1D(dim_out * 2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
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
        sample = sample.moveaxis(-1, -2)
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
        x = x.moveaxis(-1, -2)
        return x

# =========================================================
# 2. 训练主流程 (Diffusion Policy)
# =========================================================

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- A. 参数配置 ---
    pred_horizon = 16
    obs_horizon = 2
    
    action_dim = 10  # 最终动作维度: 3(pos) + 6(rot) + 1(gripper)
    obs_dim = 13     # 最终观测维度: 3(pos) + 6(rot) + 1(gripper) + 3(obj)
    
    num_epochs = 1000
    batch_size = 64

    # --- B. 数据加载 ---
    print("Loading dataset from processed_6Ddata...")
    dataset = RealRobotDataset(
        dataset_dir="processed_6Ddata", # 指向你截图里的统一文件夹
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    print(f"Total dataset size: {len(dataset)}")

    # 获取原生的 13D/10D normalizer (已经由 Dataset 自动计算好)
    normalizer = dataset.get_normalizer()

    # --- C. 初始化模型 ---
    dp_noise_pred_net = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon,
        updownsample_type='Conv',
        sin_embedding_scale=1,
    ).to(device)

    print("Diffusion Model initialized successfully!")

    # --- D. 优化器、调度器与 EMA ---
    num_diffusion_iters = 100
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_diffusion_iters,
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    ema_dp = EMAModel(parameters=dp_noise_pred_net.parameters(), power=0.75)

    optimizer = torch.optim.AdamW(
        params=dp_noise_pred_net.parameters(),
        lr=1e-4, weight_decay=1e-6
    )

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=500,
        num_training_steps=len(dataloader) * num_epochs
    )

    os.makedirs("models", exist_ok=True)
    print("Starting training...")

    # --- E. 训练循环 ---
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = list()
            
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    # 1. 取出已经归一化好的 13D obs 和 10D action
                    obs = nbatch['obs'].to(device)         # (B, To, 13)
                    naction = nbatch['action'].to(device)  # (B, Tp, 10)
                    B = obs.shape[0]

                    # Observation as FiLM conditioning
                    obs_cond = obs.flatten(start_dim=1)  # (B, To * 13)

                    # 2. 采样噪声与 Timesteps
                    noise = torch.randn(naction.shape, device=device)  # (B, Tp, 10)
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    # 3. 前向扩散过程：给 Action 加噪
                    noisy_actions = noise_scheduler.add_noise(naction, noise, timesteps)

                    # 4. 预测噪声
                    noise_pred = dp_noise_pred_net(noisy_actions, timesteps, global_cond=obs_cond)

                    # 5. 计算损失与反向传播
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()

                    # 6. 更新 EMA
                    ema_dp.step(dp_noise_pred_net.parameters())

                    # Logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                    
            tglobal.set_postfix(loss=np.mean(epoch_loss))

            # 保存 Checkpoint
            if (epoch_idx + 1) % 200 == 0:
                ckpt_path = f"models/dp_policy_epoch_{epoch_idx+1}.ckpt"
                torch.save({
                    'model_state_dict': dp_noise_pred_net.state_dict(),
                    'ema_state_dict': [p.data for p in ema_dp.shadow],
                    'normalizer': normalizer, # 记录原生的 13D/10D Normalizer，部署时直接用！
                }, ckpt_path)

    print("Training Completed.")

if __name__ == "__main__":
    train()