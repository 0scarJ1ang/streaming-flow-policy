import os
import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Union, Literal
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# 引入你的 Dataset
from dataset import RealRobotDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================
# 0. 基础架构依赖 (U-Net, 必须存在以加载 Base Policy)
# =========================================================
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
            downsample_layer = LinearDownsample1d(dim_out) if not is_last else nn.Identity()
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                downsample_layer,
            ]))
        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            upsample_layer = LinearUpsample1d(dim_in) if not is_last else nn.Identity()
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_out*2, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_in, dim_in, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                upsample_layer,
            ]))

        self.diffusion_step_encoder = diffusion_step_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

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
# 1. CCG 专属网络架构 (适配 obs_horizon = 1, 直接使用 MLP)
# =========================================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Mish(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Mish()
        )
    def forward(self, x):
        return x + self.net(x)

class CollisionPredictionCritic(nn.Module):
    def __init__(self, action_dim, obs_dim, hidden_dim=512, depth=4):
        super().__init__()
        self.obs_dim = obs_dim
        
        self.context_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Mish(),
            nn.Linear(256, 256), nn.Mish()
        )
        
        # action + t(1) + obs_pos(3) + rel_vec(3) + dist(1) + alignment(1) = action_dim + 9
        self.geo_input_dim = action_dim + 9 
        self.geo_encoder = nn.Sequential(
            nn.Linear(self.geo_input_dim, 256), nn.Mish(), nn.LayerNorm(256)
        )
        
        self.fusion = nn.Linear(256 + 256, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4), nn.Mish(),
            nn.Linear(hidden_dim // 4, 1), nn.Sigmoid()
        )

    def forward(self, a, t, obs_pos, global_cond):
        B = a.shape[0]
        if t.dim() == 1: t = t.unsqueeze(-1)
        
        ctx_feat = self.context_encoder(global_cond)

        curr_pos = a[:, :3] 
        rel_vec = obs_pos - curr_pos
        dist = torch.norm(rel_vec, dim=-1, keepdim=True)
        
        pos_vel_norm = curr_pos / (torch.norm(curr_pos, dim=-1, keepdim=True) + 1e-7)
        obs_dir = rel_vec / (dist + 1e-7)
        alignment = (pos_vel_norm * obs_dir).sum(dim=-1, keepdim=True)
        
        geo_in = torch.cat([a, t, obs_pos, rel_vec, dist, alignment], dim=-1)
        geo_feat = self.geo_encoder(geo_in)

        x = torch.cat([ctx_feat, geo_feat], dim=-1)
        x = self.fusion(x)
        for block in self.blocks: x = block(x)
        return self.head(x)

# =========================================================
# 2. 物理转换与物理空间风险评估逻辑
# =========================================================

def quat_to_6d_tensor(quat_xyzw):
    """纯 PyTorch 实现：四元数转 6D 旋转连续特征 (严格对应 matrix_to_rotation_6d)"""
    norm = torch.norm(quat_xyzw, p=2, dim=-1, keepdim=True)
    q = quat_xyzw / (norm + 1e-8)
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # 构建旋转矩阵的前两列
    r00 = 1.0 - 2.0 * (qy**2 + qz**2)
    r10 = 2.0 * (qx*qy + qz*qw)
    r20 = 2.0 * (qx*qz - qy*qw)
    
    r01 = 2.0 * (qx*qy - qz*qw)
    r11 = 1.0 - 2.0 * (qx**2 + qz**2)
    r21 = 2.0 * (qy*qz + qx*qw)
    
    rot_6d = torch.stack([r00, r10, r20, r01, r11, r21], dim=-1)
    return rot_6d

def compute_hybrid_risk_physical(trajectory_norm, obs_pos_norm, normalizer, r_obs=0.05, sharpness=150.0):
    device = trajectory_norm.device
    
    # 无论输入是 8 维还是 10 维，前三个永远是 XYZ
    scale_3d = torch.tensor(normalizer.scale[:3], device=device, dtype=torch.float32)
    min_3d = torch.tensor(normalizer.min[:3], device=device, dtype=torch.float32)
    
    traj_pos_phys = ((trajectory_norm[..., :3] + 1) / 2) * scale_3d + min_3d
    obs_pos_phys = ((obs_pos_norm + 1) / 2) * scale_3d + min_3d
    
    # 1. 物理碰撞概率 (Prob Hit)
    dists = torch.norm(traj_pos_phys - obs_pos_phys.unsqueeze(0), p=2, dim=-1) # (Steps, B)
    min_dist = torch.min(dists, dim=0)[0] # (B,)
    prob_hit = torch.sigmoid(sharpness * (r_obs - min_dist)) 
    
    # # 2. 物理瞄准概率 (Prob Aim)
    # start_pos = traj_pos_phys[0] 
    # future_idx = min(5, traj_pos_phys.shape[0]-1)
    # future_pos = traj_pos_phys[future_idx] 
    
    # vec_move = future_pos - start_pos
    # dist_move = torch.norm(vec_move, p=2, dim=-1) + 1e-7
    # dir_move = vec_move / dist_move.unsqueeze(-1)
    
    # vec_to_obs = obs_pos_phys - start_pos
    # dist_to_obs = torch.norm(vec_to_obs, p=2, dim=-1) + 1e-7
    # dir_to_obs = vec_to_obs / dist_to_obs.unsqueeze(-1)
    
    # alignment = torch.relu((dir_move * dir_to_obs).sum(dim=-1))
    
    # dist_factor = torch.exp(-(dist_to_obs**2) / (2 * 0.3**2))
    # prob_aim = (alignment ** 2) * dist_factor
    
    return prob_hit
    # return torch.maximum(prob_hit, 0.6 * prob_aim)


# =========================================================
# 3. 训练主循环
# =========================================================
def train_ccg():
    print(f"--- Starting CCG Training ---")
    
    ckpt_path = "models/real_robot_final.ckpt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing Base Policy Checkpoint: {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    normalizer = checkpoint['normalizer']
    
    # 动态匹配你真实存储模型的维度 (非常关键！)
    ckpt_obs_dim = len(normalizer.min)
    if ckpt_obs_dim == 13:
        action_dim = 10
        obs_dim = 13
    elif ckpt_obs_dim == 11:
        action_dim = 8
        obs_dim = 11
    else:
        raise ValueError(f"Unknown Checkpoint dimensions: {ckpt_obs_dim}")
        
    print(f"[Model Setup] Action Dim: {action_dim}, Obs Dim: {obs_dim}")
    
    si_velocity_net = ConditionalUnet1D(
        input_dim=action_dim, global_cond_dim=obs_dim,
        updownsample_type='Linear', sin_embedding_scale=100
    ).to(device)
    si_velocity_net.load_state_dict(checkpoint['velocity_net'])
    si_velocity_net.eval()
    
    dataset = RealRobotDataset(dataset_dir="processed_data", pred_horizon=16, obs_horizon=1)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)
    
    critic = CollisionPredictionCritic(action_dim=action_dim, obs_dim=obs_dim).to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    
    epochs = 150
    dt = 1.0 / 16.0 
    os.makedirs("models", exist_ok=True)
    save_path = "models/ccg_d_final.pth"

    def get_drift(a, t, cond):
        t_in = torch.clamp(t, 0.02, 0.98).view(-1)
        v = si_velocity_net(sample=a.unsqueeze(1), timestep=t_in, global_cond=cond).squeeze(1)
        return v

    for epoch in range(epochs):
        epoch_losses = []
        critic.train()
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for batch in pbar:
                obs = batch['obs'].to(device)         # 数据集给的真实 obs [B, 11]
                gt_action = batch['action'].to(device) # 数据集给的真实 action [B, Seq, 8]
                B = obs.shape[0]

                # ==============================================================
                # 核心修复区：动态识别并补全四元数到 6D 旋转的格式差异
                # ==============================================================
                if gt_action.shape[-1] == 8 and action_dim == 10:
                    # 获取当前 Dataloader 的 11D 归一化器参数
                    dataset_min = torch.tensor(dataset.get_normalizer().min, device=device, dtype=torch.float32)
                    dataset_scale = torch.tensor(dataset.get_normalizer().scale, device=device, dtype=torch.float32)
                    
                    # 1. 解归一化到真实的物理数值
                    obs_phys = ((obs + 1.0) / 2.0) * dataset_scale + dataset_min
                    gt_action_phys = ((gt_action + 1.0) / 2.0) * dataset_scale[:8] + dataset_min[:8]
                    
                    # 2. 四元数 -> 6D
                    pos = gt_action_phys[..., :3]
                    quat = gt_action_phys[..., 3:7]
                    gripper = gt_action_phys[..., 7:]
                    rot_6d = quat_to_6d_tensor(quat)
                    gt_action_10d_phys = torch.cat([pos, rot_6d, gripper], dim=-1)
                    
                    obs_pos = obs_phys[..., :3]
                    obs_quat = obs_phys[..., 3:7]
                    obs_gripper = obs_phys[..., 7:8]
                    obs_obj = obs_phys[..., 8:]
                    obs_rot_6d = quat_to_6d_tensor(obs_quat)
                    obs_13d_phys = torch.cat([obs_pos, obs_rot_6d, obs_gripper, obs_obj], dim=-1)
                    
                    # 3. 重新使用 Checkpoint 中的 13D Normalizer 进行模型所需归一化
                    ckpt_min = torch.tensor(normalizer.min, device=device, dtype=torch.float32)
                    ckpt_scale = torch.tensor(normalizer.scale, device=device, dtype=torch.float32)
                    
                    gt_action = (gt_action_10d_phys - ckpt_min[:10]) / ckpt_scale[:10] * 2.0 - 1.0
                    obs = (obs_13d_phys - ckpt_min) / ckpt_scale * 2.0 - 1.0

                # 现在不管是从上面进来的转换好的 10D，还是原生的 8D，形状全都正确！
                global_cond = obs.flatten(start_dim=1)
                
                with torch.no_grad():
                    # 随机时刻采样 
                    t_start = torch.rand(B, 1, device=device) * 0.8
                    future_step = (t_start * 16).long()
                    target_idx = torch.clamp(future_step, max=gt_action.shape[1]-1)
                    idx_expanded = target_idx.unsqueeze(-1).expand(-1, -1, action_dim)
                    a_base = torch.gather(gt_action, 1, idx_expanded).squeeze(1)

                    # Rollout 找到合理的障碍物位置
                    curr_a, curr_t, path = a_base.clone(), t_start.clone().view(-1), [a_base.clone()]
                    for _ in range(5): 
                        v = get_drift(curr_a, curr_t, global_cond)
                        curr_a += v * dt * (curr_t < 0.98).float().unsqueeze(1)
                        curr_t += dt
                        path.append(curr_a.clone())
                    
                    path_tensor = torch.stack(path, dim=0) 
                    rand_step = torch.randint(0, len(path), (B,))
                    base_obs_pos = path_tensor[rand_step, torch.arange(B)][:, :3]
                    
                    final_obs_pos = torch.clamp(base_obs_pos + torch.randn_like(base_obs_pos) * 0.15, -0.95, 0.95)

                # 多样化动作扰动
                n_pert = 6
                noise_small = torch.randn_like(a_base) * 0.1
                noise_med = torch.randn_like(a_base) * 0.3
                noise_large = torch.randn_like(a_base) * 0.6
                noise_pos_only = torch.zeros_like(a_base)
                noise_pos_only[:, :3] = torch.randn(B, 3, device=device) * 0.4
                
                a_list = [
                    a_base, a_base + noise_small, a_base + noise_med, 
                    a_base + noise_large, a_base + noise_pos_only, a_base - noise_pos_only
                ]
                a_exp = torch.stack(a_list, dim=1).view(-1, action_dim)
                cond_exp = global_cond.repeat_interleave(n_pert, dim=0)
                t_exp = t_start.repeat_interleave(n_pert, dim=0)
                obs_exp = final_obs_pos.repeat_interleave(n_pert, dim=0) 

                # 采样风险标签
                K_samples = 8
                curr_a_k = a_exp.repeat_interleave(K_samples, dim=0)
                curr_t_k = t_exp.repeat_interleave(K_samples, dim=0).view(-1)
                curr_cond_k = cond_exp.repeat_interleave(K_samples, dim=0)
                
                traj = [curr_a_k.clone()]
                with torch.no_grad():
                    for _ in range(12): 
                        v = get_drift(curr_a_k, curr_t_k, curr_cond_k)
                        noise = torch.randn_like(curr_a_k)
                        curr_a_k += (v * dt + 0.05 * math.sqrt(dt) * noise) * (curr_t_k < 0.99).float().unsqueeze(1)
                        curr_t_k += dt
                        traj.append(curr_a_k.clone())
                
                traj_stack = torch.stack(traj, dim=0)
                obs_targets = obs_exp.repeat_interleave(K_samples, dim=0)
                
                # 计算物理实际风险
                target_risk = compute_hybrid_risk_physical(traj_stack, obs_targets, normalizer, r_obs=0.05)
                target_risk = target_risk.view(B * n_pert, K_samples).mean(dim=1, keepdim=True)

                # 训练 Critic
                pred_risk = critic(a_exp, t_exp, obs_exp, cond_exp)
                loss = loss_fn(pred_risk, target_risk)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.5f}")

        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

    torch.save(critic.state_dict(), save_path)
    print(f"CCG Critic Saved at: {save_path}")

if __name__ == "__main__":
    train_ccg()