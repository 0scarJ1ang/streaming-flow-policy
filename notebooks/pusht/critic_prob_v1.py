import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
from tqdm.auto import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================================================
# 1. 升级版 Critic: 双流残差网络 (Dual-Stream ResNet)
# =========================================================
class ResidualBlock(nn.Module):
    """标准的残差块，帮助训练深层网络"""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim), # 加个 LayerNorm 稳住梯度
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish(),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        return x + self.net(x) # Skip Connection

class CollisionPredictionCritic(nn.Module):
    def __init__(self, action_dim, obs_dim, obs_horizon, hidden_dim=512, depth=4):
        super().__init__()
        
        # ------------------------------------------------
        # Stream 1: 几何特征流 (处理 Action, Obstacle, Time)
        # ------------------------------------------------
        # 原始输入: Action(2) + Time(1) + Obs(2) = 5
        # 人工特征: 相对位移(2) + 距离(1) + 夹角余弦(1) = 4
        geo_input_dim = 5 + 4 
        self.geo_encoder = nn.Sequential(
            nn.Linear(geo_input_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )

        # ------------------------------------------------
        # Stream 2: 上下文流 (处理 History/Global Cond)
        # ------------------------------------------------
        cond_input_dim = obs_dim * obs_horizon
        self.ctx_encoder = nn.Sequential(
            nn.Linear(cond_input_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2)
        )

        # ------------------------------------------------
        # Main Body: 深层残差网络 (融合推理)
        # ------------------------------------------------
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        
        # 堆叠 ResBlock
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(depth)
        ])
        
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, a, t, obs_pos, global_cond):
        # --- 1. 特征工程 (Feature Engineering) ---
        # 假设 obs_pos 是绝对坐标，我们需要相对特征
        # 注意: 这里假设 a 是归一化的速度或动作，obs_pos 也是归一化的位置
        
        # A. 相对位置向量 (假设当前位置在原点，或者 obs_pos 已经是相对的)
        # 如果 obs_pos 是绝对坐标且无当前位置输入，这里只能假设模型能从 global_cond 学到当前位置
        # 但为了强化，我们直接计算模长
        rel_vec = obs_pos - a
        dist = torch.norm(rel_vec, dim=-1, keepdim=True)
        # dist = torch.norm(obs_pos, dim=-1, keepdim=True)
        
        # B. 几何对齐度 (Alignment): 动作方向是否指向障碍物
        # 这是一个极强的特征！
        a_norm = a / (torch.norm(a, dim=-1, keepdim=True) + 1e-6)
        obs_dir = obs_pos / (dist + 1e-6)
        alignment = (a_norm * obs_dir).sum(dim=-1, keepdim=True) # Cosine Similarity
        
        # C. 构造几何输入向量
        # [Action(2), Time(1), Obs(2), Rel_Vec(2), Dist(1), Alignment(1)]
        # Rel_Vec 直接用 obs_pos (如果它是相对的) 或者 obs_pos - 0
        geo_features = torch.cat([a, t, obs_pos, obs_pos, dist, alignment], dim=-1)
        
        # --- 2. 双流编码 ---
        geo_emb = self.geo_encoder(geo_features)
        ctx_emb = self.ctx_encoder(global_cond)
        
        # --- 3. 融合与推理 ---
        x = torch.cat([geo_emb, ctx_emb], dim=-1)
        x = self.fusion(x)
        
        for block in self.blocks:
            x = block(x)
            
        logits = self.head(x)
        return torch.sigmoid(logits)

# =========================================================
# 2. 核心逻辑: 混合风险计算 (High Sharpness Version)
# =========================================================
# (保持不变)
def compute_hybrid_risk(trajectory, obs_pos, r_obs=0.15, sharpness=60.0):
    dists = torch.norm(trajectory - obs_pos.unsqueeze(0), p=2, dim=-1)
    min_dist = torch.min(dists, dim=0)[0] 
    prob_hit = torch.sigmoid(sharpness * (r_obs + 0.05 - min_dist))
    
    start_pos = trajectory[0]
    future_idx = min(5, trajectory.shape[0]-1)
    future_pos = trajectory[future_idx] 
    
    vec_move = future_pos - start_pos
    dist_move = torch.norm(vec_move, p=2, dim=-1) + 1e-6
    dir_move = vec_move / dist_move.unsqueeze(-1)
    
    vec_to_obs = obs_pos - start_pos
    dist_to_obs = torch.norm(vec_to_obs, p=2, dim=-1) + 1e-6
    dir_to_obs = vec_to_obs / dist_to_obs.unsqueeze(-1)
    
    alignment = torch.relu((dir_move * dir_to_obs).sum(dim=-1))
    dist_factor = torch.exp(-(dist_to_obs**2) / (2 * 0.8**2))
    prob_aim = (alignment ** 2) * dist_factor
    
    total_risk = torch.maximum(prob_hit, 0.6 * prob_aim)
    return total_risk

# =========================================================
# 3. 训练流程 (修复循环逻辑)
# =========================================================
def train_probability_critic_robust(dataloader, epochs=200, K_samples=16, sample_t_max=0.3, save_path="critic_model.pth"):
    print(f"--- Training Robust Probability Critic (ResNet Version) ---")
    print(f"--- Save path: {save_path} ---")
    
    si_velocity_net.eval()
    si_denoiser_net.eval()
    
    # 初始化更强的模型
    # hidden_dim=512, depth=4 意味着更宽更深
    critic = CollisionPredictionCritic(
        action_dim, obs_dim, obs_horizon, 
        hidden_dim=512, depth=4
    ).to(device)
    
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-4) # ResNet 通常 LR 可以稍微小一点或保持 2e-4
    loss_fn = nn.MSELoss()
    
    num_inference_steps = 16 
    dt = 1.0 / num_inference_steps
    
    best_loss = float('inf') 
    
    def get_drift(a, t, cond):
        t_in = torch.clamp(t, 0.02, 0.98).view(-1)
        v = si_velocity_net(a.unsqueeze(1), t_in, cond).squeeze(1)
        return v

    # 外层循环: Epochs
    for epoch in range(epochs):
        epoch_losses = []
        
        # 修复点：使用 tqdm 包装 dataloader，确保跑完整个 Batch
        # 不再使用 iter() 和 range(15)
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for batch in pbar:
                obs = batch['obs'].to(device)
                gt_action = batch['action'].to(device)
                B = obs.shape[0]
                global_cond = obs.flatten(start_dim=1)
                
                # ========================================
                # A. 基础数据准备
                # ========================================
                t_start_val = torch.rand(B, 1, device=device) * sample_t_max
                
                start_idx = obs_horizon - 1
                step_offset = (t_start_val * num_inference_steps).long()
                target_idx = torch.clamp(start_idx + step_offset, max=gt_action.shape[1]-1)
                idx_expanded = target_idx.unsqueeze(-1).expand(-1, -1, 2)
                a_base = torch.gather(gt_action, 1, idx_expanded).squeeze(1) 
                
                # ========================================
                # B. 动态 Oracle 障碍物生成
                # ========================================
                with torch.no_grad():
                    path = [a_base]
                    curr_a = a_base.clone()
                    curr_t = t_start_val.clone().view(-1)
                    
                    for _ in range(10):
                        mask = (curr_t < 0.98).float()
                        if mask.sum() == 0: break
                        v = get_drift(curr_a, curr_t, global_cond)
                        curr_a += v * dt * mask.unsqueeze(1)
                        curr_t += dt * mask
                        path.append(curr_a.clone())
                    
                    path_tensor = torch.stack(path, dim=0) 
                    path_len = path_tensor.shape[0]
                    
                    p_start = path_tensor[0]
                    p_end = path_tensor[-1]
                    vec_move = p_end - p_start
                    len_move = torch.norm(vec_move, dim=1, keepdim=True) + 1e-6
                    dir_move = vec_move / len_move
                    perp_vec = torch.stack([-dir_move[:, 1], dir_move[:, 0]], dim=1) 
                    
                    strategy_probs = torch.rand(B, device=device)
                    rand_ratios = torch.rand(B, device=device) * 0.7 + 0.2
                    indices = (rand_ratios * path_len).long().clamp(0, path_len-1)
                    batch_indices = torch.arange(B, device=device)
                    base_obs_pos = path_tensor[indices, batch_indices, :] 
                    
                    final_obs_pos = base_obs_pos.clone()
                    
                    mask_near = (strategy_probs > 0.5) & (strategy_probs <= 0.8)
                    offset_dir = (torch.rand(B, 1, device=device) > 0.5).float() * 2 - 1 
                    offset_val = 0.2 + torch.rand(B, 1, device=device) * 0.1 
                    final_obs_pos[mask_near] += perp_vec[mask_near] * offset_dir[mask_near] * offset_val[mask_near]
                    
                    mask_far = (strategy_probs > 0.8)
                    offset_val_far = 0.4 + torch.rand(B, 1, device=device) * 0.4 
                    final_obs_pos[mask_far] += perp_vec[mask_far] * offset_dir[mask_far] * offset_val_far[mask_far]
                    
                    final_obs_pos += torch.randn_like(final_obs_pos) * 0.03
                    final_obs_pos = torch.clamp(final_obs_pos, -0.95, 0.95)

                # ========================================
                # C. 动作多样化
                # ========================================
                a_small = a_base + torch.randn_like(a_base) * 0.05
                a_med = a_base + torch.randn_like(a_base) * 0.15
                a_large = a_base + torch.randn_like(a_base) * 0.3
                
                avoid_dir = (torch.rand(B, 1, device=device) > 0.5).float() * 2 - 1
                avoid_scale = 0.1 + torch.rand(B, 1, device=device) * 0.3
                a_avoid = a_base + perp_vec * avoid_dir * avoid_scale
                a_suicide = a_base - perp_vec * avoid_dir * avoid_scale 
                
                a_exp = torch.stack([a_base, a_small, a_med, a_large, a_avoid, a_suicide], dim=1).view(-1, 2)
                n_pert = 6
                
                cond_exp = global_cond.repeat_interleave(n_pert, dim=0)
                t_exp = t_start_val.repeat_interleave(n_pert, dim=0)
                obs_pos_exp = final_obs_pos.repeat_interleave(n_pert, dim=0)
                
                # ========================================
                # D. Rollout & Update
                # ========================================
                curr_a = a_exp.repeat_interleave(K_samples, dim=0) 
                curr_t = t_exp.repeat_interleave(K_samples, dim=0).view(-1)
                curr_cond = cond_exp.repeat_interleave(K_samples, dim=0)
                curr_obs = obs_pos_exp.repeat_interleave(K_samples, dim=0)
                
                traj_history = [curr_a.clone()]
                
                with torch.no_grad():
                    max_steps = int(1.2 / dt) 
                    for _ in range(max_steps):
                        active = (curr_t < 1.0 - 1e-3).float().unsqueeze(1)
                        if active.sum() == 0: break
                        v = get_drift(curr_a, curr_t, curr_cond)
                        d_a = v * dt + 0.08 * math.sqrt(dt) * torch.randn_like(curr_a)
                        curr_a += d_a * active
                        curr_t += dt * active.squeeze(1)
                        traj_history.append(curr_a.clone())
                
                full_traj = torch.stack(traj_history, dim=0)
                
                risk_vals = compute_hybrid_risk(full_traj, curr_obs)
                risk_vals = risk_vals.view(B * n_pert, K_samples)
                target_prob = risk_vals.mean(dim=1, keepdim=True) 
                
                pred_prob = critic(a_exp, t_exp, obs_pos_exp, cond_exp)
                loss = loss_fn(pred_prob, target_prob)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
                
                pbar.set_postfix(loss=f"{loss.item():.6f}")
            
        # --- End of Epoch Logging ---
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.6f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = save_path.replace(".pth", "_best.pth")
            torch.save(critic.state_dict(), best_path)
        
        if (epoch + 1) % 50 == 0:
            epoch_path = save_path.replace(".pth", f"_ep{epoch+1}.pth")
            torch.save(critic.state_dict(), epoch_path)
                
    # 保存最终模型
    torch.save(critic.state_dict(), save_path)
    print(f"Training Complete. Models saved to {save_path}")
    return critic

# =========================================================
# 运行训练并保存
# =========================================================
trained_critic = train_probability_critic_robust(
    dataloader, 
    epochs=150, # Epoch 可以稍微减少，因为现在每个 Epoch 跑的数据量变大了
    K_samples=12, 
    sample_t_max=0.8, 
    save_path="Robust_critic_final.pth"
)