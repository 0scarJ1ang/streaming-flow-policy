import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# =========================================================
# 1. 保持 Risk 计算逻辑不变 (用于生成 Ground Truth)
# =========================================================
def compute_hybrid_risk(trajectory, obs_pos, r_obs=0.15, sharpness=60.0):
    # ... (代码保持不变，省略以节省空间，直接使用你提供的版本) ...
    # 1. Hit Risk
    dists = torch.norm(trajectory - obs_pos.unsqueeze(0), p=2, dim=-1)
    min_dist = torch.min(dists, dim=0)[0] 
    prob_hit = torch.sigmoid(sharpness * (r_obs + 0.05 - min_dist))
    
    # 2. Aiming Risk
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
# 2. 核心修改：增加 Critic 对比可视化
# =========================================================
def visualize_critic_vs_groundtruth(dataloader, critic_model, num_vis=3, K_samples=16, sample_t_max=0.3):
    print("--- Visualizing: Ground Truth (Rollout) vs. Critic Prediction ---")
    
    # 确保两个模型都在评估模式
    si_velocity_net.eval()
    critic_model.eval() 
    
    num_inference_steps = 16 
    dt = 1.0 / num_inference_steps
    
    def get_drift(a, t, cond):
        t_in = torch.clamp(t, 0.02, 0.98).view(-1)
        v = si_velocity_net(a.unsqueeze(1), t_in, cond).squeeze(1)
        return v

    iter_loader = iter(dataloader)
    batch = next(iter_loader)
    
    obs = batch['obs'].to(device)
    gt_action = batch['action'].to(device)
    B = obs.shape[0]
    global_cond = obs.flatten(start_dim=1)
    
    num_vis = min(num_vis, B)

    # -------------------------------------------------------
    # 循环处理样本
    # -------------------------------------------------------
    for i in range(num_vis):
        # A. 准备上下文
        t_val = torch.rand(1, device=device) * sample_t_max
        start_idx = obs_horizon - 1
        step_offset = int(t_val.item() * num_inference_steps)
        target_idx = min(start_idx + step_offset, gt_action.shape[1]-1)
        
        a_base = gt_action[i, target_idx].unsqueeze(0) # [1, 2]
        cond_i = global_cond[i].unsqueeze(0)           # [1, D]
        
        # B. 预测未来路径以生成障碍物
        path = [a_base]
        curr_a = a_base.clone()
        curr_t = t_val.clone().view(-1)
        with torch.no_grad():
            for _ in range(10):
                v = get_drift(curr_a, curr_t, cond_i)
                curr_a += v * dt
                curr_t += dt
                path.append(curr_a.clone())
        path_tensor = torch.stack(path, dim=0).squeeze(1)
        
        # 计算方向
        p_start = path_tensor[0]
        p_end = path_tensor[-1]
        vec_movement = p_end - p_start
        len_movement = torch.norm(vec_movement) + 1e-6
        dir_movement = vec_movement / len_movement

        # C. 定义 3 种障碍物场景
        scenarios = []
        # 1. Near (必经之路)
        idx_near = int(path_tensor.shape[0] * 0.7)
        scenarios.append(("Near (Hit)", path_tensor[idx_near]))
        # 2. Far (前方远处)
        pos_far = p_end + dir_movement * 0.4 
        pos_far = torch.clamp(pos_far, -0.9, 0.9)
        scenarios.append(("Far (Risk)", pos_far))
        # 3. Behind (后方安全)
        pos_behind = p_start - dir_movement * 0.4
        pos_behind = torch.clamp(pos_behind, -0.9, 0.9)
        scenarios.append(("Behind (Safe)", pos_behind))
        
        # ---------------------------------------------------
        # D. 绘图设置: 2行 x 3列
        # Row 1: GT Rollout
        # Row 2: Critic Prediction
        # ---------------------------------------------------
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        plt.suptitle(f"Sample {i} | t={t_val.item():.2f} | Top: GT Rollout, Bottom: Critic Pred", fontsize=16)
        
        for ax_idx, (name, obs_pos_base) in enumerate(scenarios):
            # 增加一点随机扰动，避免完全重合
            obs_pos = obs_pos_base + torch.randn_like(obs_pos_base) * 0.02
            
            # =============================================
            # E. 构造 4 组动作 (Action Variations)
            # =============================================
            a_variations = []
            labels = []
            
            # 1. Base (GT)
            a_variations.append(a_base) 
            labels.append("Base")
            # 2. Noise
            a_variations.append(a_base + torch.randn_like(a_base) * 0.1)
            labels.append("Noise")
            # 3. Left
            perp_vec = torch.tensor([-dir_movement[1], dir_movement[0]], device=device)
            a_variations.append(a_base + perp_vec * 0.25)
            labels.append("Left")
            # 4. Right
            a_variations.append(a_base - perp_vec * 0.25)
            labels.append("Right")
            
            a_batch = torch.cat(a_variations, dim=0) # [4, 2]
            n_vars = a_batch.shape[0]
            
            # 扩展 Context 数据以匹配 [4, ...]
            t_batch_4 = t_val.repeat(n_vars, 1)      # [4, 1]
            cond_batch_4 = cond_i.repeat(n_vars, 1)  # [4, D]
            obs_batch_4 = obs_pos.view(1, 2).repeat(n_vars, 1) # [4, 2]
            
            # =============================================
            # F. 获取 Ground Truth Risk (Monte Carlo)
            # =============================================
            # 扩展 K 倍进行采样
            curr_a = a_batch.repeat_interleave(K_samples, dim=0) # [4*K, 2]
            curr_t = t_batch_4.repeat_interleave(K_samples, dim=0).view(-1)
            curr_cond = cond_batch_4.repeat_interleave(K_samples, dim=0)
            curr_obs = obs_batch_4.repeat_interleave(K_samples, dim=0)
            
            traj_hist = [curr_a.clone()]
            with torch.no_grad():
                for _ in range(int(1.5 / dt)): 
                    active = (curr_t < 1.0 - 1e-3).float().unsqueeze(1)
                    if active.sum() == 0: break
                    v = get_drift(curr_a, curr_t, curr_cond)
                    d_a = v * dt + 0.05 * math.sqrt(dt) * torch.randn_like(curr_a)
                    curr_a += d_a * active
                    curr_t += dt * active.squeeze(1)
                    traj_hist.append(curr_a.clone())
            
            full_traj = torch.stack(traj_hist, dim=0) # [T, 4*K, 2]
            
            # 计算 GT Risk
            risk_vals = compute_hybrid_risk(full_traj, curr_obs)
            risk_vals = risk_vals.view(n_vars, K_samples)
            gt_risks = risk_vals.mean(dim=1).cpu().numpy() # [4]
            
            # =============================================
            # G. 获取 Critic Prediction (Model Inference)
            # =============================================
            # Critic 不需要 K 次采样，只需要对 4 个动作进行单次推理
            with torch.no_grad():
                # 注意输入维度: a[4,2], t[4,1], obs[4,2], cond[4,D]
                pred_risks = critic_model(a_batch, t_batch_4, obs_batch_4, cond_batch_4)
                pred_risks = pred_risks.cpu().numpy().flatten() # [4]

            # =============================================
            # H. 统一绘图逻辑
            # =============================================
            trajs_np = full_traj.cpu().numpy()
            start_pts = a_batch.cpu().numpy()
            ox, oy = obs_pos.cpu().numpy()
            
            # --- ROW 1: Ground Truth ---
            ax_gt = axes[0, ax_idx]
            ax_gt.set_title(f"GT (Monte Carlo) - {name}")
            
            # --- ROW 2: Critic Prediction ---
            ax_pred = axes[1, ax_idx]
            ax_pred.set_title(f"Critic Prediction - {name}")

            # 对每一列的两个图进行通用设置
            for ax in [ax_gt, ax_pred]:
                ax.set_xlim(-1.1, 1.1)
                ax.set_ylim(-1.1, 1.1)
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                # 画障碍物
                ax.add_patch(patches.Circle((ox, oy), radius=0.05, fc='black', alpha=0.8, zorder=10))
                ax.add_patch(patches.Circle((ox, oy), radius=0.15, ec='red', fc='none', ls='--', alpha=0.5))

            # --- 绘制内容 ---
            for v_idx in range(n_vars):
                sx, sy = start_pts[v_idx]
                
                # 1. 绘制 GT (上图)
                r_gt = gt_risks[v_idx]
                color_gt = plt.cm.RdYlGn_r(r_gt)
                
                # 画 GT 起点
                ax_gt.scatter(sx, sy, c=[color_gt], s=150, edgecolors='k', zorder=5)
                ax_gt.text(sx+0.03, sy+0.03, f"{r_gt:.2f}", fontsize=9, fontweight='bold')
                
                # 画 GT 轨迹 (Critic 图里不画轨迹，因为 Critic 没有生成轨迹)
                start_k = v_idx * K_samples
                end_k = (v_idx + 1) * K_samples
                for k in range(start_k, end_k):
                    ax_gt.plot(trajs_np[:, k, 0], trajs_np[:, k, 1], color=color_gt, alpha=0.3, linewidth=1)
                
                # 2. 绘制 Prediction (下图)
                r_pred = pred_risks[v_idx]
                color_pred = plt.cm.RdYlGn_r(r_pred) # 使用相同的色卡
                
                # 画 Pred 起点 (用同样的圆点表示，方便对比颜色)
                ax_pred.scatter(sx, sy, c=[color_pred], s=150, edgecolors='k', zorder=5, label=labels[v_idx] if ax_idx==0 else "")
                
                # 标数值：上方显示 Pred，括号里显示误差
                diff = r_pred - r_gt
                diff_str = f"{diff:+.2f}"
                text_color = 'red' if abs(diff) > 0.3 else 'black' # 误差大标红
                
                ax_pred.text(sx+0.03, sy+0.03, f"{r_pred:.2f}", fontsize=10, fontweight='bold', color='black')
                ax_pred.text(sx+0.03, sy-0.1, f"Err:{diff_str}", fontsize=8, color=text_color)
                
                # 可选：在 Critic 图里画个小箭头表示动作方向，因为没有轨迹
                # 简单的把 action 当作速度向量画出来
                ax_pred.arrow(sx, sy, a_batch[v_idx, 0].item()*0.2, a_batch[v_idx, 1].item()*0.2, 
                              head_width=0.03, color=color_pred, alpha=0.6)

            # 图例
            if ax_idx == 0:
                ax_pred.legend(loc='lower left', fontsize=8)

        plt.tight_layout()
        plt.show()

# =========================================================
# 运行
# =========================================================
# 确保传入你训练好的 critic
visualize_critic_vs_groundtruth(
    dataloader, 
    critic_model=trained_critic, 
    num_vis=3, 
    K_samples=16
)