import numpy as np
import torch
import cv2
import collections
import math
import os
import imageio

# =========================================================
# 辅助函数：绘制箭头
# =========================================================
def draw_vector_arrow(img, start_pt, vector, color, scale=100, thickness=2):
    """
    在图像上绘制表示向量的箭头
    """
    if vector is None or np.linalg.norm(vector) < 1e-6:
        return
    
    # 计算终点
    end_pt = (int(start_pt[0] + vector[0] * scale), 
              int(start_pt[1] + vector[1] * scale))
    
    # 绘制箭头
    cv2.arrowedLine(img, start_pt, end_pt, color, thickness, tipLength=0.3)

# =========================================================
# 3. 带障碍物引导的推理与可视化 (核心函数)
# =========================================================
def run_guided_inference_visualized(obstacle_pos_norm, guidance_scale=2.0):
    obs, info = env.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    
    # 获取图像尺寸
    start_img = env.render()
    img_h, img_w = start_img.shape[:2] 
    imgs = [] 
    
    # 计算障碍物像素坐标 (假设 Norm 在 [-1, 1] 之间)
    obs_pixel_x = int((obstacle_pos_norm[0] + 1) / 2 * img_w)
    obs_pixel_y = int((obstacle_pos_norm[1] + 1) / 2 * img_h)
    obs_pixel = (obs_pixel_x, obs_pixel_y)
    
    # 缩放比例
    env_window_size = getattr(env, 'window_size', 512)
    vis_scale = img_w / env_window_size
    
    obs_tensor = torch.tensor([obstacle_pos_norm], device=device, dtype=torch.float32)

    # 初始化动作
    a = obs[:action_dim]
    na = normalize_data(a, stats=stats['action'])
    na = torch.from_numpy(na).to(device, dtype=torch.float32)
    na_from_prev_chunk = na.unsqueeze(0).unsqueeze(0) 
    
    done = False
    step_idx = 0
    max_steps = 200
    dt_val = 1.0 / (pred_horizon - obs_horizon)
    arrow_vis_scale = 150.0 
    
    while not done and step_idx < max_steps:
        obs_seq = np.stack(obs_deque)
        nobs = normalize_data(obs_seq, stats=stats['obs'])
        o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten().unsqueeze(0)
        
        na = na_from_prev_chunk 
        
        for i in range(action_horizon):
            # --- A. 执行 ---
            a_cpu = na.detach().to('cpu').numpy().squeeze()
            a_real = unnormalize_data(a_cpu, stats=stats['action'])
            obs, reward, done, _, info = env.step(a_real)
            obs_deque.append(obs)
            
            step_idx += 1
            if done or step_idx >= max_steps: break
            
            # --- B. 计算 Drift & Guidance ---
            t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
            t = torch.tensor([[t_scalar]], device=device, dtype=torch.float32)
            
            # 1. Base Drift
            with torch.no_grad():
                v_pred = ema_si_velocity_net(na, t.view(-1), o_test) 
                eta_pred = ema_si_denoiser_net(na, t.view(-1), o_test)
                
                gamma = gamma_t_si(t).view(-1, 1, 1)
                g_dot = d_gamma_dt_si(t).view(-1, 1, 1)
                s_pred = -eta_pred / (gamma + 1e-6)
                
                # score_coeff = 0.5 * (0.05**2) - (gamma * g_dot)
                score_coeff = - (gamma * g_dot)
                base_drift = v_pred + score_coeff * s_pred
            
            # 2. Critic Guidance (负梯度)
            with torch.enable_grad():
                na_in = na.squeeze(1).detach().requires_grad_(True) 
                val = critic_net(na_in, t, obs_tensor, o_test)
                (grad_v,) = torch.autograd.grad(val.sum(), na_in)
                grad_v = grad_v.unsqueeze(1)
            
            # 3. Fusion
            guidance = -guidance_scale * grad_v
            total_drift = base_drift + guidance

            # --- C. 可视化 ---
            img_rgb = env.render().copy()
            
            # 画障碍物
            cv2.circle(img_rgb, obs_pixel, 6, (255, 255, 0), -1) 
            cv2.circle(img_rgb, obs_pixel, 6, (0, 0, 0), 2)
            cv2.putText(img_rgb, "OBS", (obs_pixel[0]-15, obs_pixel[1]-20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # 画 Agent
            curr_phys = unnormalize_data(na.detach().cpu().numpy().squeeze(), stats=stats['action'])
            curr_px = (int(curr_phys[0] * vis_scale), int(curr_phys[1] * vis_scale))
            cv2.circle(img_rgb, curr_px, 6, (0, 0, 255), -1)

            # 画箭头
            vec_base = base_drift.detach().cpu().numpy().squeeze()
            vec_guide = guidance.detach().cpu().numpy().squeeze()
            draw_vector_arrow(img_rgb, curr_px, vec_base, (255, 0, 0), scale=arrow_vis_scale, thickness=2)
            if guidance_scale > 0:
                draw_vector_arrow(img_rgb, curr_px, vec_guide, (0, 0, 255), scale=arrow_vis_scale, thickness=2)
            
            imgs.append(img_rgb)
            
            # --- D. Update ---
            noise = torch.randn_like(na)
            diffusion = 0.05 * math.sqrt(dt_val) * noise
            # na = na + total_drift * dt_val + diffusion
            na = na + total_drift * dt_val 
            na = na.detach()

        na_from_prev_chunk = na
        if done: break
        
    print(f"Obs: {obstacle_pos_norm}, Score: {max(rewards) if rewards else 0}")
    return imgs

# =========================================================
# 4. 批量运行并保存到文件夹
# =========================================================

# 1. 定义输出文件夹
output_dir = "guided_results_gifs"
os.makedirs(output_dir, exist_ok=True)
print(f"Results will be saved to: {output_dir}")

# 2. 定义多个测试障碍物位置 (归一化坐标 [-1, 1])
test_obstacles = [
    [0.5, 0.2],   # 原始位置
    [-0.4, -0.3], # 左下角
    [0.4, 0.4],   # 上方正中
    [0.1, 0.4],  
    [0.25, -0.25]   # 右上角
]

# 3. 循环执行
for i, obs_pos in enumerate(test_obstacles):
    print(f"\n--- Processing Case {i+1}/{len(test_obstacles)}: Obstacle at {obs_pos} ---")
    
    # 运行推理
    imgs_guided = run_guided_inference_visualized(obs_pos, guidance_scale=0.40)
    
    # 构建文件名并保存
    save_name = f"obs_case_{i}_pos_{obs_pos[0]}_{obs_pos[1]}.gif"
    save_path = os.path.join(output_dir, save_name)
    
    imageio.mimsave(save_path, imgs_guided, fps=20, loop=0)
    print(f"Saved GIF to: {save_path}")

