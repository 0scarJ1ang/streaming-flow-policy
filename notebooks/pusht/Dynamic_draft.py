import numpy as np
import torch
import cv2
import collections
import math
import os
import imageio
import pandas as pd
from IPython.display import Image, display

# =========================================================
# 辅助函数
# =========================================================

def draw_vector_arrow(img, start_pt, vector, color, scale=100, thickness=2):
    if vector is None or np.linalg.norm(vector) < 1e-6:
        return
    end_pt = (int(start_pt[0] + vector[0] * scale), 
              int(start_pt[1] + vector[1] * scale))
    cv2.arrowedLine(img, start_pt, end_pt, color, thickness, tipLength=0.3)

def draw_coordinate_axes(img, color=(100, 100, 100)):
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(img, (0, cy), (w, cy), color, 1) 
    cv2.line(img, (cx, 0), (cx, h), color, 1) 
    
    tick_color = (60, 60, 60)
    cv2.line(img, (int(w*0.75), 0), (int(w*0.75), h), tick_color, 1, cv2.LINE_AA)
    cv2.line(img, (int(w*0.25), 0), (int(w*0.25), h), tick_color, 1, cv2.LINE_AA)
    cv2.line(img, (0, int(h*0.75)), (w, int(h*0.75)), tick_color, 1, cv2.LINE_AA)
    cv2.line(img, (0, int(h*0.25)), (w, int(h*0.25)), tick_color, 1, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.4
    cv2.putText(img, "(0,0)", (cx+5, cy+15), font, scale, color, 1)

# =========================================================
# [新增] 动态障碍物管理器 (整合了 Mode 1, 2, 3)
# =========================================================
class DynamicObstacleManager:
    def __init__(self, mode, start_pos_norm=None, baseline_traj=None, dt=0.05, vis_scale=1.0, seed=None):
        """
        Mode Definition:
        0: Static
        1: Sine Wave (原 Mode 4, 频率调慢)
        2: Intercept (Ambush: 快速到达后静止)
        3: Chase (随机起点, 惯性追踪)
        """
        self.mode = mode
        self.dt = dt
        self.vis_scale = vis_scale
        
        # --- [配置参数] ---
        self.sine_amp = 60.0      
        self.sine_freq = 0.05     # 较慢的震动频率

        self.goal_center = np.array([256.0, 256.0]) 
        self.safe_radius = 200.0  # 稍微缩小一点安全半径，防止随机生成太难
        
        # Mode 3: Chase 
        self.chase_speed = 1.0       
        self.chase_turn_rate = 0.10  
        
        # --- [初始化随机数生成器] ---
        if seed is None:
            seed = 42
        self.rng = np.random.RandomState(seed)
        
        # --- [位置初始化] ---
        # 默认位置 (Mode 0, 1, 2 会用到这个或被覆盖)
        if start_pos_norm is not None:
            self.pos = unnormalize_data(np.array(start_pos_norm), stats=stats['action'])
        else:
            self.pos = np.zeros(2)

        # Mode 3: Chase 强制随机起点 (全图随机)
        if self.mode == 3:
            self.pos = self.rng.uniform(50.0, 460.0, size=2)

        self.velocity = np.zeros(2)
        self.baseline_traj = baseline_traj
        
        # --- [模式特定初始化] ---
        
        # Mode 2: Intercept (埋伏模式)
        self.target_intercept_pt = None
        self.travel_duration = 0 

        if self.mode == 2 and self.baseline_traj is not None:
            # 目标是 Agent 第 90 步所在的位置
            intercept_idx = min(len(self.baseline_traj) - 1, 90) 
            self.target_intercept_pt = self.baseline_traj[intercept_idx]
            
            # 偏移量：让它从远处飞来
            offset = np.array([80.0, 80.0]) 
            self.pos = self.target_intercept_pt + offset 
            
            # [关键] 强制它在 50 步内到达位置 (比 Agent 快)
            self.travel_duration = 50 
            self.velocity = (self.target_intercept_pt - self.pos) / self.travel_duration

        # Mode 1: Sine Wave
        if self.mode == 1 and self.baseline_traj is not None:
            mid_idx = min(len(self.baseline_traj) - 1, 60)
            self.anchor_pos = self.baseline_traj[mid_idx]
            p1 = self.baseline_traj[max(0, mid_idx-5)]
            p2 = self.baseline_traj[min(len(self.baseline_traj)-1, mid_idx+5)]
            tangent = p2 - p1
            tangent /= (np.linalg.norm(tangent) + 1e-6)
            self.perp_vec = np.array([-tangent[1], tangent[0]]) 

        # 强制安全检查 (对 Mode 3 随机生成的点也有效)
        self._enforce_spawn_safety()

    def _enforce_spawn_safety(self):
        dist = np.linalg.norm(self.pos - self.goal_center)
        if dist < self.safe_radius:
            if dist < 1e-6:
                direction = np.array([1.0, 0.0])
            else:
                direction = (self.pos - self.goal_center) / dist
            
            new_dist = self.safe_radius + 20.0
            self.pos = self.goal_center + direction * new_dist
            self.pos = np.clip(self.pos, 10, 502)

    def update(self, step_idx, agent_pos_real=None):
        if self.mode == 0: # Static
            return self.pos

        elif self.mode == 1: # Sine Wave
            val = math.sin(step_idx * self.sine_freq)
            self.pos = self.anchor_pos + self.sine_amp * val * self.perp_vec

        elif self.mode == 2: # Intercept (到达即停)
            # 计算到目标的距离
            dist_to_target = np.linalg.norm(self.target_intercept_pt - self.pos)
            
            # [关键] 停止条件：距离近 或 时间到
            if dist_to_target < 2.0 or step_idx > self.travel_duration:
                self.velocity = np.zeros(2)
            
            self.pos += self.velocity
            
        elif self.mode == 3: # Chase (带惯性)
            if agent_pos_real is not None:
                vec_to_agent = agent_pos_real - self.pos
                dist = np.linalg.norm(vec_to_agent)
                
                if dist > 1e-1:
                    desired_velocity = (vec_to_agent / dist) * self.chase_speed
                    self.velocity = (1 - self.chase_turn_rate) * self.velocity + \
                                    self.chase_turn_rate * desired_velocity
                    self.pos += self.velocity
                else:
                    self.velocity *= 0.8 # 防止重叠时震荡

        return self.pos

# =========================================================
# Streaming Inference 推理函数 (Updated)
# =========================================================
def run_ccg_dynamic_inference(obstacle_mode=1,  # [修改] 传入模式 ID
                               guidance_scale=10.0, 
                               seed=42, 
                               save_gif=True, 
                               output_dir="output/ccg_dynamic_results",
                               show_baseline=True,
                               show_axes=True):
    
    # 结果容器
    result_metrics = {
        "seed": seed,
        "mode": obstacle_mode,
        # Baseline Metrics
        "base_collision": False,
        "base_reward": 0.0,
        "base_success": False,
        "base_min_dist": 0.0,
        # Guided Metrics
        "guided_collision": False,
        "guided_reward": 0.0,
        "guided_success": False,
        "guided_min_dist": 0.0
    }

    # 内部函数：运行一次 Episode
    # [修改] 增加了 baseline_traj_real 参数，用于传给 Manager
    def run_episode(obstacle_mode_in_episode, baseline_traj_real=None, record_traj=False, baseline_pixel_traj=None, force_no_guidance=False):
        # 1. 设置随机种子 (锁死 PyTorch/Numpy)
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
            import random
            random.seed(seed)
            try: env.seed(seed)
            except: pass
        
        try: obs, info = env.reset(seed=seed)
        except: obs, info = env.reset()
            
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
        
        start_img = env.render()
        img_h, img_w = start_img.shape[:2]
        vis_scale = img_w / getattr(env, 'window_size', 512)
        
        # --- 物理参数设置 ---
        TRAIN_NORM_R = 0.06 
        PHYSICAL_RADIUS = TRAIN_NORM_R * 256.0      # ~15.36 px
        COLLISION_THRESHOLD = PHYSICAL_RADIUS * 2   # ~30.72 px
        VIS_RADIUS = max(2, int(PHYSICAL_RADIUS * vis_scale))     # ~8 px
        VIS_THRESHOLD_PIXEL = int(COLLISION_THRESHOLD * vis_scale) # ~15 px

        # ---------------------------------------------------------
        # 初始化 Dynamic Obstacle Manager
        # ---------------------------------------------------------
        obs_manager = None
        obs_real_pos_arr = None
        obs_pixel = None
        obs_tensor = None
        
        if not record_traj: # 如果不是为了录制 Baseline，就初始化障碍物
            # Mode 1 & 2 需要一个初始点作为参考（这里给个默认值 [0.5, 0.5]，
            # 实际上 Mode 1/2 会根据 baseline_traj 重置位置，Mode 3 会随机重置）
            obs_manager = DynamicObstacleManager(
                mode=obstacle_mode_in_episode,
                start_pos_norm=[0.5, 0.5], 
                baseline_traj=baseline_traj_real,
                vis_scale=vis_scale,
                seed=seed
            )
        
        min_dist_to_obs = float('inf')
        
        # 初始化 Action
        a = obs[:action_dim]
        na = normalize_data(a, stats=stats['action'])
        na = torch.from_numpy(na).to(device, dtype=torch.float32)
        na_from_prev_chunk = na.unsqueeze(0).unsqueeze(0)
        
        recorded_pixels = []
        recorded_norms = []
        imgs = []
        episode_rewards = [] 
        
        done = False
        step_idx = 0
        max_steps = 250
        dt_val = 1.0 / (pred_horizon - obs_horizon)
        
        current_max_guidance = 0.0 if force_no_guidance else guidance_scale

        while not done and step_idx < max_steps:
            obs_seq = np.stack(obs_deque)
            nobs = normalize_data(obs_seq, stats=stats['obs'])
            o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten().unsqueeze(0)
            na = na_from_prev_chunk
            
            # --- Streaming Policy Loop ---
            for i in range(action_horizon):
                a_cpu = na.detach().to('cpu').numpy().squeeze() 
                a_real = unnormalize_data(a_cpu, stats=stats['action']) 
                curr_px = (int(a_real[0] * vis_scale), int(a_real[1] * vis_scale))

                # === [关键修改] 动态更新障碍物 ===
                if obs_manager is not None:
                    # 调用 Update 获取新位置
                    obs_real_pos_arr = obs_manager.update(step_idx, agent_pos_real=a_real)
                    
                    # 更新用于渲染的像素坐标
                    obs_pixel = (int(obs_real_pos_arr[0] * vis_scale), int(obs_real_pos_arr[1] * vis_scale))
                    
                    # 更新用于 Critic 的 Tensor (需要归一化)
                    curr_norm = normalize_data(obs_real_pos_arr, stats['action'])
                    obs_tensor = torch.tensor([curr_norm], device=device, dtype=torch.float32)
                
                if record_traj:
                    recorded_pixels.append(curr_px)
                    recorded_norms.append(a_cpu) # 这里的 norms 其实就是 baseline traj
                
                # --- 碰撞检测 ---
                if obs_real_pos_arr is not None:
                    dist = np.linalg.norm(a_real - obs_real_pos_arr)
                    if dist < min_dist_to_obs:
                        min_dist_to_obs = dist
                
                # --- Step 2. 环境步 ---
                obs, reward, done, _, info = env.step(a_real)
                obs_deque.append(obs)
                episode_rewards.append(reward)
                
                step_idx += 1
                if done or step_idx >= max_steps: break
                
                # --- Step 3. Drift & Guidance (Streaming) ---
                t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                t = torch.tensor([[t_scalar]], device=device, dtype=torch.float32)
                
                with torch.no_grad():
                    v_pred = ema_si_velocity_net(na, t.view(-1), o_test) 
                    eta_pred = ema_si_denoiser_net(na, t.view(-1), o_test)
                    gamma = gamma_t_si(t).view(-1, 1, 1)
                    g_dot = d_gamma_dt_si(t).view(-1, 1, 1)
                    s_pred = -eta_pred / (gamma + 1e-6)
                    score_coeff = - (gamma * g_dot)
                    base_drift = v_pred + score_coeff * s_pred
                
                guidance = torch.zeros_like(base_drift)
                
                # 只有在非 baseline 且有 guidance scale 时才计算引导
                if obs_tensor is not None and current_max_guidance > 0:
                    with torch.enable_grad():
                        na_in = na.squeeze(1).detach().requires_grad_(True) 
                        val = critic_net(na_in, t, obs_tensor, o_test)
                        collision_prob = torch.sigmoid(val).detach()
                        k = 3
                        dynamic_scale = current_max_guidance * torch.pow(collision_prob, k)
                        (grad_v,) = torch.autograd.grad(val.sum(), na_in)
                        guidance = -dynamic_scale * grad_v.unsqueeze(1)
                
                na = na + (base_drift + guidance) * dt_val 
                na = na.detach()
                
                # --- 渲染 ---
                if save_gif and not record_traj:
                    img_rgb = env.render().copy()
                    if show_axes: draw_coordinate_axes(img_rgb)
                    if show_baseline and baseline_pixel_traj is not None:
                        pts = np.array(baseline_pixel_traj, np.int32).reshape((-1, 1, 2))
                        cv2.polylines(img_rgb, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    if obs_pixel is not None:
                        # 障碍物 (Cyan)
                        cv2.circle(img_rgb, obs_pixel, VIS_RADIUS, (255, 255, 0), -1) 
                        cv2.circle(img_rgb, obs_pixel, VIS_THRESHOLD_PIXEL, (0, 0, 255), 1) 
                        # 标记障碍物模式
                        label = ["Static", "Sine", "Ambush", "Chase"][obstacle_mode_in_episode]
                        cv2.putText(img_rgb, f"{label}", (obs_pixel[0]-20, obs_pixel[1]-15), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                        
                    # 机器人 (Red)
                    cv2.circle(img_rgb, curr_px, VIS_RADIUS, (0, 0, 255), -1)
                    
                    if obs_tensor is not None and current_max_guidance > 0:
                        vec_guide = guidance.detach().cpu().numpy().squeeze()
                        draw_vector_arrow(img_rgb, curr_px, vec_guide, (0, 0, 255), scale=150.0)
                    imgs.append(img_rgb)

            na_from_prev_chunk = na
            if done: break
        
        max_reward = max(episode_rewards) if episode_rewards else 0.0  

        # 结算
        metrics = {}
        if not record_traj:
            is_collision = False
            if obs_real_pos_arr is not None:
                is_collision = min_dist_to_obs < COLLISION_THRESHOLD
            
            is_success = (not is_collision) and (max_reward > 0.85)
            
            metrics = {
                "collision": is_collision,
                "reward": max_reward,
                "success": is_success,
                "min_dist": min_dist_to_obs
            }

        if record_traj:
            return recorded_pixels, recorded_norms
        else:
            return metrics, imgs

    # === Main Logic ===
    
    # 1. Warm-up Run (Baseline)
    # 目的：1. 获取 Agent 原始意图轨迹 (供 Mode 1/2 使用)；2. 评估 Baseline 性能
    rng_state = torch.get_rng_state()
    base_pixels, base_norms = run_episode(obstacle_mode_in_episode=0, record_traj=True)
    torch.set_rng_state(rng_state)
    
    if len(base_norms) == 0: return result_metrics, []
    
    # [关键] 将 Warm-up 得到的归一化轨迹转换为物理坐标轨迹
    # 这对 Mode 1 (Sine) 和 Mode 2 (Intercept) 至关重要
    baseline_traj_real = None
    if obstacle_mode in [1, 2]:
        baseline_traj_real = np.array([unnormalize_data(n, stats['action']) for n in base_norms])

    # 2. Baseline Evaluation (With Obstacle, No Guidance)
    # 让 Baseline Agent 面对同样的动态障碍物，看看会不会撞
    print(f"Running Baseline (Mode {obstacle_mode})...")
    base_metrics, _ = run_episode(
        obstacle_mode_in_episode=obstacle_mode, 
        baseline_traj_real=baseline_traj_real, # 传入物理轨迹供管理器使用
        record_traj=False, 
        force_no_guidance=True, 
        baseline_pixel_traj=None
    )
    
    result_metrics["base_collision"] = base_metrics["collision"]
    result_metrics["base_reward"] = base_metrics["reward"]
    result_metrics["base_success"] = base_metrics["success"]
    result_metrics["base_min_dist"] = base_metrics["min_dist"]
    
    # 3. Guided Evaluation (Streaming Policy)
    print(f"Running Guided (Mode {obstacle_mode})...")
    guided_metrics, final_imgs = run_episode(
        obstacle_mode_in_episode=obstacle_mode, 
        baseline_traj_real=baseline_traj_real,
        record_traj=False, 
        force_no_guidance=False, 
        baseline_pixel_traj=base_pixels
    )
    
    result_metrics["guided_collision"] = guided_metrics["collision"]
    result_metrics["guided_reward"] = guided_metrics["reward"]
    result_metrics["guided_success"] = guided_metrics["success"]
    result_metrics["guided_min_dist"] = guided_metrics["min_dist"]
    
    # 打印结果
    base_status = "FAIL" if base_metrics['collision'] else "PASS"
    guide_status = "FAIL" if guided_metrics['collision'] else "PASS"
    print("\n[Streaming Result]")
    print(f"Mode: {['Static', 'Sine', 'Ambush', 'Chase'][obstacle_mode]}")
    print(f"Base:    Reward={base_metrics['reward']:.2f}, Collision={base_metrics['collision']}")
    print(f"Guided:  Reward={guided_metrics['reward']:.2f}, Collision={guided_metrics['collision']}")
    print(f"Improvement: {base_status} -> {guide_status}")
    
    # 保存图像
    if save_gif and len(final_imgs) > 0:
        os.makedirs(output_dir, exist_ok=True)
        save_name = f"streaming_seed{seed}_mode{obstacle_mode}.gif"
        save_path = os.path.join(output_dir, save_name)
        imageio.mimsave(save_path, final_imgs, fps=20, loop=0)
        print(f"Saved GIF to {save_path}")
    
    return result_metrics, final_imgs

# =========================================================
# 批量测试函数 (Adapted for Mode 1/2/3)
# =========================================================
def run_batch_dynamic_experiments(start_seed=500, num_seeds=10, 
                                  guidance_scale=1.0,
                                  obstacle_mode=1, # [修改] 传入 Mode
                                  save_gif=True,
                                  output_dir="output/batch_dynamic_results"):
    
    os.makedirs(output_dir, exist_ok=True)
    mode_names = ["Static", "Sine", "Ambush", "Chase"]
    mode_name = mode_names[obstacle_mode] if obstacle_mode < 4 else "Unknown"

    print(f"\n=== Starting Batch Experiment (Mode: {mode_name}) ===")
    print(f"Seeds: {start_seed} -> {start_seed + num_seeds - 1}")
    print(f"Guidance Scale: {guidance_scale}")
    
    all_results = []
    
    for i in range(num_seeds):
        seed = start_seed + i
        print(f"\n--- Processing Seed {seed} ({i+1}/{num_seeds}) ---")
        
        metrics, _ = run_ccg_dynamic_inference(
            obstacle_mode=obstacle_mode,
            guidance_scale=guidance_scale,
            seed=seed,
            save_gif=save_gif,
            output_dir=output_dir,
            show_baseline=True
        )

        metrics['obs_pos_tuple'] = (f"Mode{obstacle_mode}", mode_name)
        all_results.append(metrics)
        
        base_status = "FAIL" if metrics['base_collision'] else "PASS"
        guide_status = "FAIL" if metrics['guided_collision'] else "PASS"
        print(f"  [Seed {seed}] Base: {base_status} -> Guided: {guide_status}")

    # === 统计分析报告 ===
    df = pd.DataFrame(all_results)
    
    print("\n" + "="*60)
    print(f"            BATCH REPORT (MODE: {mode_name})            ")
    print("="*60)
    
    if not df.empty:
        total = len(df)
        base_coll_rate = df['base_collision'].mean() * 100
        base_succ_rate = df['base_success'].mean() * 100
        guide_coll_rate = df['guided_collision'].mean() * 100
        guide_succ_rate = df['guided_success'].mean() * 100
        
        print(f"Total Episodes: {total}")
        print("-" * 40)
        print(f"{'Metric':<15} | {'Baseline':<10} | {'Guided':<10}")
        print("-" * 40)
        print(f"{'Collision Rate':<15} | {base_coll_rate:5.1f}%      | {guide_coll_rate:5.1f}%")
        print(f"{'Success Rate':<15} | {base_succ_rate:5.1f}%      | {guide_succ_rate:5.1f}%")
        print("-" * 40)

    # 尝试调用之前的 print_conditional_fail_report (如果存在)
    try:
        print_conditional_fail_report(df)
    except:
        pass
        
    return df

# =========================================================
# 执行脚本
# =========================================================

# 示例：运行 Mode 3 (Chase - 随机起点追踪)
print("\n--- Testing Mode 3: Random Chase ---")
df_res = run_batch_dynamic_experiments(
    start_seed=0,        
    num_seeds=10,        
    guidance_scale=1.0, 
    obstacle_mode=3,   # 1=Sine, 2=Ambush, 3=Chase
    save_gif=True,      
    output_dir="output/batch_mode3_chase"     
)