import torch
import torch.nn as nn
import numpy as np
import threading
import queue
import time
import math
import cv2
import sys
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

from robo_control import RoboControl

# ==============================================================================
# 0. 基础网络组件定义 (与之前保持一致)
# 此处为了简洁省略了 SinusoidalPosEmb, Conv1dBlock, ConditionalResidualBlock1D... 等类的具体定义
# 请在实际运行中把 train_ccg.py 中的基础架构类原封不动地贴在这里。
# ==============================================================================
# (此处贴入 U-Net 和 CCG Critic 的架构定义，同上文)

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
    def __init__(self, action_dim=10, obs_dim=13, hidden_dim=512, depth=4):
        super().__init__()
        self.obs_dim = obs_dim
        self.context_encoder = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.Mish(), nn.Linear(256, 256), nn.Mish()
        )
        self.geo_input_dim = action_dim + 1 + 3 + 3 + 1 + 1 
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

# 省略部分：MinMaxNormalizer, gamma_t_si, matrix_to_rotation_6d 等请照常保留

class RobotBrain:
    def __init__(self, ckpt_path="models/real_robot_final.ckpt", ccg_path="models/ccg_critic_final.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Brain] Inference Device: {self.device}")

        self.pred_horizon = 16
        self.obs_horizon = 1
        self.action_dim = 10 
        self.obs_dim = 13 
        self.dt_val = 1.0 / (self.pred_horizon - self.obs_horizon)
        self.chunk_size = 8 
        self.sigma_infer = 0.0

        # --- CCG Guidance Params ---
        self.ccg_guidance_scale = 10.0  # 比 STEG 快，可以给大一点的引导强度
        self.ccg_activation_dist = 0.4  # 距离障碍物小于 40cm 开始激活

        # 加载 Base
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.velocity_net = ConditionalUnet1D(input_dim=self.action_dim, global_cond_dim=self.obs_dim, updownsample_type='Linear', sin_embedding_scale=100).to(self.device)
        self.velocity_net.load_state_dict(checkpoint['velocity_net'])
        self.velocity_net.eval()

        self.denoiser_net = ConditionalUnet1D(input_dim=self.action_dim, global_cond_dim=self.obs_dim, updownsample_type='Linear', sin_embedding_scale=100).to(self.device)
        self.denoiser_net.load_state_dict(checkpoint['denoiser_net'])
        self.denoiser_net.eval()
        self.normalizer = checkpoint['normalizer']

        # 加载 CCG
        print(f"[Brain] Loading CCG Critic: {ccg_path}")
        self.ccg_critic = CollisionPredictionCritic(action_dim=self.action_dim, obs_dim=self.obs_dim).to(self.device)
        self.ccg_critic.load_state_dict(torch.load(ccg_path, map_location=self.device, weights_only=True))
        self.ccg_critic.eval()
        
        self.chunk_step_counter = 0
        self.cached_global_cond = None
        self.na = None
        self.step_idx = 0
        self.static_obj_pos = None
    
    def get_drift(self, x, t, global_cond):
        if t.ndim == 0: t = t.unsqueeze(0)
        v_pred = self.velocity_net(sample=x, timestep=t, global_cond=global_cond)
        eta_pred = self.denoiser_net(sample=x, timestep=t, global_cond=global_cond)
        gamma = gamma_t_si(t).view(-1, 1, 1).to(self.device)
        gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(self.device)
        s_pred = -eta_pred / (gamma + EPS)
        score_coeff = 0.5 * (self.sigma_infer ** 2) - (gamma * gamma_dot)
        return v_pred + score_coeff * s_pred

    def compute_ccg_gradient(self, na_curr, t_curr, global_cond, obs_tensor_norm):
        """
        极速求导：不需要任何时间步的循环模拟，仅仅前向传播一次 CCG 并对 risk 求导。
        我们希望风险最小，所以梯度方向是负梯度 (-risk_grad)
        """
        with torch.enable_grad():
            na_in_flat = na_curr.squeeze(1).detach().requires_grad_(True) # (B, 10)
            
            # Predict Risk (0~1)
            risk = self.ccg_critic(na_in_flat, t_curr, obs_tensor_norm, global_cond)
            
            # Autograd
            grad = torch.autograd.grad(risk.sum(), na_in_flat)[0]
            
            # 我们想向风险减小的方向移动，取反向梯度，再增加通道维度 (B, 1, 10)
            return -grad.unsqueeze(1)

    def infer(self, ee_queue, image_queue, obstacle_pos=None):
        if self.chunk_step_counter == 0:
            if ee_queue.empty(): return None
            current_ee_pose = None
            while not ee_queue.empty(): current_ee_pose = ee_queue.get()
            
            self.static_obj_pos = [0.480729, -0.047977, 0.109436] 
            raw_obs = np.concatenate([current_ee_pose, self.static_obj_pos]).astype(np.float32)
            nobs = self.normalizer.normalize(raw_obs.reshape(1, -1))
            self.cached_global_cond = torch.from_numpy(nobs).float().to(self.device)
            self.na = torch.from_numpy(nobs[:, :10]).float().to(self.device).unsqueeze(1)
            print(f"[Brain] Starting new chunk execution sequence")

        t_scalar = np.clip(self.chunk_step_counter * self.dt_val, 1e-3, 1.0 - 1e-3)
        t = torch.tensor([t_scalar], device=self.device, dtype=torch.float32)

        ccg_grad = torch.zeros_like(self.na)
        ccg_scale_curr = 0.0

        if obstacle_pos is not None and len(obstacle_pos) == 3 and obstacle_pos[0] is not None:
            # 1. 物理距离判断
            obs_tensor_phys = torch.tensor(obstacle_pos, device=self.device, dtype=torch.float32)
            curr_phys = self.normalizer.unnormalize_diff_differentiable(self.na.squeeze(1), self.device)
            dist_to_obs = torch.norm(curr_phys[0, :3] - obs_tensor_phys).item()
            
            if dist_to_obs < self.ccg_activation_dist:
                # 2. 将障碍物物理坐标转换到 Normalized 空间，供给 CCG 网络使用
                scale_3d = torch.tensor(self.normalizer.scale[:3], device=self.device, dtype=torch.float32)
                min_3d = torch.tensor(self.normalizer.min[:3], device=self.device, dtype=torch.float32)
                obs_tensor_norm = (obs_tensor_phys - min_3d) / scale_3d * 2.0 - 1.0
                obs_tensor_norm = obs_tensor_norm.unsqueeze(0) # (1, 3)
                
                # 3. Training-Free 直接计算 Risk 梯度引导动作
                ccg_grad = self.compute_ccg_gradient(self.na, t, self.cached_global_cond, obs_tensor_norm)
                severity = max(0.0, (1.0 - dist_to_obs / self.ccg_activation_dist))
                ccg_scale_curr = self.ccg_guidance_scale * severity
                print(f"CCG Guidance activated! Dist: {dist_to_obs:.3f}, Severity: {severity:.2f}")

        with torch.no_grad():
            b_drift_base = self.get_drift(self.na, t, self.cached_global_cond)
            # 添加 CCG 梯度
            final_drift = b_drift_base + ccg_scale_curr * ccg_grad
            
            self.na = self.na + final_drift * self.dt_val 
            
            na_cpu = self.na.detach().cpu().numpy().squeeze()
            action_min = self.normalizer.min[:10]
            action_scale = self.normalizer.scale[:10]
            raw_action = ((na_cpu + 1) / 2) * action_scale + action_min

        self.chunk_step_counter += 1
        if self.chunk_step_counter >= self.chunk_size:
            self.chunk_step_counter = 0
            
        return raw_action

# 主运行逻辑请保留你在原代码中的 while 循环和执行部分...
# ==============================================================================
# SECTION 4: CONTROL NODE (同你原来代码的写法即可)
# ==============================================================================




if __name__ == "__main__":
    ee_pose_queue = queue.Queue()
    image_queue = queue.Queue()
    
    brain = RobotBrain(ckpt_path="models/real_robot_final.ckpt")
    logger = open("debug.txt", 'w')
    
    print("[Control] Starting Loop...")
    rospy.init_node("inference_node")
    control_node = RoboControl()

    time.sleep(2)
    # Move to initial pose
    starting_pose = [0.320236, 0.078831, 0.416621, 0.950743, -0.298455, 0.083734, -0.289216, -0.951283, -0.106830, 3]
    control_node.execute_rot(starting_pose)
    control_node.open_gripper()
    last_action = None
    time.sleep(5)
    # exit()
    try:
        while not rospy.is_shutdown():
            # 1. Get Robot State
            # (In real usage, this should come from control_node callbacks)
            # mocking data for structure
            # current_pose = [0.32, 0.08, 0.41, 0, 0, 0, 1, 0] # x,y,z,qx,qy,qz,qw,g
            # continue
            current_pose = control_node.ee_pose
            x,y,z, qx,qy,qz,qw, g = current_pose
            p_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            pose_6d = list(map(float, matrix_to_rotation_6d(torch.tensor(p_mat).unsqueeze(0))[0]))
            d1, d2, d3, d4, d5, d6 = pose_6d
            control_node.ee_pose_queue.put([x,y,z,d1,d2,d3,d4,d5,d6,g])

            # 2. Get Obstacle Info
            # Simulate receiving data from perception node
            obstacle_data = control_node.obj_pos[0] ## assume 1 obstacle for now at ID 0
            # Example: Force an obstacle if step > 50
            # if brain.step_idx > 50:
            #     obstacle_data = [0.45, 0.0, 0.2]

            # 3. Inference
            # The 'streaming' nature is handled internally by brain.chunk_step_counter
            ee_queue_open = queue.Queue()
            if last_action is None:
                action = brain.infer(control_node.ee_pose_queue, control_node.image_queue, obstacle_pos=obstacle_data)
            else:
                ee_queue_open.put(last_action) 
                action = brain.infer(ee_queue_open, control_node.image_queue, obstacle_pos=obstacle_data)
            control_node.target_obj_pose = brain.static_obj_pos
            # 4. Execute
            if action is not None:
                print(action[:3], np.array(control_node.ee_pose[:3]))
                ### debug: save the action + the current joint state (should be very similar to prev step!)
                if last_action is not None:
                    distance_between_frame = np.linalg.norm(np.array(action[:3]) - np.array(last_action[:3]), 1)
                    # print(distance_between_frame)
                    distance_to_obj = np.linalg.norm(np.array(control_node.ee_pose[:3]) - np.array(last_action[:3]), 1)
                    # print(distance_to_obj)
                    logger.write(f"{distance_to_obj}\n")
                last_action = action
                action = list(map(float, action))
                control_node.execute_rot(action)
                # print(f"[Execute] Step {brain.step_idx} (ChunkStep {brain.chunk_step_counter-1}): Action {action[:3]}")
                brain.step_idx += 1
            time.sleep(1/30) # 20Hz Control Loop

    except KeyboardInterrupt:
        print("[Control] Stopping...")