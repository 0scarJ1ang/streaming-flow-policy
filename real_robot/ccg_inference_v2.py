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
import torch
import torch.nn as nn
import numpy as np
import threading
import queue
import time
import math
import cv2
import json
from typing import Union, Literal
from torch import Tensor
import sys
from robo_control import RoboControl # Assuming this is your local file
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray, Int32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

from robo_control import RoboControl

# ==============================================================================
# 0. 基础网络组件定义 (与之前保持一致)
# 此处为了简洁省略了 SinusoidalPosEmb, Conv1dBlock, ConditionalResidualBlock1D... 等类的具体定义
# 请在实际运行中把 train_ccg.py 中的基础架构类原封不动地贴在这里。
# ==============================================================================
# (此处贴入 U-Net 和 CCG Critic 的架构定义，同上文)
# ==============================================================================
# SECTION 1: MODEL ARCHITECTURE (Fixed)
# ==============================================================================

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
def matrix_to_rotation_6d(matrix) -> torch.Tensor:
    return torch.cat([matrix[:, :, 0], matrix[:, :, 1]], dim=1)

EPS = 1e-6

def gamma_t_si(t):
    return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)

def d_gamma_dt_si(t):
    return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))


class MinMaxNormalizer:
    """Helper for normalizing data to [-1, 1]"""
    def __init__(self, data=None):
        if data is not None:
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
            self.scale = self.max - self.min
            self.scale[self.scale == 0] = 1.0
        else:
            self.min = None; self.max = None; self.scale = None
            
    def normalize(self, x):
        norm = (x - self.min) / self.scale
        return norm * 2 - 1
    
    # [IMPORTANT] Helper for STEG to calculate physical distance inside AutoGrad
    def unnormalize_diff_differentiable(self, x_norm_tensor, device):
        """
        Differentiable unnormalization for Tensor inputs.
        Args:
            x_norm_tensor: (Batch, Dim) in [-1, 1]
        Returns:
            x_phys: Physical scale
        """
        input_dim = x_norm_tensor.shape[-1]
        current_min = self.min[:input_dim]
        current_scale = self.scale[:input_dim]
        min_t = torch.tensor(current_min, device=device, dtype=torch.float32)
        scale_t = torch.tensor(current_scale, device=device, dtype=torch.float32)
        
        return ((x_norm_tensor + 1) / 2) * scale_t + min_t

# [FIX] Trick pickle to find MinMaxNormalizer in 'dataset' module
sys.modules['dataset'] = sys.modules[__name__]

class RobotBrain:
    def __init__(self, ckpt_path="models/real_robot_final.ckpt", ccg_path="models/ccg_d_final.pth"):
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
        self.ccg_guidance_scale = 1.5  # 比 STEG 快，可以给大一点的引导强度
        self.ccg_activation_dist = 10.4  # 距离障碍物小于 40cm 开始激活

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
                # severity = max(0.0, (1.0 - dist_to_obs / self.ccg_activation_dist))
                severity = 1
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