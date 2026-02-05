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
# from robo_control import RoboControl # Assuming this is your local file
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray, Int32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

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

# ==============================================================================
# SECTION 2: HELPER UTILITIES
# ==============================================================================

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
        min_t = torch.tensor(self.min, device=device, dtype=torch.float32)
        scale_t = torch.tensor(self.scale, device=device, dtype=torch.float32)
        return ((x_norm_tensor + 1) / 2) * scale_t + min_t

# [FIX] Trick pickle to find MinMaxNormalizer in 'dataset' module
sys.modules['dataset'] = sys.modules[__name__]

EPS = 1e-6

def gamma_t_si(t):
    return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)

def d_gamma_dt_si(t):
    return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    trash = []
    rvecs = []
    tvecs = []
    for c in corners:
        nada, R, t = cv2.solvePnP(marker_points, c, mtx, distortion, False, cv2.SOLVEPNP_IPPE_SQUARE)
        rvecs.append(R)
        tvecs.append(t)
        trash.append(nada)
    return rvecs, tvecs, trash

def matrix_to_rotation_6d(matrix) -> torch.Tensor:
    return torch.cat([matrix[:, :, 0], matrix[:, :, 1]], dim=1)

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[:, 0:3], d6[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b3 = F.normalize(torch.cross(b1, a2, dim=1), dim=1)
    b2 = torch.cross(b3, b1, dim=1)
    return torch.stack((b1, b2, b3), dim=2)

# ==============================================================================
# SECTION 3: MAIN INFERENCE CLASS (With Streaming STEG)
# ==============================================================================

class RobotBrain:
    def __init__(self, ckpt_path="models/real_robot_epoch_700.ckpt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Brain] Inference Device: {self.device}")

        # --- Configuration ---
        self.pred_horizon = 16
        self.obs_horizon = 1
        self.action_dim = 10 
        self.obs_dim = 13 
        self.dt_val = 1.0 / (self.pred_horizon - self.obs_horizon)
        self.chunk_size = 8 
        
        # --- Standard Diffusion Params ---
        self.sigma_infer = 0.0

        # --- STEG Guidance Params ---
        self.steg_guidance_scale = 15.0
        self.steg_activation_dist = 0.3 
        self.steg_n_ensemble = 16
        self.steg_k_horizon = 3
        self.obstacle_radius = 0.05
        self.sigma_explore = 0.05

        # --- Load Models (Same as before) ---
        print(f"[Brain] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        self.velocity_net = ConditionalUnet1D(
            input_dim=self.action_dim, global_cond_dim=self.obs_dim * self.obs_horizon,
            updownsample_type='Linear', sin_embedding_scale=100
        ).to(self.device)
        self.velocity_net.load_state_dict(checkpoint['velocity_net'])
        self.velocity_net.eval()

        self.denoiser_net = ConditionalUnet1D(
            input_dim=self.action_dim, global_cond_dim=self.obs_dim * self.obs_horizon,
            updownsample_type='Linear', sin_embedding_scale=100
        ).to(self.device)
        self.denoiser_net.load_state_dict(checkpoint['denoiser_net'])
        self.denoiser_net.eval()

        self.normalizer = checkpoint['normalizer']
        print("[Brain] Model & Normalizer loaded successfully.")
        
        # --- Vision Setup (Same as before) ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.target_id = 1
        self.camera_matrix = np.array([[600.80, 0.0, 325.53],[0.0, 600.63, 252.90],[0.0, 0.0, 1.0]], dtype=float)
        self.dist_coeffs = np.zeros(5)

        # --- STREAMING STATE VARIABLES ---
        # 这些是实现 "In-Context Guidance" 的关键
        self.chunk_step_counter = 0     # 当前在 Chunk 的第几步 (0-7)
        self.cached_global_cond = None  # 锁定的视觉观测 (Conditioning)
        self.na = None                  # 当前的 Latent State
        self.step_idx = 0               # 全局总步数 (Debug用)
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

    def compute_steg_gradient(self, na_curr, t_curr, global_cond, obstacle_pos_tensor):
        with torch.enable_grad():
            na_in = na_curr.detach().requires_grad_(True)
            na_ens = na_in.repeat(self.steg_n_ensemble, 1, 1)
            t_ens = t_curr.repeat(self.steg_n_ensemble)
            cond_ens = global_cond.repeat(self.steg_n_ensemble, 1)

            curr_na_sim = na_ens
            curr_t_sim = t_ens
            cum_cost = torch.zeros(self.steg_n_ensemble, device=self.device)

            # Forward Simulate K steps
            for k in range(self.steg_k_horizon):
                t_input = torch.clamp(curr_t_sim, 1e-3, 0.99)
                
                # Predict drift for simulation
                v_p = self.velocity_net(curr_na_sim, t_input, cond_ens)
                eta_p = self.denoiser_net(curr_na_sim, t_input, cond_ens)
                
                gamma = gamma_t_si(t_input).view(-1, 1, 1)
                gamma_dot = d_gamma_dt_si(t_input).view(-1, 1, 1)
                s_p = -eta_p / (gamma + 1e-6)
                
                # Simulation uses small fixed sigma for diversity
                score_coeff = 0.5 * (0.05 ** 2) - (gamma * gamma_dot) 
                b_drift_sim = v_p + score_coeff * s_p
                
                # --- Cost Calculation ---
                # Unnormalize to physical space
                curr_na_sim_flat = curr_na_sim.squeeze(1) 
                phys_vals = self.normalizer.unnormalize_diff_differentiable(curr_na_sim_flat, self.device)
                sim_xyz = phys_vals[:, :3]
                
                # Distance Cost
                dist_sq = ((sim_xyz - obstacle_pos_tensor) ** 2).sum(dim=-1)
                
                # Gaussian Repulsion Cost
                cost_step = self.steg_guidance_scale * torch.exp(- dist_sq / (2 * self.obstacle_radius**2))
                cum_cost = cum_cost + cost_step * self.dt_val

                # Update State (Euler-Maruyama)
                noise_step = torch.randn_like(curr_na_sim)
                curr_na_sim = curr_na_sim + b_drift_sim * self.dt_val + \
                              0.05 * math.sqrt(self.dt_val) * noise_step
                curr_t_sim = curr_t_sim + self.dt_val
            
            # Backprop
            neg_cost = -cum_cost
            log_utility = torch.logsumexp(neg_cost, dim=0)
            grads = torch.autograd.grad(log_utility, na_in)[0]
            return grads.detach()

    def calibrate_object(self, img):
        print("[Vision] Detecting ArUco for calibration...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None and self.target_id in ids:
            idx = np.where(ids == self.target_id)[0][0]
            rvec, tvec, _ = my_estimatePoseSingleMarkers(corners, 0.028, self.camera_matrix, self.dist_coeffs)
            T_cam_obj = np.eye(4)
            T_cam_obj[:3, :3] = cv2.Rodrigues(rvec[idx])[0]
            T_cam_obj[:3, 3] = tvec[idx].flatten()
            T_base_cam = np.eye(4) 
            try:
                cam_pose = rospy.wait_for_message("cam_pose_track", Float64MultiArray, timeout=1.0).data 
                x, y, z, qx, qy, qz, qw = cam_pose
                T_base_cam[:3, 3] = [x, y, z]
                T_base_cam[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            except:
                print("[Vision] Warning: cam_pose_track not available, using Identity")
                pass
            T_base_obj = T_base_cam @ T_cam_obj
            self.static_obj_pos = T_base_obj[:3, 3]
            print(f"[Vision] Object calibrated at: {self.static_obj_pos}")
            return True
        else:
            print("[Vision] ArUco marker not found.")
            return False

    def infer(self, ee_queue, image_queue, obstacle_pos=None):
        """
        Streaming Inference:
        Call Once -> Compute One ODE Step -> Return One Action -> Keep Latent State
        """
        
        # --- 1. Chunk Initialization (Step 0) ---
        if self.chunk_step_counter == 0:
            # Wait for data if empty
            if ee_queue.empty(): return None
            
            # Drain queue to get latest
            current_ee_pose = None
            while not ee_queue.empty(): current_ee_pose = ee_queue.get()
            
            self.static_obj_pos = [0.480729, -0.047977, 0.109436] 
            if self.static_obj_pos is None:
                if image_queue.empty(): return None
                if not self.calibrate_object(image_queue.get()): return None

            # Construct Observation & Lock it (Cache Global Cond)
            raw_obs = np.concatenate([current_ee_pose, self.static_obj_pos]).astype(np.float32)
            raw_obs = raw_obs.reshape(1, -1)
            nobs = self.normalizer.normalize(raw_obs)
            self.cached_global_cond = torch.from_numpy(nobs).float().to(self.device)
            
            # Initialize Latent State
            current_state_norm = nobs[:, :10]
            self.na = torch.from_numpy(current_state_norm).float().to(self.device).unsqueeze(1)
            
            print(f"[Brain] Starting new chunk execution sequence (Step 0-7)")

        # --- 2. Step-by-Step Generation ---
        # obstacle_pos is fresh every call, allowing "Still inside chunk... based on guidance"
        
        # Prepare Current Time t
        t_scalar = np.clip(self.chunk_step_counter * self.dt_val, 1e-3, 1.0 - 1e-3)
        t = torch.tensor([t_scalar], device=self.device, dtype=torch.float32)

        # Handle Obstacle Logic
        has_obstacle = False
        obs_tensor = None
        steg_grad = torch.zeros_like(self.na)
        steg_scale_curr = 0.0

        # Safe check for obstacle input
        if obstacle_pos is not None and len(obstacle_pos) == 3 and obstacle_pos[0] is not None:
            has_obstacle = True
            obs_tensor = torch.tensor(obstacle_pos, device=self.device, dtype=torch.float32)
            
            # Distance Check
            curr_phys = self.normalizer.unnormalize_diff_differentiable(self.na.squeeze(1), self.device)
            dist_to_obs = torch.norm(curr_phys[0, :3] - obs_tensor).item()
            
            if dist_to_obs < self.steg_activation_dist:
                # Trigger STEG
                steg_grad = self.compute_steg_gradient(self.na, t, self.cached_global_cond, obs_tensor)
                severity = max(0, (1.0 - dist_to_obs / self.steg_activation_dist))
                steg_scale_curr = self.steg_guidance_scale * severity

        # --- 3. Integration Step (Model + Guidance) ---
        with torch.no_grad():
            # Base Drift
            b_drift_base = self.get_drift(self.na, t, self.cached_global_cond)
            
            # Add Guidance
            final_drift = b_drift_base + steg_scale_curr * steg_grad
            
            # Euler-Maruyama Integration
            noise = torch.randn_like(self.na)
            sigma_step = self.sigma_explore if has_obstacle else 0.0
            
            # Update Latent State (self.na is kept for next call)
            self.na = self.na + final_drift * self.dt_val + \
                      sigma_step * math.sqrt(self.dt_val) * noise

            # Decode Action
            na_cpu = self.na.detach().cpu().numpy().squeeze()
            action_min = self.normalizer.min[:10]
            action_scale = self.normalizer.scale[:10]
            raw_action = ((na_cpu + 1) / 2) * action_scale + action_min

        # --- 4. Counter Management ---
        self.chunk_step_counter += 1
        
        # Reset if chunk finished
        if self.chunk_step_counter >= self.chunk_size:
            self.chunk_step_counter = 0
            
        return raw_action

# ==============================================================================
# SECTION 4: CONTROL NODE
# ==============================================================================

# Dummy class for execution context (replace with your actual import)
class RoboControl:
    def __init__(self):
        self.ee_pose = [0]*8
        self.ee_pose_queue = queue.Queue()
        self.image_queue = queue.Queue()
    def execute_rot(self, action):
        pass # Placeholder

if __name__ == "__main__":
    ee_pose_queue = queue.Queue()
    image_queue = queue.Queue()
    
    brain = RobotBrain(ckpt_path="models/real_robot_final.ckpt")
    
    print("[Control] Starting Loop...")
    rospy.init_node("inference_node")
    control_node = RoboControl()

    time.sleep(2)
    # Move to initial pose
    starting_pose = [0.320236, 0.078831, 0.416621, 0.950743, -0.298455, 0.083734, -0.289216, -0.951283, -0.106830, 3]
    control_node.execute_rot(starting_pose)
    time.sleep(5)

    try:
        while not rospy.is_shutdown():
            # 1. Get Robot State
            # (In real usage, this should come from control_node callbacks)
            # mocking data for structure
            current_pose = [0.32, 0.08, 0.41, 0, 0, 0, 1, 0] # x,y,z,qx,qy,qz,qw,g
            x,y,z, qx,qy,qz,qw, g = current_pose
            p_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            pose_6d = list(map(float, matrix_to_rotation_6d(torch.tensor(p_mat).unsqueeze(0))[0]))
            d1, d2, d3, d4, d5, d6 = pose_6d
            control_node.ee_pose_queue.put([x,y,z,d1,d2,d3,d4,d5,d6,g])

            # 2. Get Obstacle Info
            # Simulate receiving data from perception node
            obstacle_data = [None]
            # Example: Force an obstacle if step > 50
            # if brain.step_idx > 50:
            #     obstacle_data = [0.45, 0.0, 0.2]

            # 3. Inference
            # The 'streaming' nature is handled internally by brain.chunk_step_counter
            action = brain.infer(control_node.ee_pose_queue, control_node.image_queue, obstacle_pos=obstacle_data)

            # 4. Execute
            if action is not None:
                action = list(map(float, action))
                control_node.execute_rot(action)
                print(f"[Execute] Step {brain.step_idx} (ChunkStep {brain.chunk_step_counter-1}): Action {action[:3]}")
                brain.step_idx += 1
            
            time.sleep(0.05) # 20Hz Control Loop

    except KeyboardInterrupt:
        print("[Control] Stopping...")