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
from robo_control import RoboControl
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray, Int32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R
import numpy as np
import queue

# ==============================================================================
# SECTION 1: MODEL ARCHITECTURE (Fixed to match training.py)
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

        # [FIX] Match training definition: Conv1dBlock instead of raw Conv1d+Norm+Mish
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
# SECTION 2: HELPER UTILITIES (Normalization, ArUco)
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

# [FIX] Trick pickle to find MinMaxNormalizer in 'dataset' module
sys.modules['dataset'] = sys.modules[__name__]

EPS = 1e-6

def gamma_t_si(t):
    return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)

def d_gamma_dt_si(t):
    return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    '''
    This will estimate the rvec and tvec for each of the marker corners detected by:
       corners, ids, rejectedImgPoints = detector.detectMarkers(image)
    corners - is an array of detected corners for each detected marker in the image
    marker_size - is the size of the detected markers
    mtx - is the camera matrix
    distortion - is the camera distortion matrix
    RETURN list of rvecs, tvecs, and trash (so that it corresponds to the old estimatePoseSingleMarkers())
    '''
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

# ==============================================================================
# SECTION 3: MAIN INFERENCE CLASS
# ==============================================================================

class RobotBrain:
    def __init__(self, ckpt_path="models/real_robot_epoch_700.ckpt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Brain] Inference Device: {self.device}")

        # --- Configuration ---
        self.pred_horizon = 16
        self.obs_horizon = 1
        self.action_dim = 10  # 7 pose + 1 gripper
        self.obs_dim = 13    # 8 robot + 3 obj
        self.dt_val = 1.0 / (self.pred_horizon - self.obs_horizon) # Integration step size
        self.sigma_infer = 0.0 # Deterministic ODE mode
        
        # [NEW] Chunk Streaming Params
        self.chunk_size = 8 # Matches PushT action_horizon

        # --- Load Checkpoint ---
        print(f"[Brain] Loading checkpoint: {ckpt_path}")
        # [FIX] Enable safe global loading for custom classes
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        
        # Initialize Velocity Net
        self.velocity_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon,
            updownsample_type='Linear',
            sin_embedding_scale=100,
        ).to(self.device)
        self.velocity_net.load_state_dict(checkpoint['velocity_net'])
        self.velocity_net.eval()

        # Initialize Denoiser Net
        self.denoiser_net = ConditionalUnet1D(
            input_dim=self.action_dim,
            global_cond_dim=self.obs_dim * self.obs_horizon,
            updownsample_type='Linear',
            sin_embedding_scale=100,
        ).to(self.device)
        self.denoiser_net.load_state_dict(checkpoint['denoiser_net'])
        self.denoiser_net.eval()

        # Load Normalizer Stats
        self.normalizer = checkpoint['normalizer']
        print("[Brain] Model & Normalizer loaded successfully.")

        # --- State Variables ---
        self.na_from_prev_chunk = None # Handoff variable
        self.step_idx = 0
        self.static_obj_pos = None 
        self.internal_action_queue = queue.Queue() # Buffer for open-loop actions

        # --- Vision Setup (ArUco) ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        self.target_id = 1
        self.camera_matrix = np.array([
            [600.80, 0.0, 325.53],
            [0.0, 600.63, 252.90],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        self.dist_coeffs = np.zeros(5)

    def get_drift(self, x, t, global_cond):
        """Calculates the corrected drift term b(x, t)"""
        if t.ndim == 0: t = t.unsqueeze(0)
        
        # Predict velocity and noise (eta)
        v_pred = self.velocity_net(sample=x, timestep=t, global_cond=global_cond)
        eta_pred = self.denoiser_net(sample=x, timestep=t, global_cond=global_cond)
        
        # Calculate gamma and its time derivative
        gamma = gamma_t_si(t).view(-1, 1, 1).to(self.device)
        gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(self.device)
        
        # Compute score and drift coefficient
        s_pred = -eta_pred / (gamma + EPS)
        score_coeff = 0.5 * (self.sigma_infer ** 2) - (gamma * gamma_dot)
        
        b = v_pred + score_coeff * s_pred
        return b

    def calibrate_object(self, img):
        print("[Vision] Detecting ArUco for calibration...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        
        if ids is not None and self.target_id in ids:
            idx = np.where(ids == self.target_id)[0][0]
            # PnP Solver
            rvec, tvec, _ = my_estimatePoseSingleMarkers(
                corners, 0.028, self.camera_matrix, self.dist_coeffs # 0.03m marker size
            )
            
            # T_cam_obj (Object relative to Camera)
            T_cam_obj = np.eye(4)
            T_cam_obj[:3, :3] = cv2.Rodrigues(rvec[idx])[0]
            T_cam_obj[:3, 3] = tvec[idx].flatten()

            # T_base_cam (Camera relative to Base)
            # TODO: Replace Identity matrix with real camera extrinsic calibration!
            # print("[Vision] WARNING: Using Identity Matrix for Camera Pose. Replace this!")
            T_base_cam = np.eye(4) 
            cam_pose = rospy.wait_for_message("cam_pose_track", Float64MultiArray).data 
            x, y, z, qx, qy, qz, qw = cam_pose
            T_base_cam[:3, 3] = [x, y, z]
            T_base_cam[:3, :3] = R.from_quat([qx, qy, qz, qw]).as_matrix()
            
            # T_base_obj = T_base_cam * T_cam_obj
            T_base_obj = T_base_cam @ T_cam_obj
            self.static_obj_pos = T_base_obj[:3, 3]
            print(f"[Vision] Object calibrated at: {self.static_obj_pos}")
            return True
        else:
            print("[Vision] ArUco marker not found.")
            return False

    def infer(self, ee_queue, image_queue):
        """
        Chunked Streaming Logic:
        1. If action buffer has pending actions, return one (Open Loop).
        2. If empty, grab LATEST obs, replan chunk, fill buffer (Close Loop).
        """
        # A. Check internal buffer (Fast path)
        if not self.internal_action_queue.empty():
            # next_action = self.internal_action_queue.get()
            # next_action = apply_g_term(next_action)
            return self.internal_action_queue.get()

        # B. Re-planning path (Slow path)
        if ee_queue.empty(): return None
        
        # Drain queue to get LATEST observation
        current_ee_pose = None
        while not ee_queue.empty():
            current_ee_pose = ee_queue.get()
        
        # First run calibration
        self.static_obj_pos = [0.480729496052082, -0.0479779403118259, 0.109436304152868]
        if self.static_obj_pos is None:
            if image_queue.empty(): 
                print("Waiting for image...")
                return None
            img = image_queue.get()
            if not self.calibrate_object(img): return None

        # Construct Obs
        current_gripper = 1.0 
        raw_obs = np.concatenate([
            current_ee_pose, 
            self.static_obj_pos
        ]).astype(np.float32)
        
        # Normalize (Handle shape mismatch)
        raw_obs = raw_obs.reshape(1, -1) # Ensure (1, 11)
        nobs = self.normalizer.normalize(raw_obs)
        global_cond = torch.from_numpy(nobs).float().to(self.device) # (1, 11)

        # # Warm Start Logic
        # if self.na_from_prev_chunk is None:
        #     print("[Brain] First Chunk Warm Start...")
        #     # Init from current state
        #     current_state_norm = nobs[:, :8] # (1, 8)
        #     self.na = torch.from_numpy(current_state_norm).float().to(self.device).unsqueeze(1) # (1, 1, 8)
        # else:
        #     # Continue from previous chunk endpoint
        #     self.na = self.na_from_prev_chunk

        current_state_norm = nobs[:, :10] # (1, 8)
        self.na = torch.from_numpy(current_state_norm).float().to(self.device).unsqueeze(1)
        
        # Generate Chunk (Open Loop)
        print(f"[Brain] Generating new chunk of {self.chunk_size} steps...")
        chunk_actions = []
        
        with torch.no_grad():
            for i in range(self.chunk_size):
                # 1. Decode current na to action
                na_cpu = self.na.detach().cpu().numpy().squeeze()
                action_min = self.normalizer.min[:10]
                action_scale = self.normalizer.scale[:10]
                action_01 = (na_cpu + 1) / 2
                raw_action = action_01 * action_scale + action_min
                chunk_actions.append(raw_action)

                # 2. Integrate to next step
                # t starts from 0 for each new chunk!
                t_scalar = np.clip(i * self.dt_val, 1e-3, 1.0 - 1e-3)
                t = torch.tensor([t_scalar], device=self.device, dtype=torch.float32)
                
                b_drift = self.get_drift(self.na, t, global_cond)
                self.na = self.na + b_drift * self.dt_val
        
        # Save endpoint for next chunk
        self.na_from_prev_chunk = self.na

        # Fill buffer
        for act in chunk_actions:
            self.internal_action_queue.put(act)

        # print(chunk_actions[-1])
        # return chunk_actions[-1]
        # return chunk_actions
        return self.internal_action_queue.get()

# ==============================================================================
# SECTION 4: CONTROL NODE
# ==============================================================================
import csv
import os, csv
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R


def matrix_to_rotation_6d(matrix) -> torch.Tensor:
    """
    Converts 3x3 rotation matrix to 6D representation.

    Args:
        matrix: tensor of shape (batch_size, 3, 3)

    Returns:
        d6: tensor of shape (batch_size, 6)
    """
    # Take the first two columns (assuming column-major rotation matrix)
    # matrix[:, :, 0] is the first column vector (x-axis)
    # matrix[:, :, 1] is the second column vector (y-axis)

    batch_size = matrix.shape[0]

    # Flatten the first two columns
    return torch.cat([matrix[:, :, 0], matrix[:, :, 1]], dim=1)

def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks".

    Args:
        d6: tensor of shape (batch_size, 6)

    Returns:
        matrix: tensor of shape (batch_size, 3, 3)
    """
    # Split the 6D vector into two 3D vectors
    a1 = d6[:, 0:3]
    a2 = d6[:, 3:6]

    # 1. Normalize the first vector
    b1 = F.normalize(a1, dim=1)

    # 2. Get the third vector via cross product (b1 x a2)
    # This ensures the new vector is orthogonal to b1
    b3 = torch.cross(b1, a2, dim=1)
    b3 = F.normalize(b3, dim=1)

    # 3. Get the second vector via cross product (b3 x b1)
    # Since b1 and b3 are already orthogonal and normalized, b2 will be too.
    b2 = torch.cross(b3, b1, dim=1)

    # Stack columns to form the matrix
    # Shape becomes (batch_size, 3, 3)
    rot_matrix = torch.stack((b1, b2, b3), dim=2)

    return rot_matrix

if __name__ == "__main__":
    ee_pose_queue = queue.Queue()
    image_queue = queue.Queue()
    
    brain = RobotBrain(ckpt_path="models/real_robot_final.ckpt")
    
    print("[Control] Starting Loop...")
    
    rospy.init_node("inference_node")
    control_node = RoboControl()

    
    # p1 = [0.950743152057554, -0.298455893749636, 0.0837349288037733, -0.289216675656261, -0.951283797069278, -0.10683095036458]
    # p2 = [0.29702456684121853, 0.9506287004432288, 0.08989705549351938, 0.94675637866214, -0.3054395227420242, 0.10177945475230041]
    # orig_euler = R.from_matrix(rotation_6d_to_matrix(torch.tensor([p1]))[0]).as_euler('xyz', degrees=True)
    # pred_euler = R.from_matrix(rotation_6d_to_matrix(torch.tensor([p2]))[0]).as_euler('xyz', degrees=True)
    # print(orig_euler, pred_euler)
    # with open(f"processed_data/train_data_trial_19.csv", newline="") as f:
    #     reader = csv.reader(f)
    #     aligned_data = []
    #     next(reader, None)
    #     prev_state = []
    #     timestep = 0
    #     for row in reader:
    #         t, x, y, z, rd1, rd2, rd3, rd4, rd5, rd6, g, ox, oy, oz = row
    #         # qx,qy,qz,qw = list(map(float, [qx,qy,qz,qw]))
    #         rd1, rd2, rd3, rd4, rd5, rd6 = list(map(float, [rd1, rd2, rd3, rd4, rd5, rd6]))
    #         x,y,z = list(map(float, [x,y,z]))
    #         aligned_data.append(row)
    #         brain.static_obj_pos = [ox, oy, oz]
    #         ee_pose_queue.put(np.array([x, y, z, rd1, rd2, rd3, rd4, rd5, rd6, g]))
    #         action = brain.infer(ee_pose_queue, None) ## <- we follow the first traj
    #         # print(action)

    #         _, _, _, tr1, tr2, tr3, tr4, tr5, tr6, _ = action

    #         if prev_state:
    #             pos_difference = np.linalg.norm(np.array([x,y,z], dtype=float) - np.array(action[:3], dtype=float))
    #             orig_euler = R.from_matrix(rotation_6d_to_matrix(torch.tensor([[rd1, rd2, rd3, rd4, rd5, rd6]]))[0]).as_euler('xyz', degrees=True)
    #             pred_euler = R.from_matrix(
    #                 rotation_6d_to_matrix(torch.tensor([[tr1, tr2, tr3, tr4, tr5, tr6]]))[0]).as_euler('xyz', degrees=True)


    #             print(f"Timestep: {timestep}" ,pos_difference, orig_euler, pred_euler)

    #         timestep += 1
    #         prev_state = [x, y, z, rd1, rd2, rd3, rd4, rd5, rd6, g]


    # control_node.execute_quat([0.439914347624, 0.0107089763166, 0.382976760373, -0.944773841919, -0.326438018738, 0.0267042497907, -0.0112911731737, 3])
    time.sleep(2)
    starting_pose = [0.320236173359633, 0.0788312456532665, 0.416621681042161, 0.950743152057554, -0.298455893749636, 0.0837349288037733, -0.289216675656261, -0.951283797069278, -0.10683095036458, 3]
    control_node.execute_rot(starting_pose)
    
    time.sleep(5)
    try:
        ## move to initial position
        
        while not rospy.is_shutdown():
            # Mock data (Replace with real ROS sub)
            current_pose = control_node.ee_pose
            x,y,z, qx,qy,qz,qw, g = current_pose
            p_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
            pose_6d = list(map(float,matrix_to_rotation_6d(torch.tensor(p_mat).unsqueeze(0))[0]))
            
            d1, d2, d3, d4, d5, d6 = pose_6d
            # print(pose_6d, d1, d2, d3, d4, d5, d6)
            # print([x,y,z,d1,d2,d3,d4,d5,d6,g])
            control_node.ee_pose_queue.put([x,y,z,d1,d2,d3,d4,d5,d6,g])
    
            action = brain.infer(control_node.ee_pose_queue, control_node.image_queue)

            if action is not None:
                print(action)
                action = list(map(float, action))
                control_node.execute_rot(action)
                print(f"[Execute] Step {brain.step_idx}: Move to {action[:3]}")
                brain.step_idx += 1
    
            time.sleep(1/10) # 10Hz Control Loop
    
    except KeyboardInterrupt:
        print("[Control] Stopping...")