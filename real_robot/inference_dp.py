import torch
import torch.nn as nn
import numpy as np
import time
import math
import cv2
import json
import sys
import collections
from typing import Union, Literal
from torch import Tensor
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float64MultiArray, Int32MultiArray
from geometry_msgs.msg import PoseStamped, Pose
from scipy.spatial.transform import Rotation as R
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from robo_control import RoboControl # Assuming this is your local file

# ==============================================================================
# 0. 基础网络组件定义 (匹配 DP 训练时的结构)
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
            downsample_layer = ConvDownsample1d(dim_out) if not is_last else nn.Identity()
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim=cond_dim, kernel_size=kernel_size, n_groups=n_groups),
                downsample_layer,
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            upsample_layer = ConvUpsample1d(dim_in) if not is_last else nn.Identity()
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

def matrix_to_rotation_6d(matrix) -> torch.Tensor:
    return torch.cat([matrix[:, :, 0], matrix[:, :, 1]], dim=1)

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


# ==============================================================================
# 1. DP Robot Brain
# ==============================================================================

class RobotBrainDP:
    def __init__(self, ckpt_path="models/dp_policy_epoch_100.ckpt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[Brain] Inference Device: {self.device}")

        # --- DP 参数配置 ---
        self.pred_horizon = 16
        self.obs_horizon = 2
        self.action_dim = 10 
        self.obs_dim = 13 
        
        # 静态物体坐标 (如果是动态的，请通过检测模块更新)
        self.static_obj_pos = [0.480729, -0.047977, 0.109436] 
        
        # 观测缓存队列，用于存储连续的多帧历史
        self.obs_buffer = collections.deque(maxlen=self.obs_horizon)

        # 加载模型 Checkpoint
        print(f"[Brain] Loading DP Model from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.normalizer = checkpoint['normalizer']

        # 初始化网络 (注意参数：updownsample_type='Conv', scale=1)
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=self.action_dim, 
            global_cond_dim=self.obs_dim * self.obs_horizon, 
            updownsample_type='Conv', 
            sin_embedding_scale=1
        ).to(self.device)
        self.noise_pred_net.load_state_dict(checkpoint['model_state_dict'])
        self.noise_pred_net.eval()

        # 初始化推理用的 DDPM Scheduler
        self.num_diffusion_iters = 100
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

    def push_obs(self, ee_pose):
        """将当前机器人状态拼接物体坐标后推入历史队列"""
        x, y, z, qx, qy, qz, qw, g = ee_pose
        p_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
        d1, d2, d3, d4, d5, d6 = list(map(float, matrix_to_rotation_6d(torch.tensor(p_mat).unsqueeze(0))[0]))
        
        curr_robot = [x, y, z, d1, d2, d3, d4, d5, d6, g]
        curr_obs = np.concatenate([curr_robot, self.static_obj_pos]).astype(np.float32)
        self.obs_buffer.append(curr_obs)

    def is_ready(self):
        """检查历史帧是否收集完毕"""
        return len(self.obs_buffer) == self.obs_horizon

    def infer_chunk(self):
        """预测一整个 Chunk 的动作轨迹"""
        # 1. 准备条件变量
        # 堆叠历史观测: (obs_horizon, 13)
        raw_obs = np.stack(list(self.obs_buffer))
        # 归一化
        nobs = self.normalizer.normalize(raw_obs)
        # 转为 Tensor 并压平给网络: (1, obs_horizon * 13)
        nobs_tensor = torch.from_numpy(nobs).float().to(self.device).unsqueeze(0)
        obs_cond = nobs_tensor.flatten(start_dim=1)

        # 2. 采样纯高斯噪声作为初始动作
        # (Batch, pred_horizon, action_dim) -> (1, 16, 10)
        noisy_action = torch.randn(
            (1, self.pred_horizon, self.action_dim), device=self.device)

        # 3. 反向去噪循环
        self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
        with torch.no_grad():
            for k in self.noise_scheduler.timesteps:
                # 预测残差噪声
                noise_pred = self.noise_pred_net(
                    sample=noisy_action, 
                    timestep=k, 
                    global_cond=obs_cond
                )
                # Scheduler 计算去噪一步后的动作
                noisy_action = self.noise_scheduler.step(
                    model_output=noise_pred, 
                    timestep=k, 
                    sample=noisy_action
                ).prev_sample

        # 4. 反归一化到物理空间
        naction = noisy_action.squeeze(0).cpu().numpy() # (16, 10)
        action_min = self.normalizer.min[:10]
        action_scale = self.normalizer.scale[:10]
        raw_action_chunk = ((naction + 1) / 2) * action_scale + action_min

        return raw_action_chunk


# ==============================================================================
# 2. Control Node / Main Loop
# ==============================================================================

if __name__ == "__main__":
    # 注意把这里的 ckpt 换成你刚才训练跑出来的最终文件
    brain = RobotBrainDP(ckpt_path="models/dp_policy_epoch_100.ckpt")
    logger = open("debug.txt", 'w')
    
    print("[Control] Starting Loop...")
    rospy.init_node("inference_node")
    control_node = RoboControl()

    time.sleep(2)
    
    # 移动到初始位置
    starting_pose = [0.320236, 0.078831, 0.416621, 0.950743, -0.298455, 0.083734, -0.289216, -0.951283, -0.106830, 3]
    control_node.execute_rot(starting_pose)
    control_node.open_gripper()
    
    time.sleep(5)
    
    # DP Chunk 执行参数
    action_horizon = 8     # 每次预测16帧，执行前8帧就重新规划
    action_chunk = None
    step_in_chunk = 0      # 当前执行到 chunk 的第几步
    
    last_action = None

    try:
        while not rospy.is_shutdown():
            start_time = time.time()
            
            # 1. 获取机器人状态并存入缓存
            current_pose = control_node.ee_pose
            brain.push_obs(current_pose)

            # 更新目标物体坐标（用于可视化或机械臂底层）
            control_node.target_obj_pose = brain.static_obj_pos

            # 如果缓存还不满 (刚启动)，继续等下一帧
            if not brain.is_ready():
                time.sleep(1/30)
                continue

            # 2. 判断是否需要重新预测 (Replanning)
            if action_chunk is None or step_in_chunk >= action_horizon:
                # 重新推理出一整段 16 步的动作
                action_chunk = brain.infer_chunk()
                step_in_chunk = 0 # 重置计数器
                # print("[Control] Replanning... New chunk generated.")

            # 3. 从当前 Chunk 中取出一个动作去执行
            action = action_chunk[step_in_chunk]
            step_in_chunk += 1
            
            # (Debug 记录与打印)
            if last_action is not None:
                distance_to_obj = np.linalg.norm(np.array(current_pose[:3]) - np.array(last_action[:3]), 1)
                logger.write(f"{distance_to_obj}\n")
            
            last_action = action
            action_list = list(map(float, action))
            
            # 4. 下发执行
            control_node.execute_rot(action_list)

            # 保证控制频率 30Hz
            elapsed = time.time() - start_time
            sleep_time = max(0.0, (1.0 / 30.0) - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("[Control] Stopping...")
        logger.close()