import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import glob

class MinMaxNormalizer:
    """
    用于将数据归一化到 [-1, 1] 的工具类
    """
    def __init__(self, data=None):
        # 初始化时计算 min 和 max
        if data is not None:
            self.min = np.min(data, axis=0)
            self.max = np.max(data, axis=0)
            # 防止除以0 (如果某列数值完全一样)
            self.scale = self.max - self.min
            self.scale[self.scale == 0] = 1.0
        else:
            self.min = None
            self.max = None
            self.scale = None

    def load_stats(self, min_val, max_val):
        """直接加载预计算的统计值"""
        self.min = min_val
        self.max = max_val
        self.scale = self.max - self.min
        self.scale[self.scale == 0] = 1.0

    def normalize(self, x):
        """ [x_min, x_max] -> [-1, 1] """
        # 1. 归一化到 [0, 1]
        norm = (x - self.min) / self.scale
        # 2. 映射到 [-1, 1]
        return norm * 2 - 1

    def unnormalize(self, x):
        """ [-1, 1] -> [x_min, x_max] """
        # 1. 映射回 [0, 1]
        x_01 = (x + 1) / 2
        # 2. 映射回原范围
        return x_01 * self.scale + self.min

class RealRobotDataset(Dataset):
    def __init__(self, 
                 dataset_dir="processed_6Ddata", 
                 pred_horizon=16, 
                 obs_horizon=2,  # 默认匹配 DP 的历史长度 2
                 action_horizon=8):
        """
        参数:
            dataset_dir: 存放 CSV 的文件夹路径
            pred_horizon: 预测未来多少步 (Chunk size)
            obs_horizon: 使用过去多少步作为观测 
            action_horizon: 执行动作的步数 (用于推理，这里主要用于切片)
        """
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        
        # 1. 读取所有 CSV 文件
        csv_files = sorted(glob.glob(os.path.join(dataset_dir, "train_data_trial_*.csv")))
        
        if len(csv_files) == 0:
            raise ValueError(f"在 {dataset_dir} 没找到 .csv 文件，请检查路径。")
            
        print(f"找到 {len(csv_files)} 个数据文件。正在加载...")

        self.trials = []
        all_data_list = []

        # 2. 加载数据并根据 Trial 分组
        for csv_path in csv_files:
            df = pd.read_csv(csv_path)
            
            # 提取需要的列
            # Robot State (Action): EE Pose (3) + 6D Rot (6) + Gripper (1) = 10 dims
            robot_state = df[['ee_x', 'ee_y', 'ee_z', 'pose_6d_0', 'pose_6d_1', 'pose_6d_2', 'pose_6d_3', 'pose_6d_4', 'pose_6d_5', 'gripper']].values
            
            # Condition: Object Pose (3) = 3 dims
            obj_pose = df[['obj_x', 'obj_y', 'obj_z']].values
            
            # 合并所有特征用于统计归一化参数: [robot_state, obj_pose] (13 dims)
            data_combined = np.concatenate([robot_state, obj_pose], axis=1)
            
            self.trials.append({
                'robot_state': robot_state, # (T, 10)
                'obj_pose': obj_pose        # (T, 3)
            })
            all_data_list.append(data_combined)

        # 3. 计算全局归一化参数 (Fit Normalizer)
        all_data_concat = np.concatenate(all_data_list, axis=0)
        
        # 这里的 normalizer 负责处理所有维度 (13 dims)
        # 前10维是 robot, 后3维是 obj
        self.normalizer = MinMaxNormalizer(all_data_concat)
        
        print("归一化统计完成:")
        print(f"Min: {self.normalizer.min}")
        print(f"Max: {self.normalizer.max}")

        # 4. 构建索引 (Indices)
        self.indices = []
        for i, trial in enumerate(self.trials):
            trial_len = trial['robot_state'].shape[0]
            for start_idx in range(trial_len):
                 self.indices.append((i, start_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial_idx, start_idx = self.indices[idx]
        trial = self.trials[trial_idx]
        
        robot_state = trial['robot_state'] # (T, 10)
        obj_pose = trial['obj_pose']       # (T, 3)
        total_len = robot_state.shape[0]

        # --- 1. 获取窗口索引 ---
        end_idx = start_idx + self.pred_horizon
        
        # --- 2. 提取数据 (Observation & Action) ---
        
        # 观测 (Current State + History)
        obs_start_idx = start_idx - self.obs_horizon + 1
        pad_len = 0
        
        # 如果历史帧不足，计算需要用第一帧 padding 的长度
        if obs_start_idx < 0:
            pad_len = -obs_start_idx
            obs_start_idx = 0
            
        hist_robot = robot_state[obs_start_idx : start_idx + 1]
        hist_obj = obj_pose[obs_start_idx : start_idx + 1]
        raw_obs = np.concatenate([hist_robot, hist_obj], axis=-1) # (T_obs, 13)
        
        # 在前面补全不足的历史帧
        if pad_len > 0:
            padding = np.tile(raw_obs[0], (pad_len, 1))
            raw_obs = np.concatenate([padding, raw_obs], axis=0) # (obs_horizon, 13)
        
        # 动作 (Future Trajectory): 取 [start_idx : end_idx]
        # 处理边界 Padding: 如果超出长度，复制最后一帧
        if end_idx <= total_len:
            raw_action = robot_state[start_idx : end_idx]
        else:
            valid_len = total_len - start_idx
            valid_part = robot_state[start_idx : total_len]
            last_frame = robot_state[total_len - 1]
            padding = np.tile(last_frame, (self.pred_horizon - valid_len, 1))
            raw_action = np.concatenate([valid_part, padding], axis=0)

        # --- 3. 归一化 (Normalize) ---
        
        # Obs 包含所有 13 维，直接用 normalizer
        nobs = self.normalizer.normalize(raw_obs)
        
        # Action 只包含前 10 维
        action_min = self.normalizer.min[:10]
        action_scale = self.normalizer.scale[:10]
        
        # 手动归一化 action -> [-1, 1]
        naction = (raw_action - action_min) / action_scale
        naction = naction * 2 - 1

        # --- 4. 转 Tensor ---
        nobs_tensor = torch.from_numpy(nobs).float()       # (obs_horizon, 13)
        naction_tensor = torch.from_numpy(naction).float() # (pred_horizon, 10)

        return {
            'obs': nobs_tensor,
            'action': naction_tensor
        }

    def get_normalizer(self):
        return self.normalizer

# --- 测试代码 ---
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    dataset = RealRobotDataset(dataset_dir="processed_6Ddata", pred_horizon=16, obs_horizon=2)
    print(f"Dataset 长度: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in loader:
        obs = batch['obs']
        action = batch['action']
        
        print("\nBatch Shape:")
        print(f"Obs (Input): {obs.shape}")     # 应该 [4, 2, 13] (Batch, Obs_Horizon, Dims)
        print(f"Action (Target): {action.shape}")  # 应该 [4, 16, 10] (Batch, Pred_Horizon, Dims)
        
        print(f"Obs Range: [{obs.min():.2f}, {obs.max():.2f}]")
        print(f"Action Range: [{action.min():.2f}, {action.max():.2f}]")
        break