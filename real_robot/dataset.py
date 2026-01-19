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
                 dataset_dir="processed_data", 
                 pred_horizon=16, 
                 obs_horizon=1, 
                 action_horizon=8):
        """
        参数:
            dataset_dir: 存放 CSV 的文件夹路径
            pred_horizon: 预测未来多少步 (Chunk size)
            obs_horizon: 使用过去多少步作为观测 (通常 State-based 为 1)
            action_horizon: 执行动作的步数 (用于推理，这里主要用于切片)
        """
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        
        # 1. 读取所有 CSV 文件
        # 假设文件名格式为 train_data_trial_0.csv ...
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
            # Robot State (Action): EE Pose (7) + Gripper (1) = 8 dims
            robot_state = df[['ee_x', 'ee_y', 'ee_z', 'ee_qx', 'ee_qy', 'ee_qz', 'ee_qw', 'gripper']].values
            
            # Condition: Object Pose (3) = 3 dims
            obj_pose = df[['obj_x', 'obj_y', 'obj_z']].values
            
            # 合并所有特征用于统计归一化参数: [robot_state, obj_pose] (11 dims)
            data_combined = np.concatenate([robot_state, obj_pose], axis=1)
            
            self.trials.append({
                'robot_state': robot_state, # (T, 8)
                'obj_pose': obj_pose        # (T, 3)
            })
            all_data_list.append(data_combined)

        # 3. 计算全局归一化参数 (Fit Normalizer)
        all_data_concat = np.concatenate(all_data_list, axis=0)
        
        # 这里的 normalizer 负责处理所有维度 (11 dims)
        # 前8维是 robot, 后3维是 obj
        self.normalizer = MinMaxNormalizer(all_data_concat)
        
        print("归一化统计完成:")
        print(f"Min: {self.normalizer.min}")
        print(f"Max: {self.normalizer.max}")

        # 4. 构建索引 (Indices)
        # 我们需要创建一个列表，把 (trial_index, start_time_index) 映射到 flat index
        self.indices = []
        for i, trial in enumerate(self.trials):
            trial_len = trial['robot_state'].shape[0]
            # 只有当剩余长度足以提供 obs_horizon 时才有效
            # 但实际上，为了最大化利用数据，我们通常允许滑窗滑到最后
            # 只要 start_idx < trial_len 即可
            for start_idx in range(trial_len):
                 self.indices.append((i, start_idx))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        trial_idx, start_idx = self.indices[idx]
        trial = self.trials[trial_idx]
        
        robot_state = trial['robot_state'] # (T, 8)
        obj_pose = trial['obj_pose']       # (T, 3)
        total_len = robot_state.shape[0]

        # --- 1. 获取窗口索引 ---
        # 结束索引
        end_idx = start_idx + self.pred_horizon
        
        # --- 2. 提取数据 (Observation & Action) ---
        
        # 观测 (Current State): 取 start_idx 这一帧
        # obs: [robot_state, obj_pose]
        curr_robot = robot_state[start_idx]
        curr_obj = obj_pose[start_idx]
        # 拼接成 (11,)
        raw_obs = np.concatenate([curr_robot, curr_obj], axis=0)
        
        # 动作 (Future Trajectory): 取 [start_idx : end_idx]
        # 注意: Action 通常只包含 robot_state (8 dims)，不需要 obj_pose
        # 处理边界 Padding: 如果超出长度，复制最后一帧
        if end_idx <= total_len:
            raw_action = robot_state[start_idx : end_idx]
        else:
            # 拿有效部分
            valid_len = total_len - start_idx
            valid_part = robot_state[start_idx : total_len]
            # 拿最后一帧
            last_frame = robot_state[total_len - 1]
            # 复制填充
            padding = np.tile(last_frame, (self.pred_horizon - valid_len, 1))
            raw_action = np.concatenate([valid_part, padding], axis=0)

        # --- 3. 归一化 (Normalize) ---
        
        # Obs 包含所有 11 维，直接用 normalizer
        nobs = self.normalizer.normalize(raw_obs)
        
        # Action 只包含前 8 维
        # 我们需要手动从 normalizer 里提取前 8 维的参数来归一化 action
        # 这是一个小技巧
        action_min = self.normalizer.min[:8]
        action_scale = self.normalizer.scale[:8]
        
        # 手动归一化 action -> [-1, 1]
        naction = (raw_action - action_min) / action_scale
        naction = naction * 2 - 1

        # --- 4. 转 Tensor ---
        nobs_tensor = torch.from_numpy(nobs).float()       # (11,)
        naction_tensor = torch.from_numpy(naction).float() # (pred_horizon, 8)

        return {
            'obs': nobs_tensor,
            'action': naction_tensor
        }

    def get_normalizer(self):
        return self.normalizer

# --- 测试代码 ---
if __name__ == "__main__":
    # 简单的测试
    from torch.utils.data import DataLoader
    
    # 假设你的数据在 processed_data 文件夹
    dataset = RealRobotDataset(dataset_dir="processed_data", pred_horizon=16)
    
    print(f"Dataset 长度: {len(dataset)}")
    
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    for batch in loader:
        obs = batch['obs']
        action = batch['action']
        
        print("\nBatch Shape:")
        print(f"Obs (Input): {obs.shape}")     # 应该 [4, 11]
        print(f"Action (Target): {action.shape}")  # 应该 [4, 16, 8]
        
        # 验证归一化范围
        print(f"Obs Range: [{obs.min():.2f}, {obs.max():.2f}]")
        print(f"Action Range: [{action.min():.2f}, {action.max():.2f}]")
        break