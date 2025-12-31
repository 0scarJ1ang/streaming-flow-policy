import numpy as np
import torch
import h5py
from typing import Dict, List
from tqdm import tqdm

# ================= 基础工具函数 =================

def create_sample_indices(
        episode_ends: np.ndarray,
        sequence_length: int,
        pad_before: int = 0,
        pad_after: int = 0
    ) -> np.ndarray:
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i-1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    return np.array(indices)

def sample_sequence(train_data, sequence_length, buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0: data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length: data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result

# ================= 核心 Dataset 类 =================

class RobomimicSimpleDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 obs_keys: List[str] = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
                 ):
        
        # 1. 加载原始 HDF5 数据并转换格式
        # 我们在这里手动模拟 ReplayBuffer 的行为，将所有 episode 拼接在一起
        all_obs = []
        all_actions = []
        episode_ends = []
        
        # 处理旋转表示（如果需要保持 rotation_6d 等，需额外引入 RotationTransformer）
        # 这里演示基础的拼接逻辑
        with h5py.File(dataset_path, 'r') as f:
            demos = f['data']
            curr_idx = 0
            for i in tqdm(range(len(demos)), desc="Processing Robomimic HDF5"):
                demo = demos[f'demo_{i}']
                
                # 提取并拼接 obs_keys
                obs_parts = []
                for k in obs_keys:
                    part = demo['obs'][k][:]
                    if len(part.shape) == 1: part = part[:, None] # 确保是 (T, D)
                    obs_parts.append(part)
                obs_seq = np.concatenate(obs_parts, axis=-1).astype(np.float32)
                
                action_seq = demo['actions'][:].astype(np.float32)
                
                all_obs.append(obs_seq)
                all_actions.append(action_seq)
                
                curr_idx += len(action_seq)
                episode_ends.append(curr_idx)

        # 将列表转换为连续的大数组 (N, D)
        train_data = {
            'obs': np.concatenate(all_obs, axis=0),
            'action': np.concatenate(all_actions, axis=0)
        }
        episode_ends = np.array(episode_ends)

        # 2. 计算索引 (滑窗逻辑)
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1
        )

        # 3. 计算归一化统计量 (Min-Max)
        stats = dict()
        normalized_data = dict()
        for key, data in train_data.items():
            stats[key] = {
                'min': np.min(data, axis=0),
                'max': np.max(data, axis=0)
            }
            # 归一化到 [-1, 1]
            ndata = (data - stats[key]['min']) / (np.maximum(stats[key]['max'] - stats[key]['min'], 1e-5))
            normalized_data[key] = ndata * 2 - 1

        self.indices = indices
        self.stats = stats
        self.normalized_data = normalized_data
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # 获取采样索引
        buffer_start_idx, buffer_end_idx, \
        sample_start_idx, sample_end_idx = self.indices[idx]

        # 采样归一化后的序列
        nsample = sample_sequence(
            train_data=self.normalized_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # 裁剪观测长度
        nsample['obs'] = nsample['obs'][:self.obs_horizon, :].astype(np.float32)
        
        # 转换为 Tensor
        return {k: torch.from_numpy(v) for k, v in nsample.items()}
    
class RobomimicNextObsDataset(RobomimicSimpleDataset):
    """
    Robomimic数据集变体：将下一时刻的观测值（Next Observation）映射为当前时刻的动作（Action）。
    主要用于训练碰撞检测评价器（Collision Prediction Critic）或状态转移模型。
    该类通过将 t+1 时刻的末端执行器（EEF）位置赋予 t 时刻的动作标签，实现状态到状态的监督学习。
    """
    def __init__(self, 
                 dataset_path: str,
                 pred_horizon: int,
                 obs_horizon: int,
                 action_horizon: int,
                 obs_keys: List[str] = ['object', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
                 ):
        # 1. 调用父类构造函数，完成原始HDF5数据的加载、拼接及初步归一化
        super().__init__(
            dataset_path=dataset_path,
            pred_horizon=pred_horizon,
            obs_horizon=obs_horizon,
            action_horizon=action_horizon,
            obs_keys=obs_keys
        )

        # 2. 重新定义动作空间（Action Remapping）
        # 核心逻辑：将原本的控制指令替换为未来状态（此处默认为末端执行器的 Cartesian 坐标）
        
        # 获取归一化后的观测数据序列
        # 默认假设 'robot0_eef_pos' 在 obs_keys 的起始位置，占据前 3 个维度 (x, y, z)
        normalized_obs = self.normalized_data['obs']
        
        new_action_buffer = []
        
        # 3. 提取 Episode 边界信息
        # 必须按 Episode 进行切片处理，以防止在序列偏移时发生跨轨迹的数据污染（Data Leakage）
        with h5py.File(dataset_path, 'r') as f:
            episode_ends = []
            cumulative_idx = 0
            for i in range(len(f['data'])):
                demo_length = len(f['data'][f'demo_{i}/actions'])
                cumulative_idx += demo_length
                episode_ends.append(cumulative_idx)

        # 4. 执行时间步偏移逻辑 (Temporal Shifting)
        # 目标：构建 a_t = s_{t+1} 的映射关系
        current_start = 0
        for end_idx in episode_ends:
            # 提取当前轨迹的末端执行器位置序列 (T, 3)
            eef_pos_series = normalized_obs[current_start:end_idx, :3] 
            
            # 构建偏移序列：
            # 将序列整体向前平移一个单位，末尾填充最后一帧以保持长度一致
            next_state_as_action = np.concatenate([
                eef_pos_series[1:],      # t+1 步的状态 (T-1, 3)
                eef_pos_series[[-1]]     # 边界处理：最后一帧保持原地 (1, 3)
            ], axis=0)
            
            new_action_buffer.append(next_state_as_action)
            current_start = end_idx

        # 5. 更新归一化数据缓存
        # 替换原有的 action 字段，后续 __getitem__ 将自动返回偏移后的“动作”
        self.normalized_data['action'] = np.concatenate(new_action_buffer, axis=0)
        
        print(f"[Dataset] Action remapping completed. New action shape: {self.normalized_data['action'].shape}")