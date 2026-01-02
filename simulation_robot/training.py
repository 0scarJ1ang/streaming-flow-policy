import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import h5py
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict

# =================================================================
# 第一部分：定义“大科学家级别”的大脑 (Conditional U-Net)
# =================================================================

class Conv1dBlock(nn.Module):
    """最基础的“缝纫针”：卷积 -> 组归一化 -> 激活"""
    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    def forward(self, x): return self.block(x)

class ConditionalUnet1D(nn.Module):
    """
    这是您在笔记本里看到的那个大模型。它会像织毛衣一样，
    把动作序列（Action）和环境底色（Obs）交织在一起。
    """
    def __init__(self, input_dim, global_cond_dim, down_dims=[256, 512, 1024]):
        super().__init__()
        # 时间步嵌入（虽然咱们不是扩散模型，但保留这个结构可以让脑子更灵光）
        self.diffusion_step_mlp = nn.Sequential(
            nn.Linear(256, 512), nn.Mish(), nn.Linear(512, 512)
        )
        
        # 编码器部分：把信息一点点压缩、提炼
        in_out = [(input_dim, down_dims[0])] + list(zip(down_dims[:-1], down_dims[1:]))
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            self.down_modules.append(nn.ModuleList([
                Conv1dBlock(dim_in, dim_out, kernel_size=5),
                Conv1dBlock(dim_out, dim_out, kernel_size=5),
                nn.Conv1d(dim_out, dim_out, 3, stride=2, padding=1) if ind < len(in_out)-1 else nn.Identity()
            ]))
            
        # 中间连接层
        mid_dim = down_dims[-1]
        self.mid_block1 = Conv1dBlock(mid_dim, mid_dim, kernel_size=5)
        self.mid_block2 = Conv1dBlock(mid_dim, mid_dim, kernel_size=5)
        
        # 全局环境底色注入（Obs 融合）
        self.global_cond_mlp = nn.Sequential(
            nn.Linear(global_cond_dim, mid_dim * 4), nn.Mish(), nn.Linear(mid_dim * 4, mid_dim)
        )
        
        # 最后的判别层：把大模型提炼的特征转成“安全分数”
        self.final_head = nn.Sequential(
            nn.Linear(mid_dim, 64), nn.Mish(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, sample, global_cond):
        # sample: (B, T, Da) -> 换成卷积喜欢的形状 (B, Da, T)
        x = sample.transpose(1, 2)
        
        # 注入环境信息 (底色)
        cond = self.global_cond_mlp(global_cond) # (B, mid_dim)
        
        # 逐层向下卷积提取特征
        h = []
        for m1, m2, downsample in self.down_modules:
            x = m1(x)
            x = m2(x)
            h.append(x)
            x = downsample(x)
            
        # 中间精加工
        x = self.mid_block1(x)
        x = self.mid_block2(x)
        
        # 注意：这里我们不像 U-Net 那样再向上卷积了，因为咱们只需要一个“分数”
        # 咱们直接把中间最浓缩的特征拿出来，加上环境信息
        x = x.mean(dim=-1) # (B, mid_dim)
        x = x + cond
        return self.final_head(x)

# =================================================================
# 第二部分：定义“量体裁衣”的数据集 (Dataset)
# =================================================================

class RobomimicNextObsDataset(torch.utils.data.Dataset):
    """
    正式的 Robomimic 数据集：将 t+1 时刻的 EEF 位置重映射为 t 时刻的动作。
    """
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        
        all_obs, all_actions, episode_ends = [], [], []
        with h5py.File(dataset_path, 'r') as f:
            demos = f['data']
            curr_idx = 0
            for i in tqdm(range(len(demos)), desc="读取HDF5数据"):
                demo = demos[f'demo_{i}']
                # 拼接观察值：物体位置 + 手部坐标
                obs_seq = np.concatenate([
                    demo['obs/object'][:], 
                    demo['obs/robot0_eef_pos'][:]
                ], axis=-1).astype(np.float32)
                
                # 时间步偏移：下一步的位置就是现在的“动作”
                # 下一步 = [第2步, 第3步... 最后一步, 最后一步]
                next_pos_action = np.concatenate([obs_seq[1:, -3:], obs_seq[[-1], -3:]], axis=0)
                
                all_obs.append(obs_seq)
                all_actions.append(next_pos_action)
                curr_idx += len(obs_seq)
                episode_ends.append(curr_idx)

        self.obs_data = np.concatenate(all_obs, axis=0)
        self.action_data = np.concatenate(all_actions, axis=0)
        self.episode_ends = np.array(episode_ends)
        
        # 简单的归一化：洗洗布料
        self.stats = {'obs': self._get_stat(self.obs_data), 'action': self._get_stat(self.action_data)}
        self.obs_data = self._norm(self.obs_data, self.stats['obs'])
        self.action_data = self._norm(self.action_data, self.stats['action'])

    def _get_stat(self, d): return {'min': d.min(0), 'max': d.max(0)}
    def _norm(self, d, s): return (d - s['min']) / np.maximum(s['max'] - s['min'], 1e-5) * 2 - 1

    def __len__(self): return self.episode_ends[-1] - self.pred_horizon

    def __getitem__(self, idx):
        # 简单切片，实际应用建议加入上文提到的 create_sample_indices 以处理边界
        obs = self.obs_data[idx : idx + self.obs_horizon]
        action = self.action_data[idx : idx + self.pred_horizon]
        return {'obs': torch.from_numpy(obs), 'action': torch.from_numpy(action)}

# =================================================================
# 第三部分：正式开工 (Training Loop)
# =================================================================

def train():
    # 1. 硬件和参数配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_horizon = 16
    obs_horizon = 2
    
    # 2. 准备样片
    dataset = RobomimicNextObsDataset(
        dataset_path="your_data.hdf5", # <--- 这里换成您的文件路径
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=8
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)

    # 3. 实例化大模型
    # 全局底色维度 = 观察点数 * 每一帧的特征数
    obs_dim = dataset.obs_data.shape[-1]
    action_dim = dataset.action_data.shape[-1]
    model = ConditionalUnet1D(
        input_dim=action_dim, 
        global_cond_dim=obs_dim * obs_horizon
    ).to(device)

    # 4. 优化器和损失函数 (BCE 用于预测安全/碰撞)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    criterion = nn.BCELoss()

    # 5. 开始绣花
    for epoch in range(50):
        model.train()
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            obs = batch['obs'].to(device) # (B, To, Do)
            action = batch['action'].to(device) # (B, Ta, Da)
            
            # 环境底色平铺
            global_cond = obs.view(obs.shape[0], -1) 
            
            # 专家数据都是安全的，标签给 0
            labels = torch.zeros((obs.shape[0], 1)).to(device)
            
            # 预测与反向传播
            pred_safety = model(action, global_cond)
            loss = criterion(pred_safety, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"完成一轮！平均损耗: {total_loss/len(loader):.6f}")

    torch.save(model.state_dict(), "big_critic_model.pth")
    print("奶奶，模型缝好了！保存在 big_critic_model.pth 里了。")

if __name__ == "__main__":
    train()