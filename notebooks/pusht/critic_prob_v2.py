import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import gdown
from tqdm.auto import tqdm
from dataclasses import dataclass

# =========================================================
# 0. 参数设置 (必须与 Notebook 预训练模型完全一致)
# =========================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
obs_horizon = 2
obs_dim = 5
action_dim = 2
pred_horizon = 16
action_horizon = 8

# =========================================================
# 1. 网络架构 (完全对齐 Notebook 中的 Streaming SI)
# =========================================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale=1):
        super().__init__()
        self.dim, self.scale = dim, scale
    def forward(self, x):
        x = x * self.scale
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)

class Conv1dBlock(nn.Module):
    def __init__(self, inp, out, kernel, groups=8):
        super().__init__()
        # 修复：确保 groups 不超过通道数，且能被整除
        actual_groups = groups
        if out < groups: actual_groups = out
        elif out % groups != 0: actual_groups = 1
            
        self.block = nn.Sequential(
            nn.Conv1d(inp, out, kernel, padding=kernel // 2),
            nn.GroupNorm(actual_groups, out), nn.Mish()
        )
    def forward(self, x): return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self, inp, out, cond_dim, kernel=3, groups=8):
        super().__init__()
        self.blocks = nn.ModuleList([Conv1dBlock(inp, out, kernel, groups), Conv1dBlock(out, out, kernel, groups)])
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, out * 2), nn.Unflatten(-1, (2, out, 1)))
        self.residual = nn.Conv1d(inp, out, 1) if inp != out else nn.Identity()
    def forward(self, x, cond):
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        out = embed[:, 0, ...] * out + embed[:, 1, ...]
        return self.blocks[1](out) + self.residual(x)

class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim, global_cond_dim, down_dims=[256, 512, 1024]):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(256, 100), nn.Linear(256, 1024), nn.Mish(), nn.Linear(1024, 256)
        )
        cond_dim = 256 + global_cond_dim
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        self.down_modules = nn.ModuleList([
            nn.ModuleList([ConditionalResidualBlock1D(i, o, cond_dim), ConditionalResidualBlock1D(o, o, cond_dim), nn.Identity()]) 
            for i, o in in_out
        ])
        self.up_modules = nn.ModuleList([
            nn.ModuleList([ConditionalResidualBlock1D(o*2, i, cond_dim), ConditionalResidualBlock1D(i, i, cond_dim), nn.Identity()]) 
            for i, o in reversed(in_out)
        ])
        self.mid_modules = nn.ModuleList([ConditionalResidualBlock1D(all_dims[-1], all_dims[-1], cond_dim) for _ in range(2)])
        self.final_conv = nn.Sequential(Conv1dBlock(all_dims[1], all_dims[1], 5), nn.Conv1d(all_dims[1], input_dim, 1))

    def forward(self, sample, timestep, global_cond=None):
        x = sample.moveaxis(-1, -2) # (B, T, C) -> (B, C, T)
        t = timestep if torch.is_tensor(timestep) else torch.tensor([timestep], device=x.device)
        if len(t.shape) == 0: t = t[None]
        feat = self.diffusion_step_encoder(t.expand(x.shape[0]))
        if global_cond is not None: feat = torch.cat([feat, global_cond], axis=-1)
        h = []
        for res1, res2, down in self.down_modules:
            x = res1(x, feat); x = res2(x, feat); h.append(x); x = down(x)
        for mid in self.mid_modules: x = mid(x, feat)
        for res1, res2, up in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1); x = res1(x, feat); x = res2(x, feat); x = up(x)
        return self.final_conv(x).moveaxis(-1, -2)

# =========================================================
# 2. 数据处理 (完全对齐 Notebook)
# =========================================================
def create_sequence_pointers(episode_ends, sequence_length, pad_before=0, pad_after=0):
    indices = []
    for i in range(len(episode_ends)):
        start_idx = 0 if i == 0 else episode_ends[i-1]
        episode_length = episode_ends[i] - start_idx
        for idx in range(-pad_before, episode_length - sequence_length + pad_after + 1):
            b_start = max(idx, 0) + start_idx
            b_end = min(idx + sequence_length, episode_length) + start_idx
            indices.append((b_start, b_end, 0 + (b_start - (idx + start_idx)), sequence_length - ((idx + sequence_length + start_idx) - b_end)))
    return indices

class PushTDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, pred_horizon, obs_horizon, action_horizon):
        root = zarr.open(dataset_path, 'r')
        train_data = {'action': root['data']['action'][:], 'obs': root['data']['state'][:]}
        self.stats = {k: {'min': v.min(0), 'max': v.max(0)} for k, v in train_data.items()}
        self.norm_data = {k: (v - self.stats[k]['min'])/(self.stats[k]['max'] - self.stats[k]['min']) * 2 - 1 for k, v in train_data.items()}
        self.ptrs = create_sequence_pointers(root['meta']['episode_ends'][:], pred_horizon, obs_horizon-1, action_horizon-1)
        self.pred_horizon, self.obs_horizon = pred_horizon, obs_horizon

    def __len__(self): return len(self.ptrs)
    def __getitem__(self, idx):
        b_s, b_e, s_s, s_e = self.ptrs[idx]
        res = {}
        for k, v in self.norm_data.items():
            data = np.zeros((self.pred_horizon, v.shape[1]), dtype=np.float32)
            sample = v[b_s:b_e]
            data[:s_s], data[s_e:], data[s_s:s_e] = sample[0], sample[-1], sample
            res[k] = data
        return {'obs': torch.from_numpy(res['obs'][:self.obs_horizon]), 'action': torch.from_numpy(res['action'])}

# =========================================================
# 3. Critic 模型与风险逻辑
# =========================================================
class CollisionPredictionCritic(nn.Module):
    def __init__(self, action_dim, obs_dim, obs_horizon, hidden_dim=512, depth=4):
        super().__init__()
        geo_in = 5 + 4 # a(2), t(1), obs(2) + rel(2), dist(1), align(1)
        self.geo_enc = nn.Sequential(nn.Linear(geo_in, hidden_dim//2), nn.Mish(), nn.LayerNorm(hidden_dim//2))
        self.ctx_enc = nn.Sequential(nn.Linear(obs_dim*obs_horizon, hidden_dim//2), nn.Mish(), nn.LayerNorm(hidden_dim//2))
        self.fusion = nn.Linear(hidden_dim, hidden_dim)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish()) for _ in range(depth*2)])
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, a, t, obs_pos, global_cond):
        rel = obs_pos - a
        dist = torch.norm(rel, dim=-1, keepdim=True)
        align = ( (a/(torch.norm(a, dim=-1, keepdim=True)+1e-6)) * (obs_pos/(dist+1e-6)) ).sum(-1, keepdim=True)
        x = self.fusion(torch.cat([self.geo_enc(torch.cat([a,t,obs_pos,rel,dist,align],-1)), self.ctx_enc(global_cond)], -1))
        for b in self.blocks: x = x + b(x) if isinstance(b, nn.Linear) else b(x) # 简化残差
        return torch.sigmoid(self.head(x))

def compute_hybrid_risk(traj, obs_p, r_obs=0.15, sharpness=60.0):
    dists = torch.norm(traj - obs_p.unsqueeze(0), p=2, dim=-1)
    prob_hit = torch.sigmoid(sharpness * (r_obs + 0.05 - dists.min(0)[0]))
    vec_move = traj[min(5, traj.shape[0]-1)] - traj[0]
    vec_to_obs = obs_p - traj[0]
    align = torch.relu(((vec_move/(torch.norm(vec_move,2,-1,True)+1e-6)) * (vec_to_obs/(torch.norm(vec_to_obs,2,-1,True)+1e-6))).sum(-1))
    prob_aim = (align**2) * torch.exp(-(torch.norm(vec_to_obs,2,-1)**2)/(2*0.8**2))
    return torch.maximum(prob_hit, 0.6 * prob_aim)

# =========================================================
# 4. 稳健训练流程
# =========================================================
def train_critic(dataloader, si_net, epochs=100, save_path="Robust_critic_final.pth"):
    si_net.eval()
    critic = CollisionPredictionCritic(action_dim, obs_dim, obs_horizon).to(device)
    opt = torch.optim.Adam(critic.parameters(), lr=1e-4)
    dt = 1.0 / 16

    for epoch in range(epochs):
        losses = []
        with tqdm(dataloader, desc=f"Epoch {epoch+1}") as pbar:
            for batch in pbar:
                obs, gt = batch['obs'].to(device), batch['action'].to(device)
                B, cond = obs.shape[0], obs.flatten(1)
                t_start = torch.rand(B, 1, device=device) * 0.8
                a_base = torch.gather(gt, 1, (obs_horizon-1 + (t_start*16).long()).clamp(max=gt.shape[1]-1).unsqueeze(-1).expand(-1,-1,2)).squeeze(1)

                with torch.no_grad():
                    # 生成动态障碍物逻辑 (同前)
                    curr_a, curr_t, path = a_base.clone(), t_start.view(-1), [a_base]
                    for _ in range(10):
                        v = si_net(curr_a.unsqueeze(1), curr_t, cond).squeeze(1)
                        curr_a += v * dt; curr_t += dt; path.append(curr_a.clone())
                    path_t = torch.stack(path)
                    vec_move = path_t[-1] - path_t[0]
                    perp = torch.stack([-vec_move[:,1], vec_move[:,0]], 1) / (torch.norm(vec_move,2,1,True)+1e-6)
                    obs_p = path_t[torch.randint(2, len(path_t), (B,)), torch.arange(B)] + perp * (torch.randn(B,1,device=device)*0.15)
                    
                    # 动作多样化与 SDE Rollout
                    a_exp = torch.stack([a_base, a_base + perp*0.2, a_base - perp*0.2], 1).view(-1, 2)
                    K = 12
                    c_a = a_exp.repeat_interleave(K, 0)
                    c_t = t_start.repeat_interleave(3*K, 0).view(-1)
                    c_cond = cond.repeat_interleave(3*K, 0)
                    hist = [c_a.clone()]
                    for _ in range(12):
                        v = si_net(c_a.unsqueeze(1), c_t, c_cond).squeeze(1)
                        c_a += v*dt + 0.06*math.sqrt(dt)*torch.randn_like(c_a)
                        c_t += dt; hist.append(c_a.clone())
                    target = compute_hybrid_risk(torch.stack(hist), obs_p.repeat_interleave(3*K, 0)).view(B*3, K).mean(1, True)

                pred = critic(a_exp, t_start.repeat_interleave(3, 0), obs_p.repeat_interleave(3, 0), cond.repeat_interleave(3, 0))
                loss = F.mse_loss(pred, target)
                opt.zero_grad(); loss.backward(); opt.step()
                losses.append(loss.item()); pbar.set_postfix(loss=f"{loss.item():.4f}")
        print(f"Epoch {epoch+1} Loss: {np.mean(losses):.6f}")
    torch.save(critic.state_dict(), save_path)

if __name__ == "__main__":
    dataset_path = "pusht_cchi_v7_replay.zarr.zip"
    if not os.path.exists(dataset_path): gdown.download(id="1KY1InLurpMvJDRb14L9NlXT_fEsCvVUq", output=dataset_path)
    
    dl = torch.utils.data.DataLoader(PushTDataset(dataset_path, 16, 2, 8), batch_size=64, shuffle=True)
    si_net = ConditionalUnet1D(input_dim=2, global_cond_dim=10).to(device)
    
    # 这里加载你训练好的权重！
    # si_net.load_state_dict(torch.load("path_to_your_si_model.pth"))
    
    train_critic(dl, si_net)