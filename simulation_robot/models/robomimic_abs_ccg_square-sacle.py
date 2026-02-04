import os
import ctypes
import glob
import traceback

print("🔧 启动方案：强制 mujoco_py 走 CPU(OSMesa) 后端 + 手动构建离屏 Context")

# ================= 0. 必须最先设置的环境变量（在 import mujoco_py 之前） =================
# 关键：mujoco_py 在 Linux 上会根据 MUJOCO_PY_FORCE_CPU 选择 CPU(OSMesa) 或 GPU(EGL) builder。
# 你现在看到的 “Found 8 GPUs...” 来自 eglshim.c，说明实际走了 EGL；这里强制切回 OSMesa。
os.environ.setdefault("MUJOCO_PY_FORCE_CPU", "1")
# 如需强制重编译（比如之前编译产物损坏/切换依赖后），可在命令行里加：
#   export MUJOCO_PY_FORCE_REBUILD=1

# MuJoCo 2.1（以及部分包装）也会读这个变量；保留为 osmesa。
os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")

# 强制 Mesa 软件栈（避免误用系统/驱动的 GL 实现）
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast")

# ================= 1. 保障 LD_LIBRARY_PATH 含 mujoco210/bin（否则 mujoco_py 会直接报 Missing path） =================
default_mujoco_bin = os.path.expanduser("~/.mujoco/mujoco210/bin")
if os.path.isdir(default_mujoco_bin):
    current_ld = os.environ.get("LD_LIBRARY_PATH", "")
    ld_paths = [p for p in current_ld.split(":") if p]
    if default_mujoco_bin not in ld_paths:
        os.environ["LD_LIBRARY_PATH"] = default_mujoco_bin + (":" + current_ld if current_ld else "")

# ================= 2. 依赖注入（把 OSMesa / libffi / 相关 GL 依赖先用 RTLD_GLOBAL 加载） =================
system_lib_path = "/usr/lib/x86_64-linux-gnu"

def _dlopen_first(glob_pattern: str, label: str) -> None:
    candidates = glob.glob(glob_pattern)
    if not candidates:
        print(f"⚠️ 未找到 {label}: {glob_pattern}")
        return
    target = candidates[0]
    try:
        # 有的系统没有 os.RTLD_NOW，做个兼容兜底
        rtld_now = getattr(os, "RTLD_NOW", 0)
        ctypes.CDLL(target, mode=ctypes.RTLD_GLOBAL | rtld_now)
        print(f"✅ {label} 注入成功: {target}")
    except Exception as e:
        print(f"❌ {label} 注入失败: {e} ({target})")

# libffi（不同系统版本可能是 .7 / .8）
_dlopen_first(os.path.join(system_lib_path, "libffi.so.*"), "libffi")
# OSMesa + GL（OSMesa 后端需要）
_dlopen_first(os.path.join(system_lib_path, "libOSMesa.so*"), "OSMesa")
_dlopen_first(os.path.join(system_lib_path, "libGL.so.1"), "libGL")

# mujoco210 自带的 libglewosmesa（如果存在，优先注入）
_dlopen_first(os.path.join(default_mujoco_bin, "libglewosmesa.so*"), "libglewosmesa (mujoco210)")

print("\n🧪 环境变量快照（关键项）")
for k in [
    "MUJOCO_PY_FORCE_CPU",
    "MUJOCO_GL",
    "PYOPENGL_PLATFORM",
    "LIBGL_ALWAYS_SOFTWARE",
    "MESA_LOADER_DRIVER_OVERRIDE",
    "LD_LIBRARY_PATH",
]:
    print(f"- {k}={os.environ.get(k)}")

# ================= 3. 验证与手动 Context =================
print("\n🤖 正在初始化 MuJoCo / robomimic...")

try:
    import mujoco_py
    from mujoco_py import MjRenderContextOffscreen
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    dataset_path = "/home/users/meiyi/streaming-flow-policy/notebooks/pusht/Robotic_mimic/robomimic/datasets/square/ph/low_dim_abs.hdf5"
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )

    print("🏗️ 环境创建成功，正在手动构建离屏 Context(OSMesa)...")

    sim = env.env.sim
    # OSMesa 后端不依赖 GPU device id；传 0 更兼容（避免某些实现对 -1 的 range check）
    ctx = MjRenderContextOffscreen(sim, device_id=0)
    # mujoco_py 2.1.x: 用 add_render_context 注册；离屏 context 会保存在 _render_context_offscreen
    if hasattr(sim, "add_render_context"):
        sim.add_render_context(ctx)

    print("✅ 离屏 Context 构建并绑定成功")
    print("📸 尝试渲染...")
    img = sim.render(width=256, height=256, camera_name="agentview")
    print(f"✨ 渲染成功！图像 shape={getattr(img, 'shape', None)} dtype={getattr(img, 'dtype', None)}")

except Exception:
    print("❌ 初始化 / 渲染失败，完整 traceback 如下：")
    traceback.print_exc()

import os
import shutil
import ctypes

# ================= 1. 硬性规定：必须设置环境变量 (骗过 mujoco_py 的死板检查) =================
# 这一步必须在 import 之前做，否则直接报错
mujoco_bin_path = '/home/users/meiyi/.mujoco/mujoco210/bin'
nvidia_lib_path = '/usr/lib/nvidia'

current_ld = os.environ.get('LD_LIBRARY_PATH', '')

# 只要路径不在变量里，就强行加进去
new_paths = []
if mujoco_bin_path not in current_ld:
    new_paths.append(mujoco_bin_path)
if nvidia_lib_path not in current_ld:
    new_paths.append(nvidia_lib_path)

if new_paths:
    # 把新路径拼接到最前面
    os.environ['LD_LIBRARY_PATH'] = ':'.join(new_paths) + ':' + current_ld
    print(f"🔧 环境变量检查通过: 已添加 MuJoCo 和 Nvidia 路径")

# 服务器模式必须设置
os.environ['MUJOCO_GL'] = 'egl'


# ================= 2. 物理保障：搬运核心库文件 (防止 ImportError) =================
# 即使环境变量设了，有时候 Python 还是找不到文件，所以我们要把文件搬到 Python 的“家”里
conda_env_path = '/home/users/meiyi/anaconda3/envs/streaming-flow-policy'
conda_lib_path = os.path.join(conda_env_path, 'lib')

# --- 任务 A: 搬运 libmujoco210.so ---
src_mujoco = os.path.join(mujoco_bin_path, 'libmujoco210.so')
dst_mujoco = os.path.join(conda_lib_path, 'libmujoco210.so')

if not os.path.exists(dst_mujoco):
    if os.path.exists(src_mujoco):
        shutil.copy2(src_mujoco, dst_mujoco)
        print("📦 已搬运 libmujoco210.so 到 Conda 目录")
    else:
        print("⚠️ 警告: 源文件 libmujoco210.so 不存在，跳过搬运")
else:
    print("✅ libmujoco210.so 已存在")

# --- 任务 B: 修复 libglewegl.so (软链接) ---
src_glew = os.path.join(conda_lib_path, 'libGLEW.so')
dst_glew = os.path.join(conda_lib_path, 'libglewegl.so')

if os.path.exists(src_glew) and not os.path.exists(dst_glew):
    os.symlink(src_glew, dst_glew)
    print("🔗 已创建 libglewegl.so 软链接")
else:
    print("✅ libglewegl.so 检查完毕")


# ================= 3. 验证时刻 =================
print("🚀 正在导入 mujoco_py (祈祷时刻)...")
import mujoco_py
from mujoco_py.builder import MujocoException
print("\n✨✨✨ 完美！MuJoCo 路径:", mujoco_py.utils.discover_mujoco())


import os
import h5py
import numpy as np
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from robomimic.envs.env_robosuite import EnvRobosuite

# ================= 配置 =================
# 【关键】把这里改成你下载的那个 hdf5 文件的绝对路径！
# 比如: /home/users/meiyi/downloads/low_dim_abs.hdf5
dataset_path = "/home/users/meiyi/streaming-flow-policy/notebooks/pusht/Robotic_mimic/robomimic/datasets/square/ph/low_dim_abs.hdf5" 

print(f"📂 正在读取数据集: {dataset_path}")

# ================= 1. 检查数据集维度 (验证是否是 Absolute 版本) =================
# 师兄提到：Action 应该是 10 维 (3 pos + 6 rot + 1 gripper)
try:
    with h5py.File(dataset_path, 'r') as f:
        # 通常 demo_0 是第一个演示
        action_shape = f['data/demo_0/actions'].shape
        print(f"✅ 数据集加载成功！Action 维度: {action_shape}")
        
        if action_shape[1] == 10:
            print("✨ 验证通过：是 10 维的 Absolute Action 数据！")
        elif action_shape[1] == 7:
            print("⚠️ 警告：检测到 7 维 Action，这可能是相对位置(Relative)数据，可能会导致模型训练失败！")
        else:
            print(f"❓ 未知维度: {action_shape[1]}")
except Exception as e:
    print(f"❌ 读取数据集失败: {e}")
    print("请检查路径是否正确，文件是否存在。")

# ================= 2. 初始化环境 (验证 MuJoCo 是否真的能跑) =================
# 我们尝试从数据集的元数据(metadata)中自动加载环境配置
try:
    print("\n🤖 正在初始化 Robosuite 环境 (这会调用 MuJoCo)...")
    
    # 读取环境元数据
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    
    # 创建环境
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,            # 服务器没屏幕，不开渲染
        render_offscreen=False,  # 暂时先不测试图像渲染，先测物理
        use_image_obs=False,     # 使用低维状态
    )
    
    print(f"✅ 环境初始化成功: {env.name}")
    print("Action Space:", env.action_space)
    
    # 简单测试一步交互
    obs = env.reset()
    print("✅ 环境 Reset 成功，物理引擎正常工作！")

except Exception as e:
    print(f"❌ 环境初始化失败: {e}")
    print("如果是渲染相关的报错(GL Error)，可能需要检查 MUJOCO_GL=egl 是否生效")

#----------------------------------------------------------
# Standard imports
import collections
from dataclasses import dataclass
import gdown
import os
import numpy as np
import math
import torch
from torch import Tensor
import torch.nn as nn
from tqdm.auto import tqdm
from typing import List, Literal, Sequence, Tuple, Union

# Imports for diffusion policy
import zarr
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

# Imports for the Push-T environment
import gym
from gym import spaces
import pygame
import pymunk
import pymunk.pygame_util
from pymunk.space_debug_draw_options import SpaceDebugColor
from pymunk.vec2d import Vec2d
import shapely.geometry as sg
import cv2
import skimage.transform as st
import jupyviz as jviz

# always call this first
from streaming_flow_policy.all import set_random_seed
set_random_seed(0)
#----------------------------------------------------------
from typing import List, Dict, Optional
import numpy as np
import gym
from gym.spaces import Box
from robomimic.envs.env_robosuite import EnvRobosuite

class RobomimicLowdimWrapper(gym.Env):
    def __init__(self, 
        env: EnvRobosuite,
        obs_keys: List[str]=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'],
        init_state: Optional[np.ndarray]=None,
        render_hw=(256,256),
        render_camera_name='agentview',
        clip_action: bool = True
        ):

        self.env = env
        self.obs_keys = obs_keys
        self.init_state = init_state
        self.render_hw = render_hw
        self.render_camera_name = render_camera_name
        self.clip_action = clip_action
        self.seed_state_map = dict()
        self._seed = None
        
        # setup spaces
        if self.clip_action:
            # delta 控制（或已知范围的控制）通常输入应在 [-1, 1]
            low = np.full(env.action_dimension, fill_value=-1.0, dtype=np.float32)
            high = np.full(env.action_dimension, fill_value=1.0, dtype=np.float32)
        else:
            # absolute 控制：真实动作可能超出 [-1,1]（例如 z≈1.02，旋转 rotvec 可到 2~3）
            # 这里不强行限制范围，只做 NaN/Inf 清洗
            low = np.full(env.action_dimension, fill_value=-np.inf, dtype=np.float32)
            high = np.full(env.action_dimension, fill_value=np.inf, dtype=np.float32)
        self.action_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
        obs_example = self.get_observation()
        low = np.full_like(obs_example, fill_value=-1)
        high = np.full_like(obs_example, fill_value=1)
        self.observation_space = Box(
            low=low,
            high=high,
            shape=low.shape,
            dtype=low.dtype
        )
    # [新增] 添加这个属性，方便外部直接访问 wrapper.sim
    @property
    def sim(self):
        # robomimic 的 EnvRobosuite 把原始环境存在了 .env 属性里
        # 所以我们需要穿透两层: wrapper.env -> EnvRobosuite.env -> RobosuiteRawEnv
        return self.env.env.sim
    
    
    def get_observation(self):
        raw_obs = self.env.get_observation()
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed
    
    def reset(self):
        if self.init_state is not None:
            # always reset to the same state
            # to be compatible with gym
            self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            self.env.reset()

        # return obs
        obs = self.get_observation()
        return obs
    
    def step(self, action):
        # --- 安全防护：MuJoCo/robosuite 对 NaN / Inf / 越界动作非常敏感 ---
        action = np.asarray(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        # 按需裁剪：delta 模式裁剪到 [-1,1]；absolute 模式不裁剪（避免破坏真实动作尺度）
        if getattr(self, "clip_action", True):
            action = np.clip(action, -1.0, 1.0)

        try:
            raw_obs, reward, done, info = self.env.step(action)
        except MujocoException as e:
            print(f"[MuJoCoException] step() failed: {e}")
            obs = self.reset()
            info = {"mujoco_exception": str(e)}
            return obs, 0.0, True, info
        obs = np.concatenate([
            raw_obs[key] for key in self.obs_keys
        ], axis=0)
        return obs, reward, done, info
    
    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, 
            height=h, width=w, 
            camera_name=self.render_camera_name)


def test():
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    from matplotlib import pyplot as plt

    dataset_path = '/home/cchi/dev/diffusion_policy/data/robomimic/datasets/square/ph/low_dim.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(
        dataset_path)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False, 
        render_offscreen=False,
        use_image_obs=False, 
    )
    wrapper = RobomimicLowdimWrapper(
        env=env,
        obs_keys=[
            'object', 
            'robot0_eef_pos', 
            'robot0_eef_quat', 
            'robot0_gripper_qpos'
        ]
    )

    states = list()
    for _ in range(2):
        wrapper.seed(0)
        wrapper.reset()
        states.append(wrapper.env.get_state()['states'])
    assert np.allclose(states[0], states[1])

    img = wrapper.render()
    plt.imshow(img)
    # wrapper.seed()
    # states.append(wrapper.env.get_state()['states'])
#----------------------------------------------------------
import os
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from matplotlib import pyplot as plt

# 替换为你本地的数据集路径
dataset_path = '/home/users/meiyi/streaming-flow-policy/notebooks/pusht/Robotic_mimic/robomimic/datasets/square/ph/low_dim_abs.hdf5'
#----------------------------------------------------------

# 检查路径是否存在，避免后续报错
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")

# 从数据集头部读取元数据 (Metadata)
env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
ccfg = env_meta.get('env_kwargs', {}).get('controller_configs', {})
print(f"[env_meta] controller_configs.control_delta = {ccfg.get('control_delta', None)}")
print(f"[env_meta] reward_shaping = {env_meta.get('env_kwargs', {}).get('reward_shaping', None)}")
# 关键对齐：虽然 env_args 里经常写 control_delta=True，但你这个数据集的 action[:3] 分布
# 明显是绝对 eef 位置（z 约 0.8~1.0），而不是增量（应围绕 0）。
# 如果不覆写，环境会把 (x,y,z) 当成 delta * output_max，导致机械臂一直做“向上推”的小位移，基本永远 0 分。
env_meta['env_kwargs']['controller_configs']['control_delta'] = False
CONTROL_DELTA = False
print(f"[env_meta] OVERRIDE: controller_configs.control_delta = {env_meta['env_kwargs']['controller_configs']['control_delta']}")

# 打印看看环境的基本信息
print(f"环境名称: {env_meta['env_name']}")
print(f"环境类型: {env_meta['type']}")
print(f"环境参数: {env_meta['env_kwargs']}")
#----------------------------------------------------------
import robomimic.utils.obs_utils as ObsUtils

# 1. 这一步至关重要：初始化 ObsUtils
# 我们从之前读取的 env_meta 中提取观测值的定义 (specs)
# 这样 robomimic 才知道 'robot0_eef_pos' 是 low_dim 数据，而不是 rgb 图像
# ObsUtils.initialize_obs_utils_with_obs_specs(
#     obs_modality_specs={
#         "obs": env_meta["env_obs_keys"]
#     }
# )
obs_keys=[
        'object', 
        'robot0_eef_pos', 
        'robot0_eef_quat', 
        'robot0_gripper_qpos'
    ]

ObsUtils.initialize_obs_modality_mapping_from_dict(
        {'low_dim': obs_keys})

# --- 下面是你原来的代码 (现在可以正常运行了) ---

# 使用 robomimic 的工具函数创建原始环境
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False,
    render_offscreen=False, #⚠️
    use_image_obs=False,
)

# 实例化 Wrapper
wrapper = RobomimicLowdimWrapper(
    env=env,
    obs_keys=[
        'object', 
        'robot0_eef_pos', 
        'robot0_eef_quat', 
        'robot0_gripper_qpos'
    ],
    render_hw=(256, 256),
    render_camera_name='agentview',
    clip_action=CONTROL_DELTA
)

print("环境创建并封装成功！")
print(f"Action Space: {wrapper.action_space}")
print(f"Observation Space: {wrapper.observation_space}")
#----------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np

# 1. 测试环境重置 (Reset)

obs = wrapper.reset()
print(f"1. Reset 成功 | 初始观测维度: {obs.shape}")

# 2. 测试环境交互 (Step) - 执行一个随机动作
# 我们用全 0 动作或者随机动作测试
action = np.zeros(wrapper.action_space.shape) 
next_obs, reward, done, info = wrapper.step(action)
print(f"2. Step 成功  | Step后观测维度: {next_obs.shape}, Reward: {reward}")

# 3. 测试图像渲染 (Render) - 这是最容易报错的一步 (例如缺少 OpenGL 库)
try:
    img = wrapper.render()
    
    plt.figure(figsize=(5, 5))
    plt.imshow(img)
    plt.title("Environment Sanity Check")
    plt.axis('off')
    plt.show()
    
    print("3. Render 成功 | 你应该能看到一张包含机械臂和红方块的图片。")
    
except Exception as e:
    print("\n[错误] 渲染失败！")
    print(f"错误信息: {e}")
    print("提示: 如果是在无头服务器(Headless Server)上，可能需要配置 EGL 或使用 xvfb-run。")
#----------------------------------------------------------
from myrotation_transformer_final import RotationTransformer

rotation_transformer = RotationTransformer('rotation_6d', 'axis_angle')

action_dim = 10 

print(f"已初始化 RotationTransformer，Action Dim 设置为: {action_dim}")
#----------------------------------------------------------
class MinMaxNormalizer:
    def __init__(self, data=None, min_val=None, max_val=None):
        """
        可以传入 data 自动计算 min/max，也可以直接传入已知的 min/max
        """
        if data is not None:
            # 假设 data 是 (N, Dim)
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0)
        else:
            self.min_val = np.array(min_val)
            self.max_val = np.array(max_val)
            
        # 防止除以零（如果某维度没有变化）
        self.scale = self.max_val - self.min_val
        self.scale[self.scale == 0] = 1.0 

    def normalize(self, x):
        # 归一化到 [0, 1]
        norm = (x - self.min_val) / self.scale
        # 映射到 [-1, 1]
        return norm * 2 - 1

    def denormalize(self, x):
        # 从 [-1, 1] 映射回 [0, 1]
        denorm = (x + 1) / 2
        # 还原到原始范围
        return denorm * self.scale + self.min_val
#----------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from tqdm import tqdm

# 假设 RotationTransformer 和 MinMaxNormalizer 已经在上面定义好了
# from your_utils import RotationTransformer, MinMaxNormalizer 

class RobomimicDataset(Dataset):
    def __init__(self, 
                 dataset_path, 
                 obs_keys, 
                 pred_horizon, 
                 obs_horizon, 
                 action_horizon):
        
        # 参数记录
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs_keys = obs_keys
        
        # --- 新增：初始化旋转变换器 ---
        # 用于将 Axis-Angle (3) 转为 Rotation 6D (6)
        self.rotation_transformer = RotationTransformer(
            from_rep='axis_angle', to_rep='rotation_6d')

        # 1. 读取数据到内存
        print(f"正在加载数据集: {dataset_path}")
        with h5py.File(dataset_path, 'r') as f:
            demos = f['data']
            self.all_obs = []
            self.all_actions = [] # 这里将存储转换后的 10维动作
            
            for key in tqdm(demos.keys(), desc="Loading"):
                demo = demos[key]
                
                # --- 处理 Observation (保持不变) ---
                obs_list = []
                for k in self.obs_keys:
                    obs_data = demo['obs'][k][:] 
                    obs_list.append(obs_data)
                obs_seq = np.concatenate(obs_list, axis=-1)
                self.all_obs.append(obs_seq)
                
                # --- 修改重点：处理 Action (7维 -> 10维) ---
                raw_actions = demo['actions'][:].astype(np.float32) # (T, 7)
                
                # 1. 拆分动作
                pos = raw_actions[:, :3]     # (T, 3)
                rot = raw_actions[:, 3:6]    # (T, 3) Axis-Angle
                gripper = raw_actions[:, 6:] # (T, 1)
                
                # 2. 旋转变换 (Axis-Angle -> Rotation 6D)
                # Transformer 接收 Tensor，返回 Tensor，需转回 Numpy
                rot_tensor = torch.from_numpy(rot)
                rot_6d_tensor = self.rotation_transformer.forward(rot_tensor)
                rot_6d = rot_6d_tensor.numpy() # (T, 6)
                
                # 3. 拼接成绝对位置控制所需的格式 (T, 10)
                # 3 pos + 6 rot + 1 gripper = 10 dim
                new_actions = np.concatenate([pos, rot_6d, gripper], axis=-1)
                
                self.all_actions.append(new_actions)

        # 2. 计算统计数据并初始化归一化器
        # 把所有数据拼在一起
        all_obs_concat = np.concatenate(self.all_obs, axis=0)
        all_action_concat = np.concatenate(self.all_actions, axis=0)
        
        # --- 修改重点：使用类 MinMaxNormalizer ---
        # 此时 all_action_concat 已经是 10维的数据了
        self.obs_normalizer = MinMaxNormalizer(data=all_obs_concat)
        self.action_normalizer = MinMaxNormalizer(data=all_action_concat)
        
        # 打印一下验证维度
        print(f"原始 Action 维度: 7, 转换后 Action 维度: {all_action_concat.shape[-1]}")
        
        # 3. 预处理：归一化所有数据并建立索引
        self.normalized_obs = []
        self.normalized_actions = []
        self.indices = []
        
        for i in range(len(self.all_obs)):
            # --- 修改重点：调用实例方法进行归一化 ---
            n_obs = self.obs_normalizer.normalize(self.all_obs[i])
            n_action = self.action_normalizer.normalize(self.all_actions[i])
            
            self.normalized_obs.append(n_obs)
            self.normalized_actions.append(n_action)
            
            # 建立索引 (逻辑不变)
            episode_len = n_obs.shape[0]
            # 为了保证剩下的长度够 pred_horizon，需要减去它
            for start_ts in range(episode_len - self.pred_horizon):
                self.indices.append((i, start_ts))
                
        print(f"加载完成! 样本数量: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        episode_idx, start_ts = self.indices[idx]
        
        n_obs = self.normalized_obs[episode_idx]
        n_action = self.normalized_actions[episode_idx]
        
        # 1. 取 Observation
        # 注意：这里如果 obs_horizon > 1，需要处理边界情况(padding)，
        # 但为简化起见，假设 start_ts + obs_horizon 不会越界(因为上面 range 只减了 pred_horizon)
        # 实际严谨代码通常会在上面 range 减去 max(obs_horizon, pred_horizon)
        obs_seq = n_obs[start_ts : start_ts + self.obs_horizon, :]
        
        # 2. 取 Action
        action_seq = n_action[start_ts : start_ts + self.pred_horizon, :]
        
        data = {
            'obs': torch.from_numpy(obs_seq).float(),      # shape: (obs_horizon, obs_dim)
            'action': torch.from_numpy(action_seq).float() # shape: (pred_horizon, 10)
        }
        return data

    # --- 新增：提供一个反归一化+逆变换的辅助函数，供推理时使用 ---
    def get_unnormalized_action(self, n_action_pred):
        """
        推理时使用：
        输入: 模型预测的归一化动作 (B, 10) Tensor or Numpy
        输出: 环境可执行的动作 (B, 7) Numpy
        """
        is_tensor = isinstance(n_action_pred, torch.Tensor)
        if is_tensor:
            n_action_pred = n_action_pred.detach().cpu().numpy()

        # --- 数值安全：SI/ODE 积分可能让 latent 漂出 [-1, 1] 或出现 NaN/Inf ---
        n_action_pred = np.asarray(n_action_pred, dtype=np.float32)
        n_action_pred = np.nan_to_num(n_action_pred, nan=0.0, posinf=0.0, neginf=0.0)
        n_action_pred = np.clip(n_action_pred, -1.0, 1.0)
         
        # 1. 反归一化 (10D -> 10D)
        action_10d = self.action_normalizer.denormalize(n_action_pred)
        
        # 2. 拆解
        pos = action_10d[..., :3]
        rot_6d = action_10d[..., 3:9]
        gripper = action_10d[..., 9:]
        
        # 3. 逆旋转变换 (6D -> 3D Axis-Angle)
        rot_6d_tensor = torch.from_numpy(rot_6d)
        rot_axis = self.rotation_transformer.inverse(rot_6d_tensor).numpy()
        
        # 4. 拼接 (10D -> 7D)
        action_7d = np.concatenate([pos, rot_axis, gripper], axis=-1)
        return action_7d
#----------------------------------------------------------
# 配置参数 (这些是 Diffusion Policy 的标准参数)
pred_horizon = 16    # 预测未来 16 步
obs_horizon = 2      # 看过去 2 步
action_horizon = 8   # (这一步在 dataset 里用不到，但在推理时有用)

# 1. 实例化 Dataset
dataset = RobomimicDataset(
    dataset_path=dataset_path, # 确保这个变量是你之前定义的路径
    obs_keys=[
        'object', 
        'robot0_eef_pos', 
        'robot0_eef_quat', 
        'robot0_gripper_qpos'
    ],
    pred_horizon=pred_horizon,
    obs_horizon=obs_horizon,
    action_horizon=action_horizon
)

# 2. 实例化 DataLoader
dataloader = DataLoader(
    dataset, 
    batch_size=64, 
    shuffle=True, 
    num_workers=0, # 本地调试设为 0 比较稳妥，避免多进程报错
    pin_memory=True
)

# 3. 验证一下输出形状
batch = next(iter(dataloader))
print("\n--- 数据形状检查 ---")
print(f"Obs Batch:    {batch['obs'].shape}")
print(f"Action Batch: {batch['action'].shape}")
print(f"Observation Space: {wrapper.observation_space}")
#----------------------------------------------------------
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim, scale = 1):
        super().__init__()
        self.dim = dim
        self.scale = scale # added - SFP

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

    def forward(self, x):
        return self.conv(x)

class ConvUpsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

class LinearDownsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        # Reshape input to (batch_size, -1) for fully connected layer
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, -1)  # flatten spatial dimensions
        x = self.linear(x)
        x = x.view(batch_size, channels, seq_len)  # reshape back to original dimensions
        return x

class LinearUpsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: Tensor):
        # Reshape input to (batch_size, -1) for fully connected layer
        batch_size, channels, seq_len = x.size()
        x = x.view(batch_size, -1)  # flatten spatial dimensions
        x = self.linear(x)
        x = x.view(batch_size, channels, seq_len)  # reshape back to original dimensions
        return x

class Conv1dBlock(nn.Module):
    '''
        Conv1d --> GroupNorm --> Mish
    '''

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)

class ConditionalResidualBlock1D(nn.Module):
    def __init__(self,
            in_channels,
            out_channels,
            cond_dim,
            kernel_size=3,
            n_groups=8,
                 ):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])

        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels * 2
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            nn.Unflatten(-1, (-1, 1))
        )

        # Ensure dimensions compatible
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        '''
            x : [ batch_size x in_channels x horizon ]
            cond : [ batch_size x cond_dim]

            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)

        embed = embed.reshape(
            embed.shape[0], 2, self.out_channels, 1)
        scale = embed[:,0,...]
        bias = embed[:,1,...]
        out = scale * out + bias

        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out

class ConditionalUnet1D (nn.Module):
    def __init__(self,
        input_dim,
        global_cond_dim,
        updownsample_type: Literal['Conv', 'Linear'],  # added for SFP
        sin_embedding_scale,  # added for SFP
        diffusion_step_embed_dim=256,
        down_dims=[256,512,1024],
        kernel_size=5,
        n_groups=8,
        ):
        """
        input_dim: Dim of actions.
        global_cond_dim: Dim of global conditioning applied with FiLM
          in addition to diffusion step embedding. This is usually obs_horizon * obs_dim
        diffusion_step_embed_dim: Size of positional encoding for diffusion iteration k
        down_dims: Channel size for each UNet level.
          The length of this array determines numebr of levels.
        kernel_size: Conv kernel size
        n_groups: Number of groups for GroupNorm
        """

        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]

        dsed = diffusion_step_embed_dim
        diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed, scale = sin_embedding_scale), # added - SFP
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        cond_dim = dsed + global_cond_dim

        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim=cond_dim,
                kernel_size=kernel_size, n_groups=n_groups
            ),
        ])

        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            if updownsample_type == 'Linear':  # added for SFP
                downsample_layer = LinearDownsample1d(dim_out) if not is_last else nn.Identity() #added
            elif updownsample_type == 'Conv':
                downsample_layer = ConvDownsample1d(dim_out) if not is_last else nn.Identity()
            else:
                raise ValueError(f"Unsupported updownsample_type: {updownsample_type}")
            down_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_in, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_out, dim_out, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                downsample_layer,
            ]))

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            if updownsample_type == 'Linear':  # added for SFP
                upsample_layer = LinearUpsample1d(dim_in) if not is_last  else nn.Identity()
            elif updownsample_type == 'Conv':
                upsample_layer = ConvUpsample1d(dim_in) if not is_last  else nn.Identity()
            else:
                raise ValueError(f"Unsupported updownsample_type: {updownsample_type}")
            up_modules.append(nn.ModuleList([
                ConditionalResidualBlock1D(
                    dim_out*2, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
                ConditionalResidualBlock1D(
                    dim_in, dim_in, cond_dim=cond_dim,
                    kernel_size=kernel_size, n_groups=n_groups),
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

        print("Number of parameters: {:e}".format(
            sum(p.numel() for p in self.parameters()))
        )

    def forward(self,
            sample: Tensor,
            timestep: Union[Tensor, float, int],
            global_cond=None,
        ) -> Tensor:
        """
        x: (B,T,input_dim)
        timestep: (B,) or int, diffusion step
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        # (B,T,C)
        sample = sample.moveaxis(-1,-2)
        # (B,C,T)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=sample.device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])

        global_feature = self.diffusion_step_encoder(timesteps)

        if global_cond is not None:
            global_feature = torch.cat([
                global_feature, global_cond
            ], axis=-1)

        x = sample
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        # (B,C,T)
        x = x.moveaxis(-1,-2)
        # (B,T,C)
        return x
#----------------------------------------------------------
# 基于之前的 Lift 任务设置
action_dim = 10      # 动作维度
# 动态获取 obs_dim
# dataset[0] 返回的是 {'obs': Tensor, 'action': Tensor}
# shape 是 (obs_horizon, dim)，所以取 [-1]
obs_dim = dataset[0]['obs'].shape[-1] 

print(f"✅ 自动检测到的 obs_dim: {obs_dim}")#期望是23
obs_horizon = 2      # 观测历史长度

# 实例化模型
dp_noise_pred_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim * obs_horizon,
    updownsample_type='Conv',
    sin_embedding_scale=1,
)

print("模型构建成功！")

num_diffusion_iters = 100
noise_scheduler = DDPMScheduler(
    num_train_timesteps=num_diffusion_iters,
    # the choise of beta schedule has big impact on performance
    # we found squared cosine works the best
    beta_schedule='squaredcos_cap_v2',
    # clip output to [-1,1] to improve stability
    clip_sample=True,
    # our network predicts noise (instead of denoised action)
    prediction_type='epsilon'
)

# device transfer
device = torch.device('cuda')
dp_noise_pred_net = dp_noise_pred_net.to(device)
#----------------------------------------------------------
# num_epochs = 100

# # Exponential Moving Average
# # accelerates training and improves stability
# # holds a copy of the model weights
# ema_dp = EMAModel(
#     parameters=dp_noise_pred_net.parameters(),
#     power=0.75)

# # Standard ADAM optimizer
# # Note that EMA parametesr are not optimized
# optimizer = torch.optim.AdamW(
#     params=dp_noise_pred_net.parameters(),
#     lr=1e-4, weight_decay=1e-6)

# # Cosine LR schedule with linear warmup
# lr_scheduler = get_scheduler(
#     name='cosine',
#     optimizer=optimizer,
#     num_warmup_steps=500,
#     num_training_steps=len(dataset) * num_epochs
# )

# with tqdm(range(num_epochs), desc='Epoch') as tglobal:
#     # epoch loop
#     for epoch_idx in tglobal:
#         epoch_loss = list()
#         # batch loop
#         with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
#             for nbatch in tepoch:
#                 # Note that the data is normalized in the dataset.
#                 # Device transfer
#                 nobs = nbatch['obs'].to(device)  # (B, To, O)
#                 naction = nbatch['action'].to(device)  # (B, Tp, A)
#                 B = nobs.shape[0]

#                 # Observation as FiLM conditioning
#                 obs_cond = nobs.flatten(start_dim=1)  # (B, To*O)

#                 # Sample noise to add to actions
#                 noise = torch.randn(naction.shape, device=device)  # (B, Tp, A)

#                 # sample a diffusion iteration for each data point
#                 timesteps = torch.randint(
#                     0, noise_scheduler.config.num_train_timesteps,
#                     (B,), device=device
#                 ).long()  # (B,)

#                 # Forward diffusion process: Add noise to the clean images
#                 # according to the noise magnitude at each diffusion iteration.
#                 noisy_actions = noise_scheduler.add_noise(
#                     naction, noise, timesteps)  # (B, Tp, A)

#                 # Predict the noise residual.
#                 noise_pred = dp_noise_pred_net(
#                     noisy_actions, timesteps, global_cond=obs_cond)

#                 # L2 loss
#                 loss = nn.functional.mse_loss(noise_pred, noise)

#                 # optimize
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 # step lr scheduler every batch
#                 # this is different from standard pytorch behavior
#                 lr_scheduler.step()

#                 # update Exponential Moving Average of the model weights
#                 ema_dp.step(dp_noise_pred_net.parameters())

#                 # logging
#                 loss_cpu = loss.item()
#                 epoch_loss.append(loss_cpu)
#                 tepoch.set_postfix(loss=loss_cpu)
#         tglobal.set_postfix(loss=np.mean(epoch_loss))

# # Weights of the EMA model
# # is used for inference
# ema_noise_pred_net_dp = dp_noise_pred_net
# ema_dp.copy_to(ema_noise_pred_net_dp.parameters())
# #----------------------------------------------------------
# #关掉图片渲染功能⚠️
# import collections
# import numpy as np
# import torch
# from tqdm.auto import tqdm

# # 1. 准备参数
# # 确保模型处于推理模式
# dp_noise_pred_net.eval()
# dp_noise_pred_net.to(device)

# # 设置参数 (必须与训练时一致)
# max_steps = 400  # 最大运行步数
# obs_horizon = 2
# pred_horizon = 16
# action_horizon = 8
# num_diffusion_iters = 100 

# # 2. 重置环境
# obs = wrapper.reset()

# # 初始化观测队列
# obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

# # 用于保存奖励
# rewards = list()
# done = False
# step_idx = 0

# print("开始推理 (Inference)...")

# # 3. 推理循环
# with tqdm(total=max_steps, desc="Eval Robomimic") as pbar:
#     while not done:
#         # --- A. 数据准备 ---
#         # 拼接最近 obs_horizon 步的观测数据
#         obs_seq = np.stack(obs_deque) # (2, 19)
        
#         # [修改点 1]: 使用 dataset 里的实例归一化器
#         nobs = dataset.obs_normalizer.normalize(obs_seq)
        
#         # 转 Tensor 并增加 Batch 维度
#         nobs = torch.from_numpy(nobs).to(device, dtype=torch.float32)
        
#         # 构造 Global Conditioning: (1, obs_horizon * 19)
#         obs_cond = nobs.unsqueeze(0).flatten(start_dim=1)

#         # --- B. 生成动作 (反向扩散) ---
#         with torch.no_grad():
#             # [修改点 2]: 初始化纯高斯噪音，维度改为 10 (3pos + 6rot + 1gripper)
#             # 原始代码是 7，现在必须是 10
#             na_traj = torch.randn(
#                 (1, pred_horizon, 10), 
#                 device=device
#             )

#             # 设置调度器时间步
#             noise_scheduler.set_timesteps(num_diffusion_iters)

#             # 逐步去噪
#             for k in noise_scheduler.timesteps:
#                 # 预测噪音
#                 noise_pred = dp_noise_pred_net(
#                     sample=na_traj,
#                     timestep=k,
#                     global_cond=obs_cond
#                 )

#                 # 移除噪音
#                 na_traj = noise_scheduler.step(
#                     model_output=noise_pred,
#                     timestep=k,
#                     sample=na_traj,
#                 ).prev_sample

#         # --- C. 后处理 ---
#         # 转回 CPU numpy: (1, 16, 10) -> (16, 10)
#         na_traj = na_traj.detach().to('cpu').numpy()[0]
        
#         # [修改点 3]: 调用 dataset 的辅助函数进行 反归一化 + 逆旋转变换
#         # 输入 (16, 10) -> 输出 (16, 7)
#         # 这个函数内部会自动处理: 10D反归一化 -> 6D旋转转AxisAngle -> 拼接回7D
#         action_pred = dataset.get_unnormalized_action(na_traj)

#         # --- D. 动作切片与执行 (逻辑不变) ---
#         start = obs_horizon - 1
#         end = start + action_horizon
        
#         # 取出未来 action_horizon 步的动作
#         action_chunk = action_pred[start:end, :] 
        
#         # --- E. 执行动作序列 ---
#         for action in action_chunk:
#             # 环境交互 (此时 action 已经是环境能懂的 7维 格式了)
#             obs, reward, done, info = wrapper.step(action)
            
#             # 更新历史观测
#             obs_deque.append(obs)
            
#             # 记录奖励
#             rewards.append(reward)
            
#             # 进度更新
#             step_idx += 1
#             pbar.update(1)
#             pbar.set_postfix(reward=reward)
            
#             if step_idx >= max_steps:
#                 done = True
#             if done:
#                 break

# print(f"推理结束! 总得分: {max(rewards)}")
#----------------------------------------------------------
# =========================================================
# 1. 辅助函数: Gamma 调度器 (参考 model.py)
# =========================================================
EPS = 1e-6

def gamma_t_si(t):
    # 使用简单的抛物线/气泡形状，在 t=0 和 t=1 时为 0，中间膨胀
    # 这里的系数 0.1 控制 "SI 气泡" 的大小，相对于 SFP 的管状噪声
    return 0.1 * torch.sqrt(t * (1.0 - t) + EPS)

def d_gamma_dt_si(t):
    # gamma(t) 对 t 的导数
    return 0.1 * (1.0 - 2.0 * t) / (2.0 * torch.sqrt(t * (1.0 - t) + EPS))

# =========================================================
# 2. 初始化模型 (VelocityNet 和 DenoiserNet)
# =========================================================
# SFP 参数


pred_horizon = 16
action_dim = 10       # 动作维度
# 动态获取 obs_dim
# dataset[0] 返回的是 {'obs': Tensor, 'action': Tensor}
# shape 是 (obs_horizon, dim)，所以取 [-1]
obs_dim = dataset[0]['obs'].shape[-1] 

print(f"✅ 自动检测到的 obs_dim: {obs_dim}")#期望是23
obs_horizon = 2      # 观测历史长度


# 我们需要两个网络：
# 1. si_velocity_net: 预测确定性速度场 (v)
# 2. si_denoiser_net: 预测噪声场 (eta/score)

si_velocity_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
    updownsample_type='Linear',
    sin_embedding_scale=100,
).to(device)

si_denoiser_net = ConditionalUnet1D(
    input_dim=action_dim,
    global_cond_dim=obs_dim*obs_horizon,
    updownsample_type='Linear',
    sin_embedding_scale=100,
).to(device)

print("Models initialized for Streaming SI Policy.")

# =========================================================
# 3. 优化器设置
# =========================================================
optimizer_si = torch.optim.AdamW([
    {'params': si_velocity_net.parameters()},
    {'params': si_denoiser_net.parameters()}
], lr=1e-4, weight_decay=1e-6)

# EMA (Exponential Moving Average) for better inference stability
ema_si_v = EMAModel(parameters=si_velocity_net.parameters(), power=0.75)
ema_si_eta = EMAModel(parameters=si_denoiser_net.parameters(), power=0.75)

num_epochs_si = 1200
lr_scheduler_si = get_scheduler(
    name='cosine',
    optimizer=optimizer_si,
    num_warmup_steps=500,
    num_training_steps=len(dataloader) * num_epochs_si
)
#----------------------------------------------------------
def LinearlyInterpolateTrajectory(ξ, t):
    """
    Vectorized computation of positions and velocities if each trajectory
    (from a batch of trajectories) at given times for each trajectory, using
    linear interpolation.

    ξ (Tensor, dtype=float, shape=(B, T, A)): batch of action trajectories.
    t (Tensor, dtype=float, shape=(B,)): batch of times in [0, 1].

    Returns:
        ξt   (Tensor, shape=(B, A)): positions at time t
        dξdt (Tensor, shape=(B, A)): velocities at time t
    """
    B, T, A = ξ.shape

    # Compute the lower and upper limits of the bins that the time-points lie in.
    scaled_t = t * (T - 1)  # (B,) lies in [0, T-1]
    l = scaled_t.floor().long().clamp(0, T - 2)  # (B,) lower bin limits
    u = (l + 1).clamp(0, T - 1)  # (B,) upper bin limits
    λ = scaled_t - l.float()  # fractional part, lies in [0, 1]

    # Query the values of the upper and lower bin limits.
    batch_idx = torch.arange(B, device=ξ.device)  # (B,)
    ξl = ξ[batch_idx, l, :]  # (B, A)
    ξu = ξ[batch_idx, u, :]  # (B, A)

    # Linearly interpolate between bin limits to get position.
    λ = λ.unsqueeze(-1)  # (B, 1)
    ξt = ξl + λ * (ξu - ξl)  # (B, A)

    # Compute velocity as first-order hold.
    # Note that the time interval between two bins is Δt = 1 / (T-1).
    dξdt = (ξu - ξl) * (T - 1)  # (B, A)

    return ξt, dξdt  # (B, A) and (B, A)

def SampleCFMInputsAndTargets(ξt, dξdt, t, k, σ0):
    """
    Sample inputs and targets for the conditional flow matching loss (CFM)
    given positions and velocities at time t.

    This functions performs the following sampling (Eq. 2 and 3 of the paper):
        a ~ N(ξ(t), σ₀² exp(-2kt))  # (Eq. 3 in the paper)
        v = -k (a - ξ(t)) + dξdt(t)  # (Eq. 2 in the paper)

    Args:
        ξt (Tensor, shape=(B, A)): positions at time t.
        dξdt (Tensor, shape=(B, A)): velocities at time t.
        t (Tensor, shape=(B,)): times in [0, 1].
        k (float): Stabilizing gains of the conditional flow.
        σ0 (float): initial standard deviation of the noise added to the action.

    Returns:
        a (Tensor, shape=(B, A)): noised actions at time t
        v (Tensor, shape=(B, A)): noised action velocity targets at time t
    """
    # error = σ0 * torch.exp(-k*t).unsqueeze(1) * torch.randn_like(xt)
    t = t.unsqueeze(-1)  # (B, 1)
    sampled_error = σ0 * torch.exp(-k * t) * torch.randn_like(ξt)  # (B, A)
    a = ξt + sampled_error  # (B, A) ⟸ Eq. 3 in the paper
    v = -k * sampled_error + dξdt  # (B, A) ⟸ Eq. 2 in the paper

    return a, v  # (B, A) and (B, A)
#----------------------------------------------------------
# =========================================================
# 训练循环: Conditional Streaming SI Policy
# =========================================================
sigma0 = 0.4  # SFP 初始管状半径
k = 10.0      # SFP 稳定系数


print("Starting training for Streaming SI Policy...")

import random
import time

# =========================================================
# [新增] 可恢复训练的 checkpoint（保存：权重+优化器+调度器+EMA+epoch/step+loss+随机状态）
# 说明：
# - 保持兼容：仍然保存 top-level 的 'velocity_net' / 'denoiser_net'，以免你下面推理加载报错
# - 想恢复训练：把 resume_from 设置成某个 ckpt 路径即可
# =========================================================
save_dir = "/home/users/meiyi/streaming-flow-policy/notebooks/pusht/Robotic_mimic/models"
os.makedirs(save_dir, exist_ok=True)
save_every_epochs = 200

# 想续训就填路径，例如：
# resume_from = os.path.join(save_dir, "square_state_latest_si_abs.ckpt")
resume_from = None

# # 想续训就填路径，例如：
# # resume_from = os.path.join(save_dir, "can_state_latest_si_abs.ckpt")
# # ✅ 恢复训练：指定你最近的 checkpoint（按你的需求固定为 800 epoch）
# resume_from = "/home/users/meiyi/streaming-flow-policy/notebooks/pusht/Robotic_mimic/models/square_state_600_ep_si_abs.ckpt"
# if (resume_from is not None) and (not os.path.isfile(resume_from)):
#     raise FileNotFoundError(f"[resume] checkpoint not found: {resume_from}")

global_step = 0
train_history = []  # 每个 epoch 记录摘要（loss 曲线/恢复训练用）

def _pack_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        try:
            state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
        except Exception:
            state["torch_cuda_all"] = None
    return state

def _restore_rng_state(state):
    if not isinstance(state, dict):
        return
    try:
        if state.get("python", None) is not None:
            random.setstate(state["python"])
        if state.get("numpy", None) is not None:
            np.random.set_state(state["numpy"])
        if state.get("torch", None) is not None:
            torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and state.get("torch_cuda_all", None) is not None:
            torch.cuda.set_rng_state_all(state["torch_cuda_all"])
    except Exception as e:
        print(f"[warn] failed to restore rng state: {e}")

def save_checkpoint(path, epoch_idx, global_step, history):
    ckpt = {
        # --- 推理兼容键（你下面的 load_pretrained 会用到） ---
        "velocity_net": si_velocity_net.state_dict(),
        "denoiser_net": si_denoiser_net.state_dict(),

        # --- 训练恢复所需信息 ---
        "epoch": int(epoch_idx),
        "global_step": int(global_step),
        "optimizer_si": optimizer_si.state_dict(),
        "lr_scheduler_si": lr_scheduler_si.state_dict(),
        "ema_si_v": getattr(ema_si_v, "state_dict", lambda: None)(),
        "ema_si_eta": getattr(ema_si_eta, "state_dict", lambda: None)(),
        "history": history,
        "sigma0": float(sigma0),
        "k": float(k),
        "num_epochs_si": int(num_epochs_si),
        "saved_at": float(time.time()),
        "rng_state": _pack_rng_state(),
    }
    torch.save(ckpt, path)

def _torch_load_compat(path, map_location):
    """
    PyTorch 2.6+ 默认 weights_only=True，会导致包含 optimizer / rng_state 等对象的 ckpt 无法加载。
    这里显式使用 weights_only=False，并对旧版本 torch 做参数兼容。
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # older torch 不支持 weights_only 参数
        return torch.load(path, map_location=map_location)

def load_checkpoint(path):
    ckpt = _torch_load_compat(path, map_location=device)
    si_velocity_net.load_state_dict(ckpt["velocity_net"])
    si_denoiser_net.load_state_dict(ckpt["denoiser_net"])

    if "optimizer_si" in ckpt:
        optimizer_si.load_state_dict(ckpt["optimizer_si"])
    if "lr_scheduler_si" in ckpt:
        try:
            lr_scheduler_si.load_state_dict(ckpt["lr_scheduler_si"])
        except Exception as e:
            print(f"[warn] failed to load lr_scheduler state, continue with fresh scheduler. err={e}")

    # diffusers.EMAModel 支持 load_state_dict
    if ckpt.get("ema_si_v", None) is not None:
        try:
            ema_si_v.load_state_dict(ckpt["ema_si_v"])
        except Exception as e:
            print(f"[warn] failed to load ema_si_v state. err={e}")
    if ckpt.get("ema_si_eta", None) is not None:
        try:
            ema_si_eta.load_state_dict(ckpt["ema_si_eta"])
        except Exception as e:
            print(f"[warn] failed to load ema_si_eta state. err={e}")

    if "rng_state" in ckpt:
        _restore_rng_state(ckpt["rng_state"])

    start_epoch = int(ckpt.get("epoch", -1)) + 1
    step = int(ckpt.get("global_step", 0))
    history = ckpt.get("history", [])
    return start_epoch, step, history

start_epoch = 0
if (resume_from is not None) and os.path.isfile(resume_from):
    print(f"[resume] loading checkpoint: {resume_from}")
    start_epoch, global_step, train_history = load_checkpoint(resume_from)
    print(f"[resume] start_epoch={start_epoch} global_step={global_step} history_len={len(train_history)}")

# with tqdm(range(start_epoch, num_epochs_si), desc='Epoch', total=num_epochs_si, initial=start_epoch) as tglobal:
#     for epoch_idx in tglobal:
#         epoch_loss = []
#         # 用于记录每个 epoch 的平均分项 loss
#         epoch_v_loss = []
#         epoch_eta_loss = []
        
#         with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
#             for nbatch in tepoch:
#                 # 1. 数据准备
#                 nobs = nbatch['obs'].to(device)
#                 naction = nbatch['action'].to(device)
                
#                 # 提取当前及未来的动作片段
#                 ξ = naction[:, obs_horizon-1:, :] 
#                 B = ξ.shape[0]
                
#                 # 随机采样时间 t ~ U[0, 1]
#                 t = torch.rand(B, device=device)
                
#                 # 2. 计算 Ground Truth 位置和速度 (Linearly Interpolate)
#                 ξt, dξdt = LinearlyInterpolateTrajectory(ξ, t)
                
#                 # 3. 构建 Flow Matching (FP) 基础目标 (管状分布)
#                 # a_t_fp = ξ(t) + σ(t) * ε1
#                 # 这里的 σ(t) = σ0 * exp(-kt)
#                 t_expanded = t.view(B, 1)
#                 sigma_t_fp = sigma0 * torch.exp(-k * t_expanded)
#                 noise_fp = torch.randn_like(ξt)
#                 a_t_fp = ξt + sigma_t_fp * noise_fp
                
#                 # 计算 Velocity Target (指向轨迹的向量场)
#                 # v_target = dξ/dt - k * (a_t_fp - ξ(t))
#                 # 注意：这个速度是定义在 a_t_fp 上的
#                 v_target = dξdt - k * (a_t_fp - ξt)
                
#                 # 4. 构建 Stochastic Interpolant (SI) 扰动
#                 # x_t = a_t_fp + γ(t) * z
#                 # 我们在 SFP 的管子外面再套一层 SI 的气泡
#                 gamma = gamma_t_si(t_expanded)
#                 z_noise_si = torch.randn_like(ξt)
#                 x_t_in_s = a_t_fp + gamma * z_noise_si
                
#                 # 5. 网络前向传播
#                 # 输入需要 reshape 成 (B, 1, A) 以适应 UNet 接口
#                 net_input = x_t_in_s.unsqueeze(1)
#                 global_cond = nobs.flatten(start_dim=1)
                
#                 # 预测速度
#                 v_pred = si_velocity_net(sample=net_input, timestep=t, global_cond=global_cond)
#                 v_pred = v_pred.squeeze(1)
                
#                 # 预测噪声 (Score 相关的量)
#                 eta_pred = si_denoiser_net(sample=net_input, timestep=t, global_cond=global_cond)
#                 eta_pred = eta_pred.squeeze(1)
                
#                 # 6. 计算 Loss
#                 loss_v = nn.functional.mse_loss(v_pred, v_target)
#                 loss_eta = nn.functional.mse_loss(eta_pred, z_noise_si)
                
#                 loss = loss_v + loss_eta
                
#                 # 7. 反向传播
#                 loss.backward()
#                 optimizer_si.step()
#                 optimizer_si.zero_grad()
#                 lr_scheduler_si.step()
                
#                 # 更新 EMA
#                 ema_si_v.step(si_velocity_net.parameters())
#                 ema_si_eta.step(si_denoiser_net.parameters())
                
#                 # 记录数据
#                 loss_val = loss.item()
#                 v_loss_val = loss_v.item()
#                 eta_loss_val = loss_eta.item()
                
#                 epoch_loss.append(loss_val)
#                 epoch_v_loss.append(v_loss_val)
#                 epoch_eta_loss.append(eta_loss_val)

#                 global_step += 1
                
#                 # [修改点] 这里增加了 s_loss (score loss) 的显示
#                 tepoch.set_postfix(loss=loss_val, v_loss=v_loss_val, s_loss=eta_loss_val, step=global_step)
                
#         # [修改点] Epoch 结束时也可以看平均的分项 Loss
#         ep_loss = float(np.mean(epoch_loss)) if len(epoch_loss) else 0.0
#         ep_v = float(np.mean(epoch_v_loss)) if len(epoch_v_loss) else 0.0
#         ep_s = float(np.mean(epoch_eta_loss)) if len(epoch_eta_loss) else 0.0
#         tglobal.set_postfix(
#             loss=ep_loss,
#             v_loss=ep_v,
#             s_loss=ep_s
#         )

#         train_history.append({
#             "epoch": int(epoch_idx),
#             "global_step": int(global_step),
#             "loss": ep_loss,
#             "v_loss": ep_v,
#             "s_loss": ep_s,
#         })

#         # [修改] 保存可续训 checkpoint（同时写一份 latest 方便 resume）
#         if (epoch_idx + 1) % int(save_every_epochs) == 0:
#             ckpt_path_epoch = os.path.join(save_dir, f"square_state_{epoch_idx+1}_ep_si_abs.ckpt")
#             ckpt_path_latest = os.path.join(save_dir, "square_state_latest_si_abs.ckpt")
#             save_checkpoint(ckpt_path_epoch, epoch_idx=epoch_idx, global_step=global_step, history=train_history)
#             save_checkpoint(ckpt_path_latest, epoch_idx=epoch_idx, global_step=global_step, history=train_history)
#             print(f" Saved resumable checkpoint to {ckpt_path_epoch} (and latest)")

# # 保存模型
# ckpt_path_si = os.path.join(save_dir, f"square_state_{num_epochs_si}_ep_si_abs.ckpt")
# ckpt_path_latest = os.path.join(save_dir, "square_state_latest_si_abs.ckpt")
# save_checkpoint(ckpt_path_si, epoch_idx=num_epochs_si - 1, global_step=global_step, history=train_history)
# save_checkpoint(ckpt_path_latest, epoch_idx=num_epochs_si - 1, global_step=global_step, history=train_history)
# print(f"Saved Streaming SI checkpoint to {ckpt_path_si} (and {ckpt_path_latest})")

# 准备推理用的 EMA 模型
# ema_si_velocity_net = si_velocity_net
# ema_si_denoiser_net = si_denoiser_net
# ema_si_v.copy_to(ema_si_velocity_net.parameters())
# ema_si_eta.copy_to(ema_si_denoiser_net.parameters())
#----------------------------------------------------------
# =========================================================
# 加载预训练的 Streaming SI Policy 模型 (跳过训练)
# =========================================================

load_pretrained = True
ckpt_path_si = "/home/users/meiyi/streaming-flow-policy/notebooks/pusht/Robotic_mimic/models/square_state_1200_ep_si_abs.ckpt"

# 检查文件是否存在
if load_pretrained and os.path.isfile(ckpt_path_si):
    print(f"Found pretrained checkpoint: {ckpt_path_si}")
    checkpoint = _torch_load_compat(ckpt_path_si, map_location=device)

    # 1. 加载权重到基础网络
    si_velocity_net.load_state_dict(checkpoint['velocity_net'])
    si_denoiser_net.load_state_dict(checkpoint['denoiser_net'])

    # 2. 设置推理用的模型
    # 在跳过训练的情况下，我们将直接使用加载的权重作为推理模型
    # (模拟 EMA 模型及其接口)
    ema_si_velocity_net = si_velocity_net
    ema_si_denoiser_net = si_denoiser_net

    print('Pretrained weights loaded for Streaming SI Policy. Ready for inference!')

else:
    print(f"Checkpoint {ckpt_path_si} not found. Please run the training cell below.")
    # 如果需要，这里可以初始化 EMA 模型容器，以便后续训练使用
    # ema_si_velocity_net = ... (通常在训练循环中通过 EMAModel 初始化)
#----------------------------------------------------------
import collections
import numpy as np
import torch
import math
from tqdm.auto import tqdm
from scipy.spatial.transform import Rotation as R

# =========================================================
# 辅助函数: 计算 Drift (适配 10维 Action)
# =========================================================
def get_drift(x, t, global_cond, sigma_infer, eps=1e-6):
    """
    计算修正后的漂移项 b(x, t)
    x: (B, 1, 10)
    """
    if t.ndim == 0: t = t.unsqueeze(0)
    
    # 预测 v 和 eta
    v_pred = ema_si_velocity_net(sample=x, timestep=t, global_cond=global_cond)
    eta_pred = ema_si_denoiser_net(sample=x, timestep=t, global_cond=global_cond)
    
    # 获取 gamma 及其导数
    gamma = gamma_t_si(t).view(-1, 1, 1).to(device)
    gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
    
    # 计算 score 和 漂移系数
    s_pred = -eta_pred / (gamma + eps)
    score_coeff = 0.5 * (sigma_infer ** 2) - (gamma * gamma_dot)
    
    b = v_pred + score_coeff * s_pred
    return b

# =========================================================
# 核心修复: SSIP 推理函数 (Absolute Control 适配版)
# =========================================================
def run_si_inference(sigma_infer, title):
    # 1. 重置环境
    obs = wrapper.reset() 
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    rewards = []
    # 兼容性更强的 success 统计：
    # - 对 robosuite 的稀疏奖励任务（reward_shaping=False），reward>0 基本等价于成功
    # - 同时保留 info 里的 success/task_success 检测（不同版本 key 不同）
    success_steps = 0
    _printed_success_info = False
    
    # 2. Warm Start: 基于当前观测构建初始流状态 x_0
    # -----------------------------------------------------------
    # [关键]: 必须准确地将当前 obs 转换为 10维的 latent action 格式
    # -----------------------------------------------------------
    
    # 硬编码
    # start_dim = 14 

    # Warm start：由于我们已把控制器设为 absolute（control_delta=False），这里用当前 eef pos 初始化更合理
    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    # absolute 模式：需要从观测中取 eef pos 作为初值
    key_dims = []
    for key in wrapper.obs_keys:
        dummy_obs = wrapper.env.get_observation()[key]
        dim = np.array(dummy_obs).flatten().shape[0]
        key_dims.append(dim)
    target_key = 'robot0_eef_pos'
    if target_key in wrapper.obs_keys:
        idx = wrapper.obs_keys.index(target_key)
        start_dim = sum(key_dims[:idx])
    else:
        raise ValueError(f"Obs keys 里面没有 {target_key}！")
    print(f"自动计算的 Start Dim: {start_dim}")
    curr_pos_or_delta = obs[start_dim : start_dim+3].astype(np.float32)

    norm_pos = (curr_pos_or_delta - pos_min) / (pos_max - pos_min + 1e-6)
    norm_pos = norm_pos * 2 - 1
    
    # 3. 初始化 na (Latent State)
    # 维度: (1, 1, 10)
    na = torch.zeros((1, 1, 10), device=device, dtype=torch.float32)
    
    # [Fix 1]: 覆盖位置 (保持不变)
    na[0, 0, :3] = torch.from_numpy(norm_pos).to(device)
    
    # [Fix 2 - 关键]: 强制初始化夹爪为 "张开" (-1.0)
    # 假设第 10 维 (索引 9) 是夹爪。
    # 在归一化空间中，-1 通常代表 raw action 的最小值 (Open)
    na[0, 0, 9] = -1 
    
    # [Fix 3 - 关键]: 6D Rotation 的全 0 是退化的，会在 RotationTransformer 中产生 NaN
    # 这里用“单位旋转”的 6D 表示（矩阵前两列）：[1,0,0, 0,1,0]，并映射到当前 action_normalizer 的归一化空间
    rot6d_raw = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    rot_min = dataset.action_normalizer.min_val[3:9].astype(np.float32)
    rot_max = dataset.action_normalizer.max_val[3:9].astype(np.float32)
    # 保证在数据集范围内，避免归一化超出 [-1,1]
    rot6d_raw = np.clip(rot6d_raw, rot_min, rot_max)
    rot6d_norm = (rot6d_raw - rot_min) / (rot_max - rot_min + 1e-6)
    rot6d_norm = rot6d_norm * 2 - 1
    na[0, 0, 3:9] = torch.from_numpy(rot6d_norm).to(device)

    
    # 设置为流的起点
    na_from_prev_chunk = na
    
    # =========================================================
    
    done = False
    step_idx = 0
    max_steps = 400 
    dt_val = 1.0 / (pred_horizon - obs_horizon) # 流的时间步长
    
    print(f"Running Inference: {title} (sigma={sigma_infer})")
    
    with torch.no_grad():
        with tqdm(total=max_steps, desc=title) as pbar:
            while not done and step_idx < max_steps:
                # --- A. 准备 Observation ---
                obs_seq = np.stack(obs_deque)
                # [修复点]: 使用 dataset.obs_normalizer
                nobs = dataset.obs_normalizer.normalize(obs_seq)
                o_test = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten().unsqueeze(0)
                
                # --- B. Streaming Loop ---
                na = na_from_prev_chunk
                
                for i in range(action_horizon):
                    # 1. 解码当前动作并执行
                    # [修复点]: 使用 dataset.get_unnormalized_action 自动处理反归一化+6D转回
                    a_real_full = dataset.get_unnormalized_action(na) # 返回 (1, 1, 7) numpy
                    a_real = a_real_full.squeeze() # (7,)
                    if step_idx < 5 and i == 0:
                        # 轻量 debug：确认动作幅度合理（delta 模式下应接近数据分布范围）
                        print(f"[debug] a_real (first steps) = {a_real}")
                    
                    # 环境交互
                    obs, reward, done, info = wrapper.step(a_real)
                    # success 判定（强鲁棒）：
                    # 1) 稀疏奖励：reward>0 视为成功
                    # 2) 若 info 中带 success 标记，也计入
                    r = float(reward) if reward is not None else 0.0
                    info_success = False
                    if isinstance(info, dict):
                        info_success = bool(info.get("success", False) or info.get("task_success", False))
                    if (r > 0.0) or info_success:
                        success_steps += 1
                        if (not _printed_success_info) and isinstance(info, dict):
                            _printed_success_info = True
                            print(f"[success debug] reward={r} info_keys={list(info.keys())}")
                            # 打印一些常见字段，帮助后续更精准对齐
                            for k in ["success", "task_success", "is_success", "terminated", "TimeLimit.truncated"]:
                                if k in info:
                                    print(f"[success debug] info[{k}]={info.get(k)}")
                    
                    obs_deque.append(obs)
                    rewards.append(reward)
                    
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(reward=reward)
                    
                    if done or step_idx >= max_steps: break
                    
                    # 2. 积分 (Euler Step) 计算下一个时间步的动作
                    t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                    t = torch.tensor([t_scalar], device=device, dtype=torch.float32)
                    
                    b_drift = get_drift(na, t, o_test, sigma_infer)
                    noise = torch.randn_like(na)
                    diffusion = sigma_infer * math.sqrt(dt_val) * noise
                    
                    na = na + b_drift * dt_val + diffusion
                    # --- 数值安全：限制 latent 在 [-1, 1]，并清理 NaN/Inf，避免反归一化后动作爆炸 ---
                    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0)
                    na = na.clamp(-1.0, 1.0)
                
                # 更新流的起点 (Streaming)
                na_from_prev_chunk = na
                
                if done: break

    print(f"推理结束! 最高得分: {max(rewards) if rewards else 0} | success_steps: {success_steps}")

# =========================================================
# 执行推理
# =========================================================

# # 1. ODE 模式 (平滑)
# run_si_inference(sigma_infer=0.0, title="Streaming SI (ODE Mode)")

# # 2. SDE 模式 (探索)
# run_si_inference(sigma_infer=0.02, title="Streaming SI (SDE Mode)")
#----------------------------------------------------------
#------------------------CCG Training通用------------------------------

# %% [markdown]
# ### CCG

# %%
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR

# 检测设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Add the specific numpy scalar to the allowed list (for safe loading)
# torch.serialization.add_safe_globals([np._core.multiarray.scalar])
# torch.serialization.add_safe_globals([
#     np.dtype, 
#     np._core.multiarray.scalar, 
#     np._core.multiarray._reconstruct
# ])

# =========================================================
# 0. CCG 跨任务适配辅助函数（lift / can / square）
#   - 不再写死 obs_dim / action_dim / start_dim / pred_horizon
#   - 距离与碰撞判定统一基于真实 eef_pos（从 obs 中提取），避免 delta/absolute 混淆
# =========================================================
from typing import Dict, Optional

def _infer_task_name_from_dataset_path(dataset_path: str) -> str:
    p = (dataset_path or "").lower()
    for name in ["lift", "can", "square"]:
        if f"/{name}/" in p:
            return name
    return "unknown"

def _get_obs_key_dims(wrapper) -> Dict[str, int]:
    key_dims: Dict[str, int] = {}
    raw = wrapper.env.get_observation()
    for k in wrapper.obs_keys:
        key_dims[k] = int(np.asarray(raw[k]).reshape(-1).shape[0])
    return key_dims

def _get_eef_pos_start(wrapper, target_key: str = "robot0_eef_pos") -> int:
    if target_key not in wrapper.obs_keys:
        raise ValueError(f"wrapper.obs_keys 里没有 {target_key}，当前 keys={wrapper.obs_keys}")
    key_dims = _get_obs_key_dims(wrapper)
    idx = wrapper.obs_keys.index(target_key)
    return int(sum(key_dims[k] for k in wrapper.obs_keys[:idx]))

def _get_eef_pos_from_obs(obs: np.ndarray, eef_pos_start: int) -> np.ndarray:
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    return obs[eef_pos_start : eef_pos_start + 3].astype(np.float32)

def _raw_to_norm(raw: np.ndarray, min_v: np.ndarray, max_v: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float32)
    min_v = np.asarray(min_v, dtype=np.float32)
    max_v = np.asarray(max_v, dtype=np.float32)
    denom = (max_v - min_v + 1e-6)
    norm01 = (raw - min_v) / denom
    norm01 = np.clip(norm01, 0.0, 1.0)
    return norm01 * 2.0 - 1.0

def _init_na_from_obs(
    obs: np.ndarray,
    dataset,
    eef_pos_start: int,
    control_delta: Optional[bool],
    device: torch.device,
) -> torch.Tensor:
    """
    初始化 latent action na: (1, 1, action_dim)
    - absolute: 用当前 eef_pos 作为初始 pos
    - delta: 用 0 增量作为初始 pos
    - 旋转/夹爪使用数值稳定初始化（避免 6D 退化导致 NaN）
    """
    action_dim_local = int(dataset[0]["action"].shape[-1])
    na = torch.zeros((1, 1, action_dim_local), device=device, dtype=torch.float32)

    is_delta = bool(control_delta) if control_delta is not None else True
    curr_eef_pos = _get_eef_pos_from_obs(obs, eef_pos_start)  # 物理坐标
    raw_pos = np.zeros(3, dtype=np.float32) if is_delta else curr_eef_pos

    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    na[0, 0, :3] = torch.from_numpy(_raw_to_norm(raw_pos, pos_min, pos_max)).to(device)

    # gripper: 默认第 9 维是夹爪；用 “张开”(-1) 做稳定起点
    if action_dim_local >= 10:
        na[0, 0, 9] = -1.0

    # 6D rotation: 用单位旋转的稳定 6D 表示，并映射到归一化空间
    if action_dim_local >= 9:
        rot6d_raw = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
        rot_min = dataset.action_normalizer.min_val[3:9].astype(np.float32)
        rot_max = dataset.action_normalizer.max_val[3:9].astype(np.float32)
        rot6d_raw = np.clip(rot6d_raw, rot_min, rot_max)
        na[0, 0, 3:9] = torch.from_numpy(_raw_to_norm(rot6d_raw, rot_min, rot_max)).to(device)

    na = torch.nan_to_num(na, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)
    return na

def _r_obs_norm_from_phys(dataset, r_phys: float) -> float:
    """
    物理空间半径 r_phys（米）粗略换算到归一化空间半径 r_norm。
    x_norm = (x-min)/(max-min)*2-1 => dx_norm ≈ 2*dx_phys/(max-min)
    用 xyz 的平均尺度做近似（足够用于风险打标阈值）。
    """
    pos_min = dataset.action_normalizer.min_val[:3].astype(np.float32)
    pos_max = dataset.action_normalizer.max_val[:3].astype(np.float32)
    scale = np.mean(np.maximum(pos_max - pos_min, 1e-6))
    return float(2.0 * r_phys / scale)

def _task_cfg(task_name: str) -> Dict[str, float]:
    cfg: Dict[str, float] = dict(
        collision_threshold_phys=0.02,
        warning_dist_phys=0.05,
        activation_dist=0.2,
        obstacle_offset_xy=0.02,
        obstacle_offset_z=0.0,
        obstacle_noise_std_norm=0.10,
        table_z=0.8,
    )
    if task_name == "lift":
        cfg.update(collision_threshold_phys=0.02, warning_dist_phys=0.05, activation_dist=0.25)
    elif task_name == "can":
        cfg.update(collision_threshold_phys=0.02, warning_dist_phys=0.05, activation_dist=0.20)
    elif task_name == "square":
        cfg.update(collision_threshold_phys=0.02, warning_dist_phys=0.05, activation_dist=0.20)

    cfg["task_name"] = task_name
    cfg["critic_save_path"] = f"Robust_critic_{task_name}.pth" if task_name != "unknown" else "Robust_critic.pth"
    cfg["critic_checkpoint_path"] = cfg["critic_save_path"].replace(".pth", "_latest_checkpoint.pth")
    return cfg

# =========================================================
# 1. 网络架构定义 (High Capacity ResNet Critic)
# =========================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Mish()
        )
        
    def forward(self, x):
        return x + self.net(x)

class CollisionPredictionCritic(nn.Module):
    def __init__(
        self,
        action_dim,
        obs_dim,
        obs_horizon,
        hidden_dim=1024,
        depth=6,
        eef_pos_start: Optional[int] = None,
        use_obs_eef_pos: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.eef_pos_start = eef_pos_start
        self.use_obs_eef_pos = use_obs_eef_pos
        
        # 1. Context Encoder: 处理 obs_dim 维的观测序列（不同任务 obs_dim 可能不同）
        self.context_encoder = nn.Sequential(
            nn.Conv1d(obs_dim, 256, kernel_size=3, padding=1),
            nn.Mish(),
            nn.Conv1d(256, 512, kernel_size=3, padding=1),
            nn.Mish(),
            nn.AdaptiveAvgPool1d(1), 
            nn.Flatten()             
        )
        
        # 2. Geometric Encoder
        # 输入维度计算:
        # a (10) + t (1) + obs_pos (3) + rel_vec (3) + dist (1) + alignment (1) = 19
        self.geo_input_dim = action_dim + 1 + 3 + 3 + 1 + 1 
        
        self.geo_encoder = nn.Sequential(
            nn.Linear(self.geo_input_dim, 512),
            nn.Mish(),
            nn.LayerNorm(512)
        )
        
        self.fusion = nn.Linear(512 + 512, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim) for _ in range(depth)])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.Mish(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )

    def forward(self, a, t, obs_pos, global_cond):
        """
        a: (B, 10) - Current Action (normalized)
        t: (B, 1) or (B,)
        obs_pos: (B, 3) - 虚拟障碍物的 3D 位置 (normalized)
        global_cond: (B, obs_dim * obs_horizon) - flattened observation
        """
        B = a.shape[0]
        if t.dim() == 1: t = t.unsqueeze(-1)
        
        # 1. 上下文编码
        seq_len = global_cond.shape[1] // self.obs_dim
        obs_seq = global_cond.view(B, seq_len, self.obs_dim)  # (B, T, obs_dim)
        ctx_feat = self.context_encoder(obs_seq.transpose(1, 2))

        # 2. 几何特征构造 (只关注前3维 XYZ 位置)
        # 跨任务/跨控制模式的关键：优先使用 obs 里的 eef_pos（归一化后），避免把 a[:3] 错当“当前位置”
        if self.use_obs_eef_pos and (self.eef_pos_start is not None):
            curr_pos = obs_seq[:, -1, self.eef_pos_start : self.eef_pos_start + 3]  # (B, 3)
        else:
            curr_pos = a[:, :3] # (B, 3)
        
        rel_vec = obs_pos - curr_pos # (B, 3)
        dist = torch.norm(rel_vec, dim=-1, keepdim=True) # (B, 1)
        
        # 计算对齐度 (Alignment)
        # 归一化动作向量 (只看位置移动方向)
        pos_vel_norm = curr_pos / (torch.norm(curr_pos, dim=-1, keepdim=True) + 1e-7)
        obs_dir = rel_vec / (dist + 1e-7)
        alignment = (pos_vel_norm * obs_dir).sum(dim=-1, keepdim=True)
        
        # 拼接所有几何特征
        geo_in = torch.cat([a, t, obs_pos, rel_vec, dist, alignment], dim=-1)
        geo_feat = self.geo_encoder(geo_in)

        # 3. 融合与残差推理
        x = torch.cat([ctx_feat, geo_feat], dim=-1)
        x = self.fusion(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# %%

# =========================================================
# 2. 风险评估逻辑 (Labeling Function - 3D Version)
# =========================================================

def compute_hybrid_risk_3d(trajectory, obs_pos, r_obs=0.07, sharpness=100.0):
    """
    trajectory: (Steps, Total_B, 10) - 预测的动作轨迹
    obs_pos: (Total_B, 3) - 虚拟障碍物位置
    """
    # 只取 XYZ 位置 (前3维)
    traj_pos = trajectory[..., :3] # (Steps, B, 3)
    
    # 1. 碰撞概率 (Prob Hit) - 轨迹上任意一点与障碍物的最小距离
    # dists: (Steps, B)
    dists = torch.norm(traj_pos - obs_pos.unsqueeze(0), p=2, dim=-1)
    min_dist = torch.min(dists, dim=0)[0] # (B,)
    
    # r_obs 是碰撞半径，在归一化空间中 0.05 大约对应实际空间的 2.5cm-5cm
    prob_hit = torch.sigmoid(sharpness * (r_obs - min_dist))
    
    # 2. 瞄准概率 (Prob Aim) - 初始移动方向是否指向障碍物
    start_pos = traj_pos[0] # (B, 3)
    future_idx = min(5, traj_pos.shape[0]-1)
    future_pos = traj_pos[future_idx] 
    
    vec_move = future_pos - start_pos
    dist_move = torch.norm(vec_move, p=2, dim=-1) + 1e-7
    dir_move = vec_move / dist_move.unsqueeze(-1)
    
    vec_to_obs = obs_pos - start_pos
    dist_to_obs = torch.norm(vec_to_obs, p=2, dim=-1) + 1e-7
    dir_to_obs = vec_to_obs / dist_to_obs.unsqueeze(-1)
    
    alignment = torch.relu((dir_move * dir_to_obs).sum(dim=-1))
    
    # 距离衰减因子: 如果离得远，即使对准了也不算太危险
    dist_factor = torch.exp(-(dist_to_obs**2) / (2 * 0.5**2))
    
    prob_aim = (alignment ** 2) * dist_factor
    
    # Visualize
    # if 0.6 *prob_aim[0] > prob_hit[0]:
    #     print("Probaim", 0.6 *prob_aim[0])
    # if prob_hit[0] > 0.6 * prob_aim[0]:
    #     print("Prob_____hit", prob_hit[0])


    return torch.maximum(prob_hit, 0.6 * prob_aim)

# =========================================================
# 3. 核心训练函数 (含恢复逻辑)
# =========================================================

def train_robust_critic(
    dataloader,
    si_velocity_net,
    action_dim,
    obs_dim,
    obs_horizon,
    pred_horizon,
    eef_pos_start: Optional[int] = None,
    epochs=150,
    K_samples=12,
    sample_t_max=0.8,
    r_obs_norm: float = 0.07,
    obstacle_noise_std_norm: float = 0.10,
    save_path="Robust_critic.pth",
):
    
    print(f"--- Initialization ---")
    checkpoint_path = save_path.replace(".pth", "_latest_checkpoint.pth")
    best_model_path = save_path.replace(".pth", "_best.pth")
    
    # 初始化 Critic（跨任务：支持不同 obs_dim；几何特征默认取 obs 里的 eef_pos）
    critic = CollisionPredictionCritic(
        action_dim, obs_dim, obs_horizon,
        hidden_dim=1024, depth=6,
        eef_pos_start=eef_pos_start,
        use_obs_eef_pos=True,
    ).to(device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.MSELoss()
    
    start_epoch = 0
    best_loss = float('inf')
    # 与推理侧 dt_val 保持一致（更稳，也便于跨任务复用）
    dt = 1.0 / max(1, (pred_horizon - obs_horizon))

    # 加载断点
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        critic.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f"Restarting at Epoch {start_epoch+1}")

    # 定义 Drift 函数 (使用你已经训练好的 si_velocity_net)
    def get_drift(a, t, cond):
        t_in = torch.clamp(t, 0.02, 0.98).view(-1)
        # 输入需要 unsqueeze 适配 UNet 接口: (B, 1, 10)
        v = si_velocity_net(sample=a.unsqueeze(1), timestep=t_in, global_cond=cond).squeeze(1)
        return v

    for epoch in range(start_epoch, epochs):
        epoch_losses = []
        critic.train()
        
        with tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}', leave=False) as pbar:
            for batch in pbar:
                obs = batch['obs'].to(device)
                gt_action = batch['action'].to(device) # (B, 16, 10)
                
                B = obs.shape[0]
                global_cond = obs.flatten(start_dim=1)
                
                # ==========================================
                # A. 动态生成虚拟障碍物 (Target Sampling)
                # ==========================================
                with torch.no_grad():
                    # 随机选择一个起始时间 t
                    t_start = torch.rand(B, 1, device=device) * sample_t_max
                    
                    # 根据 t 映射到未来的某个时间步索引
                    # obs_horizon-1 是当前的最后一帧，我们往后看
                    # 用 pred_horizon 自适配，不再写死 16
                    future_step = (t_start * max(1, (pred_horizon - 1))).long()
                    target_idx = torch.clamp(obs_horizon - 1 + future_step, max=gt_action.shape[1]-1)
                    
                    # 取出该时间步的 GT Action (B, 1, 10) -> (B, 10)
                    # gather 需要 index 维度匹配
                    idx_expanded = target_idx.unsqueeze(-1).expand(-1, -1, action_dim)
                    a_base = torch.gather(gt_action, 1, idx_expanded).squeeze(1)

                    # 为了获得更鲁棒的障碍物位置，我们可以从当前点 rollout 一小段
                    curr_a, curr_t, path = a_base.clone(), t_start.clone().view(-1), [a_base.clone()]
                    warm_roll_steps = min(5, max(1, pred_horizon - 1))
                    for _ in range(warm_roll_steps):
                        v = get_drift(curr_a, curr_t, global_cond)
                        curr_a += v * dt * (curr_t < 0.98).float().unsqueeze(1)
                        curr_t += dt
                        path.append(curr_a.clone())
                    
                    path_tensor = torch.stack(path, dim=0) # (Steps, B, 10)
                    
                    # 随机从 rollout 的路径中选一个点作为障碍物中心
                    rand_step = torch.randint(0, len(path), (B,))
                    base_obs_full = path_tensor[rand_step, torch.arange(B)] # (B, 10)
                    
                    # [关键] 障碍物只定义为 XYZ (前3维)，并加入噪声
                    base_obs_pos = base_obs_full[:, :3]
                    final_obs_pos = torch.clamp(
                        base_obs_pos + torch.randn_like(base_obs_pos) * float(obstacle_noise_std_norm),
                        -0.95, 0.95
                    ) # (B, 3)

                # ==========================================
                # B. 构造多样化动作 (Action Perturbation)
                # ==========================================
                n_pert = 6
                
                # 在 10D 空间中，我们使用高斯噪声来产生多样性
                # a_base: (B, 10)
                noise_small = torch.randn_like(a_base) * 0.1
                noise_med = torch.randn_like(a_base) * 0.3
                noise_large = torch.randn_like(a_base) * 0.6
                
                # 也可以尝试只扰动位置 (前3维)
                noise_pos_only = torch.zeros_like(a_base)
                noise_pos_only[:, :3] = torch.randn(B, 3, device=device) * 0.4
                
                a_list = [
                    a_base,                  # 原始 GT
                    a_base + noise_small,    # 小扰动
                    a_base + noise_med,      # 中扰动
                    a_base + noise_large,    # 大扰动
                    a_base + noise_pos_only, # 仅位置扰动
                    a_base - noise_pos_only  # 反向位置扰动
                ]
                
                # (B * n_pert, 10)
                a_exp = torch.stack(a_list, dim=1).view(-1, action_dim)
                
                # 扩展其他条件
                cond_exp = global_cond.repeat_interleave(n_pert, dim=0)
                t_exp = t_start.repeat_interleave(n_pert, dim=0)
                obs_exp = final_obs_pos.repeat_interleave(n_pert, dim=0) # (B*n_pert, 3)

                # ==========================================
                # C. Rollout 采样风险标签 (K Samples per action)
                # ==========================================
                # 对每个 perturbed action，再采样 K 条轨迹 (考虑 SDE 的随机性)
                curr_a_k = a_exp.repeat_interleave(K_samples, dim=0)
                curr_t_k = t_exp.repeat_interleave(K_samples, dim=0).view(-1)
                curr_cond_k = cond_exp.repeat_interleave(K_samples, dim=0)
                
                traj = [curr_a_k.clone()]
                with torch.no_grad():
                    rollout_steps = min(12, max(1, pred_horizon - 1))
                    for _ in range(rollout_steps): # Rollout to estimate collision risk
                        v = get_drift(curr_a_k, curr_t_k, curr_cond_k)
                        # SDE 噪声项: 0.05 * sqrt(dt)
                        noise = torch.randn_like(curr_a_k)
                        curr_a_k += (v * dt + 0.05 * math.sqrt(dt) * noise) * (curr_t_k < 0.99).float().unsqueeze(1)
                        curr_t_k += dt
                        traj.append(curr_a_k.clone())
                
                # 计算 Risk Label
                # traj_stack: (Steps, Total_B, 10)
                traj_stack = torch.stack(traj, dim=0)
                
                # 扩展 obs_exp 以匹配 K_samples
                obs_targets = obs_exp.repeat_interleave(K_samples, dim=0)
                
                # 使用 3D 风险函数
                target_risk = compute_hybrid_risk_3d(traj_stack, obs_targets, r_obs=float(r_obs_norm))
                
                # 聚合 K 条轨迹的风险 (取平均) -> (B * n_pert, 1)
                target_risk = target_risk.view(B * n_pert, K_samples).mean(dim=1, keepdim=True)

                # ==========================================
                # D. 训练步
                # ==========================================
                # Critic 输入: a(10), t(1), obs(3), cond(flat)
                pred_risk = critic(a_exp, t_exp, obs_exp, cond_exp)
                
                loss = loss_fn(pred_risk, target_risk)
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
                optimizer.step()
                
                epoch_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.5f}")

        # E. 存储逻辑
        scheduler.step()
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.7f}")

        # 保存断点
        ckpt = {
            'epoch': epoch, 'model_state_dict': critic.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), 'best_loss': best_loss
        }
        torch.save(ckpt, checkpoint_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(critic.state_dict(), best_model_path)
            print("Successfully saved best model.")

        # if (epoch + 1) % 20 == 0:
        #     torch.save(critic.state_dict(), save_path.replace(".pth", f"_ep{epoch+1}.pth"))

    torch.save(critic.state_dict(), save_path)
    return critic

# =========================================================
# 调用示例:
# 确保 action_dim=10, obs_dim=19
_ccg_task_name = _infer_task_name_from_dataset_path(dataset_path)
_ccg_cfg = _task_cfg(_ccg_task_name)
_ccg_action_dim = int(dataset[0]["action"].shape[-1])
_ccg_obs_dim = int(dataset[0]["obs"].shape[-1])
_ccg_eef_pos_start = _get_eef_pos_start(wrapper, "robot0_eef_pos")
_ccg_r_obs_norm = _r_obs_norm_from_phys(dataset, float(_ccg_cfg["collision_threshold_phys"]))

trained_critic = train_robust_critic(
    dataloader,
    si_velocity_net, # 传入之前训练好的 policy
    action_dim=_ccg_action_dim,
    obs_dim=_ccg_obs_dim,
    obs_horizon=obs_horizon,
    pred_horizon=pred_horizon,
    eef_pos_start=_ccg_eef_pos_start,
    r_obs_norm=_ccg_r_obs_norm,
    obstacle_noise_std_norm=float(_ccg_cfg["obstacle_noise_std_norm"]),
    save_path=str(_ccg_cfg["critic_save_path"]),
    epochs=300,
)


# %% [markdown]
# ### CCG Experiment At Scale

# %%
import os
import random  # [新增] 用于固定Python内置随机源
import numpy as np
import torch
import torch.nn as nn
import cv2
import collections
import math
import imageio
from tqdm.auto import tqdm

# =========================================================
# 假设前置依赖已定义 (get_drift, world_to_pixel_mujoco 等)
# =========================================================
def get_drift(x, t, global_cond, sigma_infer, eps=1e-6):
    """
    计算修正后的漂移项 b(x, t) = v + score_correction
    x: (B, 1, 10)
    """
    if t.ndim == 0: t = t.unsqueeze(0)
    
    # 预测 v 和 eta
    # 注意: 这里的 unsqueeze/squeeze 取决于你的网络定义，
    # 假设网络接受 (B, 1, 10) 并返回 (B, 1, 10) 或 (B, 10)
    # 这里为了通用性，确保输入是 (B, 1, 10)
    v_pred = ema_si_velocity_net(sample=x, timestep=t, global_cond=global_cond)
    eta_pred = ema_si_denoiser_net(sample=x, timestep=t, global_cond=global_cond)
    
    # 统一维度处理: 假设网络输出可能是 (B, 10) 或 (B, 1, 10)
    if v_pred.ndim == 2: v_pred = v_pred.unsqueeze(1)
    if eta_pred.ndim == 2: eta_pred = eta_pred.unsqueeze(1)

    # 获取 gamma 及其导数
    gamma = gamma_t_si(t).view(-1, 1, 1).to(device)
    gamma_dot = d_gamma_dt_si(t).view(-1, 1, 1).to(device)
    
    # 计算 score 和 漂移系数
    s_pred = -eta_pred / (gamma + eps)
    score_coeff = 0.5 * (sigma_infer ** 2) - (gamma * gamma_dot)
    
    b = v_pred + score_coeff * s_pred
    return b.squeeze(1) # This is really important otherwise there will be bug!!! This is the only differnece to the original get_drift() function.
def world_to_pixel_mujoco(sim, pos_3d, camera_name, img_h, img_w):
    try:
        cam_id = sim.model.camera_name2id(camera_name)
        cam_pos = sim.data.cam_xpos[cam_id]
        cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
        pos_rel = pos_3d - cam_pos
        pos_cam_opengl = cam_mat.T @ pos_rel
        x, y, z = pos_cam_opengl[0], -pos_cam_opengl[1], -pos_cam_opengl[2]
        if z < 0.01: return None
        fovy = sim.model.cam_fovy[cam_id]
        f = 0.5 * img_h / np.tan(fovy * np.pi / 360)
        cx, cy = img_w / 2, img_h / 2
        u, v = int(cx + f * x / z), int(cy + f * y / z)
        if 0 <= u < img_w and 0 <= v < img_h: return (u, v)
        return None
    except Exception: return None

# =========================================================
# 3. 修改后的推理函数
#    - 修复了进度条
#    - [新增] 全面固定随机种子
#    - [修改] 碰撞阈值统一为 0.03
# =========================================================

def run_ccg_phys_vis_inference(
    obstacle_pos_phys, 
    critic_model_path,       
    guidance_scale=20.0,     
    power_k=3.0,             
    activation_dist=0.6,
    seed=0,                  # 支持传入随机种子
    collect_trajectory=False # 是否收集轨迹 (用于 Base Run)
):
    # =========================================================
    # [关键修改 1] 全面固定随机种子 (RNG Seeding)
    # =========================================================
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    wrapper.seed(seed)
    # =========================================================

    # 确保跨任务配置已就绪（即使你跳过了上面的训练 cell 也能单独跑推理/实验）
    global _ccg_cfg, _ccg_action_dim, _ccg_obs_dim, _ccg_eef_pos_start
    if "_ccg_cfg" not in globals():
        _ccg_task_name = _infer_task_name_from_dataset_path(dataset_path)
        _ccg_cfg = _task_cfg(_ccg_task_name)
    if "_ccg_action_dim" not in globals():
        _ccg_action_dim = int(dataset[0]["action"].shape[-1])
    if "_ccg_obs_dim" not in globals():
        _ccg_obs_dim = int(dataset[0]["obs"].shape[-1])
    if "_ccg_eef_pos_start" not in globals():
        _ccg_eef_pos_start = _get_eef_pos_start(wrapper, "robot0_eef_pos")

    # --- 1. 初始化模型 ---
    critic_net = CollisionPredictionCritic(
        action_dim=_ccg_action_dim,
        obs_dim=_ccg_obs_dim,
        obs_horizon=obs_horizon,
        eef_pos_start=_ccg_eef_pos_start,
        use_obs_eef_pos=True,
    ).to(device)
    # checkpoint = torch.load(critic_model_path, map_location=device, weights_only=False)
    # critic_net = HighCapacityCritic(action_dim=10, obs_dim=19, obs_horizon=2, hidden_dim=1024, depth=6).to(device)
    checkpoint = torch.load(critic_model_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    critic_net.load_state_dict(state_dict)
    critic_net.eval()
    
    # --- 2. 准备障碍物数据 ---
    if obstacle_pos_phys is None:
        obstacle_pos_phys = np.array([10.0, 10.0, 10.0]) # Dummy position

    # A. 归一化坐标
    # 注意：这里默认假设是 absolute 控制（动作 pos 的 min/max 对应物理工作空间）
    if bool(CONTROL_DELTA) if CONTROL_DELTA is not None else False:
        print("[warn] CONTROL_DELTA=True：当前 CCG 障碍物归一化仍使用 action_normalizer 的 pos 范围，可能与物理空间不一致。建议使用 *_abs 数据集。")
    pos_min = dataset.action_normalizer.min_val[:3]
    pos_max = dataset.action_normalizer.max_val[:3]
    norm_obs_pos = (obstacle_pos_phys - pos_min) / (pos_max - pos_min) * 2 - 1
    obs_pos_tensor = torch.tensor(norm_obs_pos, device=device, dtype=torch.float32).unsqueeze(0)

    # B. 物理坐标
    obs_phys_tensor = torch.tensor(obstacle_pos_phys, device=device, dtype=torch.float32)

    # --- 3. 环境重置 ---
    # (wrapper.seed已经在上面设置过了，这里直接reset)
    obs = wrapper.reset()
    obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)
    imgs = [wrapper.render()]
    rewards = []
    
    trajectory_log = [] 
    has_collided = False
    
    # [关键修改 2] 物理碰撞判定阈值改为 0.025
    COLLISION_THRESHOLD = float(_ccg_cfg["collision_threshold_phys"])

    # 初始 latent action（跨任务：从 obs 自动初始化；避免 start_dim 写死）
    na = _init_na_from_obs(obs, dataset, _ccg_eef_pos_start, CONTROL_DELTA, device)
    na_from_prev_chunk = na
    
    # 状态变量
    cached_grad = torch.zeros_like(na).squeeze(1)
    cached_prob = 0.0          
    current_dynamic_scale = 0.0 
    
    done = False
    step_idx = 0
    max_steps = 400
    dt_val = 1.0 / (pred_horizon - obs_horizon)
    sigma_infer = 0.05 

    with torch.no_grad():
        # 进度条
        pbar = tqdm(total=max_steps, desc=f"Seed {seed}", disable=collect_trajectory)
        
        while not done and step_idx < max_steps:
            
            # --- 准备 Observation ---
            obs_seq = np.stack(obs_deque)
            nobs = dataset.obs_normalizer.normalize(obs_seq)
            cond_flat = torch.from_numpy(nobs).to(device, dtype=torch.float32).flatten(start_dim=0).unsqueeze(0)
            o_test = cond_flat 
            
            na = na_from_prev_chunk
            
            # --- 执行 Action Chunk ---
            for i in range(action_horizon):
                t_scalar = np.clip(i * dt_val, 1e-3, 1.0 - 1e-3)
                t_tensor = torch.tensor([t_scalar], device=device, dtype=torch.float32)

                # --- A. 物理距离计算 ---
                # 统一使用真实 eef_pos（来自 obs），避免 delta/absolute 语义混淆
                curr_xyz_np = _get_eef_pos_from_obs(obs, _ccg_eef_pos_start)
                curr_xyz = torch.tensor(curr_xyz_np, device=device, dtype=torch.float32)
                dist_to_obs = torch.norm(curr_xyz - obs_phys_tensor).item()

                if collect_trajectory:
                    trajectory_log.append(curr_xyz_np.copy())

                # [使用统一后的阈值判定]
                if dist_to_obs < COLLISION_THRESHOLD:
                    has_collided = True

                # --- B. CCG 引导计算 ---
                effective_scale = 0.0 if collect_trajectory else guidance_scale

                if dist_to_obs < activation_dist and effective_scale > 0:
                    with torch.enable_grad():
                        na_in = na.squeeze(1).detach().requires_grad_(True)
                        risk_prob = critic_net(na_in, t_tensor, obs_pos_tensor, cond_flat)
                        (g_risk,) = torch.autograd.grad(risk_prob, na_in)
                        
                        prob_val = risk_prob.item()
                        dynamic_scale = effective_scale * (prob_val ** power_k)
                        
                        cached_grad = g_risk.detach()
                        cached_prob = prob_val
                        current_dynamic_scale = dynamic_scale
                else:
                    cached_grad = torch.zeros_like(na).squeeze(1)
                    cached_prob = 0.0
                    current_dynamic_scale = 0.0

                # --- C. Drift 计算 ---
                base_drift = get_drift(na, t_tensor, o_test, sigma_infer)
                total_drift = base_drift - current_dynamic_scale * cached_grad
                
                noise = torch.randn_like(total_drift)
                na = na + total_drift.unsqueeze(1) * dt_val + sigma_infer * math.sqrt(dt_val) * noise.unsqueeze(1)
                
                # --- D. Step & Render ---
                a_real = dataset.get_unnormalized_action(na).squeeze()
                obs, reward, done, _ = wrapper.step(a_real)
                obs_deque.append(obs)
                rewards.append(reward)
                
                if not collect_trajectory:
                    img_rgb = wrapper.render().copy()
                    
                    # === 可视化部分 ===
                    if img_rgb.shape[0] > 64:
                        h, w = img_rgb.shape[:2]
                        sim_obj = getattr(wrapper, 'sim', None) 
                        if sim_obj:
                            pos_obs_3d = obstacle_pos_phys
                            pos_agent_3d = curr_xyz_np.copy()
                            pos_obs_ground_3d = pos_obs_3d.copy()
                            pos_obs_ground_3d[2] = float(_ccg_cfg["table_z"]) # table_z

                            px_obs = world_to_pixel_mujoco(sim_obj, pos_obs_3d, "agentview", h, w)
                            px_obs_ground = world_to_pixel_mujoco(sim_obj, pos_obs_ground_3d, "agentview", h, w)
                            px_agent = world_to_pixel_mujoco(sim_obj, pos_agent_3d, "agentview", h, w)
                            
                            if px_obs is not None and px_obs_ground is not None:
                                cv2.line(img_rgb, px_obs, px_obs_ground, (255, 255, 0), 1, cv2.LINE_AA)
                                cv2.circle(img_rgb, px_obs_ground, 3, (255, 255, 0), 1)
                            if px_agent is not None and px_obs is not None:
                                cv2.line(img_rgb, px_agent, px_obs, (200, 200, 200), 1, cv2.LINE_AA)

                            if px_obs is not None:
                                cv2.circle(img_rgb, px_obs, 6, (255, 0, 255), -1) 
                                cv2.putText(img_rgb, "OBS", (px_obs[0]+8, px_obs[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                            
                            if px_agent is not None:
                                cv2.circle(img_rgb, px_agent, 6, (0, 255, 0), -1) 
                                if current_dynamic_scale > 1e-2:
                                    grad_vec_3d = (-cached_grad * current_dynamic_scale).cpu().numpy().squeeze()[:3]
                                    vis_len = 2.0
                                    pos_end = pos_agent_3d + grad_vec_3d * vis_len
                                    px_end = world_to_pixel_mujoco(sim_obj, pos_end, "agentview", h, w)
                                    if px_end is not None:
                                        cv2.arrowedLine(img_rgb, px_agent, px_end, (0, 0, 255), 2, tipLength=0.3)
                        
                        # Info Panel
                        overlay = img_rgb.copy()
                        cv2.rectangle(overlay, (5, 5), (170, 55), (0, 0, 0), -1)
                        alpha = 0.6
                        cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0, img_rgb)
                        risk_color = (0, 255, 0) if cached_prob < 0.5 else (0, 0, 255)
                        cv2.putText(img_rgb, f"Risk: {cached_prob:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, risk_color, 1, cv2.LINE_AA)
                        cv2.putText(img_rgb, f"Scale: {current_dynamic_scale:.1f}", (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

                        # Red Border Warning
                        # [关键修改 3] 可视化里的阈值也统一为 0.025
                        COLLISION_DIST = float(_ccg_cfg["collision_threshold_phys"])
                        WARNING_DIST = float(_ccg_cfg["warning_dist_phys"])
                        if dist_to_obs < WARNING_DIST:
                            severity = np.clip((WARNING_DIST - dist_to_obs) / (WARNING_DIST - COLLISION_DIST), 0.0, 1.0)
                            border_color = (0, 0, int(50 + 205 * severity))
                            thickness = int(2 + 13 * severity)
                            cv2.rectangle(img_rgb, (0, 0), (w-1, h-1), border_color, thickness)
                            if dist_to_obs < COLLISION_DIST:
                                text = "COLLISION!"
                                text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
                                cv2.putText(img_rgb, text, ((w - text_size[0]) // 2, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

                    imgs.append(img_rgb)
                
                step_idx += 1
                pbar.update(1)
                
                if done or step_idx >= max_steps: break
            
            na_from_prev_chunk = na
            if done: break
            
        pbar.close()

    max_r = max(rewards) if rewards else 0
    return imgs, max_r, np.array(trajectory_log), has_collided


# =========================================================
# 4. 实验主循环
# =========================================================

def run_experiment_100_trials():
    # 确保跨任务配置已就绪（避免依赖上面的训练 cell）
    global _ccg_cfg
    if "_ccg_cfg" not in globals():
        _ccg_task_name = _infer_task_name_from_dataset_path(dataset_path)
        _ccg_cfg = _task_cfg(_ccg_task_name)

    # 1. 建立文件夹结构
    base_dir = "experiment_results"
    sub_dirs = {
        "collision": os.path.join(base_dir, "1_collision"),
        "failure":   os.path.join(base_dir, "2_failure_no_collision"),
        "success":   os.path.join(base_dir, "3_success_no_collision")
    }
    
    # 清理或创建文件夹
    for p in sub_dirs.values():
        os.makedirs(p, exist_ok=True)
    
    stats = {
        "collision": 0,
        "failure": 0,
        "success": 0
    }
    
    total_needed = 100
    valid_count = 0
    current_seed = 0
    
    print(f"开始 100 次实验: Base Pass -> Set Obstacle -> CCG Run")
    
    pbar = tqdm(total=total_needed, desc="Valid Trials")
    
    while valid_count < total_needed:
        current_seed += 1
        
        # --- Phase 1: Base Run (无障碍物，无引导) ---
        _, max_r_base, traj_base, _ = run_ccg_phys_vis_inference(
            obstacle_pos_phys=None, # 无障碍
            critic_model_path=str(_ccg_cfg["critic_checkpoint_path"]),
            guidance_scale=0.0,     # 无引导
            seed=current_seed,
            collect_trajectory=True # 只跑数值，不渲染
        )
        
        # 如果 Base Policy 本身就失败，跳过这个种子
        if max_r_base < 1.0:
            continue
            
        # --- Phase 2: 设置障碍物 ---
        traj_len = len(traj_base)
        if traj_len < 10: continue # 轨迹太短异常处理
        
        # 保持你的逻辑：在轨迹抓取阶段 (2% - 4%) 放置障碍物
        idx_start = int(traj_len * 0.02)
        idx_end = int(traj_len * 0.04)
        
        if idx_start >= idx_end: idx_end = idx_start + 1
        
        obs_idx = np.random.randint(idx_start, idx_end)
        base_pos = traj_base[obs_idx]

        offset_xy = np.random.uniform(
            -float(_ccg_cfg["obstacle_offset_xy"]),
            float(_ccg_cfg["obstacle_offset_xy"]),
            size=2,
        )
        random_offset = np.array(
            [offset_xy[0], offset_xy[1], float(_ccg_cfg["obstacle_offset_z"])],
            dtype=np.float32,
        )
        
        obstacle_pos_selected = base_pos + random_offset
        #obstacle_pos_selected = traj_base[obs_idx]
        
        # --- Phase 3: CCG Run (有障碍物，有引导) ---
        imgs, max_r_ccg, _, has_collided = run_ccg_phys_vis_inference(
            obstacle_pos_phys=obstacle_pos_selected,
            critic_model_path=str(_ccg_cfg["critic_checkpoint_path"]),
            guidance_scale=100.0,    # 开启强引导
            power_k=2.0,
            activation_dist=0.2,
            seed=current_seed,
            collect_trajectory=False # 需要渲染并保存 GIF
        )
        
        # --- Phase 4: 分类与保存 ---
        category = ""
        save_path = ""
        
        if has_collided:
            category = "collision"
            stats["collision"] += 1
            save_path = os.path.join(sub_dirs["collision"], f"seed_{current_seed}_collided.gif")
        elif max_r_ccg < 1.0:
            category = "failure"
            stats["failure"] += 1
            save_path = os.path.join(sub_dirs["failure"], f"seed_{current_seed}_failed.gif")
        else:
            category = "success"
            stats["success"] += 1
            save_path = os.path.join(sub_dirs["success"], f"seed_{current_seed}_success.gif")
            
        # 保存 GIF
        imageio.mimsave(save_path, imgs, fps=15)
        
        valid_count += 1
        pbar.update(1)
        pbar.set_postfix(
            Coll=f"{stats['collision']}", 
            Fail=f"{stats['failure']}", 
            Succ=f"{stats['success']}"
        )

    pbar.close()
    
    # --- 输出最终统计 ---
    print("\n" + "="*40)
    print("实验结束！统计结果如下：")
    print(f"总有效实验次数: {total_needed}")
    print(f"1. 碰撞 (Collision):              {stats['collision']} ({stats['collision']}%)")
    print(f"2. 避障但在任务中失败 (Failure):    {stats['failure']} ({stats['failure']}%)")
    print(f"3. 成功避障并完成任务 (Success):    {stats['success']} ({stats['success']}%)")
    print("="*40)
    print(f"结果已保存在 {base_dir} 目录下")

# 运行实验
run_experiment_100_trials()

"""
CUDA_VISIBLE_DEVICES=1 nohup python -u /home/users/meiyi/streaming-flow-policy/notebooks/pusht/Robotic_mimic/robomimic_abs_ccg_square-sacle.py > ccg_square_train.log 2>&1 &
"""