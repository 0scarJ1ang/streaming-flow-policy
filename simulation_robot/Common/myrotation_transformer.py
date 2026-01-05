import torch
import torch.nn.functional as F
import numpy as np

def pytorch_cov(x, rowvar=True, correction=1):
    """PyTorch implementation of NumPy's cov"""
    if x.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if x.dim() < 2:
        x = x.view(1, -1)
    if not rowvar and x.size(0) != 1:
        x = x.t()
    fact = 1.0 / (x.size(1) - correction)
    x -= torch.mean(x, dim=1, keepdim=True)
    return fact * x.matmul(x.t()).squeeze()

class RotationTransformer:
    def __init__(self, from_rep='axis_angle', to_rep='rotation_6d'):
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, x):
        """
        前向传播：将输入格式 (如 axis_angle) 转换为输出格式 (如 rotation_6d)
        通常用于 DataLoad 阶段
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # 1. 统一转为 Matrix
        if self.from_rep == 'axis_angle':
            matrix = self._axis_angle_to_matrix(x)
        elif self.from_rep == 'rotation_6d':
            matrix = self._rotation_6d_to_matrix(x)
        else:
            raise NotImplementedError(f"Unsupported from_rep: {self.from_rep}")

        # 2. 从 Matrix 转为目标格式
        if self.to_rep == 'rotation_6d':
            return self._matrix_to_rotation_6d(matrix)
        elif self.to_rep == 'axis_angle':
            return self._matrix_to_axis_angle(matrix)
        else:
            raise NotImplementedError(f"Unsupported to_rep: {self.to_rep}")

    def inverse(self, x):
        """
        逆向传播：将输出格式 (如 rotation_6d) 还原为输入格式 (如 axis_angle)
        通常用于推理阶段 (Model Output -> Env Action)
        """
        # 交换 from/to 并调用 forward
        temp_transformer = RotationTransformer(from_rep=self.to_rep, to_rep=self.from_rep)
        return temp_transformer.forward(x)

    # --- 核心转换函数 (PyTorch 实现) ---

    def _axis_angle_to_matrix(self, axis_angle):
        """
        Convert rotations given as axis/angle to rotation matrices.
        Args:
            axis_angle: Rotations given as a vector in axis angle form, shape (..., 3).
        Returns:
            Rotation matrices, shape (..., 3, 3).
        """
        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
        half_angles = angles * 0.5
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for zero angle, sin(0/2)/0 -> 0.5
        sin_half_angles_over_angles[small_angles] = 0.5
        
        quat = torch.cat(
            [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
        )
        return self._quaternion_to_matrix(quat)

    def _quaternion_to_matrix(self, quaternions):
        """
        Convert rotations given as quaternions to rotation matrices.
        Args:
            quaternions: quaternions with real part first, shape (..., 4).
        Returns:
            Rotation matrices, shape (..., 3, 3).
        """
        r, i, j, k = torch.unbind(quaternions, -1)
        two_s = 2.0 / (quaternions * quaternions).sum(-1)

        o = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        return o.reshape(quaternions.shape[:-1] + (3, 3))

    def _rotation_6d_to_matrix(self, d6):
        """
        Converts 6D rotation representation to 3x3 rotation matrix.
        Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
        Input: (B, 6)
        Output: (B, 3, 3)
        """
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def _matrix_to_rotation_6d(self, matrix):
        """
        Input: (B, 3, 3)
        Output: (B, 6) - The first two columns of the rotation matrix
        """
        return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))

    def _matrix_to_axis_angle(self, matrix):
        """
        Convert rotations given as rotation matrices to axis/angle.
        Args:
            matrix: Rotation matrices, shape (..., 3, 3).
        Returns:
            Rotations given as a vector in axis angle form, shape (..., 3).
        """
        return self._quaternion_to_axis_angle(self._matrix_to_quaternion(matrix))
    
    def _matrix_to_quaternion(self, matrix):
        """
        Convert rotation matrices to quaternions.
        """
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        
        batch_dim = matrix.shape[:-2]
        m = matrix.view(batch_dim + (9,))
        
        w = torch.sqrt(1.0 + m[..., 0] + m[..., 4] + m[..., 8]) / 2.0
        x = (m[..., 7] - m[..., 5]) / (4.0 * w)
        y = (m[..., 2] - m[..., 6]) / (4.0 * w)
        z = (m[..., 3] - m[..., 1]) / (4.0 * w)
        
        # This is a simplified implementation, for full robustness against numerical instability
        # (e.g. when trace is close to -1), check pytorch3d implementation.
        # But for Diffusion Policy context, this usually suffices if data is clean.
        
        q = torch.stack([w, x, y, z], dim=-1)
        return q

    def _quaternion_to_axis_angle(self, quaternions):
        """
        Convert rotations given as quaternions to axis/angle.
        """
        norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., :1])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for zero angle, sin(0/2)/0 -> 0.5
        sin_half_angles_over_angles[small_angles] = 0.5
        return quaternions[..., 1:] / sin_half_angles_over_angles