import torch
import torch.nn.functional as F
import numpy as np

class RotationTransformer:
    def __init__(self, from_rep='axis_angle', to_rep='rotation_6d'):
        self.from_rep = from_rep
        self.to_rep = to_rep

    def forward(self, x):
        """
        前向传播：用于 Dataset 加载阶段
        将输入 (如 axis_angle) -> 输出 (如 rotation_6d)
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        
        # 1. 先转为 3x3 矩阵
        if self.from_rep == 'axis_angle':
            matrix = self._axis_angle_to_matrix(x)
        elif self.from_rep == 'rotation_6d':
            matrix = self._rotation_6d_to_matrix(x)
        else:
            raise NotImplementedError(f"不支持的输入格式: {self.from_rep}")

        # 2. 再从 3x3 矩阵转为目标格式
        if self.to_rep == 'rotation_6d':
            return self._matrix_to_rotation_6d(matrix)
        elif self.to_rep == 'axis_angle':
            return self._matrix_to_axis_angle(matrix)
        else:
            raise NotImplementedError(f"不支持的输出格式: {self.to_rep}")

    def inverse(self, x):
        """
        逆向传播：用于推理阶段
        将输出 (如 rotation_6d) -> 还原为输入 (如 axis_angle)
        """
        # 巧妙利用：逆变换 = 交换源和目标后的 forward
        temp_transformer = RotationTransformer(from_rep=self.to_rep, to_rep=self.from_rep)
        return temp_transformer.forward(x)

    # ================= 核心算法实现 =================
    
    def _axis_angle_to_matrix(self, axis_angle):
        """将轴角转换为旋转矩阵"""
        # 计算角度
        angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
        half_angles = angles * 0.5
        eps = 1e-6
        small_angles = angles.abs() < eps
        
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        sin_half_angles_over_angles[small_angles] = 0.5
        
        # 转换为四元数 [w, x, y, z]
        quat = torch.cat(
            [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
        )
        return self._quaternion_to_matrix(quat)

    def _quaternion_to_matrix(self, quaternions):
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
        """6D -> 3x3 Matrix (Gram-Schmidt 正交化)"""
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    def _matrix_to_rotation_6d(self, matrix):
        """3x3 Matrix -> 6D (取前两列)"""
        return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))

    def _matrix_to_axis_angle(self, matrix):
        return self._quaternion_to_axis_angle(self._matrix_to_quaternion(matrix))
    
    def _matrix_to_quaternion(self, matrix):
        if matrix.size(-1) != 3 or matrix.size(-2) != 3:
            raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
        batch_dim = matrix.shape[:-2]
        m = matrix.view(batch_dim + (9,))
        w = torch.sqrt(1.0 + m[..., 0] + m[..., 4] + m[..., 8]) / 2.0
        x = (m[..., 7] - m[..., 5]) / (4.0 * w)
        y = (m[..., 2] - m[..., 6]) / (4.0 * w)
        z = (m[..., 3] - m[..., 1]) / (4.0 * w)
        q = torch.stack([w, x, y, z], dim=-1)
        return q

    def _quaternion_to_axis_angle(self, quaternions):
        norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., :1])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        sin_half_angles_over_angles[small_angles] = 0.5
        return quaternions[..., 1:] / sin_half_angles_over_angles