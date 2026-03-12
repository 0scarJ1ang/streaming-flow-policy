import os, csv
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import numpy as np

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

for i in range(22):
    # if i != 19:
    #     continue
    with open(f"processed_data/train_data_trial_2_{i}.csv", newline="") as f:
        reader = csv.reader(f)
        aligned_data = []
        next(reader, None)
        for row in reader:
            t,x,y,z,qx,qy,qz,qw,g,ox,oy,oz = row
            # qx,qy,qz,qw = list(map(float, [qx,qy,qz,qw]))
            p_mat = R.from_quat([qx,qy,qz,qw]).as_matrix()

            pose_6d = list(map(float,matrix_to_rotation_6d(torch.tensor(p_mat).unsqueeze(0))[0]))

            row = {
                'timestamp': t,
                'ee_x': x, 'ee_y': y, 'ee_z': z,
                'pose_6d_0': pose_6d[0], 'pose_6d_1': pose_6d[1], 'pose_6d_2': pose_6d[2], 'pose_6d_3': pose_6d[3],
                'pose_6d_4': pose_6d[4], 'pose_6d_5': pose_6d[5],
                'gripper': g,
                'obj_x': ox,
                'obj_y': oy,
                'obj_z': oz
            }
            aligned_data.append(row)

        df = pd.DataFrame(aligned_data)
        df.to_csv(f"reprocessed_data/train_data_trial_{i+25}.csv", index=False)
