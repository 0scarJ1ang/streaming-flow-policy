from sympy import Matrix
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from urchin import URDF

def camera_to_base(joint_angles):

    robot = URDF.load("panda_arm.urdf")
    # print(robot.visual)
    cfg = {"panda_joint1": joint_angles[0],
                         "panda_joint2": joint_angles[1],
                         "panda_joint3": joint_angles[2],
                         "panda_joint4": joint_angles[3],
                         "panda_joint5": joint_angles[4],
                         "panda_joint6": joint_angles[5],
                         "panda_joint7": joint_angles[6]}
    base_transform = robot.link_fk(cfg, "robotiq_arg2f_base_link")

    panda_robotiq = R.from_euler('z', -np.pi/2).as_matrix()
    panda_link8_to_robotiq = np.eye(4)
    panda_link8_to_robotiq[:3, :3] = panda_robotiq
    camera_matrix = np.eye(4)

    camera_matrix[:3, :3] = R.from_quat([0.0121016, -0.00319473, 0.00320956, 0.999917]).as_matrix()
    camera_matrix[:3, 3] = [0.0811728, -0.0183666, 0.0741612]

    camera_position = base_transform @ camera_matrix
    camera_translation = camera_position[:3, 3]
    camera_rotation = R.from_matrix(camera_position[:3, :3]).as_quat()

    # robot.show(cfg)
    return camera_translation, camera_rotation


# print(robot.get_transform([-0.026791146148297532, 0.042747493043678725, 0.10308154650053686, -0.8637255028298855, -0.09199988118113726, 0.891565800648028, 0.0125278986092322], "panda_link8"))
joint_angle = [0.9982257118774165, -1.1112275287930635, -1.7043121799887746, -1.9539140962332093, 0.21012915831244153, 1.7142409334845012, -1.1651942654166507]
print(camera_to_base(joint_angle))

