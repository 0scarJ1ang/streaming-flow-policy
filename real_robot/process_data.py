import numpy as np
import json
import cv2
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os  # 需要引入 os 库来创建文件夹

class DataProcessorStatic:
    def __init__(self):
        # ==========================================
        # 1. 配置参数 (已修正 ID)
        # ==========================================
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters()
        
        # 【修改点 1】 ID 改为 1
        self.target_id = 1       
        
        self.marker_length = 0.03 # 30mm
        
        # 相机内参
        self.camera_matrix = np.array([
            [600.8057861328125, 0.0, 325.5308532714844],
            [0.0, 600.634033203125, 252.90933227539062],
            [0.0, 0.0, 1.0]
        ], dtype=float)
        
        self.dist_coeffs = np.array([
            0.13546444475650787, -0.47212305665016174, 
            8.851876191329211e-05, 0.0010161105310544372, 
            0.43829241394996643
        ], dtype=float)

    def to_seconds(self, timestamp):
        """【修改点 2】自动将纳秒转为秒"""
        # 如果时间戳大于 1e16 (例如 1.7e18)，说明是纳秒
        if np.any(timestamp > 1e16):
            return timestamp / 1e9
        return timestamp

    def find_nearest_timestamp(self, target_time, sorted_times):
        idx = np.searchsorted(sorted_times, target_time, side="left")
        if idx > 0 and (idx == len(sorted_times) or np.abs(target_time - sorted_times[idx-1]) < np.abs(target_time - sorted_times[idx])):
            return sorted_times[idx-1]
        return sorted_times[idx]

    def load_json_data(self, path):
        print(f"正在加载 {path} ...")
        with open(path, 'r') as f:
            data = json.load(f)
        
        # 将 key 转为 float 并进行单位检测
        # 注意：这里先转成 float 列表处理，再重建 map
        keys_float = np.array([float(k) for k in data.keys()])
        keys_seconds = self.to_seconds(keys_float)
        
        # 重建映射: 秒级时间戳 -> 原始字符串 Key
        # (我们必须保留原始 Key 才能回查字典)
        data_map = {}
        original_keys = list(data.keys())
        for i, t_sec in enumerate(keys_seconds):
            data_map[t_sec] = original_keys[i]
            
        sorted_times = np.sort(keys_seconds)
        return data, data_map, sorted_times

    def calculate_static_object_pose(self, images, image_times, cam_pose_dict, cam_times_sorted, cam_key_map):
        print("正在计算静态物体位置 (Static Object Pose)...")
        valid_positions = []
        check_limit = min(len(images), 100) 
        
        print(f"DEBUG: 检查前 {check_limit} 帧 (时间范围: {image_times[0]:.2f} ~ {image_times[-1]:.2f}s)")

        for i in range(check_limit):
            img = images[i]
            t_img = image_times[i]
            
            # --- ArUco 检测 ---
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            
            if ids is None:
                continue
            elif self.target_id not in ids:
                continue 
            
            idx = np.where(ids == self.target_id)[0][0]
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
            
            T_cam_obj = np.eye(4)
            T_cam_obj[:3, :3] = cv2.Rodrigues(rvec[idx])[0]
            T_cam_obj[:3, 3] = tvec[idx].flatten()
            
            # --- 时间匹配 ---
            t_cam = self.find_nearest_timestamp(t_img, cam_times_sorted)
            time_diff = abs(t_img - t_cam)
            
            if time_diff > 0.1: 
                continue
            
            # --- 计算 ---
            if len(valid_positions) == 0:
                print(f"Frame {i}: --- 首次匹配成功！---")
                print(f"  Img: {t_img:.2f}s, Cam: {t_cam:.2f}s, Diff: {time_diff*1000:.1f}ms")

            cam_data_raw = cam_pose_dict[cam_key_map[t_cam]] 
            
            # 【修复点】处理嵌套列表结构 [[x,y,z], [qx,qy,qz,qw]]
            cam_pos = cam_data_raw[0]   # 第一个列表是位置
            cam_quat = cam_data_raw[1]  # 第二个列表是四元数
            
            T_base_cam = np.eye(4)
            T_base_cam[:3, 3] = cam_pos
            T_base_cam[:3, :3] = R.from_quat(cam_quat).as_matrix()
            
            T_base_obj = T_base_cam @ T_cam_obj
            valid_positions.append(T_base_obj[:3, 3])
            
            if len(valid_positions) >= 10:
                break
            
        if not valid_positions:
            print("错误：无法计算物体位置，请检查 ID 或时间同步。")
            return None
            
        avg_pos = np.mean(valid_positions, axis=0)
        print(f"【计算成功】物体位置 (Base Frame): {avg_pos}")
        return avg_pos

    def process(self, image_path, image_time_path, ee_pose_path, gripper_path, cam_pose_path, output_csv):
        print(f"\n=== 开始处理 ===")
        
        # 1. 加载所有数据 & 统一转为秒
        images = np.load(image_path)
        
        image_times_raw = np.load(image_time_path).astype(float)
        image_times = self.to_seconds(image_times_raw)
        
        ee_dict, ee_map, ee_times = self.load_json_data(ee_pose_path)
        cam_dict, cam_map, cam_times = self.load_json_data(cam_pose_path)
        
        # 加载 gripper
        with open(gripper_path, 'r') as f:
            gripper_raw = json.load(f)
        
        # Gripper 处理
        gripper_keys = np.array([float(k) for k in gripper_raw.keys()])
        gripper_keys = self.to_seconds(gripper_keys)
        
        # 转成 (time, val) 列表并按时间排序
        gripper_sorted = sorted(zip(gripper_keys, gripper_raw.values()), key=lambda x: x[0])
        gripper_times_arr = np.array([x[0] for x in gripper_sorted])
        gripper_vals_arr = np.array([x[1] for x in gripper_sorted])

        # ---------------------------------------------------------
        # Step 1: 算静态位置
        # ---------------------------------------------------------
        static_obj_pos = self.calculate_static_object_pose(
            images, image_times, cam_dict, cam_times, cam_map
        )
        
        if static_obj_pos is None:
            return 

        # ---------------------------------------------------------
        # Step 2: 生成对齐数据
        # ---------------------------------------------------------
        print(f"正在生成 CSV (共 {len(ee_times)} 行)...")
        aligned_data = []
        
        for t_robot in ee_times:
            # 机器人数据
            pose_data = ee_dict[ee_map[t_robot]]
            ee_pos = pose_data[0]
            ee_quat = pose_data[1]
            
            # Gripper 匹配
            idx_g = np.searchsorted(gripper_times_arr, t_robot)
            if idx_g == 0:
                g_val = gripper_vals_arr[0]
            elif idx_g == len(gripper_times_arr):
                g_val = gripper_vals_arr[-1]
            else:
                t_prev, t_next = gripper_times_arr[idx_g-1], gripper_times_arr[idx_g]
                g_val = gripper_vals_arr[idx_g-1] if abs(t_robot - t_prev) < abs(t_robot - t_next) else gripper_vals_arr[idx_g]

            row = {
                'timestamp': t_robot,
                'ee_x': ee_pos[0], 'ee_y': ee_pos[1], 'ee_z': ee_pos[2],
                'ee_qx': ee_quat[0], 'ee_qy': ee_quat[1], 'ee_qz': ee_quat[2], 'ee_qw': ee_quat[3],
                'gripper': g_val,
                'obj_x': static_obj_pos[0],
                'obj_y': static_obj_pos[1],
                'obj_z': static_obj_pos[2]
            }
            aligned_data.append(row)

        df = pd.DataFrame(aligned_data)
        df.to_csv(output_csv, index=False)
        print(f"完成！已保存至 {output_csv}")

if __name__ == "__main__":
    processor = DataProcessorStatic()
    
    # 1. 确保输出目录存在，不存在则创建
    output_dir = "processed_data"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已创建输出目录: {output_dir}")

    # 2. 循环处理 Trial 0 到 24
    total_trials = 25
    
    for trial_id in range(total_trials):
        print(f"\n{'='*15} 正在处理 Trial {trial_id} / {total_trials - 1} {'='*15}")
        
        # 构造当前 trial 的文件路径
        # 假设输入文件都在 item_grasping 目录下
        try:
            processor.process(
                image_path=f"item_grasping/images_{trial_id}.npy",
                image_time_path=f"item_grasping/image_time_{trial_id}.npy",
                ee_pose_path=f"item_grasping/ee_pose_{trial_id}.json",
                gripper_path=f"item_grasping/gripper_{trial_id}.json",
                cam_pose_path="item_grasping/cam_pose_all.json",  # 这个文件是所有 trial 共用的
                output_csv=f"{output_dir}/train_data_trial_{trial_id}.csv"
            )
        except FileNotFoundError as e:
            print(f"⚠️ 跳过 Trial {trial_id}: 找不到文件 - {e}")
        except Exception as e:
            print(f"❌ Trial {trial_id} 处理出错: {e}")

    print("\n✅ 所有任务处理完毕！")