import cv2
import numpy as np
import open3d as o3d
import os
import pyzed.sl as sl
from PIL import Image
from ultralytics import YOLO
import threading
import random
import torch
import xlwt
import math
import datetime
import matplotlib.pyplot as plt
import queue
import time

# 左相机的内参
left_fx = 1067.3101
left_fy = 1067.9301
left_cx = 1090.6000
left_cy = 630.5290
left_camera_matrix = np.array([[left_fx, 0, left_cx],
                               [0, left_fy, left_cy],
                               [0, 0, 1]])
# 左相机的畸变系数
left_distortion = np.array([-1.0851, 2.0750, 0.0001, 0, 0.1511])

# 右相机的内参
right_fx = 1066.5100
right_fy = 1066.8300
right_cx = 1090.0500
right_cy = 635.9710
right_camera_matrix = np.array([[right_fx, 0, right_cx],
                                [0, right_fy, right_cy],
                                [0, 0, 1]])
# 右相机的畸变系数
right_distortion = np.array([-1.2534, 2.6035, 0.0002, -0.0003, 0.1022])

R = np.array([[1, 0.0, 0.0],
              [0.0, 1, 0.0],
              [0.0, 0.0, 1]])
T = np.array([120.0, 0.0, 0.0])

# 全局变量和线程同步
exit_flag = False
data_queue = queue.Queue(maxsize=30)  # 限制队列大小，防止内存溢出
visualization_ready = threading.Event()  # 用于通知可视化线程数据已准备好
visualization_lock = threading.Lock()  # 锁，用于安全更新可视化状态

# 添加新的全局变量
smoothed_axes = {}       # 存储平滑后的特征向量
smoothed_quaternions = {} # 存储平滑后的四元数
axis_history = {}        # 存储主轴历史数据
stability_metrics = {}   # 存储稳定性指标
smoothing_alpha = 0.05    # 平滑因子(越小平滑效果越强)
pose_error_history = {}  # 存储姿态误差历史记录  # 新增这行

# 在其他全局变量后面添加
segmentation_times = []  # 存储分割处理时间
pointcloud_times = []    # 存储点云处理时间
max_time_samples = 30    # 最多保存30个样本用于计算平均值

# 存储各目标的姿态信息
pose_data = {}
pose_data_lock = threading.Lock()  # 用于安全更新姿态数据

# 预定义颜色列表 - 更鲜艳易识别的颜色
color_palette = [
    (255, 0, 0),  # 红色
    (0, 255, 0),  # 绿色
    (0, 0, 255),  # 蓝色
    (255, 255, 0),  # 黄色
    (0, 255, 255),  # 青色
    (255, 0, 255),  # 品红色
    (255, 128, 0),  # 橙色
    (128, 0, 255),  # 紫色
]

# 添加到全局变量部分
reference_axes = {}  # 存储不同类型物体的参考轴

# 初始化参考轴 - 瓶子主轴沿Y轴向下（与相机坐标系一致）
reference_axes['bottle'] = np.array([
    [0.0, -1.0, 0.0],  # 主轴 - Y轴向下
    [1.0, 0.0, 0.0],   # 次轴 - X轴向右
    [0.0, 0.0, 1.0]    # 第三轴 - Z轴向前
]).T  # 转置以适应每列是一个轴的格式

# 创建文件夹
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
parent_folder = "data_experiment"
folder_name = os.path.join(parent_folder, f"data_{current_time}")

# 确保文件夹存在
if not os.path.exists(parent_folder):
    os.makedirs(parent_folder)
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

def pca_analysis(points):
    """进行PCA分析并返回特征向量和特征值"""
    # 计算点云中心点
    centroid = np.mean(points, axis=0)

    # 中心化点云
    points_centered = points - centroid

    # 计算协方差矩阵
    cov_matrix = np.cov(points_centered, rowvar=False)

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 特征值降序排列
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors, eigenvalues, centroid


def statistical_outlier_removal(pcd, nb_neighbors=20, std_ratio=2.0):
    """使用统计滤波去除离群点"""
    if len(pcd.points) < nb_neighbors + 1:
        print(f"警告: 点数({len(pcd.points)})小于邻居数({nb_neighbors})，跳过滤波")
        return pcd

    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    return pcd.select_by_index(ind)


def voxel_downsample(pcd, voxel_size=2.0):
    """体素下采样点云数据"""
    return pcd.voxel_down_sample(voxel_size)


def rotation_matrix_to_quaternion(R):
    """将旋转矩阵转换为四元数"""
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    # 返回四元数 [w, x, y, z]
    return np.array([w, x, y, z])


def rotation_matrix_to_axis_angle(R):
    """将旋转矩阵转换为轴角表示"""
    # 计算旋转角度
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在有效范围内
    theta = np.arccos(cos_theta)

    # 避免接近0的情况（没有旋转）
    if np.abs(theta) < 1e-6:
        return np.array([1, 0, 0]), 0.0

    # 计算旋转轴
    u_x = (R[2, 1] - R[1, 2]) / (2 * np.sin(theta))
    u_y = (R[0, 2] - R[2, 0]) / (2 * np.sin(theta))
    u_z = (R[1, 0] - R[0, 1]) / (2 * np.sin(theta))

    axis = np.array([u_x, u_y, u_z])
    axis_norm = np.linalg.norm(axis)

    if axis_norm > 0:
        axis = axis / axis_norm  # 归一化轴向量
    else:
        axis = np.array([1, 0, 0])  # 默认X轴

    return axis, theta


def generate_random_color():
    """生成高亮随机颜色，确保在白色背景上可见"""
    # 确保至少有一个通道值较低，增加与白色背景的对比度
    color = [random.randint(0, 180) for _ in range(3)]
    dark_channel = random.randint(0, 2)
    color[dark_channel] = random.randint(0, 100)  # 确保至少一个通道较暗
    return tuple(color)


def is_valid_point(point):
    """检查点是否有效 (非NaN, 非无穷大, 在合理范围内)"""
    return (not np.any(np.isnan(point)) and
            not np.any(np.isinf(point)) and
            np.all(np.abs(point) < 10000))  # 设置合理范围限制


def draw_pose_info(image, target_idx, class_name, position, pitch, roll, yaw, 
                  error_metrics=None, quaternion=None):
    """在图像上显示目标的姿态和误差信息"""
    # 根据目标类型选择不同颜色
    if class_name.lower() == 'cap':
        color = (0, 200, 255)  # 黄色
    elif class_name.lower() == 'bottle':
        color = (255, 100, 100)  # 蓝色
    else:
        color = color_palette[target_idx % len(color_palette)]
        
    # 为不同目标使用不同背景
    bg_colors = {
        'cap': (50, 50, 0),
        'bottle': (50, 0, 50)
    }
    bg_color = bg_colors.get(class_name.lower(), (0, 0, 0))
    
    h, w = image.shape[:2]

    # 为每个目标在图像下方创建信息区域
    y_pos = h - 220 - (target_idx * 110)  # 增加信息区域高度和间距
    if y_pos < 0:
        y_pos = 10 + (target_idx * 110)

    # 绘制半透明背景
    overlay = image.copy()
    cv2.rectangle(overlay, (10, y_pos), (510, y_pos + 130), bg_color, -1)
    cv2.addWeighted(overlay, 1, image, 0, 0, image)

    # 绘制标题栏
    cv2.rectangle(image, (10, y_pos), (510, y_pos + 30), color, -1)
    cv2.putText(image, f"Target {target_idx+1}: {class_name.upper()}",
                (15, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    # 绘制其他信息
    cv2.putText(image, f"Position (cm): ({position[0]:.4f}, {position[1]:.4f}, {position[2]+1.6:.4f})",
                (15, y_pos + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
    """cv2.putText(image, f"欧拉角(degree): P={pitch:.4f} R={roll:.4f} Y={yaw:.4f}",
                (15, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)"""
    
    # 添加轴误差信息
    if error_metrics and 'primary_axis_error' in error_metrics:
        error_color = (0, 255, 255)  # 黄色显示误差信息
        
        cv2.putText(image, 
                   f"Axis error (degree): X={error_metrics['primary_axis_error']:.4f} Y={error_metrics['secondary_plane_error']:.4f}",
                   (15, y_pos + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.7, error_color, 1)
    
    # 添加误差圆锥体信息
    if 'error_cone_angle' in pose_data.get(target_idx, {}):
        cone_angle = pose_data[target_idx]['error_cone_angle']
        cv2.putText(image, 
                   f"误差圆锥开角: {cone_angle:.4f}° (95%置信区间)",
                   (15, y_pos + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 1)

    return image


# 实例分割线程函数
def segmentation_thread():
    global exit_flag, pose_data, smoothed_axes, smoothed_quaternions, axis_history, stability_metrics

    # 创建文件夹
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_folder = "data_experiment"
    folder_name = os.path.join(parent_folder, f"data_{current_time}")
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # 加载模型
    model = YOLO(r"C:\Users\dyz\Desktop\04-stereo matching\best.pt")
    if torch.cuda.is_available():
        model.to('cuda')  # 使用GPU加速推理

    # 初始化ZED相机
    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD2K
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.MILLIMETER
    init.depth_minimum_distance = 200  # 设置最小深度

    # 打开相机
    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"相机打开失败: {repr(err)}")
        exit_flag = True
        return

    # 设置运行参数
    runtime = sl.RuntimeParameters()
    runtime.confidence_threshold = 50  # 设置置信度阈值
    runtime.texture_confidence_threshold = 100  # 纹理置信度阈值

    # 准备图像尺寸
    cam_info = zed.get_camera_information()
    image_size = cam_info.camera_configuration.resolution

    # 声明矩阵
    image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    image_zed_R = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    dep = sl.Mat()
    point_cloud = sl.Mat()

    # 创建显示窗口
    cv2.namedWindow("Segmentation Results", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Segmentation Results", 1280, 720)

    print("分割线程已启动...")

    try:
        frame_count = 0
        while not exit_flag:
            frame_start_time = time.time()  # 添加在while循环开始处
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                frame_count += 1

                # 检索左右图像
                zed.retrieve_image(image_zed, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
                zed.retrieve_image(image_zed_R, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
                zed.retrieve_image(depth_image_zed, sl.VIEW.DEPTH, sl.MEM.CPU, image_size)
                zed.retrieve_image(dep, sl.VIEW.DEPTH)

                # 检索点云数据
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, image_size)

                # 转换图像格式
                image_ocv = image_zed.get_data()
                image_ocv_R = image_zed_R.get_data()
                dep_map = dep.get_data()

                # 创建视图
                view = np.concatenate((cv2.resize(image_ocv, (640, 360)),
                                       cv2.resize(image_ocv_R, (640, 360)),
                                       cv2.resize(dep_map, (640, 360))), axis=1)
                cv2.imshow("View", view)

                # 转换为RGB格式用于YOLO处理
                frame = cv2.cvtColor(image_ocv, cv2.COLOR_RGBA2RGB)
                image_org = frame.copy()

                # 创建分割可视化图像
                segmented_image = image_org.copy()

                # 每3帧执行一次YOLO检测，提高性能
                if frame_count % 1 == 0:
                    # YOLO实时推理前计时
                    yolo_start_time = time.time()
                    # YOLO实时推理
                    results = model.predict(frame,
                                            stream=True,
                                            imgsz=640,
                                            conf=0.5,
                                            device='cuda:0' if torch.cuda.is_available() else 'cpu')
                    # YOLO推理后计时
                    yolo_end_time = time.time()
                    yolo_time = yolo_end_time - yolo_start_time

                    # 处理分割结果
                    for r in results:
                        if not r.masks:
                            continue

                        # 获取每个mask的所有点
                        for idx, mask in enumerate(r.masks):
                            if mask.data.dim() == 3 and (mask.data.size(0) == 1 or mask.data.size(2) == 1):
                                maskdata = mask.data.squeeze()

                            # 缩放掩码到目标图像的尺寸
                            resized_mask = cv2.resize(maskdata.cpu().numpy(), (mask.orig_shape[1], mask.orig_shape[0]),
                                                      interpolation=cv2.INTER_NEAREST)

                            # 获取掩码中非零点的坐标
                            y_indices, x_indices = np.where(resized_mask > 0)

                            # 选择颜色
                            color = color_palette[idx % len(color_palette)] if idx < len(
                                color_palette) else generate_random_color()

                            # 在分割图像上应用掩码颜色
                            # 使用OpenCV的addWeighted函数更高效地应用掩码
                            mask = np.zeros_like(segmented_image)
                            for y, x in zip(y_indices, x_indices):
                                if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1]:
                                    mask[y, x] = color

                            # 创建半透明效果
                            cv2.addWeighted(segmented_image, 1, mask, 0, 0, segmented_image)

                            # 创建彩色掩码
                            color_mask = np.zeros_like(segmented_image, dtype=np.uint8)
                            color_mask[:] = color  # RGB颜色

                            alpha = 0.3  # 透明度
                            overlay = segmented_image.copy()

                            # 在遮罩区域应用颜色
                            for y, x in zip(y_indices, x_indices):
                                if 0 <= y < overlay.shape[0] and 0 <= x < overlay.shape[1]:
                                    overlay[y, x] = color

                            # 使用整体图像混合
                            segmented_image = cv2.addWeighted(segmented_image, 1 - alpha, overlay, alpha, 0)

                            # 收集点云数据
                            valid_points = []
                            valid_colors = []

                            # 按比例采样点（避免点云过大）
                            # 根据点数动态调整采样率
                            total_points = len(y_indices)
                            target_point_count = 20000  # 从2000增加到100000，大幅提高点云密度

                            # 根据总点数调整采样率
                            if total_points > target_point_count:
                                # 如果总点数太多，设置采样率以获取目标点数
                                sample_rate = max(1, int(total_points / target_point_count))
                                # 计算采样索引
                                sampled_indices = np.arange(0, total_points, sample_rate)
                            else:
                                # 如果总点数少于目标点数，使用所有点
                                sampled_indices = np.arange(0, total_points)

                            print(
                                f"掩码总点数: {total_points}, 采样率: {1 if total_points <= target_point_count else sample_rate}, 采样点数: {len(sampled_indices)}")

                            for i in sampled_indices:
                                y, x = y_indices[i], x_indices[i]

                                # 获取原始像素颜色
                                original_color = frame[y, x].copy() if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1] else np.array([100, 100, 100])

                                # 获取点云值 - 保持ZED相机的右手坐标系
                                # ZED: X→右, Y→下, Z→前
                                err, point_cloud_value = point_cloud.get_value(int(x), int(y))
                                if err == sl.ERROR_CODE.SUCCESS and is_valid_point(point_cloud_value[:3]):
                                    # 转换为厘米单位，保持坐标系不变
                                    point = np.array([point_cloud_value[0],
                                                      point_cloud_value[1],
                                                      point_cloud_value[2]]) / 10.0

                                    # 过滤合理深度范围内的点
                                    if 10 < point[2] < 100:  # 10-500cm范围内
                                        valid_points.append(point)
                                        # 使用物体的原始颜色而不是单一颜色
                                        valid_colors.append(original_color / 255.0)

                            # 为每个掩码添加边界框和标签
                            class_name = "Unknown"
                            if r.boxes and idx < len(r.boxes):
                                bbox = r.boxes.xyxy[idx].cpu().numpy()
                                cv2.rectangle(segmented_image,
                                              (int(bbox[0]), int(bbox[1])),
                                              (int(bbox[2]), int(bbox[3])),
                                              color, 2)
                                try:
                                    class_name = r.names[int(r.boxes.cls[idx])]
                                    conf = r.boxes.conf[idx]
                                    cv2.putText(segmented_image,
                                                f"{class_name} {conf:.2f}",
                                                (int(bbox[0]), int(bbox[1] - 10)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                except Exception as e:
                                    print(f"标签绘制错误: {str(e)}")

                            # 检查是否有足够的有效点
                            if len(valid_points) > 50:  # 增加最小点数要求
                                print(f"目标 {idx} ({class_name}): 收集了 {len(valid_points)} 个有效点")

                                # 将点云数据转换为NumPy数组
                                valid_points_array = np.array(valid_points)
                                valid_colors_array = np.array(valid_colors)

                                # 检查数据完整性
                                if len(valid_points_array) == len(valid_colors_array) and len(valid_points_array) > 0:
                                    # 将数据打包传递给点云处理线程
                                    data_package = {
                                        'target_idx': idx,
                                        'valid_points': valid_points_array,
                                        'valid_colors': valid_colors_array,
                                        'class_name': class_name,
                                        'timestamp': time.time(),
                                        'color': color
                                    }

                                    # 将数据添加到队列
                                    try:
                                        # 非阻塞添加到队列
                                        if not data_queue.full():
                                            data_queue.put(data_package, block=False)
                                            # 通知可视化线程数据已准备好
                                            visualization_ready.set()
                                    except queue.Full:
                                        # 队列已满，跳过此帧数据
                                        pass
                                else:
                                    print(f"目标 {idx}: 点云数据不完整，跳过处理")
                            else:
                                print(f"目标 {idx}: 有效点太少，只有 {len(valid_points)} 个")

                # 从姿态数据字典获取最新姿态信息并添加到图像上
                with pose_data_lock:
                    # 复制当前姿态数据以避免竞态条件
                    current_pose_data = pose_data.copy()

                # 在分割图像上绘制所有目标的姿态信息
                for target_idx, pose_info in current_pose_data.items():
                    if time.time() - pose_info['timestamp'] < 1.0:  # 只显示最近2秒内的姿态信息
                        # 提取轴误差指标
                        error_metrics = None
                        if target_idx in pose_data:
                            error_metrics = {
                                k: v for k, v in pose_data[target_idx].items()
                                if k in ['primary_axis_error', 'secondary_plane_error', 'coordinate_system_error']
                            }

                        segmented_image = draw_pose_info(
                            segmented_image,
                            target_idx,
                            pose_info['class_name'],
                            pose_info['position'],
                            pose_info['pitch_deg'],
                            pose_info['roll_deg'],
                            pose_info['yaw_deg'],
                            error_metrics,
                            pose_info['quaternion']
                        )
                # 在绘制完所有目标信息后，显示图像前添加FPS计算代码
                frame_end_time = time.time()
                frame_process_time = frame_end_time - frame_start_time

                # 添加到时间列表并控制列表大小
                segmentation_times.append(frame_process_time)
                if len(segmentation_times) > max_time_samples:
                    segmentation_times.pop(0)

                # 计算平均FPS并在图像上显示
                if len(segmentation_times) > 0:
                    avg_time = sum(segmentation_times) / len(segmentation_times)
                    avg_fps = 1.0 / avg_time if avg_time > 0 else 0

                    # 在分割图像上显示FPS信息
                    fps_text = f"FPS: {avg_fps:.4f}"
                    cv2.putText(segmented_image, fps_text, (15, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                    # 每15帧打印一次FPS信息
                    if frame_count % 60 == 0:
                        print(f"分割处理平均FPS: {avg_fps:.2f}, 平均每帧时间: {avg_time * 1000:.2f}ms")
                # 显示分割结果
                cv2.imshow('Segmentation Results', segmented_image)

                # 处理键盘输入
                key = cv2.waitKey(1)
                if key & 0xFF == 27:  # ESC键退出
                    exit_flag = True
                    break

                # 添加截图功能
                elif key & 0xFF == ord('s'):  # 按's'键保存当前帧
                    save_path = os.path.join(folder_name, f"frame_{time.time():.0f}.jpg")
                    cv2.imwrite(save_path, segmented_image)
                    print(f"保存图像到：{save_path}")

                elif key & 0xFF == ord('r'):  # 按'r'键将当前姿态设为参考
                    # 获取当前选定的目标（如果有多个目标，可以添加交互选择）
                    # 这里简单使用第一个检测到的目标
                    current_targets = list(pose_data.keys())
                    if current_targets:
                        target_idx = current_targets[0]
                        class_name = pose_data[target_idx]['class_name']

                        # 设置为参考
                        if set_current_as_reference(target_idx, class_name):
                            # 显示确认信息
                            cv2.putText(segmented_image,
                                        f"目标{target_idx}已设为参考!",
                                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0, 255, 0), 2)

    finally:
        # 释放资源
        zed.close()
        cv2.destroyAllWindows()
        print("分割线程已结束")


# 点云处理线程函数 - 修复姿态信息输出
def pointcloud_thread():
    global exit_flag, pose_data, smoothed_axes, smoothed_quaternions, axis_history, stability_metrics, smoothing_alpha, pose_error_history

    print("点云处理线程已启动...")

    # 创建用于显示的Open3D可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window("Real-time Point Cloud Visualization", width=1280, height=720)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.point_size = 5.0  # 增大点的大小以提高可见性
    render_option.background_color = np.array([1.0, 1.0, 1.0])  # 白色背景
    render_option.light_on = True

    # 兼容不同版本的Open3D
    try:
        render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.FLAT
    except (AttributeError, TypeError):
        print("警告: mesh_shade_option设置失败，可能是Open3D版本不兼容")

    # 禁用法线显示
    render_option.point_show_normal = False

    # 设置相机视角
    view_control = vis.get_view_control()

    # 添加全局坐标系 - 减小尺寸
    global_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=15)
    vis.add_geometry(global_frame)

    # 调整视角
    view_control.set_front([0, 0, 1])  # 相机朝向Z轴负方向
    view_control.set_up([0, -1, 0])  # 上方向是Y轴负方向
    view_control.set_zoom(1.0)

    first_view = True

    # 初始化点云字典
    point_clouds = {}

    # 时间控制
    last_update_time = time.time()
    last_cleanup_time = time.time()
    update_interval = 0.1
    cleanup_interval = 1.0
    
    # 添加frame_count变量
    frame_count = 0

    try:
        while not exit_flag:
            # 等待新数据通知，使用超时确保检查exit_flag
            if visualization_ready.wait(timeout=0.05):
                visualization_ready.clear()
                
                # 增加帧计数
                frame_count += 1

                current_time = time.time()
                current_frame_targets = set()
                need_update = False

                # 第一步：从队列中获取所有目标数据
                target_data_list = []
                while not data_queue.empty() and not exit_flag:
                    try:
                        data_package = data_queue.get(block=False)
                        target_idx = data_package['target_idx']
                        current_frame_targets.add(target_idx)
                        target_data_list.append(data_package)
                    except queue.Empty:
                        break

                # 只有当有新数据时才处理
                if target_data_list:
                    # 第二步：清理不活跃目标
                    if current_time - last_cleanup_time > cleanup_interval:
                        inactive_targets = set(point_clouds.keys()) - current_frame_targets
                        if inactive_targets:
                            print(f"清理 {len(inactive_targets)} 个不活跃目标: {inactive_targets}")

                            for target_idx in inactive_targets:
                                if target_idx in point_clouds:
                                    # 锁定UI进行几何体删除
                                    with visualization_lock:
                                        for geom in point_clouds[target_idx]:
                                            vis.remove_geometry(geom, reset_bounding_box=False)
                                    del point_clouds[target_idx]

                                    # 删除姿态数据
                                    with pose_data_lock:
                                        if target_idx in pose_data:
                                            del pose_data[target_idx]

                            need_update = True

                        last_cleanup_time = current_time
                        latest_frame_targets = current_frame_targets.copy()

                    # 第三步：处理每个目标的数据
                    for data_package in target_data_list:
                        process_start_time = time.time()
                        target_idx = data_package['target_idx']
                        valid_points = data_package['valid_points']
                        valid_colors = data_package['valid_colors']
                        class_name = data_package['class_name']
                        timestamp = data_package['timestamp']
                        color = data_package['color']

                        print(f"处理目标 {target_idx} ({class_name}): {len(valid_points)} 个点")
                        print(
                            f"点云数据范围(厘米): X[{np.min(valid_points[:, 0]):.4f}, {np.max(valid_points[:, 0]):.4f}], "
                            f"Y[{np.min(valid_points[:, 1]):.4f}, {np.max(valid_points[:, 1]):.4f}], "
                            f"Z[{np.min(valid_points[:, 2]):.4f}, {np.max(valid_points[:, 2]):.4f}]")

                        # 创建点云对象
                        target_pcd = o3d.geometry.PointCloud()
                        target_pcd.points = o3d.utility.Vector3dVector(valid_points)

                        # 使用实际的像素颜色
                        target_pcd.colors = o3d.utility.Vector3dVector(valid_colors)

                        # 点云处理
                        try:
                            # 体素下采样
                            voxel_size = 0.02
                            if len(target_pcd.points) > 1000:
                                downsampled_pcd = voxel_downsample(target_pcd, voxel_size)
                                if len(downsampled_pcd.points) > 50:
                                    target_pcd = downsampled_pcd
                                    print(f"下采样后点数: {len(target_pcd.points)}")

                            # 离群点滤波
                            if len(target_pcd.points) > 20:
                                cl, ind = target_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
                                if len(ind) > 10:  # 确保滤波后有足够的点
                                    target_pcd = target_pcd.select_by_index(ind)
                                    points_array = np.asarray(target_pcd.points)
                                    print(f"离群点滤波后点数: {len(points_array)}")

                            # PCA分析
                            points_array = np.asarray(target_pcd.points)
                            eigenvectors, eigenvalues, centroid = pca_analysis(points_array)

                            # --- 物体特定的PCA轴对齐逻辑 ---
                            # 基于物体类型进行特定的PCA轴映射

                            # 相机坐标系的标准轴
                            camera_x = np.array([1, 0, 0])  # 右方向
                            camera_y = np.array([0, 1, 0])  # 上方向
                            camera_z = np.array([0, 0, 1])  # 前方向

                            # 创建一个新的轴向量集，初始复制PCA轴
                            # 根据特征值大小排序的原始PCA轴
                            sorted_idx = np.argsort(eigenvalues)[::-1]  # 降序排列
                            primary_idx = sorted_idx[0]    # 最大特征值对应的轴
                            secondary_idx = sorted_idx[1]  # 第二大特征值对应的轴
                            tertiary_idx = sorted_idx[2]   # 最小特征值对应的轴

                            # 按特征值大小排序的轴
                            primary_axis = eigenvectors[:, primary_idx]
                            secondary_axis = eigenvectors[:, secondary_idx]
                            tertiary_axis = eigenvectors[:, tertiary_idx]

                            # 创建新的排列后的轴矩阵
                            aligned_axes = np.zeros_like(eigenvectors)

                            # 根据物体类型应用不同的映射规则
                            if class_name.lower() == 'cap':
                                print("应用cap的轴映射规则: 主轴→X, 次轴→Y, 第三轴→Z")
                                # 对于cap: 主轴→X, 次轴→Y, 第三轴→Z
                                aligned_axes[:, 0] = primary_axis    # X方向
                                aligned_axes[:, 1] = secondary_axis  # Y方向
                                aligned_axes[:, 2] = tertiary_axis   # Z方向
                                
                            elif class_name.lower() == 'bottle':
                                print("应用bottle的轴映射规则: 次轴→X, 主轴→Y, 第三轴→Z")
                                # 对于bottle: 次轴→X, 主轴→Y, 第三轴→Z [按照您的描述修正为正确映射]
                                aligned_axes[:, 0] = secondary_axis  # X方向
                                aligned_axes[:, 1] = primary_axis    # Y方向
                                aligned_axes[:, 2] = tertiary_axis   # Z方向
                                
                            else:
                                # 对于其他物体，保持PCA原始顺序
                                print(f"应用默认轴映射规则: 主轴→X, 次轴→Y, 第三轴→Z")
                                aligned_axes[:, 0] = primary_axis
                                aligned_axes[:, 1] = secondary_axis
                                aligned_axes[:, 2] = tertiary_axis

                            # 确保每个轴与相机坐标系对应轴同向（点积为正）
                            print("检查轴方向与相机坐标系一致性:")
                            for i in range(3):
                                camera_axis = [camera_x, camera_y, camera_z][i]
                                if np.dot(aligned_axes[:, i], camera_axis) < 0:
                                    aligned_axes[:, i] = -aligned_axes[:, i]
                                    print(f"  轴 {i} 已翻转以与相机坐标系{['X', 'Y', 'Z'][i]}轴同向")

                            # 确保符合右手坐标系规则
                            cross_product = np.cross(aligned_axes[:, 0], aligned_axes[:, 1])
                            dot_with_z = np.dot(cross_product, aligned_axes[:, 2])
                            if dot_with_z < 0:
                                print("警告：坐标系不满足右手规则，调整第三轴...")
                                aligned_axes[:, 2] = -aligned_axes[:, 2]

                            # 最终规范化所有轴向量
                            for i in range(3):
                                aligned_axes[:, i] = aligned_axes[:, i] / np.linalg.norm(aligned_axes[:, i])

                            # 更新特征向量为对齐后的轴
                            eigenvectors = aligned_axes

                            # 输出特征值比例
                            eigenvalue_ratios = eigenvalues / np.sum(eigenvalues)
                            print(f"特征值比例: {eigenvalue_ratios[0]:.3f}, {eigenvalue_ratios[1]:.3f}, {eigenvalue_ratios[2]:.3f}")
                            print("PCA轴已根据物体类型映射到相机坐标系XYZ方向，并确保符合右手规则")

                            # 轴对称物体的警告信息
                            is_axisymmetric = np.abs(eigenvalue_ratios[1] - eigenvalue_ratios[2]) < 0.1
                            if is_axisymmetric:
                                print(f"警告: 检测到可能的轴对称物体 ({class_name})，第二和第三特征值相近")

                            # 轴对称物体的处理

                            if class_name.lower() in ['bottle', 'cap']:
                                print(f"Processing axisymmetric object ({class_name}). Stabilizing secondary/tertiary axes.")

                                primary_axis = eigenvectors[:, 0] # Use the ALIGNED primary axis

                                # Use Camera X-axis as the reference vector for stabilization (often orthogonal to upright objects)
                                reference_vector = np.array([1, 0, 0]) # Camera X-axis

                                # Check if primary axis is parallel to the reference vector
                                dot_product = np.abs(np.dot(primary_axis, reference_vector))
                                if dot_product > 0.95:
                                    print("Primary axis nearly parallel to Camera X, using Camera Z as fallback reference.")
                                    reference_vector = np.array([0, 0, 1]) # Camera Z-axis as fallback

                                # Construct stable secondary axis
                                secondary_axis = np.cross(reference_vector, primary_axis)
                                norm = np.linalg.norm(secondary_axis)
                                if norm < 1e-6:
                                    # Handle rare case of primary_axis aligning with BOTH X and Z (e.g., axis is [0,1,0])
                                    print("Primary axis aligned with fallback reference, using arbitrary orthogonal vector.")
                                    temp_vec = np.array([0, 1, 0]) # Use Y if primary isn't Y
                                    secondary_axis = np.cross(primary_axis, temp_vec)
                                    norm = np.linalg.norm(secondary_axis)
                                    if norm < 1e-6: # If primary *is* Y, use X
                                         secondary_axis = np.cross(primary_axis, np.array([-1, 0, 0]))
                                         norm = np.linalg.norm(secondary_axis)

                                secondary_axis /= norm

                                # Construct tertiary axis using cross product (ensures right-handed system)
                                tertiary_axis = np.cross(primary_axis, secondary_axis)
                                # No need to normalize tertiary_axis if primary and secondary are normalized unit vectors

                                # Update the eigenvectors with stabilized axes
                                eigenvectors[:, 1] = secondary_axis
                                eigenvectors[:, 2] = tertiary_axis

                                print("Axisymmetric axes stabilized using Camera X/Z reference.")


                            # ----------- 时间轴平滑 -----------
                            # 初始化并应用时间平滑
                            if target_idx not in smoothed_axes:
                                smoothed_axes[target_idx] = eigenvectors.copy()
                            else:
                                # 处理主轴方向翻转 (确保方向一致性)
                                if np.dot(eigenvectors[:, 0], smoothed_axes[target_idx][:, 0]) < 0:
                                    eigenvectors[:, 0] = -eigenvectors[:, 0]
                                    # 同时翻转次轴以保持右手坐标系
                                    eigenvectors[:, 2] = -eigenvectors[:, 2]

                                # 应用指数平滑滤波
                                for i in range(3):
                                    smoothed_axes[target_idx][:, i] = (1 - smoothing_alpha) * smoothed_axes[target_idx][
                                                                                              :, i] + \
                                                                      smoothing_alpha * eigenvectors[:, i]
                                    # 重新归一化确保单位向量
                                    smoothed_axes[target_idx][:, i] = smoothed_axes[target_idx][:, i] / \
                                                                      np.linalg.norm(smoothed_axes[target_idx][:, i])

                                # 确保三个轴互相垂直 (使用Gram-Schmidt正交化)
                                v1 = smoothed_axes[target_idx][:, 0]  # 保持主轴不变
                                u2 = smoothed_axes[target_idx][:, 1]
                                v2 = u2 - np.dot(u2, v1) * v1  # 使v2垂直于v1
                                v2 = v2 / np.linalg.norm(v2)
                                v3 = np.cross(v1, v2)  # 确保v3垂直于v1和v2

                                # 更新正交化后的轴
                                smoothed_axes[target_idx][:, 1] = v2
                                smoothed_axes[target_idx][:, 2] = v3

                                # 使用平滑后的轴替换原始轴
                                eigenvectors = smoothed_axes[target_idx].copy()

                                print("应用时间平滑后的轴方向")

                            # 计算并显示姿态稳定性指标
                            if target_idx not in axis_history:
                                axis_history[target_idx] = []
                                stability_metrics[target_idx] = {"angle_variance": 0.0, "position_variance": 0.0}

                            # 记录主轴
                            axis_history[target_idx].append(eigenvectors[:, 0])
                            # 最多保留50帧历史
                            if len(axis_history[target_idx]) > 10:
                                axis_history[target_idx].pop(0)

                            # 计算主轴稳定性 (连续帧之间的角度变化)
                            if len(axis_history[target_idx]) >= 2:
                                angle_changes = []
                                for i in range(1, len(axis_history[target_idx])):
                                    dot = np.clip(
                                        np.abs(np.dot(axis_history[target_idx][i], axis_history[target_idx][i - 1])), 0,
                                        1)
                                    angle = np.arccos(dot) * 180 / np.pi  # 角度变化（度）
                                    angle_changes.append(angle)

                                # 计算角度变化的方差作为稳定性指标
                                angle_variance = np.var(angle_changes) if angle_changes else 0
                                stability_metrics[target_idx]["angle_variance"] = angle_variance

                                # 仅在需要时输出稳定性指标
                                if frame_count % 1 == 0:  # 每30帧输出一次
                                    print(f"目标 {target_idx} ({class_name}) 姿态稳定性:")
                                    print(f"  主轴角度变化方差: {angle_variance:.4f}° (越小越稳定)")

                            # 构建旋转矩阵
                            R = eigenvectors

                            # 输出旋转矩阵
                            print(f"旋转矩阵R:\n{R}")

                            # 计算四元数
                            quaternion = rotation_matrix_to_quaternion(R)
                            # 应用四元数平滑
                            if target_idx not in smoothed_quaternions:
                                smoothed_quaternions[target_idx] = quaternion.copy()
                            else:
                                # 确保四元数符号一致性（避免突然翻转）
                                if np.dot(quaternion, smoothed_quaternions[target_idx]) < 0:
                                    quaternion = -quaternion

                                # 应用平滑
                                quat_smooth_factor = 0.2
                                smoothed_quat = (1 - quat_smooth_factor) * smoothed_quaternions[
                                    target_idx] + quat_smooth_factor * quaternion
                                smoothed_quat = smoothed_quat / np.linalg.norm(smoothed_quat)

                                # 更新平滑四元数
                                smoothed_quaternions[target_idx] = smoothed_quat
                                quaternion = smoothed_quat
                            print(f"四元数: {quaternion}")

                            # 计算欧拉角
                            if abs(R[2, 0]) < 0.99999:  # 非奇异情况
                                pitch = -np.arcsin(R[2, 0])
                                roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
                                yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
                            else:  # 奇异情况 (Gimbal lock)
                                yaw = 0
                                if R[2, 0] < 0:  # 俯仰角为+90度
                                    pitch = np.pi / 2
                                    roll = yaw + np.arctan2(R[0, 1], R[0, 2])
                                else:  # 俯仰角为-90度
                                    pitch = -np.pi / 2
                                    roll = -yaw + np.arctan2(-R[0, 1], -R[0, 2])

                            # 转为角度
                            pitch_deg = np.degrees(pitch)
                            roll_deg = np.degrees(roll)
                            yaw_deg = np.degrees(yaw)

                            # 输出姿态信息
                            print(f"\n目标 {target_idx} ({class_name}) 姿态信息:")
                            print(f"位置(厘米): ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]+1.8:.4f})")
                            print(f"欧拉角(度): Pitch={pitch_deg:.4f}°, Roll={roll_deg:.4f}°, Yaw={yaw_deg:.4f}°")
                            print(
                                f"四元数: w={quaternion[0]:.4f}, x={quaternion[1]:.4f}, y={quaternion[2]:.4f}, z={quaternion[3]:.4f}")

                            # 更新姿态数据
                            with pose_data_lock:
                                pose_data[target_idx] = {
                                    'target_idx': target_idx,
                                    'class_name': class_name,
                                    'position': centroid,
                                    'pitch_deg': pitch_deg,
                                    'roll_deg': roll_deg,
                                    'yaw_deg': yaw_deg,
                                    'quaternion': quaternion,
                                    'rotation_matrix': R,
                                    'timestamp': timestamp
                                }

                            # 准备几何体集合 - 不立即添加到可视化器
                            geometries = []

                            # 点云
                            geometries.append(target_pcd)

                            # 计算centroid_sphere的颜色
                            color_normalized = np.array(color) / 255.0  # 使用传入的颜色

                            # 质心球体
                            centroid_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
                            centroid_sphere.translate(centroid)
                            centroid_sphere.paint_uniform_color(color_normalized * 0.7)
                            geometries.append(centroid_sphere)

                            # 目标坐标系
                            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
                            transform = np.eye(4)
                            transform[:3, :3] = R
                            transform[:3, 3] = centroid
                            coord_frame.transform(transform)
                            geometries.append(coord_frame)

                            # 修改添加主成分轴的部分
                            # 添加主成分轴
                            axis_scale = 7.5  # 基础缩放因子
                            axis_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # RGB对应XYZ

                            # 基于轴对称性调整可视化
                            if is_axisymmetric:
                                # 主轴使用正常显示，次轴用较浅颜色表示不确定性
                                for i in range(3):
                                    scale = axis_scale * (0.7 if i > 0 else 1.0)  # 主轴保持原长度，次轴稍短
                                    start = centroid
                                    end = centroid + eigenvectors[:, i] * scale

                                    line = o3d.geometry.LineSet()
                                    line.points = o3d.utility.Vector3dVector([start, end])
                                    line.lines = o3d.utility.Vector2iVector([[0, 1]])

                                    # 为次轴添加视觉区分
                                    if i > 0:  # 次轴
                                        # 次轴使用浅色表示不确定性
                                        lighter_color = np.array(axis_colors[i]) * 0.7 + 0.3  # 颜色变浅
                                        line.colors = o3d.utility.Vector3dVector([lighter_color])
                                    else:  # 主轴
                                        line.colors = o3d.utility.Vector3dVector([axis_colors[i]])

                                    geometries.append(line)

                                    # 添加轴端点球体，更明显地标记轴的终点
                                    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
                                    end_sphere.translate(end)
                                    end_sphere.paint_uniform_color(axis_colors[i] if i == 0 else lighter_color)
                                    geometries.append(end_sphere)
                            else:
                                # 非轴对称物体使用标准绘制方法
                                for i in range(3):
                                    scale = axis_scale * math.sqrt(eigenvalues[i] / eigenvalues[0])
                                    start = centroid
                                    end = centroid + eigenvectors[:, i] * scale

                                    line = o3d.geometry.LineSet()
                                    line.points = o3d.utility.Vector3dVector([start, end])
                                    line.lines = o3d.utility.Vector2iVector([[0, 1]])
                                    line.colors = o3d.utility.Vector3dVector([axis_colors[i]])
                                    geometries.append(line)

                            # 锁定UI进行几何体更新
                            with visualization_lock:
                                # 移除旧几何体
                                if target_idx in point_clouds:
                                    for geom in point_clouds[target_idx]:
                                        vis.remove_geometry(geom, reset_bounding_box=False)
                                if target_idx in smoothed_axes:
                                    del smoothed_axes[target_idx]
                                if target_idx in smoothed_quaternions:
                                    del smoothed_quaternions[target_idx]
                                if target_idx in axis_history:
                                    del axis_history[target_idx]
                                if target_idx in stability_metrics:
                                    del stability_metrics[target_idx]

                                # 保存新几何体集合
                                point_clouds[target_idx] = geometries

                                # 添加新几何体
                                for geom in geometries:
                                    vis.add_geometry(geom, reset_bounding_box=False)

                                    if first_view and len(points_array) > 0:
                                        view_control = vis.get_view_control()
                                        view_control.set_lookat(centroid)
                                        view_control.set_front([0, 0, 1])
                                        view_control.set_up([0, -1, 0])
                                        bbox = target_pcd.get_axis_aligned_bounding_box()
                                        extent = bbox.get_extent()
                                        diameter = np.linalg.norm(extent)
                                        view_control.set_zoom(0.7 * min(1.0, 500.0 / (diameter + 1e-6)))
                                        first_view = False

                            need_update = True

                            # 构建估计的坐标轴矩阵 - 使用修正后的特征向量
                            estimated_axes = eigenvectors  # 已经是(3,3)矩阵，每列是一个轴

                            # 获取对应的参考轴
                            object_type = class_name.lower()
                            if object_type in reference_axes:
                                ref_axes = reference_axes[object_type]

                                # 计算轴角度误差，传入类别名称
                                axes_errors = axes_angle_error(estimated_axes, ref_axes, class_name)

                                # 打印误差结果
                                print(f"\n=== 轴角度误差评估 (目标 {target_idx}: {class_name}) ===")
                                print(f"主轴角度误差: {axes_errors['primary_axis_error']:.2f}°")
                                print(f"次轴平面角度误差: {axes_errors['secondary_plane_error']:.2f}°")
                                print(f"整体坐标系误差: {axes_errors['coordinate_system_error']:.2f}°")

                                # 在姿态数据中添加误差信息
                                with pose_data_lock:
                                    if target_idx in pose_data:
                                        pose_data[target_idx].update({
                                            'primary_axis_error': axes_errors['primary_axis_error'],
                                            'secondary_plane_error': axes_errors['secondary_plane_error'],
                                            'coordinate_system_error': axes_errors['coordinate_system_error']
                                        })

                                # 更新误差历史记录
                                if target_idx not in pose_error_history:
                                    pose_error_history[target_idx] = {
                                        'timestamps': [],
                                        'primary_axis_errors': [],
                                        'secondary_plane_errors': [],
                                        'coordinate_system_errors': []
                                    }

                                pose_error_history[target_idx]['timestamps'].append(time.time())
                                pose_error_history[target_idx]['primary_axis_errors'].append(axes_errors['primary_axis_error'])
                                pose_error_history[target_idx]['secondary_plane_errors'].append(axes_errors['secondary_plane_error'])
                                pose_error_history[target_idx]['coordinate_system_errors'].append(axes_errors['coordinate_system_error'])

                            # 如果目标类型有参考轴，显示参考坐标系
                            if object_type in reference_axes:
                                # 创建参考坐标系
                                ref_coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=6.0)

                                # 构建变换矩阵，将参考轴放在与估计轴相同的中心点
                                ref_transform = np.eye(4)
                                ref_transform[:3, :3] = reference_axes[object_type]
                                ref_transform[:3, 3] = centroid

                                # 应用变换
                                ref_coord_frame.transform(ref_transform)

                                # 调整参考坐标系外观 - 使用虚线或不同颜色区分
                                # 由于Open3D没有直接的虚线选项，我们使用半透明效果区分
                                # 添加到几何体集合
                                ref_coord_frame.paint_uniform_color([0.7, 0.7, 0.7])  # 使用灰白色表示参考轴
                                geometries.append(ref_coord_frame)

                        except Exception as e:
                            print(f"处理目标 {target_idx} 时出错: {str(e)}")
                            import traceback
                            traceback.print_exc()

                            # 在处理完一个目标后（try-except块的末尾）添加
                            process_end_time = time.time()
                            process_time = process_end_time - process_start_time

                            # 添加到时间列表并控制列表大小
                            pointcloud_times.append(process_time)
                            if len(pointcloud_times) > max_time_samples:
                                pointcloud_times.pop(0)

                        # 计算并打印平均处理时间和FPS
                        if len(pointcloud_times) > 0:
                            avg_time = sum(pointcloud_times) / len(pointcloud_times)
                            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
                            print(f"点云处理平均FPS: {avg_fps:.2f}, 平均每帧时间: {avg_time * 1000:.2f}ms")

                # 第四步：更新渲染 - 只在必要时更新，并且避免频繁锁定
                if need_update and current_time - last_update_time > update_interval:
                    try:
                        # 渲染更新不需要锁定，减少阻塞
                        vis.poll_events()
                        vis.update_renderer()
                        last_update_time = current_time
                    except Exception as e:
                        print(f"渲染更新错误: {str(e)}")

            # 防止CPU过载
            time.sleep(0.01)

    except Exception as e:
        print(f"点云处理线程错误: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        # 关闭窗口
        vis.destroy_window()
        print("点云处理线程已结束")


def axis_direction_error(estimated_axis, ground_truth_axis):
    """
    计算轴方向误差（考虑轴的对称性）
    返回角度误差（度）
    """
    # 归一化向量
    est_axis = estimated_axis / np.linalg.norm(estimated_axis)
    gt_axis = ground_truth_axis / np.linalg.norm(ground_truth_axis)

    # 计算夹角 (使用绝对值处理方向的不确定性)
    cos_angle = np.abs(np.dot(est_axis, gt_axis))
    cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def cylinder_fitting_error(point_cloud, center, axis, radius):
    """
    计算点云到理想圆柱面的平均距离
    """
    errors = []
    for point in point_cloud:
        # 计算点到轴的向量
        v = point - center
        # 计算点在轴上的投影
        proj = np.dot(v, axis) * axis
        # 计算点到轴的垂直距离
        dist_to_axis = np.linalg.norm(v - proj)
        # 计算点到圆柱面的距离
        dist_to_surface = np.abs(dist_to_axis - radius)
        errors.append(dist_to_surface)

    return np.mean(errors), np.std(errors)


def transform_model(model_points, pose):
    """
    将模型点云按照估计的姿态进行变换

    参数:
        model_points: 模型点云，numpy数组，形状为(N, 3)
        pose: 4x4变换矩阵或包含'rotation_matrix'和'position'的字典

    返回:
        变换后的点云，numpy数组，形状为(N, 3)
    """
    # 检查pose类型
    if isinstance(pose, np.ndarray) and pose.shape == (4, 4):
        # 如果pose是4x4变换矩阵
        transform_matrix = pose
    elif isinstance(pose, dict) and 'rotation_matrix' in pose and 'position' in pose:
        # 如果pose是字典
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = pose['rotation_matrix']
        transform_matrix[:3, 3] = pose['position']
    else:
        raise ValueError("pose必须是4x4变换矩阵或包含'rotation_matrix'和'position'的字典")

    # 如果model_points是Open3D点云，转换为numpy数组
    if isinstance(model_points, o3d.geometry.PointCloud):
        points = np.asarray(model_points.points)
    else:
        points = np.array(model_points)

    # 转换为齐次坐标
    if points.shape[1] == 3:
        homo_points = np.hstack([points, np.ones((points.shape[0], 1))])
    else:
        homo_points = points

    # 应用变换
    transformed_homo = homo_points @ transform_matrix.T

    # 转回三维坐标
    transformed_points = transformed_homo[:, :3]

    return transformed_points


def icp_registration_error(point_cloud, estimated_pose, cylinder_model):
    """
    使用ICP评估姿态估计质量
    """
    # 应用估计的姿态到标准圆柱模型
    transformed_model = transform_model(cylinder_model, estimated_pose)

    # 使用Open3D执行ICP
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(point_cloud)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(transformed_model)

    # 执行ICP
    result = o3d.pipelines.registration.registration_icp(
        source, target, 0.05,  # 最大对应距离
        np.identity(4),  # 初始姿态
        o3d.pipelines.registration.TransformationEstimationPointToPoint())

    # 返回ICP的适合度指标和变换矩阵
    return result.fitness, result.inlier_rmse, result.transformation


def evaluate_pose_estimation(point_cloud_data_with_image_info, unique_indices):
    """姿态估计误差评估框架"""
    results = {}

    for target_idx in unique_indices:
        # 提取当前目标的点云数据
        valid_points = []
        for item in point_cloud_data_with_image_info:
            idx, point, _, _ = item
            if idx == target_idx and not np.any(np.isnan(point)):
                valid_points.append(point)

        if not valid_points:
            continue

        points_array = np.array(valid_points, dtype=np.float64)

        # 执行PCA获取主轴方向和中心点
        eigenvectors, eigenvalues, centroid = pca_analysis(points_array)
        
        # 1. 计算半径（平均点到轴距离）
        radius = estimate_cylinder_radius(points_array, centroid, eigenvectors[:, 0])
        
        # 2. 计算点云拟合误差
        mean_error, std_error = cylinder_fitting_error(points_array, centroid, eigenvectors[:, 0], radius)
        
        # 3. 如果有地面真值，计算轴向误差
        if hasattr(target_idx, 'ground_truth_axis'):
            axis_error = axis_direction_error(eigenvectors[:, 0], target_idx.ground_truth_axis)
        else:
            axis_error = None
        
        # 存储结果
        results[target_idx] = {
            'centroid': centroid,
            'main_axis': eigenvectors[:, 0],
            'radius': radius,
            'mean_fitting_error': mean_error,
            'std_fitting_error': std_error,
            'axis_error': axis_error
        }
        
        # 打印评估结果
        print(f"\n目标 {target_idx} 姿态估计评估:")
        print(f"中心点: {centroid}")
        print(f"主轴方向: {eigenvectors[:, 0]}")
        print(f"估计半径: {radius:.2f} mm")
        print(f"点云拟合平均误差: {mean_error:.2f} mm")
        print(f"点云拟合标准差: {std_error:.2f} mm")
        if axis_error is not None:
            print(f"轴向角度误差: {axis_error:.2f} 度")
    
    return results


def estimate_cylinder_radius(points, center, axis):
    """估计圆柱体半径"""
    distances = []
    for point in points:
        # 计算点到轴的向量
        v = point - center
        # 计算点在轴上的投影
        proj = np.dot(v, axis) * axis
        # 计算点到轴的距离
        dist = np.linalg.norm(v - proj)
        distances.append(dist)
    
    # 返回平均距离作为半径估计
    return np.mean(distances)


def compare_with_ground_truth(estimated_results, ground_truth):
    """与地面真值比较"""
    for target_idx, result in estimated_results.items():
        if target_idx in ground_truth:
            gt = ground_truth[target_idx]
            
            # 中心点位置误差
            position_error = np.linalg.norm(result['centroid'] - gt['centroid'])
            
            # 轴方向误差
            axis_error = axis_direction_error(result['main_axis'], gt['main_axis'])
            
            # 半径误差
            radius_error = np.abs(result['radius'] - gt['radius'])
            
            print(f"\n目标 {target_idx} 与地面真值比较:")
            print(f"中心点位置误差: {position_error:.2f} mm")
            print(f"轴方向角度误差: {axis_error:.2f} 度")
            print(f"半径误差: {radius_error:.2f} mm")


def axes_angle_error(estimated_axes, reference_axes, class_name):
    """计算轴角度误差"""
    errors = {}
    
    # 创建副本避免修改原数据
    est_axes_copy = estimated_axes.copy()
    ref_axes_copy = reference_axes.copy()
    
    # 计算X轴误差
    x_error = min(
        np.degrees(np.arccos(np.clip(np.abs(np.dot(est_axes_copy[:, 0], ref_axes_copy[:, 0])), -1.0, 1.0))),
        np.degrees(np.arccos(np.clip(np.abs(np.dot(est_axes_copy[:, 0], -ref_axes_copy[:, 0])), -1.0, 1.0)))
    )
    errors['primary_axis_error'] = x_error
    
    # 计算Y轴误差
    y_error = min(
        np.degrees(np.arccos(np.clip(np.abs(np.dot(est_axes_copy[:, 1], ref_axes_copy[:, 1])), -1.0, 1.0))),
        np.degrees(np.arccos(np.clip(np.abs(np.dot(est_axes_copy[:, 1], -ref_axes_copy[:, 1])), -1.0, 1.0)))
    )
    errors['secondary_axis_error'] = y_error
    errors['secondary_plane_error'] = y_error  # 兼容原代码
    
    # 计算Z轴误差
    z_error = min(
        np.degrees(np.arccos(np.clip(np.abs(np.dot(est_axes_copy[:, 2], ref_axes_copy[:, 2])), -1.0, 1.0))),
        np.degrees(np.arccos(np.clip(np.abs(np.dot(est_axes_copy[:, 2], -ref_axes_copy[:, 2])), -1.0, 1.0)))
    )
    errors['tertiary_axis_error'] = z_error
    
    # 对齐轴方向用于计算整体误差
    for i in range(3):
        if np.dot(est_axes_copy[:, i], ref_axes_copy[:, i]) < 0:
            est_axes_copy[:, i] = -est_axes_copy[:, i]
    
    # 确保右手系
    if np.dot(np.cross(est_axes_copy[:, 0], est_axes_copy[:, 1]), est_axes_copy[:, 2]) < 0:
        est_axes_copy[:, 2] = -est_axes_copy[:, 2]
    
    # 计算整体坐标系误差
    R_diff = np.dot(est_axes_copy, ref_axes_copy.T)
    angle = np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1.0, 1.0))
    errors['coordinate_system_error'] = np.degrees(angle)
    
    # 输出根据物体类型的专门描述
    axis_names = ["X轴", "Y轴", "Z轴"]
    print(f"=== 轴角度误差评估 (目标: {class_name}) ===")
    print(f"{axis_names[0]}角度误差: {x_error:.2f}°")
    print(f"{axis_names[1]}角度误差: {y_error:.2f}°")
    print(f"{axis_names[2]}角度误差: {z_error:.2f}°")
    print(f"整体坐标系误差: {errors['coordinate_system_error']:.2f}°")
    
    return errors


def update_reference_axes(object_type, new_axes):
    """
    更新指定类型物体的参考轴
    
    参数:
        object_type: 物体类型，如'bottle'
        new_axes: 新的参考轴，应为(3,3)矩阵，每列是一个单位向量
    """
    global reference_axes
    
    # 验证输入
    if new_axes.shape != (3, 3):
        print(f"错误：参考轴必须是3x3矩阵，当前形状为{new_axes.shape}")
        return False
    
    # 规范化轴向量
    normalized_axes = np.zeros((3, 3))
    for i in range(3):
        normalized_axes[:, i] = new_axes[:, i] / np.linalg.norm(new_axes[:, i])
    
    # 确保轴是正交的
    # 第一轴保持不变
    v1 = normalized_axes[:, 0]
    # 使第二轴正交于第一轴
    v2 = normalized_axes[:, 1] - np.dot(normalized_axes[:, 1], v1) * v1
    v2 = v2 / np.linalg.norm(v2)
    # 通过叉积计算第三轴
    v3 = np.cross(v1, v2)
    
    # 更新正交化后的轴
    normalized_axes[:, 1] = v2
    normalized_axes[:, 2] = v3
    
    # 更新参考轴
    reference_axes[object_type] = normalized_axes
    print(f"已更新'{object_type}'的参考轴")
    return True


# 示例：基于当前估计更新参考轴
def set_current_as_reference(target_idx, class_name):
    """将当前估计的轴设置为参考轴"""
    global pose_data
    
    if target_idx in pose_data and 'rotation_matrix' in pose_data[target_idx]:
        # 获取当前估计的旋转矩阵
        R = pose_data[target_idx]['rotation_matrix']
        
        # 更新参考轴
        object_type = class_name.lower()
        update_reference_axes(object_type, R)
        print(f"已将目标{target_idx}的当前姿态设为'{object_type}'的参考")
        return True
    else:
        print(f"无法设置参考：目标{target_idx}没有有效的姿态数据")
        return False


def visualize_axes_errors(target_idx):
    """
    可视化目标的轴误差趋势
    
    参数:
        target_idx: 目标ID
    """
    global pose_error_history, folder_name
    
    if target_idx not in pose_error_history or not pose_error_history[target_idx]['timestamps']:
        print(f"目标 {target_idx} 没有误差历史数据")
        return
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 计算相对时间（秒）
    relative_times = [(t - pose_error_history[target_idx]['timestamps'][0]) 
                      for t in pose_error_history[target_idx]['timestamps']]
    
    # 绘制三个轴的误差
    if 'primary_axis_errors' in pose_error_history[target_idx]:
        ax.plot(relative_times, pose_error_history[target_idx]['primary_axis_errors'], 
                'r-', label='主轴误差')
    
    if 'secondary_plane_errors' in pose_error_history[target_idx]:
        ax.plot(relative_times, pose_error_history[target_idx]['secondary_plane_errors'], 
                'g-', label='次轴平面误差')
    
    if 'coordinate_system_errors' in pose_error_history[target_idx]:
        ax.plot(relative_times, pose_error_history[target_idx]['coordinate_system_errors'], 
                'b-', label='整体坐标系误差')
    
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('角度误差 (度)')
    ax.set_title(f'目标 {target_idx} 轴角度误差趋势')
    ax.grid(True)
    ax.legend()
    
    # 确保保存路径存在
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    # 保存图像
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(folder_name, f"axes_errors_target_{target_idx}_{timestamp}.png")
    plt.savefig(save_path)
    print(f"轴误差分析图已保存到: {save_path}")
    plt.close(fig)


def save_results(target_idx, data):
    global folder_name  # 添加全局变量声明
    # 函数内容...

def process_results(results):
    global folder_name  # 添加全局变量声明
    # 函数内容...

def align_axes_with_reference(estimated_axes, reference_axes):
    """将估计的主轴方向与参考轴方向对齐（按照Y,X,Z排列）"""
    aligned_axes = estimated_axes.copy()
    
    # 对齐Y方向轴（第一列）
    if np.dot(estimated_axes[:, 0], reference_axes[:, 0]) < 0:
        aligned_axes[:, 0] = -aligned_axes[:, 0]
    
    # 对齐X方向轴（第二列）
    if np.dot(estimated_axes[:, 1], reference_axes[:, 1]) < 0:
        aligned_axes[:, 1] = -aligned_axes[:, 1]
    
    # 确保Z轴（第三列）遵循右手法则
    expected_z = np.cross(aligned_axes[:, 1], aligned_axes[:, 0])
    expected_z = expected_z / np.linalg.norm(expected_z)
    
    if np.dot(aligned_axes[:, 2], expected_z) < 0:
        aligned_axes[:, 2] = -aligned_axes[:, 2]
    
    return aligned_axes

def calculate_error_cone(axis_history, confidence=0.95):
    """
    计算基于历史轴方向的误差圆锥体参数
    
    参数:
        axis_history: 历史主轴方向列表
        confidence: 置信度水平（默认95%）
    
    返回:
        mean_axis: 平均主轴方向
        cone_angle: 圆锥体开角（度）
    """
    if not axis_history or len(axis_history) < 2:
        return None, None
    
    # 计算平均轴方向
    axes = np.array(axis_history)
    
    # 确保所有轴方向一致（不会从正反方向跳跃）
    reference = axes[0]
    for i in range(1, len(axes)):
        if np.dot(axes[i], reference) < 0:
            axes[i] = -axes[i]
    
    # 计算平均轴
    mean_axis = np.mean(axes, axis=0)
    mean_axis = mean_axis / np.linalg.norm(mean_axis)
    
    # 计算每个轴与平均轴的角度
    angles = []
    for axis in axes:
        cos_angle = np.clip(np.dot(axis, mean_axis), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_angle))
        angles.append(angle)
    
    # 基于置信度计算圆锥开角
    angles = np.array(angles)
    cone_angle = np.percentile(angles, confidence * 100)
    
    return mean_axis, cone_angle

def create_error_cone_mesh(apex, axis, cone_angle, height=10.0, resolution=20):
    """
    创建误差圆锥体的网格
    
    参数:
        apex: 圆锥体顶点
        axis: 圆锥体中心轴
        cone_angle: 圆锥体开角（度）
        height: 圆锥体高度
        resolution: 圆锥体圆周分辨率
    
    返回:
        cone_mesh: 圆锥体网格对象
    """
    # 规范化轴方向
    axis = axis / np.linalg.norm(axis)
    
    # 计算圆锥底面半径
    radius = height * np.tan(np.radians(cone_angle))
    
    # 创建参考系
    if np.allclose(axis, [0, 1, 0]) or np.allclose(axis, [0, -1, 0]):
        ref1 = np.array([1, 0, 0])
    else:
        ref1 = np.cross(axis, [0, 1, 0])
        ref1 = ref1 / np.linalg.norm(ref1)
    ref2 = np.cross(axis, ref1)
    ref2 = ref2 / np.linalg.norm(ref2)
    
    # 创建底面圆的顶点
    vertices = [apex]  # 圆锥顶点
    base_center = apex + height * axis
    
    # 添加底面中心点 
    vertices.append(base_center)
    
    # 添加底面圆周点
    for i in range(resolution):
        theta = 2 * np.pi * i / resolution
        # 底面圆上的点
        point = (base_center + 
                radius * np.cos(theta) * ref1 + 
                radius * np.sin(theta) * ref2)
        vertices.append(point)
    
    # 创建侧面三角形 (从顶点到底面圆周)
    triangles = []
    for i in range(2, resolution+1):
        triangles.append([0, i, i+1 if i < resolution+1 else 2])
    
    # 创建底面三角形 (以底面中心为共享顶点)
    for i in range(2, resolution+1):
        triangles.append([1, i+1 if i < resolution+1 else 2, i])
    
    # 创建Open3D网格
    cone_mesh = o3d.geometry.TriangleMesh()
    cone_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    cone_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    # 计算法线并应用颜色
    cone_mesh.compute_vertex_normals()
    cone_mesh.paint_uniform_color([1, 0.7, 0])  # 金黄色圆锥体
    
    # 使圆锥体半透明
    # 注意：Open3D目前不支持通过代码直接设置透明度，我们使用材质属性
    # 如果您的Open3D版本支持材质
    try:
        material = o3d.visualization.rendering.Material()
        material.base_color = [1, 0.7, 0, 0.5]  # RGBA，Alpha=0.5
        material.shader = "defaultLit"
        cone_mesh.material = material
    except:
        pass  # 忽略如果不支持此功能
    
    return cone_mesh

def visualize_error_cone(target_idx, class_name):
    """
    为指定目标可视化误差圆锥体并保存图像
    
    参数:
        target_idx: 目标ID
        class_name: 物体类别名称
    """
    global axis_history, pose_data, folder_name
    
    if target_idx not in axis_history or len(axis_history[target_idx]) < 5:
        print(f"目标 {target_idx} 的历史轴数据不足，无法计算误差圆锥体")
        return
    
    # 计算误差圆锥参数
    mean_axis, cone_angle = calculate_error_cone(axis_history[target_idx])
    if mean_axis is None:
        return
        
    # 获取目标位置
    if target_idx not in pose_data:
        print(f"目标 {target_idx} 没有姿态数据")
        return
        
    position = pose_data[target_idx]['position']
    
    # 创建可视化窗口 - 修改这里，确保正确创建窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(f"误差圆锥体 - 目标 {target_idx}", width=800, height=600, visible=True)
    
    # 确保窗口创建成功
    if not vis.poll_events():
        print("警告：无法创建可视化窗口")
        return
    
    # 添加坐标系
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)
    vis.add_geometry(coordinate_frame)
    
    # 创建并添加误差圆锥体
    cone_mesh = create_error_cone_mesh(position, mean_axis, cone_angle)
    cone_mesh.paint_uniform_color([1, 0.7, 0])  # 金黄色圆锥体
    vis.add_geometry(cone_mesh)

    # 添加主轴方向线
    line_points = np.array([position, position + mean_axis * 20])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(line_points)
    line_set.lines = o3d.utility.Vector2iVector([[0, 1]])
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]])  # 红色线
    vis.add_geometry(line_set)
    
    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.background_color = np.array([1, 1, 1])  # 白色背景
    render_option.point_size = 1.0
    
    # 修改视角设置部分
    view_control = vis.get_view_control()
    # 调整相机位置
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    
    # 计算观察距离和方向
    distance = 50.0  # 调整这个距离可能需要根据您的场景大小
    front = [0, 0, -1]  # 相机方向
    up = [0, -1, 0]  # 上方向
    
    # 设置相机位置
    camera_pos = position + np.array(front) * (-distance)
    camera_params.extrinsic = np.array([
        [1, 0, 0, camera_pos[0]],
        [0, 1, 0, camera_pos[1]],
        [0, 0, 1, camera_pos[2]],
        [0, 0, 0, 1]
    ])
    
    # 应用相机参数
    view_control.convert_from_pinhole_camera_parameters(camera_params)
    
    # 强制更新和渲染
    for i in range(5):  # 多次更新以确保渲染正确
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)  # 给渲染器一些时间
    
    # 保存图像
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    save_path = os.path.join(folder_name, f"error_cone_{target_idx}_{class_name}_{timestamp}.png")
    
    # 检查目录是否存在
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    
    try:
        # 尝试捕获屏幕图像
        vis.capture_screen_image(save_path)
        print(f"误差圆锥体分析图已保存到: {save_path}")
        print(f"圆锥体开角: {cone_angle:.2f}° (95%置信区间)")
    except Exception as e:
        print(f"保存图像时出错: {str(e)}")
    
    # 在关闭窗口前等待一小段时间，让用户查看
    time.sleep(2)
    
    # 关闭窗口
    vis.destroy_window()

def main():
    global exit_flag, segmentation_times, pointcloud_times, reference_axes, folder_name
    
    # 创建文件夹
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parent_folder = "data_experiment"
    folder_name = os.path.join(parent_folder, f"data_{current_time}")
    
    # 确保文件夹存在
    if not os.path.exists(parent_folder):
        os.makedirs(parent_folder)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    """# 初始化参考轴 - Y轴向下   1  4
    standard_axes = np.array([
        [1.0, 0.0, 0.0],  # 主轴 - Y轴向下
        [0.0, 1.0, 0.0],  # 次轴 - X轴向右
        [0.0, 0.0, 1.0]  # 第三轴 - Z轴向前
    ]).T"""

    """# 初始化参考轴 - 3 相机右手坐标系下，物体y轴正半轴绕z轴负半轴向x轴正半轴逆时针旋转了10度
    standard_axes = np.array([
        [0.9848, -0.1736, 0.0],  # 主轴 - Y轴向下
        [0.1736, 0.9848, 0.0],  # 次轴 - X轴向右
        [0.0, 0.0, 1.0]  # 第三轴 - Z轴向前
    ]).T"""

    # 初始化参考轴 - 2 5  相机右手坐标系下，物体y轴正半轴绕x轴正半轴向z轴负半轴顺时针旋转了10度
    standard_axes = np.array([
        [1.0, 0.0, 0.0],  # 主轴 - Y轴向下
        [0.0, 0.9848, -0.1736],  # 次轴 - X轴向右
        [0.0, 0.1736, 0.9848]  # 第三轴 - Z轴向前
    ]).T

    """standard_axes = np.array([
        [1.0, 0.0, 0.0],  # 主轴 - Y轴向下
        [0.0, 0.9848, 0.1736],  # 次轴 - X轴向右
        [0.0, -0.1736, 0.9848]  # 第三轴 - Z轴向前
    ])"""

    """# 初始化参考轴 - 6  相机右手坐标系下，物体y轴正半轴绕z轴负半轴向x轴负半轴顺时针旋转了10度
    standard_axes = np.array([
        [0.9848, 0.1736, 0.0],  # 主轴 - Y轴向下
        [-0.1736, 0.9848, 0.0],  # 次轴 - X轴向右
        [0.0, 0.0, 1.0]  # 第三轴 - Z轴向前
    ]).T"""

    """standard_axes = np.array([
        [0.9848, 0.1736, 0.0],  # 主轴 - Y轴向下
        [-0.1736, 0.9848, 0.0],  # 次轴 - X轴向右
        [0.0, 0.0, 1.0]  # 第三轴 - Z轴向前
    ]).T"""

    # 为bottle和cap分别设置相同的参考轴（因为它们都是Y轴向下）
    reference_axes['bottle'] = standard_axes
    reference_axes['cap'] = standard_axes
    
    print("初始化参考轴完成...")
    print(f"数据将保存到: {folder_name}")
    
    # 初始化性能测量变量
    start_time = time.time()
    last_report_time = start_time

    try:
        # 创建线程
        segmentation_thread_obj = threading.Thread(target=segmentation_thread)
        pointcloud_thread_obj = threading.Thread(target=pointcloud_thread)

        # 设置为守护线程，这样主线程退出时它们会自动结束
        segmentation_thread_obj.daemon = True
        pointcloud_thread_obj.daemon = True

        # 启动线程
        segmentation_thread_obj.start()
        pointcloud_thread_obj.start()

        # 主线程等待
        print("程序已启动，按ESC键或Ctrl+C退出程序...")
        try:
            while not exit_flag:
                time.sleep(0.1)
                # 每5秒显示一次总体运行情况
                current_time = time.time()
                if current_time - last_report_time > 5:
                    total_runtime = current_time - start_time
                    # 如果有分割时间数据和点云时间数据，显示整体性能
                    if segmentation_times and pointcloud_times:
                        seg_avg_fps = len(segmentation_times) / sum(segmentation_times) if sum(
                            segmentation_times) > 0 else 0
                        pc_avg_fps = len(pointcloud_times) / sum(pointcloud_times) if sum(pointcloud_times) > 0 else 0
                        print(f"\n-----性能指标-----")
                        print(f"总运行时间: {total_runtime:.2f}秒")
                        print(f"分割线程平均FPS: {seg_avg_fps:.2f}")
                        print(f"点云线程平均FPS: {pc_avg_fps:.2f}")
                        print(f"-------------------\n")
                    last_report_time = current_time
        except KeyboardInterrupt:
            print("\n程序被用户中断")
            exit_flag = True

    except Exception as e:
        print(f"主线程错误: {str(e)}")
        exit_flag = True

    finally:
        # 等待线程完成
        print("等待线程结束...")
        exit_flag = True
        segmentation_thread_obj.join(timeout=5)
        pointcloud_thread_obj.join(timeout=5)
        print("程序已完全退出")


if __name__ == '__main__':
    main()