import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation
import trimesh
import os

def rotation_matrix_to_quaternion(R):
    """将3x3旋转矩阵转换为单位四元数。

    参数:
        R : 3x3 numpy数组（旋转矩阵）

    返回:
        q : 四元数 [w, x, y, z] 其中 w 是标量部分（非负）
    """
    assert R.shape == (3, 3), "矩阵必须是3x3"

    tr = R[0, 0] + R[1, 1] + R[2, 2]

    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S

    quat = np.array([w, x, y, z])
    # 确保w非负，如果为负则取反
    if w < 0:
        quat = -quat
    # 注意：这里取反后，四元数仍然表示相同的旋转，但是通常我们使用标量部分为正的表示

    return quat


def quaternion_angular_distance(q1, q2):
    """
    计算两个单位四元数之间的最小角度差（弧度）

    参数:
        q1, q2: 单位四元数 [w, x, y, z]

    返回:
        角度差（弧度），范围 [0, π]
    """
    # 确保单位四元数
    q1 = np.asarray(q1) / np.linalg.norm(q1)
    q2 = np.asarray(q2) / np.linalg.norm(q2)

    # 点积（考虑四元数双覆盖性质）
    dot = np.abs(np.dot(q1, q2))

    # 限制在有效范围 [0, 1]
    dot = np.clip(dot, 0.0, 1.0)

    # 角度差（弧度）
    return 2 * np.arccos(dot)


def find_quaternion_outliers(quaternions, threshold=30):
    """
    从一组四元数中找出离群点

    参数:
        quaternions: 四元数列表，每个为 [w, x, y, z]
        threshold: 角度阈值（度），默认30°

    返回:
        (离群点索引列表, 平均角度差列表, 参考四元数)
    """
    # 转换为numpy数组
    quats = np.array(quaternions)

    # 1. 计算参考四元数（均值四元数）
    mean_quat = np.mean(quats, axis=0)
    mean_quat /= np.linalg.norm(mean_quat)  # 归一化

    # 2. 计算每个四元数与参考四元数的角度差
    angular_diffs = []
    for q in quats:
        angle_rad = quaternion_angular_distance(mean_quat, q)
        angle_deg = np.degrees(angle_rad)
        angular_diffs.append(angle_deg)

    # 3. 计算平均角度差和标准差
    mean_angle = np.mean(angular_diffs)
    std_angle = np.std(angular_diffs)

    # 4. 找出离群点（超过阈值）
    outliers = []
    for i, angle in enumerate(angular_diffs):
        # 阈值可以是固定值或基于统计的动态值
        if angle > threshold:  # 固定阈值
            outliers.append(i)
        # 或者使用动态阈值：if angle > mean_angle + 2 * std_angle

    return outliers, angular_diffs, mean_quat

def ransac_quaternion(quaternions, threshold_deg=30, max_iterations=1000, min_inliers_ratio=0.5):
    """
    使用RANSAC从一组四元数中找出内点和离群点

    参数:
        quaternions: 四元数列表，每个为 [w, x, y, z]
        threshold_deg: 角度阈值（度）
        max_iterations: 最大迭代次数
        min_inliers_ratio: 最小内点比例

    返回:
        (内点索引列表, 离群点索引列表, 参考四元数)
    """
    # 转换为numpy数组
    quats = np.array(quaternions)
    n = quats.shape[0]

    # 角度阈值转换为弧度
    threshold_rad = np.radians(threshold_deg)

    best_inliers = []
    best_num_inliers = 0
    best_model = None

    for _ in range(max_iterations):
        # 1. 随机选择一个样本作为模型假设
        sample_idx = np.random.randint(0, n)
        model = quats[sample_idx]

        # 2. 计算所有四元数到该模型的角度差
        inliers = []
        for i in range(n):
            dist = quaternion_angular_distance(model, quats[i])
            if dist <= threshold_rad:
                inliers.append(i)

        # 3. 检查是否满足最小内点比例要求
        num_inliers = len(inliers)
        if num_inliers >= min_inliers_ratio * n:
            # 4. 使用所有内点优化模型（计算平均四元数）
            inlier_quats = quats[inliers]
            mean_quat = np.mean(inlier_quats, axis=0)
            mean_quat /= np.linalg.norm(mean_quat)  # 归一化

            # 5. 使用优化后的模型重新计算内点
            refined_inliers = []
            for i in range(n):
                dist = quaternion_angular_distance(mean_quat, quats[i])
                if dist <= threshold_rad:
                    refined_inliers.append(i)

            num_refined_inliers = len(refined_inliers)

            # 6. 更新最佳模型
            if num_refined_inliers > best_num_inliers:
                best_num_inliers = num_refined_inliers
                best_inliers = refined_inliers
                best_model = mean_quat

                # 提前终止条件：所有点都是内点
                if best_num_inliers == n:
                    break

    # 7. 识别离群点
    if best_inliers:
        outliers = [i for i in range(n) if i not in best_inliers]
        return best_inliers, outliers, best_model
    else:
        # 没有找到足够的内点
        return [], list(range(n)), None

def auto_threshold(quaternions):
    """
    基于数据统计自动计算RANSAC阈值

    参数:
        quaternions: 四元数列表

    返回:
        自动计算的角度阈值（度）
    """
    # 计算所有两两角度差
    n = len(quaternions)
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dist = quaternion_angular_distance(quaternions[i], quaternions[j])
            dists.append(np.degrees(dist))

    # 计算统计量（使用中位数和MAD，对离群点更鲁棒）
    median = np.median(dists)
    mad = np.median(np.abs(dists - median))  # 中位数绝对偏差

    # 返回 median + 3*MAD 作为阈值
    return median + 3 * mad


def euclidean_distance(p1, p2):
    """
    计算两点之间的欧氏距离

    参数:
        p1, p2: 3D 点 [x, y, z]

    返回:
        欧氏距离
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def ransac_translation(translations, threshold=0.1, max_iterations=1000, min_inliers_ratio=0.5):
    """
    使用RANSAC从一组平移数据中找出内点和离群点

    参数:
        translations: 3D点列表，每个为 [x, y, z]
        threshold: 距离阈值
        max_iterations: 最大迭代次数
        min_inliers_ratio: 最小内点比例

    返回:
        (内点索引列表, 离群点索引列表, 参考点)
    """
    # 转换为numpy数组
    points = np.array(translations)
    n = points.shape[0]

    best_inliers = []
    best_num_inliers = 0
    best_model = None

    for _ in range(max_iterations):
        # 1. 随机选择一个样本作为模型假设
        sample_idx = np.random.randint(0, n)
        model = points[sample_idx]

        # 2. 计算所有点到该模型的距离
        inliers = []
        for i in range(n):
            dist = euclidean_distance(model, points[i])
            if dist <= threshold:
                inliers.append(i)

        # 3. 检查是否满足最小内点比例要求
        num_inliers = len(inliers)
        if num_inliers >= min_inliers_ratio * n:
            # 4. 使用所有内点优化模型（计算中心点）
            inlier_points = points[inliers]
            centroid = np.mean(inlier_points, axis=0)

            # 5. 使用优化后的模型重新计算内点
            refined_inliers = []
            for i in range(n):
                dist = euclidean_distance(centroid, points[i])
                if dist <= threshold:
                    refined_inliers.append(i)

            num_refined_inliers = len(refined_inliers)

            # 6. 更新最佳模型
            if num_refined_inliers > best_num_inliers:
                best_num_inliers = num_refined_inliers
                best_inliers = refined_inliers
                best_model = centroid

                # 提前终止条件：所有点都是内点
                if best_num_inliers == n:
                    break

    # 7. 识别离群点
    if best_inliers:
        outliers = [i for i in range(n) if i not in best_inliers]
        return best_inliers, outliers, best_model
    else:
        # 没有找到足够的内点
        return [], list(range(n)), None


# ======================== 核心函数 ========================

def quaternion_to_rotation_matrix(q):
    """
    将单位四元数转换为旋转矩阵

    参数:
        q: 单位四元数 [w, x, y, z]

    返回:
        3x3旋转矩阵
    """
    # 确保四元数是单位四元数
    q = np.array(q)
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > 1e-6:
        q = q / norm

    w, x, y, z = q

    # 计算旋转矩阵元素 (公式通用)
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)]
    ])

    return R

    return R


def quaternion_spherical_mean(quaternions, tolerance=1e-6, max_iterations=100):
    """
    计算四元数的球面平均（正确方法）

    参数:
        quaternions: 四元数列表，每个四元数是[w, x, y, z]
        tolerance: 收敛容差
        max_iterations: 最大迭代次数

    返回:
        平均四元数 [w, x, y, z]
    """
    # 确保所有四元数在同一个半球
    aligned_quats = []
    for q in quaternions:
        # 检查点积符号
        if aligned_quats and np.dot(aligned_quats[0], q) < 0:
            aligned_quats.append(-np.array(q))  # 取反到同一半球
        else:
            aligned_quats.append(np.array(q))

    # 初始估计
    avg_q = np.mean(aligned_quats, axis=0)
    avg_q /= np.linalg.norm(avg_q)

    # 迭代优化
    for _ in range(max_iterations):
        # 计算对数映射
        log_sum = np.zeros(3)
        for q in aligned_quats:
            # 计算四元数相对旋转
            rot_avg = Rotation.from_quat([avg_q[1], avg_q[2], avg_q[3], avg_q[0]])
            rot_q = Rotation.from_quat([q[1], q[2], q[3], q[0]])

            # 计算相对旋转
            delta_rot = rot_avg.inv() * rot_q

            # 对数映射到切空间
            log_sum += delta_rot.as_rotvec()

        # 指数映射回四元数
        avg_delta = Rotation.from_rotvec(log_sum / len(aligned_quats))
        new_avg_rot = rot_avg * avg_delta
        new_avg_q = new_avg_rot.as_quat()
        avg_q = np.array([new_avg_q[3], new_avg_q[0], new_avg_q[1], new_avg_q[2]])

        # 检查收敛
        if np.linalg.norm(log_sum) < tolerance:
            break

    return avg_q


def quaternion_to_euler(q):
    """
    将单位四元数转换为欧拉角（偏航、俯仰、滚转）

    参数:
        q : 四元数 [w, x, y, z]

    返回:
        (yaw, pitch, roll) : 欧拉角（弧度）
    """
    q = np.array(q)
    norm = np.linalg.norm(q)
    if abs(norm - 1.0) > 1e-6:
        q = q / norm
    w, x, y, z = q

    # 1. 计算滚转角 (roll, x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # 2. 计算俯仰角 (pitch, y-axis rotation)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        # 处理万向节死锁情况
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # 3. 计算偏航角 (yaw, z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return yaw, pitch, roll


def euler_to_rotation_matrix(yaw, pitch, roll):
    """
    将欧拉角转换为旋转矩阵（OpenCV坐标系）

    参数:
        yaw : 偏航角（绕Z轴旋转，弧度）
        pitch : 俯仰角（绕Y轴旋转，弧度）
        roll : 滚转角（绕X轴旋转，弧度）

    返回:
        R : 3x3旋转矩阵
    """
    # 绕Z轴旋转（偏航角）
    R_z = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 绕Y轴旋转（俯仰角）
    R_y = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    # 绕X轴旋转（滚转角）
    R_x = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    # 组合旋转矩阵（Z-Y-X顺序）
    R = R_z @ R_y @ R_x

    return R

if __name__ == '__main__':
    gt_pose_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\pre_pose_with_optimization_adaptive.txt')
    gt_pose_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\pre_pose_with_optimization_adaptive_2.txt')
    gt_pose_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\pre_pose_with_optimization_adaptive_2.txt')
    gt_pose_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\pre_pose_with_optimization_adaptive.txt')
    gt_pose_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\pre_pose_with_optimization_adaptive.txt')
    gt_pose_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_optimization_adaptive.txt')

    num = gt_pose_1.shape[0]
    global_pose = []
    obje_size = []
    obje_size.append(513.2765)
    obje_size.append(352.7896)
    obje_size.append(262.2528)
    obje_size.append(518.7096)
    obje_size.append(238.095)
    obje_size.append(212.3694)
    # 读取三维模型，并计算包围盒中心坐标
    bounding_box_center_list = []
    for index in  range(0,6):
        path_ = "part_0" + str(index + 1)
        mesh = trimesh.load(os.path.join(r'F:\Data_pose\00QEL_Assembly_Datasets',path_, r'model\part_0' + str(index + 1) + '.obj'))
        centroid4 = mesh.center_mass
        bounding_box = mesh.bounding_box
        bounding_box_center = bounding_box.centroid
        print(bounding_box_center)
        bounding_box_center_list.append(bounding_box_center)
    # 物体之间最远的距离
    max_distance = np.linalg.norm(np.array(bounding_box_center_list[0]) - np.array(bounding_box_center_list[4]))
    # 物体的最大半径
    max_size = 518.7096
    # 1,2,3,4,5,6
    part_id = 1
    ids = []
    for i in range(num):
        similar_quats = []
        list_translation = []
        Rotation_matrix = []
        R1 = gt_pose_1[i][:9].reshape(3, 3)
        Q1 = rotation_matrix_to_quaternion(R1)
        t1 = gt_pose_1[i][9:].reshape(1, 3)

        R2 = gt_pose_2[i][:9].reshape(3, 3)
        Q2 = rotation_matrix_to_quaternion(R2)
        t2 = gt_pose_2[i][9:].reshape(1, 3)

        R3 = gt_pose_3[i][:9].reshape(3, 3)
        Q3 = rotation_matrix_to_quaternion(R3)
        t3 = gt_pose_3[i][9:].reshape(1, 3)

        R4 = gt_pose_4[i][:9].reshape(3, 3)
        Q4 = rotation_matrix_to_quaternion(R4)
        t4 = gt_pose_4[i][9:].reshape(1, 3)

        R5 = gt_pose_5[i][:9].reshape(3, 3)
        Q5 = rotation_matrix_to_quaternion(R5)
        t5 = gt_pose_5[i][9:].reshape(1, 3)

        R6 = gt_pose_6[i][:9].reshape(3, 3)
        Q6 = rotation_matrix_to_quaternion(R6)
        t6 = gt_pose_6[i][9:].reshape(1, 3)

        Rotation_matrix.append(R1)
        Rotation_matrix.append(R2)
        Rotation_matrix.append(R3)
        Rotation_matrix.append(R4)
        Rotation_matrix.append(R5)
        Rotation_matrix.append(R6)

        similar_quats.append(Q1)
        similar_quats.append(Q2)
        similar_quats.append(Q3)
        similar_quats.append(Q4)
        similar_quats.append(Q5)
        similar_quats.append(Q6)

        # ===========通过均值检测离群点，效果一般==============
        # outliers, angles, mean_q = find_quaternion_outliers(similar_quats, threshold=5)

        # print("参考四元数:", mean_q)
        # print("角度差(度):", angles)
        # print("离群点索引:", outliers)

        # 自动阈值有问题
        # auto_thresh = auto_threshold(similar_quats)
        # print("自动阈值：", auto_thresh)
        inliers, outliers, model = ransac_quaternion(
            similar_quats,
            threshold_deg= 2,    # 5
            max_iterations=1000
        )
        # print("==============================")
        print("数据id： ", i)
        print("参考四元数:", model)
        print("旋转内点索引:", inliers)
        print("旋转离群点索引:", outliers)

        list_translation.append(t1)
        list_translation.append(t2)
        list_translation.append(t3)
        list_translation.append(t4)
        list_translation.append(t5)
        list_translation.append(t6)
        # print(list_translation)
        t_inliers, t_outliers, t_model = ransac_translation(list_translation,
                                                            threshold=0.05,
                                                            max_iterations=1000)
        print("参考平移:", t_model)
        print("平移内点索引:", t_inliers)
        print("平移离群点索引:", t_outliers)

        Rotation_temp = Rotation_matrix[part_id - 1]
        t_temp = list_translation[part_id - 1]
        Q_temp = similar_quats[part_id - 1]

        R_new =Rotation_temp
        t_new =t_temp
        # 检索所有外点
        for index in outliers:
            # 提取特定的目标
            if index == (part_id - 1):
                # 旋转和平移索引
                r_valid_ = []
                t_valid = []
                distance_valid = []
                size_valid = []
                # 遍历所有内点
                for ii in inliers:
                    r_valid_.append(similar_quats[ii])
                    t_valid.append(list_translation[ii])
                    d = np.linalg.norm(np.array(bounding_box_center_list[ii]) - np.array(bounding_box_center_list[index]))
                    
                    # 修改后
                    distance_valid.append(1 - d / max_distance)
                    size_valid.append(obje_size[ii] / max_size)
                if len(r_valid_)==0:
                    R_new = quaternion_to_rotation_matrix(Q_temp)
                    t_new = t_temp
                else:
                    ids.append(i)
                    yaw_list = []
                    pitch_list = []
                    roll_list = []

                    # 权重的归一化
                    weight_ = []
                    for iii in range(0, len(distance_valid)):
                        weight = distance_valid[iii] * size_valid[iii]
                        weight_.append(weight)
                    total = sum(np.array(weight_))
                    weight_normal_ = np.array(weight_) / total
                    # 求内点的平均值，计算旋转误差
                    for every_quat in r_valid_:
                        yaw1, pitch1, roll1 = quaternion_to_euler(every_quat)
                        yaw1 = math.degrees(yaw1)
                        pitch1 = math.degrees(pitch1)
                        roll1 = math.degrees(roll1)
                        yaw_list.append(yaw1)
                        pitch_list.append(pitch1)
                        roll_list.append(roll1)
                        print("yaw_11", yaw1)
                        print("pitch_11", pitch1)
                        print("roll_11", roll1)
                    # yaw_avg = np.mean(np.array(yaw_list), axis=0)
                    # pitch_avg = np.mean(np.array(pitch_list), axis=0)
                    # roll_avg = np.mean(np.array(roll_list), axis=0)
                    # 加权求和
                    yaw_avg = np.average(np.array(yaw_list), weights=weight_normal_, axis= 0) * np.sum(weight_normal_)
                    pitch_avg = np.average(np.array(pitch_list), weights=weight_normal_, axis= 0) * np.sum(weight_normal_)
                    roll_avg = np.average(np.array(roll_list), weights=weight_normal_, axis= 0) * np.sum(weight_normal_)

                    print("yaw_new", yaw_avg)
                    print("pitch_new", pitch_avg)
                    print("roll_new", roll_avg)

                    # 求内点的平均值，计算平移误差
                    yaw_avg = math.radians(yaw_avg)
                    pitch_avg = math.radians(pitch_avg)
                    roll_avg = math.radians(roll_avg)

                    R_new = euler_to_rotation_matrix(yaw_avg, pitch_avg, roll_avg)

                    yaw_old, pitch_old, roll_old = quaternion_to_euler(Q_temp)
                    yaw_old = math.degrees(yaw_old)
                    pitch_old = math.degrees(pitch_old)
                    roll_old = math.degrees(roll_old)
                    print("yaw_old", yaw_old)
                    print("pitch_old", pitch_old)
                    print("roll_old", roll_old)
                    print("t_valid", len(t_valid))
                    print("weight_normal_", len(weight_normal_))
                    t_new = np.average(np.array(t_valid), weights=weight_normal_, axis= 0) * np.sum(weight_normal_)
                    # t_new = np.mean(t_valid, axis=0)

                # print(R_2_new)
        pp = []
        for x in R_new[:3, :3]:
            for y in x:
                pp.append(y)
        for xx in t_new[0, :3]:
            pp.append(xx)
        global_pose.append(pp)

        print("==============================")
    save_path = os.path.join(r"F:\Data_pose\00QEL_Assembly_Datasets", "part_0" + str(part_id), "pose","pre_pose_with_ransac_new_2.txt")
    np.savetxt(save_path, global_pose,  fmt='%.6f')
    save_path_ids = os.path.join(r"F:\Data_pose\00QEL_Assembly_Datasets", "part_0" + str(part_id), "pose", "ids_modify_new_2.txt")
    np.savetxt(save_path_ids, ids, fmt='%d')