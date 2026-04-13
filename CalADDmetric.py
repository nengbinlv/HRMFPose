import numpy as np
import os
import os.path as osp
from utils.utils import load_ply, project
import cv2
from scipy import spatial
from itertools import combinations
from random import choice
import math

def add_metric(pose_pred, pose_targets, diameter, mesh_model, syn=False, percentage=0.1):
    diameter = diameter * percentage
    model_pred = (
            np.dot(mesh_model["pts"], pose_pred[:, :3].T) + pose_pred[:, 3] * 1000
    )
    model_targets = (
            np.dot(mesh_model["pts"], pose_targets[:, :3].T) + pose_targets[:, 3] * 1000
    )

    if syn:
        mean_dist_index = spatial.cKDTree(model_pred)
        mean_dist, _ = mean_dist_index.query(model_targets, k=1)
        mean_dist = np.mean(mean_dist)
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
    # print(mean_dist)
    return (mean_dist < diameter)

def cm_degree_5_metric(pose_pred, pose_targets, icp=False):
    translation_distance = (
        np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 100
    )
    rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))
    # print(angular_distance)
    return translation_distance < 5 and angular_distance < 5

def projection_2d(pose_pred, pose_targets, K, mesh_model, threshold=5):
    model_2d_pred = project(mesh_model["pts"] / 1000, K, pose_pred)
    model_2d_targets = project(mesh_model["pts"] / 1000, K, pose_targets)
    proj_mean_diff = np.mean(
        np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1)
    )

    return proj_mean_diff < threshold

def calculate_tra_and_rot(pose, pred_pose, add):
    if add == False:
        return 0,0,0,0,0,0
    # pred_pose = pose_reverse(pred_pose, pose)
    rot = pose[:, :3]
    tra = pose[:, 3:].reshape(1, 3)
    pred_rot = pred_pose[:, :3]
    pred_tra = pred_pose[:, 3:].reshape(1, 3)
    tra_error = (tra - pred_tra) * 1000

    # print(float(tra[:,0]))

    x_error = math.fabs(tra_error[:, 0])
    y_error = math.fabs(tra_error[:, 1])
    z_error = math.fabs(tra_error[:, 2])

    sy = math.sqrt(rot[2, 1] * rot[2, 1] + rot[2, 2] * rot[2, 2])
    alpha = math.atan2(rot[2, 1], rot[2, 2])
    beta = math.atan2(-rot[2, 0], sy)
    gamma = math.atan2(rot[1, 0], rot[0, 0])

    pred_sy = math.sqrt(pred_rot[2, 1] * pred_rot[2, 1] + pred_rot[2, 2] * pred_rot[2, 2])
    pred_alpha = math.atan2(pred_rot[2, 1], pred_rot[2, 2])
    pred_beta = math.atan2(-pred_rot[2, 0], pred_sy)
    pred_gamma = math.atan2(pred_rot[1, 0], pred_rot[0, 0])

    alpha_error = math.fabs((math.fabs(alpha) - math.fabs(pred_alpha)) * 180 / math.pi)
    beta_error = math.fabs((math.fabs(beta) - math.fabs(pred_beta)) * 180 / math.pi)
    gamma_error = math.fabs((math.fabs(gamma) - math.fabs(pred_gamma)) * 180 / math.pi)

    return x_error, y_error, z_error, alpha_error, beta_error, gamma_error

def Mono6D():
    mesh_model = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\HingeBase\model\HingeBase.ply" )
    # Bracket  79.737
    # connector 73.223
    # HingeBase 81.240385
    # L-Holder 76.3413387
    # PoleClamp 75.6405979
    # SideClamp 80.11866
    # Stopper 117.0897
    # T-Holder  77.62087
    diameter = 81.240385

    gt_poses = np.loadtxt(r'F:\Data_pose\Mono6D\HingeBase\pose\gt_Pose_HingeBase_ours.txt')
    pre_poses = np.loadtxt(r'F:\Data_pose\Mono6D\HingeBase\pose\pre_Pose_HingeBase_induspose_train.txt')
    # K = np.array([[567.53720406, 0.0, 312.66570357], [0.0, 569.36175922, 257.1729701], [0.0, 0.0, 1.0]])

    # K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
    #                                [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
    #                                [0.0, 0.0, 1.0]])
    K = np.array([[2209.878296, 0.0, 349.751312],
                                   [0.0, 2210.376676, 254.828051],
                                   [0.0, 0.0, 1.0]])
    num = gt_poses.shape[0]
    ADD = []
    cm_degree_metric = []
    project_2d_metric = []
    x_error = []
    y_error = []
    z_error = []
    a_error = []
    b_error = []
    c_error = []

    for i in range(num):
        R = gt_poses[i][:9].reshape(3, 3)
        t = gt_poses[i][9:].reshape(3, 1)
        gt_pose = np.concatenate([R, t], axis=1)
        R_ = pre_poses[i][:9].reshape(3, 3)
        t_ = pre_poses[i][9:].reshape(3, 1)
        pre_pose = np.concatenate([R_, t_], axis=1)
        add = add_metric(pre_pose, gt_pose, diameter, mesh_model)
        if add == True:
            print(i)
        ADD.append(add)

        cm_degree = cm_degree_5_metric(pre_pose, gt_pose)
        cm_degree_metric.append(cm_degree)

        p2d = projection_2d(pre_pose, gt_pose, K, mesh_model)
        project_2d_metric.append(p2d)

        x, y, z, a, b, c = calculate_tra_and_rot(gt_pose, pre_pose, add)
        x_error.append(x)
        y_error.append(y)
        z_error.append(z)
        a_error.append(a)
        b_error.append(b)
        c_error.append(c)

    add_mean = np.mean(ADD)
    cm_degree_metric_mean = np.mean(cm_degree_metric)
    project_2d_metric_mean = np.mean(project_2d_metric)
    x_error_mean = np.mean(x_error)
    y_error_mean = np.mean(y_error)
    z_error_mean = np.mean(z_error)
    a_error_mean = np.mean(a_error)
    b_error_mean = np.mean(b_error)
    c_error_mean = np.mean(c_error)
    print("ADD: ", add_mean)
    print("cm_degree: ", cm_degree_metric_mean)
    print("project_2d: ", project_2d_metric_mean)
    print("x_error_mean: ", x_error_mean)
    print("y_error_mean: ", y_error_mean)
    print("z_error_mean: ", z_error_mean)
    print("a_error_mean: ", a_error_mean)
    print("b_error_mean: ", b_error_mean)
    print("c_error_mean: ", c_error_mean)

def Qel6D():
    mesh_model = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_06\model\part_06.ply" )
    # part_01  513.2765
    # part_02  352.7896
    # part_03  262.2528
    # part_04   518.7096
    # part_05  238.095
    # part_06  212.3694
    diameter = 212.3694

    gt_poses = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\gt_Pose_part_06_ours.txt')
    pre_poses = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_ransac_new_2_opti.txt')

    # pre_poses_with_ransac = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_ransac_andOp.txt')
    # ids = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\ids_modify.txt')

    # K = np.array([[567.53720406, 0.0, 312.66570357], [0.0, 569.36175922, 257.1729701], [0.0, 0.0, 1.0]])

    K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                                   [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                                   [0.0, 0.0, 1.0]])
    num = gt_poses.shape[0]
    ADD = []
    cm_degree_metric = []
    project_2d_metric = []
    x_error = []
    y_error = []
    z_error = []
    a_error = []
    b_error = []
    c_error = []
    t_err = []
    R_err = []
    global_pose = []
    for i in range(num):
        R = gt_poses[i][:9].reshape(3, 3)
        t = gt_poses[i][9:].reshape(3, 1)
        gt_pose = np.concatenate([R, t], axis=1)
        # if i in ids:
        #     R_ = pre_poses_with_ransac[i][:9].reshape(3, 3)
        #     t_ = pre_poses_with_ransac[i][9:].reshape(3, 1)
        # else:
        R_ = pre_poses[i][:9].reshape(3, 3)
        t_ = pre_poses[i][9:].reshape(3, 1)
        pre_pose = np.concatenate([R_, t_], axis=1)
        add = add_metric(pre_pose, gt_pose, diameter, mesh_model)
        # if add == False:
        #     print(i)
        pp = []
        for x in pre_pose[:3, :3]:
            for y in x:
                pp.append(y)
        for xx in pre_pose[:3, 3]:
            pp.append(xx)
        global_pose.append(pp)

        ADD.append(add)

        cm_degree = cm_degree_5_metric(pre_pose, gt_pose)
        cm_degree_metric.append(cm_degree)

        p2d = projection_2d(pre_pose, gt_pose, K, mesh_model)
        project_2d_metric.append(p2d)

        x, y, z, a, b, c = calculate_tra_and_rot(gt_pose, pre_pose, add)
        x_error.append(x)
        y_error.append(y)
        z_error.append(z)
        a_error.append(a)
        b_error.append(b)
        c_error.append(c)
        if add == True:
            t_error, R_error = cal_Rt_metric(gt_pose, pre_pose, add)
            t_err.append(t_error)
            R_err.append(R_error)
    # np.savetxt(r"F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_ransac_andOp_select.txt", global_pose, fmt='%.6f')
    add_mean = np.mean(ADD)
    cm_degree_metric_mean = np.mean(cm_degree_metric)
    project_2d_metric_mean = np.mean(project_2d_metric)
    x_error_mean = np.mean(x_error)
    y_error_mean = np.mean(y_error)
    z_error_mean = np.mean(z_error)

    a_error_mean = np.mean(a_error)
    b_error_mean = np.mean(b_error)
    c_error_mean = np.mean(c_error)

    t_mean = np.mean(t_err)
    R_mean = np.mean(R_err)
    print("ADD: ", add_mean)
    print("cm_degree: ", cm_degree_metric_mean)
    print("project_2d: ", project_2d_metric_mean)
    print("x_error_mean: ", x_error_mean)
    print("y_error_mean: ", y_error_mean)
    print("z_error_mean: ", z_error_mean)
    print("a_error_mean: ", a_error_mean)
    print("b_error_mean: ", b_error_mean)
    print("c_error_mean: ", c_error_mean)
    print("t_mean: ", t_mean)
    print("R_mean: ", R_mean)

def CalProjection():
    mesh_model = []
    mesh_model_1 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_01\model\part_01.ply")
    mesh_model_2 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_02\model\part_02.ply")
    mesh_model_3 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_03\model\part_03.ply")
    mesh_model_4 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_04\model\part_04.ply")
    mesh_model_5 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_05\model\part_05.ply")
    mesh_model_6 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_06\model\part_06.ply")
    mesh_model.append(mesh_model_1)
    mesh_model.append(mesh_model_2)
    mesh_model.append(mesh_model_3)
    mesh_model.append(mesh_model_4)
    mesh_model.append(mesh_model_5)
    mesh_model.append(mesh_model_6)
    K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                  [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                  [0.0, 0.0, 1.0]])
    gt_pose = []
    gt_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\gt_Pose_part_01_ours.txt')
    gt_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\gt_Pose_part_02_ours.txt')
    gt_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\gt_Pose_part_03_ours.txt')
    gt_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\gt_Pose_part_04_ours.txt')
    gt_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\gt_Pose_part_05_ours.txt')
    gt_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\gt_Pose_part_06_ours.txt')
    gt_pose.append(gt_poses_1)
    gt_pose.append(gt_poses_2)
    gt_pose.append(gt_poses_3)
    gt_pose.append(gt_poses_4)
    gt_pose.append(gt_poses_5)
    gt_pose.append(gt_poses_6)

    pre_pose = []
    pre_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_pose.append(pre_poses_1)
    pre_pose.append(pre_poses_2)
    pre_pose.append(pre_poses_3)
    pre_pose.append(pre_poses_4)
    pre_pose.append(pre_poses_5)
    pre_pose.append(pre_poses_6)

    for index in range(6):
        num = gt_pose[index].shape[0]
        gt_poses = gt_pose[index]
        pre_poses = pre_pose[index]
        mesh_model_ = mesh_model[index]
        for pixel in range(1,6):
            project_2d_metric = []
            for i in range(num):
                R = gt_poses[i][:9].reshape(3, 3)
                t = gt_poses[i][9:].reshape(3, 1)
                gt_pose_ = np.concatenate([R, t], axis=1)
                R_ = pre_poses[i][:9].reshape(3, 3)
                t_ = pre_poses[i][9:].reshape(3, 1)
                pre_pose_ = np.concatenate([R_, t_], axis=1)
                p2d = projection_2d(pre_pose_, gt_pose_, K, mesh_model_, pixel)
                project_2d_metric.append(p2d)
            print("part", index+1)
            print("pixel", pixel)
            project_2d_metric_mean = np.mean(project_2d_metric)
            print("projection: ", project_2d_metric_mean)
def cal_Rt_metric(pose_pred, pose_targets, add):
    # if add == False:
    #     return 0, 0
    translation_distance = (
        np.linalg.norm(pose_pred[:, 3] - pose_targets[:, 3]) * 1000
    )
    rotation_diff = np.dot(pose_pred[:, :3], pose_targets[:, :3].T)
    trace = np.trace(rotation_diff)
    trace = trace if trace <= 3 else 3
    angular_distance = np.rad2deg(np.arccos((trace - 1.0) / 2.0))

    return translation_distance, angular_distance

def CalRt():
    mesh_model = []
    mesh_model_1 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_01\model\part_01.ply")
    mesh_model_2 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_02\model\part_02.ply")
    mesh_model_3 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_03\model\part_03.ply")
    mesh_model_4 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_04\model\part_04.ply")
    mesh_model_5 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_05\model\part_05.ply")
    mesh_model_6 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_06\model\part_06.ply")
    mesh_model.append(mesh_model_1)
    mesh_model.append(mesh_model_2)
    mesh_model.append(mesh_model_3)
    mesh_model.append(mesh_model_4)
    mesh_model.append(mesh_model_5)
    mesh_model.append(mesh_model_6)
    K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                  [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                  [0.0, 0.0, 1.0]])
    gt_pose = []
    gt_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\gt_Pose_part_01_ours.txt')
    gt_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\gt_Pose_part_02_ours.txt')
    gt_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\gt_Pose_part_03_ours.txt')
    gt_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\gt_Pose_part_04_ours.txt')
    gt_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\gt_Pose_part_05_ours.txt')
    gt_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\gt_Pose_part_06_ours.txt')
    gt_pose.append(gt_poses_1)
    gt_pose.append(gt_poses_2)
    gt_pose.append(gt_poses_3)
    gt_pose.append(gt_poses_4)
    gt_pose.append(gt_poses_5)
    gt_pose.append(gt_poses_6)

    pre_pose = []
    pre_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_pose.append(pre_poses_1)
    pre_pose.append(pre_poses_2)
    pre_pose.append(pre_poses_3)
    pre_pose.append(pre_poses_4)
    pre_pose.append(pre_poses_5)
    pre_pose.append(pre_poses_6)

    # part_01  513.2765
    # part_02  352.7896
    # part_03  262.2528
    # part_04   518.7096
    # part_05  238.095
    # part_06  212.3694
    diameter = []
    diameter.append(513.2765)
    diameter.append(352.7896)
    diameter.append(262.2528)
    diameter.append(518.7096)
    diameter.append(238.095)
    diameter.append(212.3694)

    for index in range(6):
        num = gt_pose[index].shape[0]
        gt_poses = gt_pose[index]
        pre_poses = pre_pose[index]
        mesh_model_ = mesh_model[index]
        diameter_ = diameter[index]
        t_error_ = []
        R_error_ = []
        for i in range(num):
            R = gt_poses[i][:9].reshape(3, 3)
            t = gt_poses[i][9:].reshape(3, 1)
            gt_pose_ = np.concatenate([R, t], axis=1)
            R_ = pre_poses[i][:9].reshape(3, 3)
            t_ = pre_poses[i][9:].reshape(3, 1)
            pre_pose_ = np.concatenate([R_, t_], axis=1)
            add = add_metric(pre_pose_, gt_pose_, diameter_, mesh_model_)
            if add == True:
                t_error, R_error = cal_Rt_metric(pre_pose_, gt_pose_, add)
                t_error_.append(t_error)
                R_error_.append(R_error)
        print("part", index+1)
        t_error_result = np.mean(np.array(t_error_))
        R_error_result = np.mean(np.array(R_error_))
        print("t_error_result: ", t_error_result)
        print("R_error_result: ", R_error_result)

def CalRtSaveError():
    mesh_model = []
    mesh_model_1 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_01\model\part_01.ply")
    mesh_model_2 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_02\model\part_02.ply")
    mesh_model_3 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_03\model\part_03.ply")
    mesh_model_4 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_04\model\part_04.ply")
    mesh_model_5 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_05\model\part_05.ply")
    mesh_model_6 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_06\model\part_06.ply")
    mesh_model.append(mesh_model_1)
    mesh_model.append(mesh_model_2)
    mesh_model.append(mesh_model_3)
    mesh_model.append(mesh_model_4)
    mesh_model.append(mesh_model_5)
    mesh_model.append(mesh_model_6)
    K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                  [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                  [0.0, 0.0, 1.0]])
    gt_pose = []
    gt_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\gt_Pose_part_01_ours.txt')
    gt_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\gt_Pose_part_02_ours.txt')
    gt_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\gt_Pose_part_03_ours.txt')
    gt_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\gt_Pose_part_04_ours.txt')
    gt_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\gt_Pose_part_05_ours.txt')
    gt_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\gt_Pose_part_06_ours.txt')
    gt_pose.append(gt_poses_1)
    gt_pose.append(gt_poses_2)
    gt_pose.append(gt_poses_3)
    gt_pose.append(gt_poses_4)
    gt_pose.append(gt_poses_5)
    gt_pose.append(gt_poses_6)

    pre_pose = []
    pre_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_pose.append(pre_poses_1)
    pre_pose.append(pre_poses_2)
    pre_pose.append(pre_poses_3)
    pre_pose.append(pre_poses_4)
    pre_pose.append(pre_poses_5)
    pre_pose.append(pre_poses_6)

    # part_01  513.2765
    # part_02  352.7896
    # part_03  262.2528
    # part_04   518.7096
    # part_05  238.095
    # part_06  212.3694
    diameter = []
    diameter.append(513.2765)
    diameter.append(352.7896)
    diameter.append(262.2528)
    diameter.append(518.7096)
    diameter.append(238.095)
    diameter.append(212.3694)

    for index in range(6):
        num = gt_pose[index].shape[0]
        gt_poses = gt_pose[index]
        pre_poses = pre_pose[index]
        mesh_model_ = mesh_model[index]
        diameter_ = diameter[index]
        t_error_ = []
        R_error_ = []
        for i in range(num):
            R = gt_poses[i][:9].reshape(3, 3)
            t = gt_poses[i][9:].reshape(3, 1)
            gt_pose_ = np.concatenate([R, t], axis=1)
            R_ = pre_poses[i][:9].reshape(3, 3)
            t_ = pre_poses[i][9:].reshape(3, 1)
            pre_pose_ = np.concatenate([R_, t_], axis=1)
            add = add_metric(pre_pose_, gt_pose_, diameter_, mesh_model_)
            t_error, R_error = cal_Rt_metric(pre_pose_, gt_pose_, add)
            t_error_.append(t_error)
            R_error_.append(R_error)

        print("part", index+1)
        np.savetxt(os.path.join(r"F:\Data_pose\00QEL_Assembly_Datasets\error_save", str(index+1) + "_ours++_R.txt"), R_error_, fmt='%.6f')
        np.savetxt(os.path.join(r"F:\Data_pose\00QEL_Assembly_Datasets\error_save", str(index+1) + "_ours++_t.txt"), t_error_, fmt='%.6f')

def CalADD():
    mesh_model = []
    mesh_model_1 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_01\model\part_01.ply")
    mesh_model_2 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_02\model\part_02.ply")
    mesh_model_3 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_03\model\part_03.ply")
    mesh_model_4 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_04\model\part_04.ply")
    mesh_model_5 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_05\model\part_05.ply")
    mesh_model_6 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_06\model\part_06.ply")
    mesh_model.append(mesh_model_1)
    mesh_model.append(mesh_model_2)
    mesh_model.append(mesh_model_3)
    mesh_model.append(mesh_model_4)
    mesh_model.append(mesh_model_5)
    mesh_model.append(mesh_model_6)
    K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                  [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                  [0.0, 0.0, 1.0]])
    gt_pose = []
    gt_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\gt_Pose_part_01_ours.txt')
    gt_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\gt_Pose_part_02_ours.txt')
    gt_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\gt_Pose_part_03_ours.txt')
    gt_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\gt_Pose_part_04_ours.txt')
    gt_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\gt_Pose_part_05_ours.txt')
    gt_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\gt_Pose_part_06_ours.txt')
    gt_pose.append(gt_poses_1)
    gt_pose.append(gt_poses_2)
    gt_pose.append(gt_poses_3)
    gt_pose.append(gt_poses_4)
    gt_pose.append(gt_poses_5)
    gt_pose.append(gt_poses_6)

    pre_pose = []
    pre_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_pose.append(pre_poses_1)
    pre_pose.append(pre_poses_2)
    pre_pose.append(pre_poses_3)
    pre_pose.append(pre_poses_4)
    pre_pose.append(pre_poses_5)
    pre_pose.append(pre_poses_6)

    # part_01  513.2765
    # part_02  352.7896
    # part_03  262.2528
    # part_04   518.7096
    # part_05  238.095
    # part_06  212.3694
    diameter = []
    diameter.append(513.2765)
    diameter.append(352.7896)
    diameter.append(262.2528)
    diameter.append(518.7096)
    diameter.append(238.095)
    diameter.append(212.3694)

    for index in range(6):
        num = gt_pose[index].shape[0]
        gt_poses = gt_pose[index]
        pre_poses = pre_pose[index]
        mesh_model_ = mesh_model[index]
        diameter_ = diameter[index]
        ADD_ = []
        for i in range(num):
            R = gt_poses[i][:9].reshape(3, 3)
            t = gt_poses[i][9:].reshape(3, 1)
            gt_pose_ = np.concatenate([R, t], axis=1)
            R_ = pre_poses[i][:9].reshape(3, 3)
            t_ = pre_poses[i][9:].reshape(3, 1)
            pre_pose_ = np.concatenate([R_, t_], axis=1)
            add = add_metric(pre_pose_, gt_pose_, diameter_, mesh_model_)
            ADD_.append(add)
        print("part", index+1)
        ADD_result = np.mean(np.array(ADD_))
        print("ADD_result: ", ADD_result)
def CalCmDegree():
    mesh_model = []
    mesh_model_1 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_01\model\part_01.ply")
    mesh_model_2 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_02\model\part_02.ply")
    mesh_model_3 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_03\model\part_03.ply")
    mesh_model_4 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_04\model\part_04.ply")
    mesh_model_5 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_05\model\part_05.ply")
    mesh_model_6 = load_ply(r"F:\Data_pose\00QEL_Assembly_Datasets\part_06\model\part_06.ply")
    mesh_model.append(mesh_model_1)
    mesh_model.append(mesh_model_2)
    mesh_model.append(mesh_model_3)
    mesh_model.append(mesh_model_4)
    mesh_model.append(mesh_model_5)
    mesh_model.append(mesh_model_6)
    K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                  [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                  [0.0, 0.0, 1.0]])
    gt_pose = []
    gt_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\gt_Pose_part_01_ours.txt')
    gt_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\gt_Pose_part_02_ours.txt')
    gt_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\gt_Pose_part_03_ours.txt')
    gt_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\gt_Pose_part_04_ours.txt')
    gt_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\gt_Pose_part_05_ours.txt')
    gt_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\gt_Pose_part_06_ours.txt')
    gt_pose.append(gt_poses_1)
    gt_pose.append(gt_poses_2)
    gt_pose.append(gt_poses_3)
    gt_pose.append(gt_poses_4)
    gt_pose.append(gt_poses_5)
    gt_pose.append(gt_poses_6)

    pre_pose = []
    pre_poses_1 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_01\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_2 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_02\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_3 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_03\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_4 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_04\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_5 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_05\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_poses_6 = np.loadtxt(r'F:\Data_pose\00QEL_Assembly_Datasets\part_06\pose\pre_pose_with_ransac_new_2_opti.txt')
    pre_pose.append(pre_poses_1)
    pre_pose.append(pre_poses_2)
    pre_pose.append(pre_poses_3)
    pre_pose.append(pre_poses_4)
    pre_pose.append(pre_poses_5)
    pre_pose.append(pre_poses_6)

    # part_01  513.2765
    # part_02  352.7896
    # part_03  262.2528
    # part_04   518.7096
    # part_05  238.095
    # part_06  212.3694
    diameter = []
    diameter.append(513.2765)
    diameter.append(352.7896)
    diameter.append(262.2528)
    diameter.append(518.7096)
    diameter.append(238.095)
    diameter.append(212.3694)

    for index in range(6):
        num = gt_pose[index].shape[0]
        gt_poses = gt_pose[index]
        pre_poses = pre_pose[index]
        mesh_model_ = mesh_model[index]
        diameter_ = diameter[index]
        cm_degree_ = []
        for i in range(num):
            R = gt_poses[i][:9].reshape(3, 3)
            t = gt_poses[i][9:].reshape(3, 1)
            gt_pose_ = np.concatenate([R, t], axis=1)
            R_ = pre_poses[i][:9].reshape(3, 3)
            t_ = pre_poses[i][9:].reshape(3, 1)
            pre_pose_ = np.concatenate([R_, t_], axis=1)
            cm_degree = cm_degree_5_metric(pre_pose_, gt_pose_)
            cm_degree_.append(cm_degree)
        print("part", index+1)
        cm_degree_result = np.mean(np.array(cm_degree_))
        print("cm_degree_result: ", cm_degree_result)

def CalADDMono6D():
    mesh_model = []
    mesh_model_1 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\Bracket\model\Bracket.ply")
    mesh_model_2 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\HingeBase\model\HingeBase.ply")
    mesh_model_3 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\L-Holder\model\L-Holder.ply")
    mesh_model_4 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\Stopper\model\Stopper.ply")
    mesh_model_5 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\T-Holder\model\T-Holder.ply")

    mesh_model.append(mesh_model_1)
    mesh_model.append(mesh_model_2)
    mesh_model.append(mesh_model_3)
    mesh_model.append(mesh_model_4)
    mesh_model.append(mesh_model_5)

    K = np.array([[2209.878296, 0.0, 349.751312],
                  [0.0, 2210.376676, 254.828051],
                  [0.0, 0.0, 1.0]])

    gt_pose = []
    gt_poses_1 = np.loadtxt(r'F:\Data_pose\Mono6D\Bracket\pose\gt_Pose_Bracket_ours.txt')
    gt_poses_2 = np.loadtxt(r'F:\Data_pose\Mono6D\HingeBase\pose\gt_Pose_HingeBase_ours.txt')
    gt_poses_3 = np.loadtxt(r'F:\Data_pose\Mono6D\L-Holder\pose\gt_Pose_L-Holder_ours.txt')
    gt_poses_4 = np.loadtxt(r'F:\Data_pose\Mono6D\Stopper\pose\gt_Pose_Stopper_ours.txt')
    gt_poses_5 = np.loadtxt(r'F:\Data_pose\Mono6D\T-Holder\pose\gt_Pose_T-Holder_ours.txt')

    gt_pose.append(gt_poses_1)
    gt_pose.append(gt_poses_2)
    gt_pose.append(gt_poses_3)
    gt_pose.append(gt_poses_4)
    gt_pose.append(gt_poses_5)

    pre_pose = []
    pre_poses_1 = np.loadtxt(r'F:\Data_pose\Mono6D\Bracket\pose\pre_Pose_Bracket_contourpose.txt')
    pre_poses_2 = np.loadtxt(r'F:\Data_pose\Mono6D\HingeBase\pose\pre_Pose_HingeBase_contourpose.txt')
    pre_poses_3 = np.loadtxt(r'F:\Data_pose\Mono6D\L-Holder\pose\pre_Pose_L-Holder_contourpose.txt')
    pre_poses_4 = np.loadtxt(r'F:\Data_pose\Mono6D\Stopper\pose\pre_Pose_Stopper_contourpose.txt')
    pre_poses_5 = np.loadtxt(r'F:\Data_pose\Mono6D\T-Holder\pose\pre_Pose_T-Holder_contourpose.txt')

    pre_pose.append(pre_poses_1)
    pre_pose.append(pre_poses_2)
    pre_pose.append(pre_poses_3)
    pre_pose.append(pre_poses_4)
    pre_pose.append(pre_poses_5)

    # Bracket  79.737
    # HingeBase 81.240385
    # L-Holder 76.3413387
    # Stopper 117.0897
    # T-Holder  77.62087

    diameter = []
    diameter.append(79.737)
    diameter.append(81.240385)
    diameter.append(76.3413387)
    diameter.append(117.0897)
    diameter.append(77.62087)

    for index in range(5):
        num = gt_pose[index].shape[0]
        gt_poses = gt_pose[index]
        pre_poses = pre_pose[index]
        mesh_model_ = mesh_model[index]
        diameter_ = diameter[index]
        ADD_ = []
        for i in range(num):
            R = gt_poses[i][:9].reshape(3, 3)
            t = gt_poses[i][9:].reshape(3, 1)
            gt_pose_ = np.concatenate([R, t], axis=1)
            R_ = pre_poses[i][:9].reshape(3, 3)
            t_ = pre_poses[i][9:].reshape(3, 1)
            pre_pose_ = np.concatenate([R_, t_], axis=1)
            add = add_metric(pre_pose_, gt_pose_, diameter_, mesh_model_)
            ADD_.append(add)
        print("obj： ", index+1)
        ADD_result = np.mean(np.array(ADD_))
        print("ADD_result: ", ADD_result)

def CalRtMono6D():
    mesh_model = []
    mesh_model_1 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\Bracket\model\Bracket.ply")
    mesh_model_2 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\HingeBase\model\HingeBase.ply")
    mesh_model_3 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\L-Holder\model\L-Holder.ply")
    mesh_model_4 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\Stopper\model\Stopper.ply")
    mesh_model_5 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\T-Holder\model\T-Holder.ply")

    mesh_model.append(mesh_model_1)
    mesh_model.append(mesh_model_2)
    mesh_model.append(mesh_model_3)
    mesh_model.append(mesh_model_4)
    mesh_model.append(mesh_model_5)

    K = np.array([[2209.878296, 0.0, 349.751312],
                  [0.0, 2210.376676, 254.828051],
                  [0.0, 0.0, 1.0]])

    gt_pose = []
    gt_poses_1 = np.loadtxt(r'F:\Data_pose\Mono6D\Bracket\pose\gt_Pose_Bracket_ours.txt')
    gt_poses_2 = np.loadtxt(r'F:\Data_pose\Mono6D\HingeBase\pose\gt_Pose_HingeBase_ours.txt')
    gt_poses_3 = np.loadtxt(r'F:\Data_pose\Mono6D\L-Holder\pose\gt_Pose_L-Holder_ours.txt')
    gt_poses_4 = np.loadtxt(r'F:\Data_pose\Mono6D\Stopper\pose\gt_Pose_Stopper_ours.txt')
    gt_poses_5 = np.loadtxt(r'F:\Data_pose\Mono6D\T-Holder\pose\gt_Pose_T-Holder_ours.txt')

    gt_pose.append(gt_poses_1)
    gt_pose.append(gt_poses_2)
    gt_pose.append(gt_poses_3)
    gt_pose.append(gt_poses_4)
    gt_pose.append(gt_poses_5)

    pre_pose = []
    pre_poses_1 = np.loadtxt(r'F:\Data_pose\Mono6D\Bracket\pose\pre_pose_with_optimization_adaptive_3.txt')
    pre_poses_2 = np.loadtxt(r'F:\Data_pose\Mono6D\HingeBase\pose\pre_pose_with_optimization_adaptive.txt')
    pre_poses_3 = np.loadtxt(r'F:\Data_pose\Mono6D\L-Holder\pose\pre_pose_with_optimization_adaptive.txt')
    pre_poses_4 = np.loadtxt(r'F:\Data_pose\Mono6D\Stopper\pose\pre_pose_with_optimization_adaptive_2.txt')
    pre_poses_5 = np.loadtxt(r'F:\Data_pose\Mono6D\T-Holder\pose\pre_pose_with_optimization_adaptive.txt')

    pre_pose.append(pre_poses_1)
    pre_pose.append(pre_poses_2)
    pre_pose.append(pre_poses_3)
    pre_pose.append(pre_poses_4)
    pre_pose.append(pre_poses_5)

    # Bracket  79.737
    # HingeBase 81.240385
    # L-Holder 76.3413387
    # Stopper 117.0897
    # T-Holder  77.62087

    diameter = []
    diameter.append(79.737)
    diameter.append(81.240385)
    diameter.append(76.3413387)
    diameter.append(117.0897)
    diameter.append(77.62087)

    for index in range(5):
        num = gt_pose[index].shape[0]
        gt_poses = gt_pose[index]
        pre_poses = pre_pose[index]
        mesh_model_ = mesh_model[index]
        diameter_ = diameter[index]
        ADD_ = []
        x_error = []
        y_error = []
        z_error = []
        a_error = []
        b_error = []
        c_error = []
        for i in range(num):
            R = gt_poses[i][:9].reshape(3, 3)
            t = gt_poses[i][9:].reshape(3, 1)
            gt_pose_ = np.concatenate([R, t], axis=1)
            R_ = pre_poses[i][:9].reshape(3, 3)
            t_ = pre_poses[i][9:].reshape(3, 1)
            pre_pose_ = np.concatenate([R_, t_], axis=1)
            add = add_metric(pre_pose_, gt_pose_, diameter_, mesh_model_)
            ADD_.append(add)
            if add == True:
                x, y, z, a, b, c = calculate_tra_and_rot(gt_pose_, pre_pose_, add)
                x_error.append(x)
                y_error.append(y)
                z_error.append(z)
                a_error.append(a)
                b_error.append(b)
                c_error.append(c)
        x_error_mean = np.mean(x_error)
        y_error_mean = np.mean(y_error)
        z_error_mean = np.mean(z_error)
        a_error_mean = np.mean(a_error)
        b_error_mean = np.mean(b_error)
        c_error_mean = np.mean(c_error)
        print("obj： ", index+1)
        # ADD_result = np.mean(np.array(ADD_))
        print("x_error_mean: ", x_error_mean)
        print("y_error_mean: ", y_error_mean)
        print("z_error_mean: ", z_error_mean)
        print("a_error_mean: ", a_error_mean)
        print("b_error_mean: ", b_error_mean)
        print("c_error_mean: ", c_error_mean)
def CalMetricMono6D():
    mesh_model = []
    mesh_model_1 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\Bracket\model\Bracket.ply")
    mesh_model_2 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\HingeBase\model\HingeBase.ply")
    mesh_model_3 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\L-Holder\model\L-Holder.ply")
    mesh_model_4 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\Stopper\model\Stopper.ply")
    mesh_model_5 = load_ply(r"E:\MetalDataset\Mono-6D\Re_modify\T-Holder\model\T-Holder.ply")

    mesh_model.append(mesh_model_1)
    mesh_model.append(mesh_model_2)
    mesh_model.append(mesh_model_3)
    mesh_model.append(mesh_model_4)
    mesh_model.append(mesh_model_5)

    K = np.array([[2209.878296, 0.0, 349.751312],
                  [0.0, 2210.376676, 254.828051],
                  [0.0, 0.0, 1.0]])

    gt_pose = []
    gt_poses_1 = np.loadtxt(r'F:\Data_pose\Mono6D\Bracket\pose\gt_Pose_Bracket_ours.txt')
    gt_poses_2 = np.loadtxt(r'F:\Data_pose\Mono6D\HingeBase\pose\gt_Pose_HingeBase_ours.txt')
    gt_poses_3 = np.loadtxt(r'F:\Data_pose\Mono6D\L-Holder\pose\gt_Pose_L-Holder_ours.txt')
    gt_poses_4 = np.loadtxt(r'F:\Data_pose\Mono6D\Stopper\pose\gt_Pose_Stopper_ours.txt')
    gt_poses_5 = np.loadtxt(r'F:\Data_pose\Mono6D\T-Holder\pose\gt_Pose_T-Holder_ours.txt')

    gt_pose.append(gt_poses_1)
    gt_pose.append(gt_poses_2)
    gt_pose.append(gt_poses_3)
    gt_pose.append(gt_poses_4)
    gt_pose.append(gt_poses_5)

    pre_pose = []
    pre_poses_1 = np.loadtxt(r'F:\Data_pose\Mono6D\Bracket\pose\pre_Pose_Bracket_induspose_train.txt')
    pre_poses_2 = np.loadtxt(r'F:\Data_pose\Mono6D\HingeBase\pose\pre_Pose_HingeBase_induspose_train.txt')
    pre_poses_3 = np.loadtxt(r'F:\Data_pose\Mono6D\L-Holder\pose\pre_Pose_L-Holder_induspose_train.txt')
    pre_poses_4 = np.loadtxt(r'F:\Data_pose\Mono6D\Stopper\pose\pre_Pose_Stopper_induspose_train.txt')
    pre_poses_5 = np.loadtxt(r'F:\Data_pose\Mono6D\T-Holder\pose\pre_Pose_T-Holder_induspose_train.txt')

    pre_pose.append(pre_poses_1)
    pre_pose.append(pre_poses_2)
    pre_pose.append(pre_poses_3)
    pre_pose.append(pre_poses_4)
    pre_pose.append(pre_poses_5)

    # Bracket  79.737
    # HingeBase 81.240385
    # L-Holder 76.3413387
    # Stopper 117.0897
    # T-Holder  77.62087

    diameter = []
    diameter.append(79.737)
    diameter.append(81.240385)
    diameter.append(76.3413387)
    diameter.append(117.0897)
    diameter.append(77.62087)

    for index in range(5):
        num = gt_pose[index].shape[0]
        gt_poses = gt_pose[index]
        pre_poses = pre_pose[index]
        mesh_model_ = mesh_model[index]
        diameter_ = diameter[index]
        ADD_ = []
        project_2d_metric = []
        cm_degree_ = []
        x_error = []
        y_error = []
        z_error = []
        a_error = []
        b_error = []
        c_error = []
        for i in range(num):
            R = gt_poses[i][:9].reshape(3, 3)
            t = gt_poses[i][9:].reshape(3, 1)
            gt_pose_ = np.concatenate([R, t], axis=1)
            R_ = pre_poses[i][:9].reshape(3, 3)
            t_ = pre_poses[i][9:].reshape(3, 1)
            pre_pose_ = np.concatenate([R_, t_], axis=1)
            add = add_metric(pre_pose_, gt_pose_, diameter_, mesh_model_)
            ADD_.append(add)
            p2d = projection_2d(pre_pose_, gt_pose_, K, mesh_model_, 5)
            project_2d_metric.append(p2d)
            cm_degree = cm_degree_5_metric(pre_pose_, gt_pose_)
            cm_degree_.append(cm_degree)

            if add == True:
                x, y, z, a, b, c = calculate_tra_and_rot(gt_pose_, pre_pose_, add)
                x_error.append(x)
                y_error.append(y)
                z_error.append(z)
                a_error.append(a)
                b_error.append(b)
                c_error.append(c)


        print("obj： ", index+1)
        ADD_result = np.mean(np.array(ADD_))
        projection_2d_result = np.mean(np.array(project_2d_metric))
        cm_degree_result = np.mean(np.array(cm_degree_))
        print("ADD_result: ", ADD_result)
        print("projection_2d_result: ", projection_2d_result)
        print("cm_degree_result: ", cm_degree_result)
        x_error_mean = np.mean(x_error)
        y_error_mean = np.mean(y_error)
        z_error_mean = np.mean(z_error)
        a_error_mean = np.mean(a_error)
        b_error_mean = np.mean(b_error)
        c_error_mean = np.mean(c_error)
        # ADD_result = np.mean(np.array(ADD_))
        print("x_error_mean: ", x_error_mean)
        print("y_error_mean: ", y_error_mean)
        print("z_error_mean: ", z_error_mean)
        print("a_error_mean: ", a_error_mean)
        print("b_error_mean: ", b_error_mean)
        print("c_error_mean: ", c_error_mean)

if __name__ == '__main__':

    # Mono6D()
    # Qel6D()
    # CalProjection()
    # CalRt()
    # CalADD()
    # CalCmDegree()

    # CalADDMono6D()
    # CalRtMono6D()

    # CalRtSaveError()

    CalMetricMono6D()


