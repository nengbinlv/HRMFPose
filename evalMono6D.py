import torch
import tqdm
import numpy as np
import os
import os.path as osp

from numpy.matlib import empty

from utils.utils import load_ply, project
import cv2
from scipy import spatial
from itertools import combinations
from random import choice
import math
from config import rtDic, diameters
# from visPose.visualize_pose import visualizeById_2
import itertools
from sklearn.neighbors import KernelDensity


cuda = torch.cuda.is_available()

indexs = []
index = 0
index_2 = 0
for idx in range(701, 923):
    idx = int(idx)
    indexs.append(idx)

count = os.listdir(osp.join(r"E:\MetalDataset\Mono-6D\Re_modify\Bracket", "photo_cut", "val"))
img_path = osp.join(r"E:\MetalDataset\Mono-6D\Re_modify\Bracket", "photo_cut", "val")
test_inds = [ind.replace(".png", "") for ind in count]
test_inds.sort()
img_path_list = []
for idx in test_inds:
    # idx = int(idx)
    img_name = "{}.png".format(idx)
    img_path_list.append(img_name)
g_id = 0

class evaluator:
    def __init__(self, args, model, test_loader, device):
        self.args = args
        self.model = model
        self.cad_path = osp.join(os.getcwd(), "cad")
        self.mesh_model = load_ply(
            osp.join(
                self.cad_path, "{}.ply".format(args.class_type)
            )
        )
        self.keyponits = np.loadtxt(os.path.join(os.getcwd(), "keypoints/{}.txt".format(args.class_type))) / 1000
        self.device = device
        self.pts_3d = self.mesh_model["pts"]  # * 1000
        self.data_loader = test_loader

        self.index = 1
        self.threshold = args.threshold
        self.proj_2d = []
        self.proj_2d_mean = []
        self.add = []
        self.x_error_all = []
        self.y_error_all = []
        self.z_error_all = []
        self.alpha_error_all = []
        self.beta_error_all = []
        self.gama_error_all = []
        # ablation
        # self.error_num = 0
        self.diameter = diameters[args.class_type]

        # for PECP
        example = ""
        for i in range(0, self.keyponits.shape[0]):
            if i < 10:
                example = example + str(i)
            if i >= 10:
                example = example + chr(97 + i - 10)
        self.list_all = list(combinations(example, 4))

    def grid_voting(self, points, grid_size=3.0):
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        max_votes = -1
        best_center = None

        for x in np.arange(x_min, x_max, grid_size):
            for y in np.arange(y_min, y_max, grid_size):
                count = sum(1 for px, py in points
                            if x <= px < x + grid_size and y <= py < y + grid_size)
                if count > max_votes:
                    max_votes = count
                    best_center = (x + grid_size / 2, y + grid_size / 2)
        return best_center

    def kde_voting(self, points, bandwidth=0.5):
        points_arr = np.array(points)
        # 提取 x 和 y 的范围
        x_min, y_min = points_arr.min(axis=0)
        x_max, y_max = points_arr.max(axis=0)

        # 生成独立的 x 和 y 网格向量
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)

        # 生成二维网格
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T

        # 核密度估计
        kde = KernelDensity(bandwidth=bandwidth).fit(points_arr)
        log_density = kde.score_samples(grid_points)
        max_idx = np.argmax(log_density)
        return tuple(grid_points[max_idx])

    def heatmap_mu_cov(self, heatmap, img):
        # 转换为彩色图像（假设原热图是单通道）
        heatmap_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        # 设定概率阈值（根据数据分布调整）
        threshold = 0.2
        c, _,_ = heatmap.shape
        points_2d = []

        for k in range(c):
            # 创建掩膜（mask），筛选有效区域
            mask = heatmap[k] > threshold
            # 计算仅考虑有效区域的权重和坐标
            if np.any(mask):
                # 有效区域的权重总和
                total_weight = np.sum(heatmap[k][mask])

                # 生成网格坐标（仅考虑有效区域）
                rows, cols = heatmap[k].shape
                y_coords, x_coords = np.mgrid[0:rows, 0:cols]
                x_valid = x_coords[mask]
                y_valid = y_coords[mask]
                heatmap_valid = heatmap[k][mask]

                # 计算均值坐标
                mu_x = np.sum(x_valid * heatmap_valid) / total_weight
                mu_y = np.sum(y_valid * heatmap_valid) / total_weight

                # 计算协方差矩阵
                x_diff = x_valid - mu_x
                y_diff = y_valid - mu_y
                cov_xx = np.sum(x_diff ** 2 * heatmap_valid) / total_weight
                cov_yy = np.sum(y_diff ** 2 * heatmap_valid) / total_weight
                cov_xy = np.sum(x_diff * y_diff * heatmap_valid) / total_weight
                cov_matrix = np.array([[cov_xx, cov_xy], [cov_xy, cov_yy]])

                # 特征值分解（确定椭圆方向与轴长）
                eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
                angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))

                # 置信椭圆缩放因子（例如 95% 置信度对应卡方临界值 5.991）
                confidence_level = 5.991  # 卡方分布 95% 置信度（2自由度）
                scale = np.sqrt(confidence_level)

                # 椭圆轴长
                major_axis = scale * np.sqrt(eigenvalues[0])
                minor_axis = scale * np.sqrt(eigenvalues[1])

                # 绘制关键点（红色圆点）
                cv2.circle(heatmap_color, (int(mu_x), int(mu_y)), 1, (0, 0, 255), -1)
                cv2.putText(heatmap_color, str(k), (int(mu_x), int(mu_y)), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                            (0, 255, 255), 1,
                            cv2.LINE_AA)
                # 绘制置信椭圆（绿色）
                cv2.ellipse(
                    heatmap_color,
                    center=(int(mu_x), int(mu_y)),
                    axes=(int(major_axis), int(minor_axis)),
                    angle=angle,
                    startAngle=0,
                    endAngle=360,
                    color=(0, 255, 0),
                    thickness=1
                )
                points_2d.append((int(mu_x), int(mu_y)))
            else:
                points_2d.append((int(0), int(0)))
        # 显示结果
        cv2.imshow("Keypoint with Uncertainty", heatmap_color)
        cv2.waitKey(1)
        # cv2.destroyAllWindows()

        return points_2d
    def evaluate(self):
        self.model.eval()

        with torch.no_grad():
            global_pose =[]
            gt_pose_save = []
            for data in tqdm.tqdm(self.data_loader, leave=False, desc="val"):
                if cuda:
                    # img, heatmap, K, pose, gray_edge = [x.to(self.device) for x in data]
                    img, heatmap, K, pose = [x.to(self.device) for x in data]
                else:
                    # img, heatmap, K, pose, gray_edge  = data
                    img, heatmap, K, pose = data

                # pred_heatmap, pred_contour, pre_vis, pre_grah, pre_mask, = self.model(img, gt_edge = gray_edge)
                pred_heatmap, pred_contour, pre_vis, pre_grah, pre_mask, = self.model(img)
                # pred_heatmap, pred_contour, pre_grah, pre_mask = self.model(img)
                vis_list = []
                score_list = []
                for i in range(8):
                    id_keypoints = "pred_vis_v" + str(i + 1)
                    pred_vis_v = pre_vis[id_keypoints]
                    score, pre = torch.max(pred_vis_v.data, 1)

                    vis_list.append(pre.data.item())
                    score_list.append(score.detach().cpu().numpy())
                    # vis_list.append(1)
                pred_heatmap = pred_heatmap[:, :8, :, :]

                keypoints_2d, predict_2d = self.map_2_points(heatmap, pred_heatmap)
                # keypoints_2d, predict_2d = self.map_2_points_vote(heatmap, pred_heatmap)
                # # 取出
                # pred_contour = pred_contour[3]
                pred_contour_new = pred_contour.detach().cpu().numpy()
                pred_contour_new = pred_contour_new.squeeze()

                img_trans = ((pred_contour_new - pred_contour_new.min()) * (
                            1 / (pred_contour_new.max() - pred_contour_new.min()) * 255)).astype('uint8')
                # ret, binary = cv2.threshold(img_trans, 150, 255, cv2.THRESH_BINARY)
                cv2.imshow("binary", pred_contour_new)
                cv2.waitKey(1)

                # contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                pred_heatmap_new = pred_heatmap.detach().cpu().numpy()
                pred_heatmap_new = pred_heatmap_new.squeeze()

                pre_points2d = self.heatmap_mu_cov(pred_heatmap_new, img_trans)
                pre_points2d = np.array(pre_points2d)

                # 显示
                # cv2.imshow("pred_contour_new",pred_contour_new)
                cv2.imshow("pred_heatmap_new1", pred_heatmap_new[0])
                cv2.imshow("pred_heatmap_new2", pred_heatmap_new[1])
                # cv2.imshow("pred_heatmap_new3", pred_heatmap_new[2])
                # cv2.imshow("pred_heatmap_new4", pred_heatmap_new[3])
                # cv2.waitKey(0)
                img_fused = np.zeros(img_trans.shape, dtype=np.uint8)
                for heatmap in pred_heatmap_new:
                    heatmap_1 = ((heatmap - heatmap.min()) * (
                            1 / (heatmap.max() - heatmap.min()) * 255)).astype('uint8')

                    xx = cv2.addWeighted(heatmap_1, 0.125 * 2, img_trans, 0.125 / 2, 0)
                    img_fused = img_fused + xx

                cv2.imshow("fused", img_fused)
                cv2.waitKey(1)

                global g_id
                img_name = osp.join(img_path, img_path_list[g_id])
                img_input = cv2.imread(img_name)

                cv2.imwrite(
                    osp.join("F:\\Data_pose\\Mono6D\\Bracket", "pre_edge", img_path_list[g_id]),
                    img_trans)

                # gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
                # cv2.drawContours(img_input, contours, -1, (0, 0, 255), 1)
                image_pred = img_input.copy()
                iamge_project = img_input.copy()
                image_vote = img_input.copy()

                predict_2d = predict_2d.detach().cpu()
                predict_2d_new = predict_2d.squeeze()
                predict_2d_new = predict_2d_new.numpy()
                src_RGB = cv2.cvtColor(img_trans, cv2.COLOR_GRAY2BGR)
                image_edge_score = img_trans.astype(np.float32) / 255
                edge_score_list = []
                index = 0
                for point in predict_2d_new:
                    # print("edge value: ",image_edge_score[int(point[1]),int(point[0])])
                    edge_score_list.append(image_edge_score[int(point[1]), int(point[0])])
                    if vis_list[index] == 1:
                        cv2.circle(img_input, (int(point[0]), int(point[1])), 1, (0, 0, 255), 1)
                        cv2.putText(img_input, str(index), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX,
                                    0.3, (0, 0, 255), 1,
                                    cv2.LINE_AA)
                        cv2.circle(src_RGB, (int(point[0]), int(point[1])), 1, (0, 0, 255), 1)

                    else:
                        cv2.circle(img_input, (int(point[0]), int(point[1])), 1, (255, 0, 0), 1)
                        cv2.putText(img_input, str(index), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX,
                                    0.3,
                                    (255, 0, 0), 1,
                                    cv2.LINE_AA)
                        cv2.circle(src_RGB, (int(point[0]), int(point[1])), 1, (255, 0, 0), 1)
                    index = index + 1
                cv2.imshow("src_RGB", src_RGB)
                cv2.waitKey(1)
                # cv2.imwrite("E:\\lnbCode\\ContourPose-main\\data\\test\\scene2\\keypointsResult\\" + img_name,img_input)
                keypoints_2d_new = keypoints_2d.squeeze()
                ii = 0
                for point in keypoints_2d_new:
                    cv2.circle(img_input, (int(point[0]), int(point[1])), 1, (0, 255, 0), 1)
                    cv2.putText(img_input, str(ii), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                                (0, 255, 0), 1,
                                cv2.LINE_AA)
                    ii = ii + 1

                cv2.imshow("img", img_input)
                cv2.waitKey(1)

                pre_mask = pre_mask.detach().cpu().numpy()
                pre_mask = pre_mask.squeeze()
                pre_mask_img = ((pre_mask - pre_mask.min()) * (1 / (pre_mask.max() - pre_mask.min()) * 255)).astype(
                    'uint8')
                pre_mask[pre_mask <= 0] = 0
                pre_mask[pre_mask > 0] = 1

                pred_contour_new[pred_contour_new <= 0] = 0
                pred_contour_new[pred_contour_new > 0] = 1

                cv2.imshow("pre_mask_0_1", pre_mask)
                cv2.waitKey(1)

                cv2.imwrite(
                    osp.join("F:\\Data_pose\\Mono6D\\Bracket", "pre_mask", img_path_list[g_id]),
                    pre_mask_img)

                pre_grah = pre_grah.detach().cpu().numpy()
                pre_grah = pre_grah.reshape((-1, 2, pred_contour_new.shape[0], pred_contour_new.shape[1]))

                num_pts = predict_2d_new.shape[0]
                ret, pre_mask_img = cv2.threshold(pre_mask_img, 150, 255, cv2.THRESH_BINARY)
                pre_mask_img = pre_mask_img / 255
                cv2.imshow("pre_mask_img", pre_mask_img)
                i_edge = 0

                edge_map = {}

                for start_idx in range(0, num_pts - 1):
                    for end_idx in range(start_idx + 1, num_pts):
                        # pred, red
                        start = np.int16(np.round(predict_2d_new[start_idx]))
                        edge_x = pre_grah[i_edge, 0][pre_mask == 1.].mean()
                        edge_y = pre_grah[i_edge, 1][pre_mask == 1.].mean()
                        edge = np.array([edge_x, edge_y])
                        edge_map[(start_idx, end_idx)] = edge
                        end = np.int16(np.round(predict_2d_new[start_idx] + edge))
                        image_pred = cv2.line(img_input, tuple(start), tuple(end), (0, 255, 255), 1)
                        i_edge = i_edge + 1
                        # cv2.imshow("image_pred", image_pred)
                        # cv2.waitKey(0)
                # cv2.imshow("pre_mask_img", pre_mask_img)
                cv2.imshow("image_pred", image_pred)
                cv2.waitKey(1)
                vote_2d = []
                # for k in range(num_pts):
                #     point_2d = predict_2d_new[k]
                #     points_2d_vote = []
                #     for (i, j), value in edge_map.items():
                #         if i != k and j == k:
                #             p_i = predict_2d_new[i]
                #             edge = p_i + value
                #             if vis_list[k] == 0:
                #                 points_2d_vote.append(edge)
                #         if j != k and i == k:
                #             p_i = predict_2d_new[j]
                #             edge = p_i - value
                #             if vis_list[k] == 0:
                #                 points_2d_vote.append(edge)
                #     # for point in points_2d_vote:
                #     # image_vote = cv2.circle(image_vote, (int(point[0]), int(point[1])), 1, (0, 255, 255), 1)
                #     # cv2.imshow("image_vote", image_vote)
                #     # cv2.waitKey(0)
                #     if vis_list[k] == 1:
                #         points_2d_vote.append(point_2d)
                #
                #     if len(points_2d_vote) == 0:
                #         result = point_2d
                #     else:
                #         result = self.grid_voting(points_2d_vote)
                #
                #     # result = (result + point_2d) / 2
                #     # result = self.kde_voting(points_2d_vote)
                #     # vote_2d.append(result)
                #
                #     if vis_list[k] == 0:
                #         vote_2d.append(result)
                #     else:
                #         vote_2d.append(point_2d)
                    # image_vote = cv2.circle(image_vote, (int(result[0]), int(result[1])), 1, (255, 0, 255), 1)
                    # cv2.imshow("image_vote_1", image_vote)
                    # cv2.waitKey(0)

                # vote_2d = np.array(vote_2d)
                # vote_2d = np.expand_dims(vote_2d, axis=0)
                # vote_2d = torch.tensor(vote_2d, dtype=torch.float)

                pre_points2d = np.expand_dims(pre_points2d, axis=0)
                pre_points2d = torch.tensor(pre_points2d, dtype=torch.float)

                pre_pose, gt_pose = self.calculate_metric(keypoints_2d, predict_2d, K)
                # pre_pose = self.calculate_metric(keypoints_2d, vote_2d, K)
                # pre_pose = self.calculate_metric(keypoints_2d, pre_points2d, K)

                # pre_pose = self.calculate_metric_visble(keypoints_2d, predict_2d, K, vis_list, score_list, edge_score_list)

                pp = []
                for x in pre_pose[:, :3]:
                    for y in x:
                        # print(y)
                        pp.append(y)
                for xx in pre_pose[:, 3]:
                    pp.append(xx)
                global_pose.append(pp)

                pp_gt = []
                for x in gt_pose[:, :3]:
                    for y in x:
                        # print(y)
                        pp_gt.append(y)
                for xx in gt_pose[:, 3]:
                    pp_gt.append(xx)
                gt_pose_save.append(pp_gt)

                g_id = g_id + 1

                #根据位姿重新投影关键点
                k = K.detach().cpu().numpy()
                k = k.squeeze()
                pts2d = project(self.keyponits, k, pre_pose)
                pts2d = pts2d.squeeze()

                ii = 0
                for point in pts2d:
                    cv2.circle(iamge_project, (int(point[0]), int(point[1])), 1, (0, 255, 0), 1)
                    cv2.putText(iamge_project, str(ii), (int(point[0]), int(point[1])), cv2.FONT_HERSHEY_COMPLEX, 0.3,
                                (0, 255, 0), 1,
                                cv2.LINE_AA)
                    ii = ii + 1
                cv2.imshow("iamge_project", iamge_project)
                cv2.waitKey(1)

            # np.savetxt(r"F:\Data_pose\Mono6D\Bracket\pose\pre_Pose_Bracket_ours_2.txt", global_pose, fmt='%.6f')
            # np.savetxt(r"F:\Data_pose\Mono6D\HingeBase\pose\gt_Pose_HingeBase_ours.txt", gt_pose_save,
            #            fmt='%.6f')
                # self.calculate_metric_PECP(keypoints_2d, predict_2d, K, pose, pred_contour)
            for i in range(len(self.add)):
                if self.add[i] == 0:
                    print(img_path_list[i])
            proj_2d_mean = np.mean(self.proj_2d)
            add_mean = np.mean(self.add)
            x = np.mean(self.x_error_all)
            y = np.mean(self.y_error_all)
            z = np.mean(self.z_error_all)
            alpha = np.mean(self.alpha_error_all)
            beta = np.mean(self.beta_error_all)
            gamma = np.mean(self.gama_error_all)
            print("model class type:{}:2D- {}  ADD-{}".format(self.args.class_type, proj_2d_mean, add_mean))
            print('x error:{} mm, y error:{} mm, z error:{} mm'.format(x, y, z))
            print("translation error:{} mm".format((x ** 2 + y ** 2 + z ** 2) ** 0.5))
            print('alpha error:{} °, beta error:{} °, gamaa error:{} °'.format(alpha, beta, gamma))
            print("rotation error:{} mm".format((alpha ** 2 + beta ** 2 + gamma ** 2) ** 0.5))

    def map_2_points(self, heatmap, pred_heatmap):

        def extract_coords(input_map):
            flat_map = input_map.view(input_map.shape[0], input_map.shape[1], -1)
            max_idx = torch.argmax(flat_map, dim=2)
            width = input_map.shape[3]
            x = (max_idx / width).int().unsqueeze(dim=2)
            y = (max_idx % width).unsqueeze(dim=2)
            return torch.cat((y, x), dim=2)

        gt_points = extract_coords(heatmap)
        pred_points = extract_coords(pred_heatmap)
        return gt_points, pred_points

    def map_2_points_vote(self, heatmap, pred_heatmap):
        # 通过计算坐标权重来求最终过的图像坐标
        def get_keypoints_from_heatmap(heatmap):
            """
            从热图中获取关键点的坐标，基于高斯分布的期望值。
            参数：
            heatmap：预测的热图，形状为 (batch_size, num_keypoints, height, width)
            返回：
            keypoints：关键点的坐标，形状为 (batch_size, num_keypoints, 2)
            """
            batch_size, num_keypoints, height, width = heatmap.shape
            # 创建坐标网格
            x_coords = torch.arange(width, device=heatmap.device)
            y_coords = torch.arange(height, device=heatmap.device)
            x_grid, y_grid = torch.meshgrid(y_coords, x_coords)
            # 初始化关键点坐标
            keypoints = torch.zeros((batch_size, num_keypoints, 2), device=heatmap.device)
            for b in range(batch_size):
                for c in range(num_keypoints):
                    # 获取当前关键点的热图
                    current_heatmap = heatmap[b, c, :, :]
                    # 计算坐标权重
                    x_weighted = current_heatmap * x_grid
                    y_weighted = current_heatmap * y_grid
                    # 计算期望坐标
                    heatmap_sum = current_heatmap.sum()
                    if heatmap_sum == 0:
                        # 如果热图总和为零，返回零坐标
                        x_expectation = 0
                        y_expectation = 0
                    else:
                        x_expectation = x_weighted.sum() / heatmap_sum
                        y_expectation = y_weighted.sum() / heatmap_sum
                    # 存储关键点坐标
                    keypoints[b, c, 0] = x_expectation
                    keypoints[b, c, 1] = y_expectation
            return keypoints
        gt_points = get_keypoints_from_heatmap(heatmap)
        pred_points = get_keypoints_from_heatmap(pred_heatmap)
        return gt_points, pred_points

    def calculate_tra_and_rot(self, pose, pred_pose):
        if self.add[-1] == False:
            return 0
        pred_pose = self.pose_reverse(pred_pose, pose)
        rot = pose[:, :3]
        tra = pose[:, 3:].reshape(1, 3)
        pred_rot = pred_pose[:, :3]
        pred_tra = pred_pose[:, 3:].reshape(1, 3)
        tra_error = (tra - pred_tra) * 1000

        x_error = math.fabs(tra_error[:, 0])
        y_error = math.fabs(tra_error[:, 1])
        z_error = math.fabs(tra_error[:, 2])

        self.x_error_all.append(x_error)
        self.y_error_all.append(y_error)
        self.z_error_all.append(z_error)

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

        self.alpha_error_all.append(alpha_error)
        self.beta_error_all.append(beta_error)
        self.gama_error_all.append(gamma_error)

    def calculate_metric(self, keypoints2d, predict2d, K):

        batch_size = keypoints2d.shape[0]
        keypoints_num = keypoints2d.shape[1]

        for i in range(batch_size):
            keypoints = keypoints2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            predict = predict2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            #self.set_error_points(predict)
            k = K[i].detach().cpu().numpy()
            gt_pose = self.pnp(self.keyponits, keypoints, k)
            pred_pose = self.pnp(self.keyponits, predict, k)

            # render_img = visualizeById_2("qel",pred_pose)
            # cv2.imshow("render_img", render_img)
            # global  index_2
            # idx = indexs[index_2]
            # img_name = "{}.jpg".format(idx, 1)
            # img_input = cv2.imread("E:\\lnbCode\\ContourPose-main\\data\\test\\scene2\\photo_cut\\" + img_name)
            # index_2 = index_2 + 1
            # edge_img = cv2.Canny(render_img, 150, 200, 5, L2gradient=True)
            # # element = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
            # # edge_img = cv2.dilate(edge_img, element)
            # edge_final = edge_img
            # mask_inv = cv2.bitwise_not(edge_final)
            # edge_final = cv2.cvtColor(edge_final, cv2.COLOR_GRAY2BGR)
            #
            # edge_final[:, :, 0] = 0
            # edge_final[edge_final > 0] = 150
            # edge_final[:, :, 2] = 0
            # background_img = cv2.bitwise_and(img_input, img_input, mask=mask_inv)
            # last_img = cv2.add(edge_final, background_img)
            # cv2.imshow("last_img", last_img)
            # cv2.imwrite("E:\\lnbCode\\ContourPose-main\\data\\test\\scene2\\visResult\\" + img_name, last_img)
            # cv2.waitKey(1)

            self.projection_2d(pred_pose, gt_pose, k)
            if self.args.class_type in ["obj1", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29', 'obj33']:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)
            self.calculate_tra_and_rot(gt_pose, pred_pose)
        return pred_pose, gt_pose

    def calculate_metric_visble(self, keypoints2d, predict2d, K, visble, score, edge_score):

        batch_size = keypoints2d.shape[0]
        keypoints_num = keypoints2d.shape[1]

        for i in range(batch_size):
            keypoints = keypoints2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            predict = predict2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            #self.set_error_points(predict)
            k = K[i].detach().cpu().numpy()
            gt_pose = self.pnp(self.keyponits, keypoints, k)

            # pred_pose = self.pnp(self.keyponits, predict, k)
            points_2d = []
            points_3d = []
            score_ = []
            score = np.array(score)
            score_ = (score - np.min(score)) / (np.max(score) - np.min(score))
            #
            score_ = score_ * np.array(edge_score)

            score_tt = score_.tolist()
            # 获取前 N 个元素的索引, 选择前6个关键点
            top_n = 8
            # 为什么报错？？？
            top_n_indices = [i for i, _ in sorted(enumerate(score_tt), key=lambda x: -x[1])[:top_n]]

            for id in top_n_indices:
                points_2d.append(predict[id])
                points_3d.append(self.keyponits[id])
            # for v in range(len(visble)):
            #     if visble[v] == 1:
            #         points_2d.append(predict[v])
            #         points_3d.append(self.keyponits[v])

            if len(points_2d) >= 4:
                pred_pose = self.pnp(np.array(points_3d), np.array(points_2d), k)
            else:
                pred_pose = self.pnp(self.keyponits, predict, k)

            self.projection_2d(pred_pose, gt_pose, k)
            if self.args.class_type in ["obj1", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29', 'obj33']:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)
            self.calculate_tra_and_rot(gt_pose, pred_pose)
        return pred_pose

    def set_error_points(self, predict):
        error_list = np.random.choice(10, self.error_num, replace=False)
        for num in error_list:
            w = np.random.randint(0, 480)
            h = np.random.randint(0, 640)
            predict[num] = [w, h]

    def calculate_metric_PECP(self, keypoints2d, predict2d, K, pose, pred_contour):

        batch_size = keypoints2d.shape[0]
        keypoints_num = keypoints2d.shape[1]

        for i in range(batch_size):
            predict = predict2d[i].detach().cpu().numpy().reshape(keypoints_num, -1)
            # self.set_error_points(predict)
            k = K[i].detach().cpu().numpy()
            gt_pose = pose[i].detach().cpu().numpy()
            contour = pred_contour[i][0].detach().cpu().numpy()
            contour[contour >= 0] = 1
            contour[contour < 0] = 0
            contour = np.asarray(contour).astype(np.uint8)
            foreground = np.sum(contour)
            if foreground > 1000:
                # Gaussian convolution can be used here
                edge_heatmap = contour
                pred_pose = self.PECP(self.keyponits, predict, k, edge_heatmap, self.list_all)
            else:
                pred_pose = self.pnp(self.keyponits, predict, k)
            self.projection_2d(pred_pose, gt_pose, k)
            if self.args.class_type in ["obj1", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29', 'obj33']:
                self.add_metric(pred_pose, gt_pose, syn=True)
            else:
                self.add_metric(pred_pose, gt_pose)
            self.calculate_tra_and_rot(gt_pose, pred_pose)

    def PECP(self, points_3d, points_2d, K, target_contour, list_all):
        match_dict = {}
        for i in range(points_3d.shape[0]):
            match_keypoints = np.concatenate((points_2d[i], points_3d[i]), axis=0)
            # 1 2 3 4 5 6 7 8 9 a b c
            if i >= 10:
                match_dict[chr(97 + i - 10)] = match_keypoints
            else:
                match_dict[str(i)] = match_keypoints

        list_score = np.zeros(points_2d.shape[0])

        # Number of iterations obtained by calculation
        iteration_time = 400

        for i in range(iteration_time):
            temp_list = choice(list_all)
            keypoints_2d = np.zeros((temp_list.__len__(), 2))
            keypoints_3d = np.zeros((temp_list.__len__(), 3))
            for j in range(temp_list.__len__()):
                keypoints_2d[j] = match_dict[temp_list[j]][:2]
                keypoints_3d[j] = match_dict[temp_list[j]][2:]
            _, R_exp, t = cv2.solvePnP(keypoints_3d, keypoints_2d, K, distCoeffs=np.zeros(shape=[5, 1], dtype="float64"),
                                       flags=cv2.SOLVEPNP_EPNP)
            R, _ = cv2.Rodrigues(R_exp)
            pose = np.concatenate([R, t], axis=-1)
            # 2d points
            valid_2d = project(self.valid_3d, K, pose).astype(int)
            sum, valid_2d = self.get_confidence(target_contour, valid_2d)
            score = sum - 0.33 * valid_2d.shape[0]
            if score > 0:
                for t in temp_list:
                    if t >= 'a':
                        index = ord(t) - 97 + 10
                    else:
                        index = int(t)
                    list_score[index] = list_score[index] + score
        max = 0
        total = points_2d.shape[0] + 1
        error_num = 0
        for k in range(4, total):
            k = k - error_num
            top_index = self.top_K_idx(list_score, k)
            keypoints_2d = np.zeros((top_index.shape[0], 2))
            keypoints_3d = np.zeros((top_index.shape[0], 3))
            j = 0
            for idx in top_index:
                if idx >= 10:
                    temp_idx = chr(97 + idx - 10)
                else:
                    temp_idx = str(idx)
                keypoints_2d[j] = match_dict[temp_idx][:2]
                keypoints_3d[j] = match_dict[temp_idx][2:]
                j = j + 1
            if top_index.shape[0] == 4:
                _, R_exp, t = cv2.solvePnP(keypoints_3d, keypoints_2d, K,
                                           distCoeffs=np.zeros(shape=[5, 1], dtype="float64"),
                                           flags=cv2.SOLVEPNP_EPNP)
                R, _ = cv2.Rodrigues(R_exp)
                pose = np.concatenate([R, t], axis=-1)
            else:
                pose = self.pnp(keypoints_3d, keypoints_2d, K)
            valid_2d = project(self.valid_3d, K, pose).astype(int)  # 得到2d点
            sum, valid_2d = self.get_confidence(target_contour, valid_2d)
            if sum >= max:
                max = sum
                pose_final = pose
            else:
                list_score[top_index[-1]] = -1
                total = total - 1
                error_num = error_num + 1
                continue
        ransac_pose = self.pnp(points_3d, points_2d, K)
        valid_2d = project(self.valid_3d, K, ransac_pose).astype(int)  # 得到2d点
        sum, valid_2d = self.get_confidence(target_contour, valid_2d)
        if sum >= max:
            pose_final = ransac_pose
        return pose_final

    def get_confidence(self, target_contour, valid_2d):
        valid_2d[valid_2d[:, 0] >= 640] = 0
        valid_2d[valid_2d[:, 0] < 0] = 0
        valid_2d[valid_2d[:, 1] >= 480] = 0
        valid_2d[valid_2d[:, 1] < 0] = 0
        valid_2d = np.unique(valid_2d, axis=0)
        sum = np.sum(target_contour[valid_2d[:, 1], valid_2d[:, 0]])
        return sum, valid_2d

    def pnp(self, points_3d, points_2d, camera_matrix):

        try:
            dist_coeffs = self.dist_coeffs
        except:
            dist_coeffs = np.zeros(shape=[5, 1], dtype="float64")

        assert (
                points_3d.shape[0] == points_2d.shape[0]
        ), "points 3D and points 2D must have same number of vertices"
        points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
        points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
        camera_matrix = camera_matrix.astype(np.float64)

        if points_2d.shape[0] < 5:
            _, R_exp, t = cv2.solvePnP(
                points_3d, points_2d, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
            )
        else:
            if self.args.class_type in ["obj2", "obj32"]:
                _, R_exp, t, inliers = cv2.solvePnPRansac(
                    points_3d, points_2d, camera_matrix, dist_coeffs, iterationsCount=1000, reprojectionError=5,
                    flags=cv2.SOLVEPNP_ITERATIVE)
            else:
                _, R_exp, t, inliers = cv2.solvePnPRansac(
                    points_3d, points_2d, camera_matrix, dist_coeffs, iterationsCount=1000, reprojectionError=5,
                    flags=cv2.SOLVEPNP_EPNP)

        R, _ = cv2.Rodrigues(R_exp)
        return np.concatenate([R, t], axis=-1)

    def projection_2d(self, pose_pred, pose_targets, K):
        # rot pos in z
        pose_pred = self.pose_reverse(pose_pred, pose_targets)

        model_2d_pred = project(self.mesh_model["pts"] / 1000, K, pose_pred)
        model_2d_targets = project(self.mesh_model["pts"] / 1000, K, pose_targets)
        proj_mean_diff = np.mean(
            np.linalg.norm(model_2d_pred - model_2d_targets, axis=-1)
        )

        if proj_mean_diff < self.threshold:
            self.proj_2d_mean.append(proj_mean_diff)
        self.proj_2d.append(proj_mean_diff < self.threshold)

    def pose_reverse(self, pose_pred, pose_targets):
        if self.args.class_type in ["obj1", "obj2", 'obj5', 'obj14', 'obj17', 'obj18', 'obj24', 'obj26', 'obj29',
                                    'obj33']:
            rot = rtDic[self.args.class_type]
            pose_pred2 = np.dot(pose_pred, rot)
            ori = np.linalg.norm(pose_targets - pose_pred)
            new = np.linalg.norm(pose_targets - pose_pred2)
            if new < ori:
                pose_pred = pose_pred2
        return pose_pred

    def add_metric(self, pose_pred, pose_targets, syn=False, percentage=0.1):
        diameter = self.diameter * percentage
        model_pred = (
                np.dot(self.mesh_model["pts"], pose_pred[:, :3].T) + pose_pred[:, 3] * 1000
        )
        model_targets = (
                np.dot(self.mesh_model["pts"], pose_targets[:, :3].T) + pose_targets[:, 3] * 1000
        )

        if syn:
            mean_dist_index = spatial.cKDTree(model_pred)
            mean_dist, _ = mean_dist_index.query(model_targets, k=1)
            mean_dist = np.mean(mean_dist)
        else:
            mean_dist = np.mean(np.linalg.norm(model_pred - model_targets, axis=-1))
        self.add.append(mean_dist < diameter)

    def top_K_idx(self, data, k):
        data = np.array(data)
        idx = data.argsort()[-k:][::-1]
        return idx
