# 实验室发动机铸件的位姿估计

import cv2
import numpy as np
from models.networkAssembly import ContourPose
import os
import torch
from PIL import Image
import glob
from utils.poseVisRender import render_mesh_on_image

class poseEstimation:
    def __init__(self, k, keypoints3d, model_dir):
        super(poseEstimation, self).__init__()
        self.k = k
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("using {} device".format(self.device))
        self.keypoints3d = keypoints3d
        ContourNet = ContourPose(heatmap_dim=self.keypoints3d.shape[0] + 1,
                                 graph_dim=(self.keypoints3d.shape[0] * (self.keypoints3d.shape[0] - 1) // 2) * 2)
        ContourNet = ContourNet.to(self.device)

        pretrained_model = torch.load(model_dir)
        ContourNet.load_state_dict(pretrained_model['net'], strict=False)
        ContourNet.eval()
        self.model = ContourNet

    def extract_coords(self, input_map):
        flat_map = input_map.view(input_map.shape[0], input_map.shape[1], -1)
        max_idx = torch.argmax(flat_map, dim=2)
        width = input_map.shape[3]
        x = (max_idx / width).int().unsqueeze(dim=2)
        y = (max_idx % width).unsqueeze(dim=2)
        return torch.cat((y, x), dim=2)

    def pnp(self, points_3d, points_2d, camera_matrix):

        dist_coeffs = np.zeros(shape=[5, 1], dtype="float64")
        assert (
                points_3d.shape[0] == points_2d.shape[0]
        ), "points 3D and points 2D must have same number of vertices"
        points_2d = np.ascontiguousarray(points_2d.astype(np.float64))
        points_3d = np.ascontiguousarray(points_3d.astype(np.float64))
        camera_matrix = camera_matrix.astype(np.float64)

        _, R_exp, t, inliers = cv2.solvePnPRansac(
            points_3d, points_2d, camera_matrix, dist_coeffs, iterationsCount=1000, reprojectionError=5,
            flags=cv2.SOLVEPNP_ITERATIVE)
        R, _ = cv2.Rodrigues(R_exp)
        return np.concatenate([R, t], axis=-1)
    def project(self, xyz, K, RT):
        xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
        xyz = np.dot(xyz, K.T)
        xy = xyz[:, :2] / xyz[:, 2:]
        return xy

    def poseCal(self, img):
        img_show = img
        img = img / 255.0
        # 归一化？？
        img -= [0.419, 0.427, 0.424]
        img /= [0.184, 0.206, 0.197]
        img = torch.tensor(img, dtype=torch.float32).permute((2, 0, 1)).to(self.device)
        img = img.unsqueeze(0)
        pred_heatmap, pred_contour, pre_vis, pre_grah,  pre_mask =  self.model(img)

        # pred_heatmap_new = pred_heatmap.detach().cpu().numpy()
        # pred_heatmap_new = pred_heatmap_new.squeeze()
        # pred_contour_new = pred_contour.detach().cpu().numpy()
        # pred_contour_new = pred_contour_new.squeeze()
        # img_trans = ((pred_contour_new - pred_contour_new.min()) * (
        #             1 / (pred_contour_new.max() - pred_contour_new.min()) * 255)).astype('uint8')
        # # ret, binary = cv2.threshold(img_trans, 150, 255, cv2.THRESH_BINARY)
        # cv2.imshow("binary", pred_contour_new)
        # cv2.imshow("img_trans", img_trans)

        # # cv2.imshow("pred_contour_new",pred_contour_new)
        # cv2.imshow("pred_heatmap_new1", pred_heatmap_new[0])
        # cv2.imshow("pred_heatmap_new2", pred_heatmap_new[1])
        pred_heatmap = pred_heatmap[:, :8, :, :]
        predict_2d = self.extract_coords(pred_heatmap)
        predict = predict_2d[0].detach().cpu().numpy().reshape(8, -1)
        pre_pose = self.pnp(self.keypoints3d, predict, self.k)

        return pre_pose, predict

if __name__ == '__main__':
    #====相机内参，缩小了3.825倍=====
    k = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                               [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                               [0.0, 0.0, 1.0]])
    # ====3d关键点=====
    corners = np.loadtxt("keypoints/part_02.txt")
    #======模型路径====
    model_dir = r"weights\part_02\150.pkl"
    #=====位姿估计初始化======
    ps = poseEstimation(k, corners, model_dir)

    #=====读取图像=====
    img_paths = glob.glob(os.path.join(r"assets", "test*"))
    for img_path in  img_paths:
        img = np.array(Image.open(img_path))
        # 转成bgr，如果是cv读取，可能不需要转换
        img = img[:, :, ::-1]
        
        img = cv2.resize(img, dsize=(640, 535))
        pre_pose, pre_points2d = ps.poseCal(img)
        print("pre_pose", pre_pose)
        vis_img = render_mesh_on_image(
            img,
            r'cad\part_02.ply',
            pre_pose,
            ps.k,
            color=(255, 0, 0),
            alpha=0.72,
            draw_edges=False,
            edge_color=(25, 25, 25),
            edge_thickness=1,
        )

        for point in pre_points2d:
            cv2.circle(vis_img, (int(point[0]), int(point[1])), 2, (0, 255, 255), 2)

        xy_ = ps.project(ps.keypoints3d, ps.k, pre_pose)
        for point in xy_:
            cv2.circle(vis_img, (int(point[0]), int(point[1])), 2, (255, 0, 0), 2)

        cv2.imshow("img_show", vis_img)
        cv2.waitKey(0)

