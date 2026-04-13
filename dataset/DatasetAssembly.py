import random
from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import glob, pickle
import numpy as np
import os.path as osp
from utils.utils import project
import yaml
import cv2
from utils.transforms import rotate_img
from utils.heatmap import generate_heatmap, generate_heatmap_train, generate_heatmap_train_oneMap
from tqdm import tqdm

class MyDataset(Dataset):
    def __init__(self, root, cls, is_train=True, scene=None, index=None):
        super(MyDataset, self).__init__()
        self.root = root
        self.test_root = osp.join(self.root, "test")
        self.cls = cls
        # 为了索引objxx
        self.objID = self.cls[3:]
        self.data_paths = []
        self.is_training = is_train
        # 关键点
        self.corners = np.loadtxt(os.path.join(os.getcwd(), "keypoints\\{}.txt".format(cls))) / 1000  # KEYPOINTS
        self.element = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        # 背景图像，用于数据增强
        self.bg_imgs_path = os.path.join(os.getcwd(), "dataset\\bg_imgs.npy")
        self.sun_path = os.path.join(root, "SUN2012pascalformat")
        self.get_bg_imgs()
        self.bg_imgs = np.load(self.bg_imgs_path).astype(np.str)

        if is_train:
            self.path = osp.join(self.root, "train", self.cls)
            # 相机内参
            self.K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                               [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                               [0.0, 0.0, 1.0]])
            # 位姿真值
            self.train_pose = yaml.load(open(osp.join(self.path, 'gt_2.yml'), 'r'),Loader=yaml.FullLoader)
            self.render_train_pose = yaml.load(open(osp.join(self.path, 'render', 'pose_final.yml'), 'r'),
                                               Loader=yaml.FullLoader)
            self.data_paths = self.get_train_data_path(self.root, self.cls)

            # 新增加，直接把数据载入到内存中
            # self.img = []
            # self.mask = []
            # self.pose = []
            # self.K_temp = []
            # self.gt_contour = []
            # self.gt_state = []
            # for path in self.data_paths:
            #     img, mask, pose, K, gt_contour, gt_state = self.get_data(path)
            #     self.img.append(img)
            #     self.mask.append(mask)
            #     self.pose.append(pose)
            #     self.K_temp.append(K)
            #     self.gt_contour.append(gt_contour)
            #     self.gt_state.append(gt_state)

        else:
            # 测试阶段
            self.path = osp.join(self.root, self.cls)
            self.K = np.array([[1.83094488e+03 / 3.825, 0.0, 1.19980612e+03 / 3.825],
                               [0.0, 1.83114709e+03 / 3.825, 1.02603620e+03 / 3.825],
                               [0.0, 0.0, 1.0]])
            self.train_pose = yaml.load(open(osp.join(self.path, 'gt.yml'), 'r'), Loader=yaml.FullLoader)
            self.data_paths = self.get_test_data_path()

            # 新增加，直接把数据载入到内存中
            # self.img = []
            # self.pose = []
            # self.K_temp = []
            # for path in self.data_paths:
            #     img, pose, K = self.get_data(path)
            #     self.img.append(img)
            #     self.pose.append(pose)
            #     self.K_temp.append(K)

# 获取训练数据list
    def get_train_data_path(self, root, cls):
        paths_list = []
        paths = {}

        count = os.listdir(osp.join(self.path, "photo_cut", "train"))
        train_inds = [ind.replace(".png", "") for ind in count]
        train_inds.sort()

        train_img_path = osp.join(self.path, "photo_cut", "train")
        mask_path = osp.join(self.path, "mask")  # use test data train
        edge_path = osp.join(self.path, "edge_occ")

        render_dir = osp.join(self.path, "render")
        render_img_path = osp.join(render_dir, "rgb_bg")
        render_mask_path = osp.join(render_dir, "mask")
        render_edge_dir = osp.join(render_dir, "edge_occ")
        render_num = len(glob.glob(osp.join(render_img_path, "*.png")))
        render_count = os.listdir(render_img_path)
        render_train_inds = [ind.replace(".png", "") for ind in render_count]
        render_train_inds.sort()
        #获取图像路径
        for idx in train_inds:
            # idx = int(idx)
            img_name = "{}.png".format(idx)
            mask_name = "{}.png".format(idx)
            paths["img_path"] = osp.join(train_img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["edge_path"] = osp.join(edge_path, img_name)
            paths["type"] = "true"

            paths_list.append(paths.copy())
        #获取渲染图像
        for idx in render_train_inds:
            img_name = "{}.png".format(idx)
            paths["img_path"] = osp.join(render_img_path, img_name)
            paths["mask_path"] = osp.join(render_mask_path, img_name)
            paths["edge_path"] = osp.join(render_edge_dir, img_name)
            paths["type"] = "render"
            paths_list.append(paths.copy())

        return paths_list

    def get_test_data_path(self):
        paths_list = []
        paths = {}

        test_img_path = osp.join(self.path, "photo_cut", "val")
        count = os.listdir(osp.join(self.path, "photo_cut", "val"))
        test_inds = [ind.replace(".png", "") for ind in count]
        test_inds.sort()
        # print(test_inds)
        mask_path = osp.join(self.path, "mask")
        edge_path = osp.join(self.path, "edge_occ")

        for idx in test_inds:
            # idx = int(idx)
            # print(idx)
            img_name = "{}.png".format(idx)
            mask_name = "{}.png".format(idx)
            paths["img_path"] = osp.join(test_img_path, img_name)
            paths["mask_path"] = osp.join(mask_path, mask_name)
            paths["edge_path"] = osp.join(edge_path, img_name)
            paths["type"] = "test"
            paths_list.append(paths.copy())

        return paths_list
    # 用于读取数据
    def get_data(self, path):
        img = np.array(Image.open(path["img_path"]))
        img = img[:,:,::-1]
        img = cv2.resize(img, dsize=(640, 535))
        if path["type"] == "true":
            gt_contour = np.array(Image.open(path["edge_path"]).convert('L') )
            gt_contour = cv2.resize(gt_contour, dsize=(640, 535))
            # 膨胀？
            gt_contour = cv2.dilate(gt_contour, kernel=self.element)

            # mask = (np.asarray(cv2.imread(path["img_path"], 0)) != 0).astype(np.uint8)
            mask = np.array(Image.open(path["mask_path"]).convert('L'))
            mask = cv2.resize(mask, dsize=(640, 535))

            idx = int(osp.basename(path["img_path"]).replace(".png", ""))
            instance_gt = self.train_pose[int(idx)][0]
            # 根据索找K
            K = self.K
            # 位姿真值
            R = np.array(instance_gt['cam_R_m2c']).reshape(3, 3)
            t = np.array(instance_gt['cam_t_m2c']).reshape(3, 1)
            pose = np.concatenate([R, t], axis=1)  # 合成3x4位姿矩阵

            return img, mask, pose, K, gt_contour
        if path["type"] == "render":
            idx = int(osp.basename(path["img_path"]).replace(".png", ""))
            instance_gt = self.render_train_pose[int(idx)][0]
            # 根据索找K
            K = self.K
            # 位姿真值
            R = np.array(instance_gt['cam_R_m2c']).reshape(3, 3)
            t = np.array(instance_gt['cam_t_m2c']).reshape(3, 1)
            pose = np.concatenate([R, t], axis=1)  # 合成3x4位姿矩阵

            K = self.K
            gt_contour = np.array(Image.open(path["edge_path"]).convert('L'))
            gt_contour = cv2.resize(gt_contour, dsize=(640, 535))
            # mask = (np.asarray(cv2.imread(path["img_path"], 0)) != 0).astype(np.uint8)
            mask = np.array(Image.open(path["mask_path"]).convert('L'))
            mask = cv2.resize(mask, dsize=(640, 535))
            # edges = cv2.Canny(mask, 100, 200)
            # gt_contour = cv2.bitwise_or(gt_contour, edges)
            gt_contour = cv2.dilate(gt_contour, kernel=self.element)

            return img, mask, pose, K, gt_contour

        elif path["type"] == "test":
            idx = int(osp.basename(path["img_path"]).replace(".png", ""))
            K = self.K
            instance_gt = self.train_pose[int(idx)][0]  # test
            R = np.array(instance_gt['cam_R_m2c']).reshape(3, 3)
            t = np.array(instance_gt['cam_t_m2c']).reshape(3, 1)
            pose = np.concatenate([R, t], axis=1)  # 合成3x4位姿矩阵
            return img, pose, K

    def get_bg_imgs(self):
        if os.path.exists(self.bg_imgs_path):
            return

        img_paths = glob.glob(os.path.join(self.sun_path, 'JPEGImages\\*'))
        bg_imgs = []

        for img_path in tqdm(img_paths):
            bg_imgs.append(img_path)

        np.save(self.bg_imgs_path, bg_imgs)

    def random_background(self, img, mask):
        self.get_bg_imgs()
        random_img_path = random.choice(self.bg_imgs)
        random_img = cv2.imread(random_img_path)
        row, col = img.shape[:2]
        w ,h = mask.shape[:2]
        if row < w + 1 or col < h + 1:   #row < 481 or col < 641
            random_img = cv2.resize(random_img, dsize=(960, int((960 / row) * col)))  # 960
        random_img = self.random_crop(random_img)  # crop 480*640
        random_img = self.random_filp(random_img)  # random flip

        mask_img = cv2.bitwise_and(img, img, mask=mask)
        maskCopy = np.array(mask)
        np.place(maskCopy, maskCopy > 0, 255)
        mask_inv = cv2.bitwise_not(maskCopy)
        background = cv2.bitwise_and(random_img, random_img, mask=mask_inv)
        background = cv2.medianBlur(background, 3)
        last_img = cv2.add(mask_img, background)
        last_img = cv2.medianBlur(last_img, 3)
        return last_img

    def random_translation(self, img, edge, mask, pt2d, is_render=False):
        if is_render:
            # 修改了，之前的太大了
            random_y = random.randint(-20, 50)
            random_x = random.randint(-20, 60)
        else:
            random_y = random.randint(-20, 50)
            random_x = random.randint(-20, 60)
        M = np.float32([[1, 0, random_x], [0, 1, random_y]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        edge = cv2.warpAffine(edge, M, (edge.shape[1], edge.shape[0]))
        mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        pt2d[:, 0] = pt2d[:, 0] + random_x
        pt2d[:, 1] = pt2d[:, 1] + random_y
        pt2d = pt2d[:, :2]

        return img, edge, mask, pt2d

    def random_rotation_and_resize(self, img, edge, mask, pt2d, is_render=False):
        if not is_render:
            ratio = random.uniform(0.8, 1.1)
        else:
            ratio = random.uniform(0.8, 1.1)
        flag = random.randint(1, 4)
        if flag == 1:
            random_angle = random.randint(0, 360)
        else:
            random_angle = 0
        height = img.shape[1]
        width = img.shape[0]
        mat = cv2.getRotationMatrix2D((height * 0.5, width * 0.5), random_angle, ratio)
        img = cv2.warpAffine(img, mat, (img.shape[1], img.shape[0]))
        edge = cv2.warpAffine(edge, mat, (edge.shape[1], edge.shape[0]))
        mask = cv2.warpAffine(mask, mat, (mask.shape[1], mask.shape[0]))

        last_row = np.asarray([[0, 0, 1]], dtype=np.float32)
        R = np.concatenate([mat, last_row], axis=0).transpose()
        last_col = np.ones((pt2d.shape[0], 1), dtype=np.float32)
        pt2d = np.concatenate([pt2d, last_col], axis=1)
        pt2d = np.float32(np.matmul(pt2d, R))
        pt2d = pt2d[:, :2]

        return img, edge, mask, pt2d

    def random_crop(self, img):
        h, w = img.shape[:2]
        y = np.random.randint(0, h - 535)  # 480
        x = np.random.randint(0, w - 640)  # 640
        image = img[y:y + 535, x:x + 640, :]  # 480 # 640
        # cv2.imshow("",image)
        # cv2.waitKey(0)
        return image

    def random_filp(self, image):
        flip_prop = np.random.randint(low=0, high=3)
        axis = np.random.randint(0, 2)
        if flip_prop == 0:
            image = cv2.flip(image, axis)
        return image

    def augment(self, img, mask, gt_contour, pose, K):
        img = np.asarray(img).astype(np.uint8)
        if True:
            # randomly mask out to add occlusion
            R = np.eye(3, dtype=np.float32)
            R_orig = pose[:3, :3]
            T_orig = pose[:3, 3]

            img, mask, gt_contour, R = rotate_img(img, mask, gt_contour, T_orig, K, -30, 30)

            new_R = np.dot(R, R_orig)
            pose[:3, :3] = new_R

        return img, mask, gt_contour, pose

    def get_heatmap_train(self, pt2d, img, keypointsVisible):
        heatmap = generate_heatmap_train(pt2d, keypointsVisible, img.shape[0], img.shape[1])
        return heatmap

    def get_heatmap(self, pose, K, keypoints, img):
        keypoints_2d = project(keypoints, K, pose)
        heatmap = generate_heatmap(keypoints_2d, img.shape[0], img.shape[1])

        return heatmap

    def get_keypointsVisible(self, pt2d, mask):
        kpVisble = np.zeros(pt2d.shape[0])
        mask = cv2.dilate(mask, kernel=self.element)
        index = 0
        for p in range(pt2d.shape[0]):
            # 可见
            if pt2d[p][1] <= mask.shape[0] and pt2d[p][0] <= mask.shape[1] and pt2d[p][1] > 0 and pt2d[p][0] > 0:
                # 在mask内
                if mask[int(pt2d[p][1]),int(pt2d[p][0])] > 0 :
                    kpVisble[p] = 1
                else:
                    kpVisble[p] = 0
            # 不可见
            else:
                kpVisble[p] = 0
        # kpVisble = kpVisble.reshape(-1, 2)
        return kpVisble

    def keypoints_to_graph(self, mask, pt2d):
        # 注意这里的mask
        mask = mask[0] / 255
        num_pts = pt2d.shape[0]
        num_edges = num_pts * (num_pts - 1) // 2
        graph = np.zeros((num_edges, 2, mask.shape[0], mask.shape[1]),
                         dtype=np.float32)
        edge_idx = 0
        for start_idx in range(0, num_pts - 1):
            start = pt2d[start_idx]
            for end_idx in range(start_idx + 1, num_pts):
                end = pt2d[end_idx]
                edge = end - start
                graph[edge_idx, 0][mask == 1.] = edge[0]
                graph[edge_idx, 1][mask == 1.] = edge[1]
                edge_idx += 1
        graph = graph.reshape((num_edges * 2, mask.shape[0], mask.shape[1]))

###########################可视化
        # graph_pred = graph.reshape((-1, 2, mask.shape[0], mask.shape[1]))
        # image_pred = mask.copy()
        # i_edge = 0
        # for start_idx in range(0, num_pts - 1):
        #     for end_idx in range(start_idx + 1, num_pts):
        #         # pred, red
        #         start = np.int16(np.round(pt2d[start_idx]))
        #         edge_x = graph_pred[i_edge, 0][mask == 1.].mean()
        #         edge_y = graph_pred[i_edge, 1][mask == 1.].mean()
        #         edge = np.array([edge_x, edge_y])
        #         end = np.int16(np.round(pt2d[start_idx] + edge))
        #         image_pred = cv2.line(image_pred, tuple(start), tuple(end), (255), 1)
        #         i_edge = i_edge + 1
        # cv2.imshow("image_pred",image_pred)
        # cv2.waitKey(0)

        return graph
    # 读取数据时的入口函数
    def __getitem__(self, index):

        path = self.data_paths[index]

        if self.is_training:
            img, mask, pose, K, gt_contour = self.get_data(path)
            # img = self.img[index]
            # mask = self.mask[index]
            # pose = self.pose[index]
            # K = self.K_temp[index]
            # gt_contour = self.gt_contour[index]
            # gt_state = self.gt_state[index]
        else:
            img, pose, K = self.get_data(path)
            # cv2.imshow("data_img_test", img)

            # img = self.img[index]
            # pose = self.pose[index]
            # K = self.K_temp[index]
            heatmap = self.get_heatmap(pose, K, self.corners, img)

        if self.is_training:
            # 进行数据增强
            img, mask, gt_contour, pose = self.augment(img, mask, gt_contour, pose, K)

            element = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            mask_dilate = cv2.dilate(mask, kernel=element)

            keypoints_2d = project(self.corners, K, pose)

            if path["type"] == "true":
                # random rotation and resize
                # 随机旋转
                img, gt_contour, mask, keypoints_2d = self.random_rotation_and_resize(img, gt_contour, mask, keypoints_2d)
                # random translation
                # 随机平移
                img, gt_contour, mask, keypoints_2d = self.random_translation(img, gt_contour, mask, keypoints_2d)

                keypointsVisible = self.get_keypointsVisible(keypoints_2d, mask)
                heatmap = self.get_heatmap_train(keypoints_2d, img, keypointsVisible)
                heatmap_oneMap = generate_heatmap_train_oneMap(keypoints_2d, keypointsVisible, img.shape[0], img.shape[1])
                heatmap = np.concatenate([heatmap, heatmap_oneMap], axis=0)
                # cv2.imshow("h", heatmap[8])
                # cv2.waitKey(0)
                mask_ = mask.reshape((1, mask.shape[0], mask.shape[1]))
                graph = self.keypoints_to_graph(mask_, keypoints_2d)
            else:
                # 对于渲染图像
                img, gt_contour, mask, keypoints_2d = self.random_rotation_and_resize(img, gt_contour, mask, keypoints_2d, is_render=True)
                img, gt_contour, mask, keypoints_2d = self.random_translation(img, gt_contour, mask, keypoints_2d, is_render=True)
                keypointsVisible = self.get_keypointsVisible(keypoints_2d, mask)
                heatmap = self.get_heatmap_train(keypoints_2d, img, keypointsVisible)
                heatmap_oneMap = generate_heatmap_train_oneMap(keypoints_2d, keypointsVisible, img.shape[0], img.shape[1])
                heatmap = np.concatenate([heatmap, heatmap_oneMap], axis=0)
                mask_ = mask.reshape((1, mask.shape[0], mask.shape[1]))
                graph = self.keypoints_to_graph(mask_, keypoints_2d)


            # random background
            # 随机背景
            # img = self.random_background(img, mask)

            # random light （模拟随机光照）
            alpha = random.uniform(0.8, 1.2)
            beta = random.randint(-5, 5)
            # 对像素取绝对值的函数
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        if self.is_training:
            gt_contour = gt_contour / 255
            # 扩展维度
            gt_contour = np.expand_dims(gt_contour, axis=2)

            mask = mask / 255
            gt_mask = np.expand_dims(mask, axis=2)

            mask_dilate = mask_dilate / 255
            gt_mask_dilate= np.expand_dims(mask_dilate, axis=2)
            # keypointsVisible = np.expand_dims(keypointsVisible, axis=1)

        img = img / 255.0
        # 归一化？？
        img -= [0.419, 0.427, 0.424]
        img /= [0.184, 0.206, 0.197]

        # 转换为tensor
        heatmap = torch.tensor(heatmap, dtype=torch.float32)
        # 转换维度？permute
        img = torch.tensor(img, dtype=torch.float32).permute((2, 0, 1))
        if self.is_training:
            # contour的类型为int
            # 报错
            gt_contour = torch.tensor(gt_contour, dtype=torch.int8).permute((2, 0, 1))

            # keypointsVisible = torch.tensor(keypointsVisible, dtype=torch.float).permute((1, 0))
            keypointsVisible = torch.tensor(keypointsVisible, dtype=torch.float)
            graph = torch.tensor(graph, dtype=torch.float)

            gt_mask = torch.tensor(gt_mask, dtype=torch.int8).permute((2, 0, 1))
            gt_mask_dilate = torch.tensor(gt_mask_dilate, dtype=torch.int8).permute((2, 0, 1))

            return img, heatmap, K, pose, gt_contour, keypointsVisible, graph, gt_mask, gt_mask_dilate
        return img, heatmap, K, pose

    def __len__(self):
        return len(self.data_paths)
