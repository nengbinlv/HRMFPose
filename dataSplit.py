import os
import random
from shutil import copy2
import numpy as np

# 步骤一未作改动，需要带入自己的文件路径file_path 、new_file_path
def splitimg():
    # 图片文件夹路径
    file_path = r"E:\PoseTrackingCode\ContourPose-main\data\train\obj_02\rgb"
    # 新文件存放路径
    new_file_path = r"E:\PoseTrackingCode\ContourPose-main\data\train\obj_02\photo_cut"
    # 划分数据比例
    split_rate = [0.3, 0.7, 0]
    class_names = os.listdir(file_path)
    # 目标文件夹下创建文件夹
    split_names = ['train', 'val', 'test']
    # print(class_names)  # ['00000.jpg', '00001.jpg', '00002.jpg'... ]
    current_all_data = os.listdir(file_path)

    # 判断是否存在目标文件夹，不存在则创建---->创建train\val\test文件夹
    if os.path.isdir(new_file_path):
        pass
    else:
        os.makedirs(new_file_path)

    for split_name in split_names:
        split_path = os.path.join(new_file_path, split_name)
        # D:/Code/Data/GREENTdata/train, val, test
        if os.path.isdir(split_path):
            pass
        else:
            os.makedirs(split_path)

        # 按照比例划分数据集，并进行数据图片的复制
        for class_name in class_names:
            current_data_path = file_path  # D:/Code/Data/centerlinedata/tem_voc/JPEGImages/
            current_data_length = len(class_names)  # 文件夹下的图片个数
            current_data_index_list = list(range(current_data_length))
            random.shuffle(current_data_index_list)

            train_stop_flag = current_data_length * split_rate[0]
            val_stop_flag = current_data_length * (split_rate[0] + split_rate[1])

    current_idx = 0
    train_num = 0
    val_num = 0
    test_num = 0
    # 图片复制到文件夹中
    train_list = []
    val_list = []
    test_list = []
    for i in current_data_index_list:
        src_img_path = os.path.join(current_data_path, current_all_data[i])
        if current_idx <= train_stop_flag:
            newpath = os.path.join(os.path.join(new_file_path, 'train'), current_all_data[i])
            os.rename(src_img_path, newpath)
            train_num += 1
            # train_list.append(src_img_path)
        elif (current_idx > train_stop_flag) and (current_idx <= val_stop_flag):
            newpath = os.path.join(os.path.join(new_file_path, 'val'), current_all_data[i])
            os.rename(src_img_path, newpath)
            # copy2(src_img_path, newpath)
            val_num += 1
        else:
            newpath = os.path.join(os.path.join(new_file_path, 'test'), current_all_data[i])
            os.rename(src_img_path, newpath)
            # copy2(src_img_path, newpath)
            test_num += 1
        current_idx += 1
    # np.savetxt(os.path.join(file_path, "train.txt"),train_list)
    print("Done!", train_num, val_num, test_num)


if __name__ == '__main__':
    splitimg()