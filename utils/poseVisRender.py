from scipy.spatial.transform import Rotation
import numpy as np
import pyrender
import trimesh
import cv2
import copy
import os
import pickle
import yaml
from pyrender.constants import RenderFlags

# 设置最小极角阈值（约0.1度）
EPSILON_THETA = np.deg2rad(0.1)

def spherical_to_cartesian(r, theta, phi):
    y = -r * np.sin(theta)
    z = -r * np.cos(theta) * np.cos(phi)
    x = r * np.cos(theta) * np.sin(phi)
    return np.array([x, y, z])

def handle_pole_rotation(theta, phi):
    R_phi = Rotation.from_euler('z', phi).as_matrix()  # Z轴旋转
    R_theta = Rotation.from_euler('x', -theta).as_matrix()  # 负号修正Y轴方向
    R = R_theta @ R_phi
    correction = np.diag([1, -1, -1])  # Y轴和Z轴方向翻转
    return correction @ R

def look_at_rotation(theta, phi):
    forward = -spherical_to_cartesian(1, theta, phi)
    forward /= np.linalg.norm(forward)

    right = np.cross([0, -1, 0], forward)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    return np.vstack([right, down , forward]).T  # 3x3旋转矩阵

def rotation_matrix_roll(roll):
    """绕视线轴（Z轴）旋转的矩阵（Roll角）"""
    return np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1]
    ])

def compose_rotation(theta, phi, psi):
    # 初始朝向矩阵
    R_look = look_at_rotation(theta, phi)
    # 绕Z轴的自身旋转
    R_self = Rotation.from_euler('z', psi).as_matrix()
    # R_self = rotation_matrix_roll(psi)
    # 组合旋转（注意顺序：先朝向物体，再自身旋转）
    # 全局旋转在前，局部旋转在后
    return R_look @ R_self
    # return R_self @ R_look

def build_camera_pose(r, theta, phi, psi):
    # 获取位置和旋转
    position = spherical_to_cartesian(r, theta, phi)
    R = compose_rotation(theta, phi, psi)

    # 构建位姿矩阵（世界坐标系 → 相机坐标系）
    pose = np.eye(4)

    pose[:3, :3] = R
    pose[:3, 3] = -R.T @ position  # 相机原点在世界坐标系中的位置
    # print("R: ", R)
    # print("RT: ", R.T)
    return pose

def load_mesh(obj_file_path):
    return trimesh.load(obj_file_path)

def load_model_scene(mesh, visFlag):
    # 加载.obj 模型文件
    try:
        mesh = mesh

    except Exception as e:
        print(f"加载模型文件时出错: {e}")
        return
    # # 高光指数，控制高光锐利度
    # roughness = 1.0 - (mesh.visual.material.kwargs['ns'] / 1000.0)
    # # 漫反射系数，控制材质的基底颜色
    # kd = mesh.visual.material.kwargs['kd']
    # # 环境光系数，控制材质对环境光的反射强度
    # ka = mesh.visual.material.kwargs['ka']
    # # 镜面反射系数，控制高光颜色
    # ks = mesh.visual.material.kwargs['ks']
    #
    # ke = mesh.visual.material.kwargs['ke']
    # ni = mesh.visual.material.kwargs['ni']
    # d = mesh.visual.material.kwargs['d']
    # 创建场景
    # scene = pyrender.Scene(ambient_light=(kd[0],ka[1],ka[2]))  # 呈现黄色
    scene = pyrender.Scene()

    # if mesh.visual.kind == 'texture':
    #     # 如果材质是纹理材质
    #     print("材质类型：纹理材质")
    #     print("材质图像：", mesh.visual.material.image)
    #     print("材质 UV 坐标：", mesh.visual.uv)
    # elif mesh.visual.kind == 'color':
    #     # 如果材质是颜色材质
    #     print("材质类型：颜色材质")
    #     print("材质颜色：", mesh.visual.material.main_color)
    # else:
    #     print("材质类型：未知")

    # meterial = pyrender.MetallicRoughnessMaterial(baseColorFactor=[255, 0, 0, 255])
    # meterial = pyrender.MetallicRoughnessMaterial(baseColorFactor = mesh.visual.main_color) #mesh.visual.material.main_color
    if isinstance(mesh, trimesh.Scene):
        # 分割多材质子网格
        for name, submesh in mesh.geometry.items():
            material = submesh.visual.material
            # texture = Image.open(material.baseColorTexture) if material.baseColorTexture else None
            # basecolor = material.main_color

            pbr_material = create_material_from_trimesh(submesh, visFlag)

            # pbr_material = pyrender.MetallicRoughnessMaterial(
            #     baseColorFactor=basecolor,
            #     # baseColorTexture=texture,
            #     metallicFactor=0.5,
            #     roughnessFactor=0.5
            # )

            pyrender_submesh = pyrender.Mesh.from_trimesh(submesh, material=pbr_material)
            scene.add(pyrender_submesh)
    elif isinstance(mesh, trimesh.Trimesh):
        # 单材质处理（同上）baseColorFactor
        # # 高光指数，控制高光锐利度
        # roughness = 1.0 - (mesh.visual.material.kwargs['ns'] / 1000.0)

        # 漫反射系数，控制材质的基底颜色
        # kd = mesh.visual.material.kwargs['kd']
        # 环境光系数，控制材质对环境光的反射强度
        # ka = mesh.visual.material.kwargs['ka']
        # 镜面反射系数，控制高光颜色
        # ks = mesh.visual.material.kwargs['ks']

        # ke = mesh.visual.material.kwargs['ke']
        # ni = mesh.visual.material.kwargs['ni']
        # d = mesh.visual.material.kwargs['d']

        # meterial = pyrender.MetallicRoughnessMaterial(baseColorFactor=mesh.visual.material.main_color) #

        # metallicFactor参数很重要
        # meterial = pyrender.MetallicRoughnessMaterial(baseColorFactor=(kd[0], kd[1], kd[2], 1.0),
        #                                               roughnessFactor=roughness,metallicFactor = 0.5, emissiveFactor = ke) # ,roughnessFactor=roughness,
        # mesh = pyrender.Mesh.from_trimesh(mesh, material = meterial)

        material = create_material_from_trimesh(mesh, visFlag)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
        scene.add(mesh)

        print("geometry mesh is None")

    # mesh = pyrender.Mesh.from_trimesh(mesh, material=meterial)

    return scene

def create_material_from_trimesh(obj_mesh, visFlag):
    """从Trimesh网格对象创建兼容的Pyrender材质"""
    # 尝试从Trimesh网格获取材质信息
    material = obj_mesh.visual.material if hasattr(obj_mesh.visual, 'material') else None

    # 初始化材质参数
    base_color = [0.8, 0.8, 0.8, 1.0]  # 基础颜色
    if visFlag == 1:
        base_color = [0.0, 1.0, 0.0, 1.0]
    else:
        base_color = [0.88, 0.0, 0.0, 1.0]
    metallic_factor = 0.2  # 金属度（0表示非金属）
    roughness_factor = 0.6  # 粗糙度（0表示光滑，1表示粗糙）

    # 从Trimesh材质中提取参数
    if material:
        # 处理OBJ材质属性
        if hasattr(material, 'kwargs'):
            kwargs = material.kwargs
            # 使用漫反射颜色作为基础颜色
            if 'kd' in kwargs:
                kd = kwargs['kd']
                # base_color = [kd[0], kd[1], kd[2], 1.0]
                base_color = base_color

        # 处理颜色材质
        if hasattr(material, 'main_color'):
            rgba = material.main_color
            # base_color = [c / 255.0 for c in rgba]  # 转换为0-1范围
            base_color = base_color

    # 创建Pyrender的MetallicRoughnessMaterial
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=base_color,
        metallicFactor=metallic_factor,
        roughnessFactor=roughness_factor
    )

def render(scene, pose):
    # 小论文相机
    # fx = 1.83094488e+03 / 3.825
    # fy = 1.83114709e+03 / 3.825
    # cx = 1.19980612e+03 / 3.825
    # cy = 1.02603620e+03 / 3.825
    # 实验室发动机相机

    # fx = 1.81823106e+03 / 3.825
    # fy = 1.81873518e+03 / 3.825
    # cx = 1.19548175e+03 / 3.825
    # cy = 1.00709320e+03 / 3.825

    # Mono6D 数据集
    fx = 2209.878296
    fy = 2210.376676
    cx = 349.751312
    cy = 254.828051

    # MP6D数据集
    # fx = 567.53720406
    # fy = 569.3617592
    # cx = 312.66570357
    # cy = 257.1729701

    camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, zfar=10000)
    camera_pose = np.array([[1.0, 0.0, 0., 0],  # 示例外参矩阵，可根据实际调整
                            [0.0, 1.0, 0.0, 0],
                            [0., 0.0, 1.0, 0],
                            [0.0, 0.0, 0.0, 1.0]])
    obj_pose = pose
    # 将第二行和第三行求反
    obj_pose[[1, 2]] *= -1
    # 然后求逆
    obj_pose = np.linalg.inv(obj_pose)

    camera_node = pyrender.Node(camera=camera, matrix=obj_pose)

    # print("obj_pose", obj_pose)
    # scene.add(camera, pose=camera_pose)
    scene.add_node(camera_node)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
    scene.add(light, pose=obj_pose)
    # scene.set_pose(light, obj_pose)

    scene.bg_color = [0.0, 0.0, 0.0, 255]
    # 渲染场景
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=480)  # 535   480
    try:
        flags = RenderFlags.SHADOWS_DIRECTIONAL # RenderFlags.ALL_WIREFRAME   FLIP_WIREFRAME
        color, depth = r.render(scene, flags)
    except Exception as e:
        print(f"渲染场景时出错: {e}")
        r.delete()
        return
    r.delete()
    return color, depth