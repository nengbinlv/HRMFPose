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

def project_points(xyz, K, RT):
    xyz = np.asarray(xyz, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    RT = np.asarray(RT, dtype=np.float64)
    xyz = np.dot(xyz, RT[:, :3].T) + RT[:, 3:].T
    xy = np.dot(xyz, K.T)
    xy = xy[:, :2] / xy[:, 2:]
    return xy, xyz[:, 2]

def _to_trimesh(mesh_or_path):
    if isinstance(mesh_or_path, str):
        mesh_or_path = trimesh.load(mesh_or_path, force='mesh', process=False)

    if isinstance(mesh_or_path, trimesh.Scene):
        geometries = [geom for geom in mesh_or_path.geometry.values()]
        if not geometries:
            raise ValueError('The provided scene does not contain any geometry.')
        mesh_or_path = trimesh.util.concatenate(geometries)

    if not isinstance(mesh_or_path, trimesh.Trimesh):
        raise TypeError('mesh_or_path must be a file path, trimesh.Trimesh, or trimesh.Scene.')

    return mesh_or_path

def _cv_to_gl_transform():
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

def _build_material_from_bgr(color):
    b, g, r = color
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[r / 255.0, g / 255.0, b / 255.0, 1.0],
        metallicFactor=0.05,
        roughnessFactor=0.55,
    )

def render_mesh_on_image(image, mesh_or_path, pose, K, color=(0, 0, 255), alpha=0.35,
                         draw_edges=True, edge_color=(0, 0, 0), edge_thickness=1):
    mesh = _to_trimesh(mesh_or_path)
    image = image.copy()
    height, width = image.shape[:2]

    try:
        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=[0.25, 0.25, 0.25])
        material = _build_material_from_bgr(color)
        pyrender_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)

        mesh_pose = _cv_to_gl_transform() @ np.asarray(pose, dtype=np.float64)
        scene.add(pyrender_mesh, pose=mesh_pose)

        K = np.asarray(K, dtype=np.float64)
        camera = pyrender.IntrinsicsCamera(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            znear=0.01,
            zfar=10000.0,
        )
        scene.add(camera, pose=np.eye(4))

        light_pose = np.eye(4)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=light_pose)

        offset_light = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.6), pose=offset_light)

        r = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
        try:
            flags = RenderFlags.RGBA | RenderFlags.SHADOWS_DIRECTIONAL
            color_rgba, _ = r.render(scene, flags=flags)
        finally:
            r.delete()

        rendered_rgb = color_rgba[:, :, :3].astype(np.float32)
        rendered_bgr = rendered_rgb[:, :, ::-1]
        rendered_alpha = (color_rgba[:, :, 3:4].astype(np.float32) / 255.0) * float(alpha)

        base = image.astype(np.float32)
        blended = rendered_bgr * rendered_alpha + base * (1.0 - rendered_alpha)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        return blended
    except Exception as e:
        print(f'Pyrender overlay failed, fallback to polygon render: {e}')

    if mesh.faces is None or len(mesh.faces) == 0:
        raise ValueError('The mesh has no triangular faces to render.')

    overlay = image.copy()
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices_2d, vertices_depth = project_points(vertices, K, pose)

    faces = np.asarray(mesh.faces, dtype=np.int64)
    face_depth = vertices_depth[faces].mean(axis=1)
    render_order = np.argsort(face_depth)[::-1]

    fill_color = tuple(int(channel) for channel in color)
    outline_color = tuple(int(channel) for channel in edge_color)

    for face_idx in render_order:
        face = faces[face_idx]
        pts = vertices_2d[face]
        if np.any(~np.isfinite(pts)):
            continue
        if np.any(vertices_depth[face] <= 1e-6):
            continue

        pts_int = np.round(pts).astype(np.int32)
        if np.all((pts_int[:, 0] < 0) | (pts_int[:, 0] >= width) | (pts_int[:, 1] < 0) | (pts_int[:, 1] >= height)):
            continue

        cv2.fillConvexPoly(overlay, pts_int, fill_color)
        if draw_edges:
            cv2.polylines(overlay, [pts_int.reshape(-1, 1, 2)], True, outline_color, edge_thickness, cv2.LINE_AA)

    blended = cv2.addWeighted(overlay, alpha, image, 1.0 - alpha, 0.0)
    return blended

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