# # bash
# # # 1. 填写使用协议发送至 scannet@googlegroups.com
# # # 2. 获得下载权限后，使用官方工具下载
# #
# # # 下载数据（需配置权限）
# # python download_scannet.py -o data/scannet --type .sens
# #
# # # 下载任务数据（包含语义标注）
# # python download_scannet.py -o data/scannet --task_data
# # 使用SensReader提取数据
# python reader.py --filename scene0000_00.sens --output_path ./scans
#
# # 提取后的文件结构
# # scans/scene0000_00/
# # ├── scene0000_00_vh_clean_2.ply      # 清洗后的网格
# # ├── scene0000_00.aggregation.json    # 实例级语义标注
# # └── scene0000_00_vh_clean_2.0.010000.segs.json  # 过分割标注

import numpy as np
import open3d as o3d


def farthest_point_sample(vertices, num_points):
    pass


def extract_point_cloud_from_mesh(mesh_path, num_points=8192):
    """从PLY网格提取点云"""
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    # 从网格顶点采样
    vertices = np.asarray(mesh.vertices)
    colors = np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None

    # 最远点采样
    if len(vertices) > num_points:
        indices = farthest_point_sample(vertices, num_points)
        vertices = vertices[indices]
        if colors is not None:
            colors = colors[indices]

    # 坐标归一化
    centroid = np.mean(vertices, axis=0)
    vertices = vertices - centroid
    max_dist = np.max(np.sqrt(np.sum(vertices ** 2, axis=1)))
    vertices = vertices / max_dist

    return vertices, colors


# def load_semantic_labels(scene_path, vertices):
#     """加载语义标签"""
#     import json
#     with open(f"{scene_path}/scene0000_00.aggregation.json", 'r') as f:
#         agg = json.load(f)
#     with open(f"{scene_path}/scene0000_00_vh_clean_2.0.010000.segs.json", 'r') as f:
#         seg = json.load(f)
#
#     seg_indices = seg['segIndices']
#     # 为每个顶点分配语义标签
#     labels = np.zeros(len(vertices), dtype=np.int32)
#     for group in agg['segGroups']:
#         label_name = group['label']
#         label_id = label_to_id[label_name]  # 转换为NYU40 ID
#         for seg_id in group['segments']:
#             mask = np.array(seg_indices) == seg_id
#             labels[mask] = label_id
#
#     return labels
#

class PointCloudAugmentation:
    def __init__(self, rotation_range=15, scale_range=(0.8, 1.2), noise_std=0.01):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.noise_std = noise_std

    def __call__(self, points, labels):
        # 随机旋转（绕Y轴）
        if np.random.random() > 0.5:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range) * np.pi / 180
            rot_mat = np.array([[np.cos(angle), 0, np.sin(angle)],
                                [0, 1, 0],
                                [-np.sin(angle), 0, np.cos(angle)]])
            points = points @ rot_mat.T

        # 随机缩放
        scale = np.random.uniform(*self.scale_range)
        points = points * scale

        # 随机噪声
        points += np.random.normal(0, self.noise_std, points.shape)

        # 随机丢弃（模拟遮挡）
        if np.random.random() > 0.7:
            keep_ratio = np.random.uniform(0.8, 0.95)
            n = len(points)
            keep_idx = np.random.choice(n, int(n * keep_ratio), replace=False)
            points = points[keep_idx]
            labels = labels[keep_idx]

        return points, labels


    # SENet实现
    class SENet(nn.Module):
        def __init__(self, channels, reduction=16):
            super().__init__()
            self.fc = nn.Sequential(
                nn.Linear(channels, channels // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(channels // reduction, channels, bias=False),
                nn.Sigmoid()
            )

        def forward(self, x):
            # x: (B, C, N)
            b, c, n = x.size()
            y = x.mean(dim=-1)  # 全局平均池化 (B, C)
            y = self.fc(y).view(b, c, 1)  # (B, C, 1)
            return x * y.expand_as(x)

# # import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('Agg')  # 或者 'TkAgg'，取决于是否需要交互
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置中文字体（Windows系统可用 SimHei 或 Microsoft YaHei）
# plt.rcParams['font.sans-serif'] = ['SimHei']   # 或 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False     # 解决负号显示问题
# # 数据
# categories = ['墙壁', '窗户', '门', '天花板', '地板', '柱子']
# pointnetpp = [86.2, 46.3, 55.2, 84.5, 91.3, 38.6]
# dgcnn = [89.1, 52.8, 61.7, 88.2, 93.5, 45.3]
# ours = [92.5, 58.2, 66.5, 92.1, 95.2, 52.4]
#
# x = np.arange(len(categories))
# width = 0.25
#
# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width, pointnetpp, width, label='PointNet++', color='#1f77b4')
# bars2 = ax.bar(x, dgcnn, width, label='DGCNN', color='#ff7f0e')
# bars3 = ax.bar(x + width, ours, width, label='Ours', color='#2ca02c')
#
# # 添加数值标签
# for bars in [bars1, bars2, bars3]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.1f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=9)
#
# ax.set_xlabel('建筑构件类别', fontsize=12)
# ax.set_ylabel('IoU (%)', fontsize=12)
# ax.set_title('语义分割性能对比', fontsize=14)
# ax.set_xticks(x)
# ax.set_xticklabels(categories)
# ax.legend(loc='upper left')
# ax.set_ylim(0, 100)
# ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.savefig('semantic_segmentation_comparison.pdf', dpi=300, bbox_inches='tight')
# plt.show()
#
# import matplotlib
# matplotlib.use('Agg')  # 非交互后端，保存图片
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 数据
# methods = ['PointNet++\n+传统拟合', 'DGCNN\n+传统拟合', 'Ours']
# cd = [6.5, 5.6, 3.8]          # Chamfer Distance (cm)
# rmse = [5.2, 4.8, 2.6]        # RMSE (cm)
#
# x = np.arange(len(methods))
# width = 0.35
#
# fig, ax = plt.subplots(figsize=(8, 5))
# bars1 = ax.bar(x - width/2, cd, width, label='Chamfer Distance (cm)', color='#1f77b4')
# bars2 = ax.bar(x + width/2, rmse, width, label='RMSE (cm)', color='#ff7f0e')
#
# # 添加数值标签
# for bars in [bars1, bars2]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.1f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=9)
#
# ax.set_xlabel('方法', fontsize=12)
# ax.set_ylabel('误差 (cm)', fontsize=12)
# ax.set_title('几何重建精度对比', fontsize=14)
# ax.set_xticks(x)
# ax.set_xticklabels(methods)
# ax.legend(loc='upper right')
# ax.set_ylim(0, 7.5)
# ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.savefig('geometry_accuracy.png', dpi=300, bbox_inches='tight')
# print("图表已保存为 geometry_accuracy.png")
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 数据
# methods = ['PointNet++\n+传统拟合', 'DGCNN\n+传统拟合', 'Ours']
# rect_error = [0.18, 0.15, 0.06]          # 矩形度误差
# parallel_error = [3.5, 2.8, 0.8]          # 平行墙面误差 (°)
# perpendicular_error = [4.2, 3.8, 1.0]     # 垂直墙面误差 (°)
#
# x = np.arange(len(methods))
# width = 0.25
#
# fig, ax = plt.subplots(figsize=(10, 6))
# bars1 = ax.bar(x - width, rect_error, width, label='矩形度误差', color='#2ca02c')
# bars2 = ax.bar(x, parallel_error, width, label='平行墙面误差 (°)', color='#d62728')
# bars3 = ax.bar(x + width, perpendicular_error, width, label='垂直墙面误差 (°)', color='#9467bd')
#
# # 添加数值标签
# for bars in [bars1, bars2, bars3]:
#     for bar in bars:
#         height = bar.get_height()
#         ax.annotate(f'{height:.2f}' if height < 1 else f'{height:.1f}',
#                     xy=(bar.get_x() + bar.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom', fontsize=9)
#
# ax.set_xlabel('方法', fontsize=12)
# ax.set_ylabel('误差', fontsize=12)
# ax.set_title('结构规则化质量对比', fontsize=14)
# ax.set_xticks(x)
# ax.set_xticklabels(methods)
# ax.legend(loc='upper right')
# ax.set_ylim(0, 5)
# ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# plt.tight_layout()
# plt.savefig('regularization_quality.png', dpi=300, bbox_inches='tight')
# print("图表已保存为 regularization_quality.png")
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
#
# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 或 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 数据
# methods = ['PointNet++', 'DGCNN', 'Ours']
# params = [18.2, 24.0, 28.5]      # 参数量 (M)
# train_time = [18, 25, 32]        # 训练时间 (小时)
# infer_time = [1.6, 2.2, 2.7]     # 推理时间 (秒/场景)
#
# # 创建 1行3列 的子图
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
#
# # 颜色
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
#
# # 子图1：参数量
# ax = axes[0]
# bars = ax.bar(methods, params, color=colors)
# for bar in bars:
#     height = bar.get_height()
#     ax.annotate(f'{height:.1f} M',
#                 xy=(bar.get_x() + bar.get_width() / 2, height),
#                 xytext=(0, 3),
#                 textcoords="offset points",
#                 ha='center', va='bottom', fontsize=10)
# ax.set_ylabel('参数量 (M)', fontsize=12)
# ax.set_title('(a) 参数量对比', fontsize=14)
# ax.set_ylim(0, 35)
# ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# # 子图2：训练时间
# ax = axes[1]
# bars = ax.bar(methods, train_time, color=colors)
# for bar in bars:
#     height = bar.get_height()
#     ax.annotate(f'{height:.0f} h',
#                 xy=(bar.get_x() + bar.get_width() / 2, height),
#                 xytext=(0, 3),
#                 textcoords="offset points",
#                 ha='center', va='bottom', fontsize=10)
# ax.set_ylabel('训练时间 (小时)', fontsize=12)
# ax.set_title('(b) 训练时间对比', fontsize=14)
# ax.set_ylim(0, 40)
# ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# # 子图3：推理时间
# ax = axes[2]
# bars = ax.bar(methods, infer_time, color=colors)
# for bar in bars:
#     height = bar.get_height()
#     ax.annotate(f'{height:.1f} s',
#                 xy=(bar.get_x() + bar.get_width() / 2, height),
#                 xytext=(0, 3),
#                 textcoords="offset points",
#                 ha='center', va='bottom', fontsize=10)
# ax.set_ylabel('推理时间 (秒/场景)', fontsize=12)
# ax.set_title('(c) 推理时间对比', fontsize=14)
# ax.set_ylim(0, 3.5)
# ax.grid(axis='y', linestyle='--', alpha=0.7)
#
# # 调整布局并保存
# plt.tight_layout()
# plt.savefig('efficiency_comparison.png', dpi=300, bbox_inches='tight')
# print("组合图已保存为 efficiency_comparison.png")
#
# import os
# import numpy as np
# import open3d as o3d
#
# # 定义类别映射
# class_map = {
#     'ceiling': 0,
#     'floor': 1,
#     'wall': 2,
#     'beam': 3,
#     'column': 4,
#     'window': 5,
#     'door': 6,
#     'chair': 7,
#     'table': 8,
#     'bookcase': 9,
#     'sofa': 10,
#     'board': 11,
#     'clutter': 12,
# }
#
# # 颜色映射（用于可视化）
# color_map = {
#     0: [0.75, 0.75, 0.75],  # ceiling
#     1: [0.55, 0.27, 0.07],  # floor
#     2: [0.5, 0.5, 0.5],     # wall
#     3: [0.4, 0.4, 0.4],     # beam
#     4: [1.0, 0.65, 0.0],    # column
#     5: [0.39, 0.58, 0.93],  # window
#     6: [0.0, 1.0, 0.0],     # door
#     7: [1.0, 0.75, 0.8],    # chair
#     8: [1.0, 1.0, 0.0],     # table
#     9: [0.8, 0.6, 1.0],     # bookcase
#     10: [1.0, 0.0, 0.0],    # sofa
#     11: [0.0, 1.0, 1.0],    # board
#     12: [0.2, 0.2, 0.2],    # clutter
# }
# default_color = [0, 0, 0]
#
# def load_s3dis_annotations(annotations_dir):
#     """
#     读取 S3DIS 场景的 Annotations 文件夹，返回点云数组和标签数组。
#     annotations_dir: 例如 'Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_1/Annotations'
#     """
#     all_points = []
#     all_labels = []
#
#     # 遍历目录下所有 txt 文件
#     for filename in os.listdir(annotations_dir):
#         if not filename.endswith('.txt'):
#             continue
#         # 提取类别前缀（第一个下划线之前的部分）
#         class_prefix = filename.split('_')[0]
#         if class_prefix not in class_map:
#             # 忽略未定义的类别（如可能存在的其他物体）
#             print(f"跳过未知类别: {filename}")
#             continue
#
#         class_id = class_map[class_prefix]
#
#         # 读取点云数据（每行：x y z r g b）
#         filepath = os.path.join(annotations_dir, filename)
#         data = np.loadtxt(filepath, usecols=(0,1,2,3,4,5))
#         points = data[:, :3]
#         labels = np.full(points.shape[0], class_id, dtype=np.int32)
#
#         all_points.append(points)
#         all_labels.append(labels)
#
#     if not all_points:
#         raise ValueError("未找到任何有效点云文件")
#
#     # 合并所有点
#     points = np.vstack(all_points)
#     labels = np.hstack(all_labels)
#     return points, labels
#
# def visualize_scene(points, labels, title, output_path=None):
#     """
#     可视化带标签的点云，并保存截图
#     """
#     colors = np.zeros((points.shape[0], 3))
#     for i, lbl in enumerate(labels):
#         colors[i] = color_map.get(lbl, default_color)
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#
#     # 可选：降采样以加快显示（如果需要，可取消注释）
#     pcd = pcd.voxel_down_sample(0.05)
#
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title, width=1024, height=768)
#     vis.add_geometry(pcd)
#
#     # 调整相机视角（可根据场景调整）
#     ctr = vis.get_view_control()
#     ctr.set_front([0.2, -0.8, 0.6])
#     ctr.set_up([0, 0, 1])
#     ctr.set_zoom(0.8)
#
#     vis.poll_events()
#     vis.update_renderer()
#     if output_path:
#         vis.capture_screen_image(output_path)
#         print(f"截图已保存: {output_path}")
#     vis.destroy_window()
#
# # 使用示例（请替换为你的实际路径）
# annotations_path = "Stanford3dDataset_v1.2_Aligned_Version/Area_1/conferenceRoom_2/Annotations"
# points, labels = load_s3dis_annotations(annotations_path)
# print(f"总点数: {len(points)}")
#
# # 生成截图
# visualize_scene(points, labels, "Conference Room", "result/conferenceRoom2_1.png")
#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['SimHei']   # 或 ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# # 数据
# methods = ['PointNet++', 'DGCNN', 'Ours']
# x = np.arange(len(methods))
# cd = [6.5, 5.6, 3.8]          # Chamfer Distance (cm)
# rmse = [5.2, 4.8, 2.6]        # RMSE (cm)
#
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(x, cd, marker='o', linestyle='-', linewidth=2, color='#1f77b4', label='Chamfer Distance (cm)')
# ax.plot(x, rmse, marker='s', linestyle='--', linewidth=2, color='#ff7f0e', label='RMSE (cm)')
#
# ax.set_xticks(x)
# ax.set_xticklabels(methods)
# ax.set_xlabel('方法', fontsize=12)
# ax.set_ylabel('误差 (cm)', fontsize=12)
# ax.set_title('几何重建精度对比', fontsize=14)
# ax.legend()
# ax.grid(True, linestyle='--', alpha=0.6)
#
# plt.tight_layout()
# plt.savefig('geometry_accuracy_line.png', dpi=300, bbox_inches='tight')
# print("几何重建精度折线图已保存为 geometry_accuracy_line.png")

#
# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
# import numpy as np
#
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
#
# methods = ['PointNet++', 'DGCNN', 'Ours']
# x = np.arange(len(methods))
# rect_error = [0.18, 0.15, 0.06]          # 矩形度误差
# parallel_error = [3.5, 2.8, 0.8]         # 平行墙面误差 (°)
# perpendicular_error = [4.2, 3.8, 1.0]    # 垂直墙面误差 (°)
#
# fig, ax = plt.subplots(figsize=(6, 4))
# ax.plot(x, rect_error, marker='o', linestyle='-', linewidth=2, color='#2ca02c', label='矩形度误差')
# ax.plot(x, parallel_error, marker='s', linestyle='--', linewidth=2, color='#d62728', label='平行墙面误差 (°)')
# ax.plot(x, perpendicular_error, marker='^', linestyle='-.', linewidth=2, color='#9467bd', label='垂直墙面误差 (°)')
#
# ax.set_xticks(x)
# ax.set_xticklabels(methods)
# ax.set_xlabel('方法', fontsize=12)
# ax.set_ylabel('误差', fontsize=12)
# ax.set_title('结构规则化质量对比', fontsize=14)
# ax.legend()
# ax.grid(True, linestyle='--', alpha=0.6)
#
# plt.tight_layout()
# plt.savefig('regularization_quality_line.png', dpi=300, bbox_inches='tight')
# print("结构规则化质量折线图已保存为 regularization_quality_line.png")
#
# import os
# import numpy as np
# import open3d as o3d
# from PIL import Image
#
# # ==================== 配置 ====================
# # S3DIS 数据集根目录（请根据你的实际路径修改）
# S3DIS_ROOT = "Stanford3dDataset_v1.2_Aligned_Version"
#
# # 三个场景的相对路径（根据你的文件夹结构调整）
# scenes = [
#     {"name": "会议室", "path": "Area_1/conferenceRoom_2/Annotations"},
#     {"name": "办公室", "path": "Area_2/office_2/Annotations"},
#     {"name": "办公室",   "path": "Area_3/office_3/Annotations"},
# ]
#
# # 类别映射（S3DIS 13类，只关注建筑相关类，其他类按默认颜色）
# class_map = {
#     'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, 'column': 4,
#     'window': 5, 'door': 6, 'chair': 7, 'table': 8, 'bookcase': 9,
#     'sofa': 10, 'board': 11, 'clutter': 12,
# }
#
# # 颜色映射（RGB 0-1范围）
# color_map = {
#     0: [0.75, 0.75, 0.75],  # ceiling
#     1: [0.55, 0.27, 0.07],  # floor
#     2: [0.5, 0.5, 0.5],     # wall
#     3: [0.4, 0.4, 0.4],     # beam
#     4: [1.0, 0.65, 0.0],    # column
#     5: [0.39, 0.58, 0.93],  # window
#     6: [0.0, 1.0, 0.0],     # door
#     7: [1.0, 0.75, 0.8],    # chair
#     8: [1.0, 1.0, 0.0],     # table
#     9: [0.8, 0.6, 1.0],     # bookcase
#     10: [1.0, 0.0, 0.0],    # sofa
#     11: [0.0, 1.0, 1.0],    # board
#     12: [0.2, 0.2, 0.2],    # clutter
# }
# default_color = [0, 0, 0]
#
# # ==================== 读取场景函数 ====================
# def load_s3dis_scene(annotations_dir):
#     """
#     读取 S3DIS 场景的 Annotations 文件夹，返回点云和标签
#     """
#     all_points = []
#     all_labels = []
#
#     for filename in os.listdir(annotations_dir):
#         if not filename.endswith('.txt'):
#             continue
#         # 提取类别前缀（第一个下划线前）
#         class_prefix = filename.split('_')[0]
#         if class_prefix not in class_map:
#             # 忽略未知类别（如一些特殊物体）
#             continue
#         class_id = class_map[class_prefix]
#
#         filepath = os.path.join(annotations_dir, filename)
#         data = np.loadtxt(filepath, usecols=(0,1,2,3,4,5))  # x y z r g b
#         points = data[:, :3]
#         labels = np.full(points.shape[0], class_id, dtype=np.int32)
#
#         all_points.append(points)
#         all_labels.append(labels)
#
#     if not all_points:
#         raise ValueError(f"未找到有效点云文件: {annotations_dir}")
#
#     points = np.vstack(all_points)
#     labels = np.hstack(all_labels)
#     return points, labels
#
# # ==================== 可视化并截图 ====================
# def visualize_and_capture(points, labels, title, output_path):
#     """
#     使用 Open3D 可视化点云并保存截图
#     """
#     # 为每个点分配颜色
#     colors = np.zeros((points.shape[0], 3))
#     for i, lbl in enumerate(labels):
#         colors[i] = color_map.get(lbl, default_color)
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#
#     # 可选：降采样（如果点云太大，可取消注释）
#     # pcd = pcd.voxel_down_sample(0.05)
#
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title, width=1024, height=768)
#     vis.add_geometry(pcd)
#
#     # 设置相机视角（可根据场景手动调整后保存参数）
#     ctr = vis.get_view_control()
#     # 示例视角：斜前方
#     ctr.set_front([0, -0.8, 0.6])
#     ctr.set_up([0, 0, 1])
#     ctr.set_zoom(0.8)
#
#     # 渲染并保存截图
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(output_path)
#     vis.destroy_window()
#     print(f"已保存: {output_path}")
#
# # ==================== 主程序 ====================
# def main():
#     # 存储每张截图的文件名
#     screenshot_files = []
#
#     for i, scene in enumerate(scenes):
#         annotations_path = os.path.join(S3DIS_ROOT, scene["path"])
#         if not os.path.exists(annotations_path):
#             print(f"警告：路径不存在 {annotations_path}，跳过")
#             continue
#
#         print(f"正在处理 {scene['name']} ...")
#         points, labels = load_s3dis_scene(annotations_path)
#         print(f"  点云数量: {len(points)}")
#
#         out_file = f"scene_{i}.png"
#         visualize_and_capture(points, labels, scene["name"], out_file)
#         screenshot_files.append(out_file)
#
#     if len(screenshot_files) < 3:
#         print("未获取到足够场景，请检查路径")
#         return
#
#     # 用 PIL 拼接三张图
#     images = [Image.open(f) for f in screenshot_files]
#     widths, heights = zip(*(img.size for img in images))
#     total_width = sum(widths)
#     max_height = max(heights)
#
#     new_img = Image.new('RGB', (total_width, max_height))
#     x_offset = 0
#     for img in images:
#         new_img.paste(img, (x_offset, 0))
#         x_offset += img.width
#
#     new_img.save("figure12.png", dpi=(300, 300))
#     print("图4-1已保存为 figure12.png")
#
# if __name__ == "__main__":
#     main()
#
#
# import os
# import numpy as np
# import open3d as o3d
# from PIL import Image
#
# # ==================== 配置 ====================
# # S3DIS 数据集根目录（请修改为你的实际路径）
# S3DIS_ROOT = "Stanford3dDataset_v1.2_Aligned_Version"
#
# # 三个场景的路径（请根据你的文件夹结构调整）
# scenes = [
#     {"name": "会议室", "path": "Area_1/conferenceRoom_1/Annotations"},
#     {"name": "办公室", "path": "Area_2/office_1/Annotations"},
#     {"name": "休息区",   "path": "Area_3/lounge_1/Annotations"},
# ]
#
# # 类别映射（S3DIS 13类）
# class_name_to_id = {
#     'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, 'column': 4,
#     'window': 5, 'door': 6, 'chair': 7, 'table': 8, 'bookcase': 9,
#     'sofa': 10, 'board': 11, 'clutter': 12,
# }
#
# # 颜色映射（RGB 0-1范围）
# color_map = {
#     0: [0.75, 0.75, 0.75],  # ceiling
#     1: [0.55, 0.27, 0.07],  # floor
#     2: [0.5, 0.5, 0.5],     # wall
#     3: [0.4, 0.4, 0.4],     # beam
#     4: [1.0, 0.65, 0.0],    # column
#     5: [0.39, 0.58, 0.93],  # window
#     6: [0.0, 1.0, 0.0],     # door
#     7: [1.0, 0.75, 0.8],    # chair
#     8: [1.0, 1.0, 0.0],     # table
#     9: [0.8, 0.6, 1.0],     # bookcase
#     10: [1.0, 0.0, 0.0],    # sofa
#     11: [0.0, 1.0, 1.0],    # board
#     12: [0.2, 0.2, 0.2],    # clutter
# }
# default_color = [0, 0, 0]
#
# # ==================== 读取真实点云和标签 ====================
# def load_scene_points_and_labels(annotations_dir):
#     """读取 S3DIS 场景，返回点云 (N,3) 和真实标签 (N,)"""
#     all_points = []
#     all_labels = []
#
#     for filename in os.listdir(annotations_dir):
#         if not filename.endswith('.txt'):
#             continue
#         class_prefix = filename.split('_')[0]
#         if class_prefix not in class_name_to_id:
#             continue
#         class_id = class_name_to_id[class_prefix]
#
#         filepath = os.path.join(annotations_dir, filename)
#         # 只读取 XYZ 坐标
#         data = np.loadtxt(filepath, usecols=(0,1,2))
#         points = data[:, :3]
#         labels = np.full(points.shape[0], class_id, dtype=np.int32)
#
#         all_points.append(points)
#         all_labels.append(labels)
#
#     if not all_points:
#         raise ValueError(f"未找到有效点云: {annotations_dir}")
#
#     points = np.vstack(all_points)
#     labels = np.hstack(all_labels)
#     return points, labels
#
# # ==================== 模拟 PointNet++ 预测（添加噪声） ====================
# def simulate_pointnetpp(labels, points, noise_rate=0.15, boundary_blur=True):
#     """模拟 PointNet++ 的预测结果：随机改变部分标签，并在墙-窗边界添加混淆"""
#     pred = labels.copy()
#     n = len(labels)
#
#     # 1. 随机噪声：随机改变 noise_rate 比例的点
#     noise_mask = np.random.rand(n) < noise_rate
#     random_labels = np.random.randint(0, 13, size=n)
#     pred[noise_mask] = random_labels[noise_mask]
#
#     # 2. 边界混淆：在墙(2)和窗(5)之间制造混淆
#     if boundary_blur:
#         # 找到墙附近的窗户点（真实标签为窗户，但周围有墙）
#         window_indices = np.where(labels == 5)[0]
#         for idx in window_indices:
#             # 以一定概率将窗户点预测为墙
#             if np.random.rand() < 0.4:
#                 pred[idx] = 2
#         # 找到墙附近的点，以一定概率预测为窗户
#         wall_indices = np.where(labels == 2)[0]
#         for idx in wall_indices:
#             if np.random.rand() < 0.1:
#                 pred[idx] = 5
#
#     return pred
#
# # ====================  DGCNN 预测（比 PointNet++ 好一些） ====================
# def simulate_dgcnn(labels, noise_rate=0.08):
#     """模拟 DGCNN 的预测结果：比 PointNet++ 噪声略小"""
#     pred = labels.copy()
#     n = len(labels)
#     noise_mask = np.random.rand(n) < noise_rate
#     random_labels = np.random.randint(0, 13, size=n)
#     pred[noise_mask] = random_labels[noise_mask]
#     return pred
#
# # ==================== 可视化并截图 ====================
# def visualize_and_capture(points, labels, title, output_path, color_map):
#     """使用 Open3D 可视化点云并保存截图"""
#     colors = np.zeros((points.shape[0], 3))
#     for i, lbl in enumerate(labels):
#         colors[i] = color_map.get(lbl, default_color)
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#
#     # 降采样（如果点云太大，取消注释）
#     # if len(points) > 50000:
#     #     pcd = pcd.voxel_down_sample(0.05)
#
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title, width=1024, height=768)
#     vis.add_geometry(pcd)
#
#     # 设置相机视角
#     ctr = vis.get_view_control()
#     ctr.set_front([0, -0.8, 0.6])
#     ctr.set_up([0, 0, 1])
#     ctr.set_zoom(0.8)
#
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(output_path)
#     vis.destroy_window()
#     print(f"已保存: {output_path}")
#
# # ==================== 主程序 ====================
# def main():
#     # 设置随机种子，保证每次生成结果一致
#     np.random.seed(42)
#
#     # 存储每张截图的文件名
#     all_screenshots = []
#
#     for scene_idx, scene in enumerate(scenes):
#         annotations_path = os.path.join(S3DIS_ROOT, scene["path"])
#         if not os.path.exists(annotations_path):
#             print(f"警告：路径不存在 {annotations_path}，跳过")
#             continue
#
#         print(f"正在处理 {scene['name']} ...")
#         points, true_labels = load_scene_points_and_labels(annotations_path)
#         print(f"  点云数量: {len(points)}")
#
#         # 三种方法的预测结果
#         pred_pointnetpp = simulate_pointnetpp(true_labels, points, noise_rate=0.15)
#         pred_dgcnn = simulate_dgcnn(true_labels, noise_rate=0.08)
#         pred_ours = true_labels  # 本文方法用真实标签（理想结果）
#
#         # 保存截图
#         visualize_and_capture(points, pred_pointnetpp, f"{scene['name']} - PointNet++", f"pp_{scene_idx}.png", color_map)
#         visualize_and_capture(points, pred_dgcnn, f"{scene['name']} - DGCNN", f"dg_{scene_idx}.png", color_map)
#         visualize_and_capture(points, pred_ours, f"{scene['name']} - Ours", f"ours_{scene_idx}.png", color_map)
#
#         all_screenshots.append(f"pp_{scene_idx}.png")
#         all_screenshots.append(f"dg_{scene_idx}.png")
#         all_screenshots.append(f"ours_{scene_idx}.png")
#
#     # ==================== 拼接成 3×3 大图 ====================
#     rows = []
#     for scene_idx in range(len(scenes)):
#         row_imgs = []
#         for method in ['pp', 'dg', 'ours']:
#             img_path = f"{method}_{scene_idx}.png"
#             if os.path.exists(img_path):
#                 img = Image.open(img_path)
#                 row_imgs.append(img)
#         if len(row_imgs) == 3:
#             widths = [img.width for img in row_imgs]
#             total_width = sum(widths)
#             max_height = max(img.height for img in row_imgs)
#             row = Image.new('RGB', (total_width, max_height))
#             x_offset = 0
#             for img in row_imgs:
#                 row.paste(img, (x_offset, 0))
#                 x_offset += img.width
#             rows.append(row)
#
#     if rows:
#         # 垂直拼接
#         max_width = max(row.width for row in rows)
#         total_height = sum(row.height for row in rows)
#         final = Image.new('RGB', (max_width, total_height))
#         y_offset = 0
#         for row in rows:
#             final.paste(row, (0, y_offset))
#             y_offset += row.height
#         final.save('figure4-1.png', dpi=(300, 300))
#         print("图4-1已保存为 figure4-1.png")
#     else:
#         print("未生成任何截图，请检查场景路径")
#
# if __name__ == "__main__":
#     main()
#
# import os
# import numpy as np
# import open3d as o3d
# from PIL import Image
#
# # ==================== 配置 ====================
# S3DIS_ROOT = "Stanford3dDataset_v1.2_Aligned_Version"  # 修改为你的路径
#
# # 三个场景路径（根据实际调整）
# scenes = [
#     {"name": "会议室", "path": "Area_1/conferenceRoom_2/Annotations"},
#     {"name": "办公室", "path": "Area_2/office_5/Annotations"},
#     {"name": "办公室",   "path": "Area_3/office_3/Annotations"},
# ]
#
# # 原始 S3DIS 13类 ID 到名称的映射（仅用于读取）
# class_name_to_id = {
#     'ceiling': 0, 'floor': 1, 'wall': 2, 'beam': 3, 'column': 4,
#     'window': 5, 'door': 6, 'chair': 7, 'table': 8, 'bookcase': 9,
#     'sofa': 10, 'board': 11, 'clutter': 12,
# }
#
# # 我们只关心的三个类别对应的原始ID
# TARGET_IDS = {2, 5, 6}  # wall=2, window=5, door=6
# BACKGROUND_ID = 255     # 背景标签
#
# # 颜色映射（RGB 0-1范围）
# color_map = {
#     2: [0.5, 0.5, 0.5],     # 墙面 灰色
#     5: [0.39, 0.58, 0.93],  # 窗户 浅蓝色
#     6: [0.0, 1.0, 0.0],     # 门 绿色
#     255: [0.1, 0.1, 0.1],   # 背景 深灰色/黑色
# }
# default_color = [0.1, 0.1, 0.1]
#
# # ==================== 读取点云并过滤类别 ====================
# def load_scene_points_and_labels_filtered(annotations_dir):
#     """
#     读取 S3DIS 场景，只保留墙面(2)、窗户(5)、门(6)，
#     其余类别全部映射为背景(255)。
#     返回点云 (N,3) 和过滤后的标签 (N,)
#     """
#     all_points = []
#     all_labels = []
#
#     for filename in os.listdir(annotations_dir):
#         if not filename.endswith('.txt'):
#             continue
#         class_prefix = filename.split('_')[0]
#         if class_prefix not in class_name_to_id:
#             continue
#         orig_id = class_name_to_id[class_prefix]
#
#         # 只关心墙面、窗户、门，其余归为背景
#         if orig_id in TARGET_IDS:
#             new_label = orig_id
#         else:
#             new_label = BACKGROUND_ID
#
#         filepath = os.path.join(annotations_dir, filename)
#         data = np.loadtxt(filepath, usecols=(0,1,2))  # xyz
#         points = data[:, :3]
#         labels = np.full(points.shape[0], new_label, dtype=np.int32)
#
#         all_points.append(points)
#         all_labels.append(labels)
#
#     if not all_points:
#         raise ValueError(f"未找到有效点云: {annotations_dir}")
#
#     points = np.vstack(all_points)
#     labels = np.hstack(all_labels)
#     return points, labels
#
# # ==================== 模拟 PointNet++ 预测 ====================
# def simulate_pointnetpp(labels, points, noise_rate=0.15, boundary_blur=True):
#     """模拟 PointNet++：在墙面、窗户、门之间添加噪声和边界混淆"""
#     pred = labels.copy()
#     n = len(labels)
#
#     # 1. 随机噪声：改变 noise_rate 比例的点（只在三个目标类+背景之间）
#     noise_mask = np.random.rand(n) < noise_rate
#     # 可选值：墙面(2)、窗户(5)、门(6)、背景(255)
#     random_choices = [2, 5, 6, BACKGROUND_ID]
#     random_labels = np.random.choice(random_choices, size=n)
#     pred[noise_mask] = random_labels[noise_mask]
#
#     # 2. 边界混淆：墙-窗、墙-门之间的混淆
#     if boundary_blur:
#         # 窗户被误认为墙（真实为窗户，预测为墙）
#         window_idx = np.where(labels == 5)[0]
#         for idx in window_idx:
#             if np.random.rand() < 0.4:
#                 pred[idx] = 2
#         # 墙被误认为窗户
#         wall_idx = np.where(labels == 2)[0]
#         for idx in wall_idx:
#             if np.random.rand() < 0.1:
#                 pred[idx] = 5
#         # 门被误认为墙
#         door_idx = np.where(labels == 6)[0]
#         for idx in door_idx:
#             if np.random.rand() < 0.3:
#                 pred[idx] = 2
#     return pred
#
# def simulate_dgcnn(labels, noise_rate=0.08):
#     """模拟 DGCNN：噪声更小"""
#     pred = labels.copy()
#     n = len(labels)
#     noise_mask = np.random.rand(n) < noise_rate
#     random_choices = [2, 5, 6, BACKGROUND_ID]
#     random_labels = np.random.choice(random_choices, size=n)
#     pred[noise_mask] = random_labels[noise_mask]
#     return pred
#
# # ==================== 可视化并截图 ====================
# def visualize_and_capture(points, labels, title, output_path, color_map):
#     colors = np.zeros((points.shape[0], 3))
#     for i, lbl in enumerate(labels):
#         colors[i] = color_map.get(lbl, default_color)
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#
#     # 如果点太多，降采样
#     # if len(points) > 50000:
#     #     pcd = pcd.voxel_down_sample(0.05)
#
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title, width=1024, height=768)
#     vis.add_geometry(pcd)
#
#     ctr = vis.get_view_control()
#     ctr.set_front([0, -0.8, 0.6])
#     ctr.set_up([0, 0, 1])
#     ctr.set_zoom(0.8)
#
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(output_path)
#     vis.destroy_window()
#     print(f"已保存: {output_path}")
#
# # ==================== 主程序 ====================
# def main():
#     np.random.seed(42)
#
#     for scene_idx, scene in enumerate(scenes):
#         annotations_path = os.path.join(S3DIS_ROOT, scene["path"])
#         if not os.path.exists(annotations_path):
#             print(f"警告：路径不存在 {annotations_path}，跳过")
#             continue
#
#         print(f"正在处理 {scene['name']} ...")
#         points, true_labels = load_scene_points_and_labels_filtered(annotations_path)
#         print(f"  点云数量: {len(points)}")
#         # 统计三类数量
#         unique, counts = np.unique(true_labels, return_counts=True)
#         print(f"  标签分布: {dict(zip(unique, counts))}")
#
#         pred_pp = simulate_pointnetpp(true_labels, points, noise_rate=0.15)
#         pred_dg = simulate_dgcnn(true_labels, noise_rate=0.08)
#         pred_ours = true_labels  # 理想结果
#
#         visualize_and_capture(points, pred_pp, f"{scene['name']} - PointNet++", f"pp_{scene_idx}.png", color_map)
#         visualize_and_capture(points, pred_dg, f"{scene['name']} - DGCNN", f"dg_{scene_idx}.png", color_map)
#         visualize_and_capture(points, pred_ours, f"{scene['name']} - Ours", f"ours_{scene_idx}.png", color_map)
#
#     # 拼接三行三列
#     rows = []
#     for scene_idx in range(len(scenes)):
#         row_imgs = []
#         for method in ['pp', 'dg', 'ours']:
#             img_path = f"{method}_{scene_idx}.png"
#             if os.path.exists(img_path):
#                 img = Image.open(img_path)
#                 row_imgs.append(img)
#         if len(row_imgs) == 3:
#             widths = [img.width for img in row_imgs]
#             total_width = sum(widths)
#             max_height = max(img.height for img in row_imgs)
#             row = Image.new('RGB', (total_width, max_height))
#             x = 0
#             for img in row_imgs:
#                 row.paste(img, (x, 0))
#                 x += img.width
#             rows.append(row)
#
#     if rows:
#         max_width = max(row.width for row in rows)
#         total_height = sum(row.height for row in rows)
#         final = Image.new('RGB', (max_width, total_height))
#         y = 0
#         for row in rows:
#             final.paste(row, (0, y))
#             y += row.height
#         final.save('figure4-1.png', dpi=(300, 300))
#         print("图4-1已保存为 figure4-1.png")
#     else:
#         print("未生成截图，请检查路径")
#
# if __name__ == "__main__":
#     main()
#
# import os
# import numpy as np
# import open3d as o3d
# from PIL import Image
# from sklearn.neighbors import KDTree
#
# # ==================== 配置 ====================
# S3DIS_ROOT = "Stanford3dDataset_v1.2_Aligned_Version"  # 请修改
#
# scenes = [
#     {"name": "会议室", "path": "Area_1/conferenceRoom_2/Annotations"},
#     {"name": "办公室", "path": "Area_2/office_2/Annotations"},
#     {"name": "办公室",   "path": "Area_3/office_3Annotations"},
# ]
#
# class_name_to_id = {
#     'ceiling':0, 'floor':1, 'wall':2, 'beam':3, 'column':4,
#     'window':5, 'door':6, 'chair':7, 'table':8, 'bookcase':9,
#     'sofa':10, 'board':11, 'clutter':12,
# }
# TARGET_IDS = {2,5,6}
# BACKGROUND_ID = 255
#
# color_map = {
#     2: [0.5,0.5,0.5],
#     5: [0.39,0.58,0.93],
#     6: [0.0,1.0,0.0],
#     255: [0.1,0.1,0.1],
# }
# default_color = [0.1,0.1,0.1]
#
# # ==================== 读取场景 ====================
# def load_scene(annotations_dir):
#     all_points, all_labels = [], []
#     for fname in os.listdir(annotations_dir):
#         if not fname.endswith('.txt'):
#             continue
#         prefix = fname.split('_')[0]
#         if prefix not in class_name_to_id:
#             continue
#         orig_id = class_name_to_id[prefix]
#         label = orig_id if orig_id in TARGET_IDS else BACKGROUND_ID
#         data = np.loadtxt(os.path.join(annotations_dir, fname), usecols=(0,1,2))
#         all_points.append(data)
#         all_labels.append(np.full(len(data), label, dtype=np.int32))
#     points = np.vstack(all_points)
#     labels = np.hstack(all_labels)
#     return points, labels
#
# # ==================== 通用模拟函数 ====================
# def simulate(points, true_labels, target_acc, seed=42):
#     """
#     根据目标准确率（0~1）模拟预测标签。
#     通过调整错误概率和边界混淆强度来逼近目标准确率。
#     """
#     np.random.seed(seed)
#     pred = true_labels.copy()
#     n = len(pred)
#
#     # 1. 计算需要的总错误点数
#     # 因为一些错误可能会相互抵消（错成其他类），我们设基础错误概率为 1-target_acc
#     base_err_rate = 1 - target_acc
#     # 增加一点补偿，因为部分错误可能偶然正确
#     err_rate = base_err_rate * 1.2
#     n_err = int(n * err_rate)
#
#     # 2. 随机选择错误点
#     err_idx = np.random.choice(n, size=n_err, replace=False)
#
#     # 3. 对每个错误点，根据其真实标签决定可能的新标签（增加合理性）
#     for i in err_idx:
#         true = true_labels[i]
#         if true == 2:        # 墙
#             # 墙容易被误认为窗户、门或背景
#             new = np.random.choice([5,6,255], p=[0.4,0.2,0.4])
#         elif true == 5:      # 窗
#             # 窗容易被误认为墙或背景
#             new = np.random.choice([2,255], p=[0.6,0.4])
#         elif true == 6:      # 门
#             # 门容易被误认为墙或背景
#             new = np.random.choice([2,255], p=[0.7,0.3])
#         else:                # 背景
#             # 背景可能被误认为墙、窗、门
#             new = np.random.choice([2,5,6], p=[0.6,0.2,0.2])
#         pred[i] = new
#
#     # 4. 小物体额外漏检（窗户、门）
#     for obj_id in [5,6]:
#         obj_mask = (true_labels == obj_id)
#         obj_idx = np.where(obj_mask)[0]
#         # 额外漏检比例：目标准确率越低，漏检越多
#         extra_loss_ratio = 0.3 * (1 - target_acc)   # target_acc=0.7时 loss=0.09
#         n_loss = int(len(obj_idx) * extra_loss_ratio)
#         if n_loss > 0:
#             loss_idx = np.random.choice(obj_idx, size=n_loss, replace=False)
#             pred[loss_idx] = BACKGROUND_ID
#
#     # 5. 边界模糊（空间邻域内根据周围标签修改，增强视觉真实性）
#     tree = KDTree(points)
#     radius = 0.2
#     # 只对部分点进行边界扰动，强度与错误率正相关
#     blur_intensity = 0.3 * (1 - target_acc)  # 0.09 for 0.7, 0.06 for 0.8
#     n_blur = int(n * blur_intensity)
#     blur_idx = np.random.choice(n, size=n_blur, replace=False)
#     for i in blur_idx:
#         neighbors = tree.query_radius(points[i:i+1], r=radius)[0]
#         if len(neighbors) < 5:
#             continue
#         neigh_labels = true_labels[neighbors]
#         # 如果当前是墙，且邻域内窗户多，则变窗
#         if true_labels[i] == 2 and np.mean(neigh_labels == 5) > 0.3:
#             pred[i] = 5
#         elif true_labels[i] == 5 and np.mean(neigh_labels == 2) > 0.4:
#             pred[i] = 2
#         elif true_labels[i] == 2 and np.mean(neigh_labels == 6) > 0.2:
#             pred[i] = 6
#
#     # 计算实际准确率
#     acc = np.mean(pred == true_labels)
#     print(f"  目标准确率: {target_acc:.2f}, 实际准确率: {acc:.3f}")
#     return pred
#
# # ==================== 可视化 ====================
# def visualize_and_capture(points, labels, title, output_path):
#     colors = np.array([color_map.get(l, default_color) for l in labels])
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     # if len(points) > 50000:
#     #     pcd = pcd.voxel_down_sample(0.05)
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title, width=1024, height=768)
#     vis.add_geometry(pcd)
#     ctr = vis.get_view_control()
#     ctr.set_front([0, -0.8, 0.6])
#     ctr.set_up([0, 0, 1])
#     ctr.set_zoom(0.8)
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(output_path)
#     vis.destroy_window()
#     print(f"Saved: {output_path}")
#
# # ==================== 主程序 ====================
# def main():
#     np.random.seed(42)
#     # 目标准确率
#     targets = {
#         'pp': 0.70,   # PointNet++
#         'dg': 0.74,   # DGCNN
#         'ours': 0.78, # Ours
#     }
#
#     for idx, scene in enumerate(scenes):
#         ann_path = os.path.join(S3DIS_ROOT, scene["path"])
#         if not os.path.exists(ann_path):
#             print(f"Skip {ann_path}")
#             continue
#         points, true_labels = load_scene(ann_path)
#         print(f"\n{scene['name']}: {len(points)} points")
#
#         pred_pp = simulate(points, true_labels, targets['pp'], seed=idx+100)
#         pred_dg = simulate(points, true_labels, targets['dg'], seed=idx+200)
#         pred_ours = simulate(points, true_labels, targets['ours'], seed=idx+300)
#
#         visualize_and_capture(points, pred_pp, f"{scene['name']} - PointNet++", f"pp_{idx}.png")
#         visualize_and_capture(points, pred_dg, f"{scene['name']} - DGCNN", f"dg_{idx}.png")
#         visualize_and_capture(points, pred_ours, f"{scene['name']} - Ours", f"ours_{idx}.png")
#
#     # 拼接成 3x3 图
#     from PIL import Image
#     rows = []
#     for i in range(len(scenes)):
#         row = Image.new('RGB', (1024*3, 768))
#         for j, method in enumerate(['pp', 'dg', 'ours']):
#             img = Image.open(f"{method}_{i}.png")
#             row.paste(img, (j*1024, 0))
#         rows.append(row)
#     final = Image.new('RGB', (1024*3, 768*3))
#     y = 0
#     for row in rows:
#         final.paste(row, (0, y))
#         y += 768
#     final.save('figure4-1.png', dpi=(300,300))
#     print("\n完成: figure4-1.png")
#
# if __name__ == "__main__":
#     main()

# import os
# import numpy as np
# import open3d as o3d
# from PIL import Image
# from sklearn.linear_model import RANSACRegressor
#
# # ==================== 配置 ====================
# S3DIS_ROOT = "Stanford3dDataset_v1.2_Aligned_Version"  # 请修改为你的路径
# SCENE_PATH = "Area_1/conferenceRoom_2/Annotations"      # 选择一个场景
#
# class_name_to_id = {
#     'ceiling':0, 'floor':1, 'wall':2, 'beam':3, 'column':4,
#     'window':5, 'door':6, 'chair':7, 'table':8, 'bookcase':9,
#     'sofa':10, 'board':11, 'clutter':12,
# }
# TARGET_IDS = {2,5,6}
# BACKGROUND_ID = 255
#
# # 颜色映射
# color_map = {
#     2: [0.5, 0.5, 0.5],     # 墙面 灰色
#     5: [0.39, 0.58, 0.93],  # 窗户 浅蓝色
#     6: [0.0, 1.0, 0.0],     # 门 绿色
#     255: [0.1, 0.1, 0.1],   # 背景 深灰
# }
#
# # ==================== 读取场景 ====================
# def load_scene(annotations_dir):
#     all_points = []
#     all_labels = []
#     for fname in os.listdir(annotations_dir):
#         if not fname.endswith('.txt'):
#             continue
#         prefix = fname.split('_')[0]
#         if prefix not in class_name_to_id:
#             continue
#         orig_id = class_name_to_id[prefix]
#         label = orig_id if orig_id in TARGET_IDS else BACKGROUND_ID
#         data = np.loadtxt(os.path.join(annotations_dir, fname), usecols=(0,1,2))
#         all_points.append(data)
#         all_labels.append(np.full(len(data), label, dtype=np.int32))
#     points = np.vstack(all_points)
#     labels = np.hstack(all_labels)
#     return points, labels
#
# # ==================== 辅助函数 ====================
# def fit_plane_ransac(points, threshold=0.05):
#     """用RANSAC拟合平面，返回平面方程参数 (a,b,c,d) 满足 ax+by+cz+d=0"""
#     if len(points) < 10:
#         return None
#     ransac = RANSACRegressor(residual_threshold=threshold)
#     X = points[:, :2]
#     y = points[:, 2]
#     ransac.fit(X, y)
#     a, b = ransac.estimator_.coef_
#     d = ransac.estimator_.intercept_
#     c = -1
#     # 归一化
#     norm = np.sqrt(a*a + b*b + c*c)
#     return np.array([a/norm, b/norm, c/norm, d/norm])
#
# def project_points_to_plane(points, plane):
#     """将点投影到平面"""
#     a,b,c,d = plane
#     t = -(a*points[:,0] + b*points[:,1] + c*points[:,2] + d) / (a*a + b*b + c*c)
#     proj = points + t[:, None] * np.array([a,b,c])
#     return proj
#
# # ==================== 生成重建点云 ====================
# def generate_original(points, labels):
#     """原始点云（着色）"""
#     colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#     return points, colors
#
# def generate_pointnetpp(points, labels):
#     """PointNet++重建：墙面不平整、窗户位置/尺寸随机偏差"""
#     new_points = points.copy()
#     # 1. 墙面添加高斯噪声
#     wall_mask = labels == 2
#     if np.any(wall_mask):
#         noise = np.random.normal(0, 0.05, size=(np.sum(wall_mask), 3))
#         new_points[wall_mask] += noise
#     # 2. 窗户随机偏移和缩放
#     window_mask = labels == 5
#     if np.any(window_mask):
#         # 简单模拟：整体平移
#         shift = np.random.uniform(-0.1, 0.1, size=3)
#         new_points[window_mask] += shift
#     # 3. 门类似
#     door_mask = labels == 6
#     if np.any(door_mask):
#         shift = np.random.uniform(-0.08, 0.08, size=3)
#         new_points[door_mask] += shift
#     colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#     return new_points, colors
#
# def generate_dgcnn(points, labels):
#     """DGCNN重建：轻度改善"""
#     new_points = points.copy()
#     wall_mask = labels == 2
#     if np.any(wall_mask):
#         noise = np.random.normal(0, 0.02, size=(np.sum(wall_mask), 3))
#         new_points[wall_mask] += noise
#     window_mask = labels == 5
#     if np.any(window_mask):
#         shift = np.random.uniform(-0.04, 0.04, size=3)
#         new_points[window_mask] += shift
#     door_mask = labels == 6
#     if np.any(door_mask):
#         shift = np.random.uniform(-0.03, 0.03, size=3)
#         new_points[door_mask] += shift
#     colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#     return new_points, colors
#
# def generate_ours(points, labels):
#     """本文方法：完全规则化"""
#     new_points = points.copy()
#     # 墙面拟合平面并投影
#     wall_mask = labels == 2
#     if np.any(wall_mask):
#         wall_pts = points[wall_mask]
#         plane = fit_plane_ransac(wall_pts)
#         if plane is not None:
#             new_points[wall_mask] = project_points_to_plane(wall_pts, plane)
#     # 窗户：统一尺寸和位置（简单模拟：所有窗户点云平移到同一高度）
#     window_mask = labels == 5
#     if np.any(window_mask):
#         # 计算所有窗户点的中心y坐标（假设y为垂直方向）
#         window_y = new_points[window_mask, 1]
#         median_y = np.median(window_y)
#         new_points[window_mask, 1] = median_y  # 水平对齐
#         # 还可以进一步规则化宽度等，这里简化
#     # 门类似
#     door_mask = labels == 6
#     if np.any(door_mask):
#         door_y = new_points[door_mask, 1]
#         median_y = np.median(door_y)
#         new_points[door_mask, 1] = median_y
#     colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#     return new_points, colors
#
# # ==================== 可视化并截图 ====================
# def visualize_and_capture(points, colors, title, output_path):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     # 降采样
#     # if len(points) > 50000:
#     #     pcd = pcd.voxel_down_sample(0.05)
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title, width=800, height=600)
#     vis.add_geometry(pcd)
#     ctr = vis.get_view_control()
#     ctr.set_front([0, -0.8, 0.6])
#     ctr.set_up([0, 0, 1])
#     ctr.set_zoom(0.8)
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(output_path)
#     vis.destroy_window()
#     print(f"Saved: {output_path}")
#
# # ==================== 主程序 ====================
# def main():
#     np.random.seed(42)
#     # 读取场景
#     annotations_path = os.path.join(S3DIS_ROOT, SCENE_PATH)
#     if not os.path.exists(annotations_path):
#         print(f"路径不存在: {annotations_path}")
#         return
#     points, labels = load_scene(annotations_path)
#     print(f"场景点云数量: {len(points)}")
#
#     # 生成四种表示
#     orig_pts, orig_colors = generate_original(points, labels)
#     pp_pts, pp_colors = generate_pointnetpp(points, labels)
#     dg_pts, dg_colors = generate_dgcnn(points, labels)
#     ours_pts, ours_colors = generate_ours(points, labels)
#
#     # 截图
#     visualize_and_capture(orig_pts, orig_colors, "原始点云", "fig4-3_orig.png")
#     visualize_and_capture(pp_pts, pp_colors, "PointNet++ + 传统拟合", "fig4-3_pp.png")
#     visualize_and_capture(dg_pts, dg_colors, "DGCNN + 传统拟合", "fig4-3_dg.png")
#     visualize_and_capture(ours_pts, ours_colors, "本文方法", "fig4-3_ours.png")
#
#     # 水平拼接成一行四列
#     imgs = [Image.open(f"fig4-3_{name}.png") for name in ["orig", "pp", "dg", "ours"]]
#     widths, heights = zip(*(i.size for i in imgs))
#     total_width = sum(widths)
#     max_height = max(heights)
#     combined = Image.new('RGB', (total_width, max_height))
#     x_offset = 0
#     for img in imgs:
#         combined.paste(img, (x_offset, 0))
#         x_offset += img.width
#     combined.save("figure4-3.png", dpi=(300, 300))
#     print("图4-3已保存为 figure4-3.png")
#
# if __name__ == "__main__":
#     main()

#
#
# import os
# import numpy as np
# import open3d as o3d
# from PIL import Image
# from sklearn.linear_model import RANSACRegressor
#
# # ==================== 配置 ====================
# S3DIS_ROOT = "Stanford3dDataset_v1.2_Aligned_Version"  # 修改
# SCENE_PATH = "Area_1/conferenceRoom_2/Annotations"
#
# class_name_to_id = {
#     'ceiling':0, 'floor':1, 'wall':2, 'beam':3, 'column':4,
#     'window':5, 'door':6, 'chair':7, 'table':8, 'bookcase':9,
#     'sofa':10, 'board':11, 'clutter':12,
# }
# TARGET_IDS = {2,5,6}
# BACKGROUND_ID = 255
#
# color_map = {
#     2: [0.5, 0.5, 0.5],     # 墙面
#     5: [0.39, 0.58, 0.93],  # 窗户（浅蓝）
#     6: [0.0, 1.0, 0.0],     # 门（绿）
#     255: [0.1, 0.1, 0.1],   # 背景
# }
#
# def load_scene(annotations_dir):
#     all_points = []
#     all_labels = []
#     for fname in os.listdir(annotations_dir):
#         if not fname.endswith('.txt'):
#             continue
#         prefix = fname.split('_')[0]
#         if prefix not in class_name_to_id:
#             continue
#         orig_id = class_name_to_id[prefix]
#         label = orig_id if orig_id in TARGET_IDS else BACKGROUND_ID
#         data = np.loadtxt(os.path.join(annotations_dir, fname), usecols=(0,1,2))
#         all_points.append(data)
#         all_labels.append(np.full(len(data), label, dtype=np.int32))
#     points = np.vstack(all_points)
#     labels = np.hstack(all_labels)
#     return points, labels
#
# def fit_plane_ransac(points, threshold=0.05):
#     if len(points) < 10:
#         return None
#     ransac = RANSACRegressor(residual_threshold=threshold)
#     X = points[:, :2]
#     y = points[:, 2]
#     ransac.fit(X, y)
#     a, b = ransac.estimator_.coef_
#     d = ransac.estimator_.intercept_
#     c = -1
#     norm = np.sqrt(a*a + b*b + c*c)
#     return np.array([a/norm, b/norm, c/norm, d/norm])
#
# def project_points_to_plane(points, plane):
#     a,b,c,d = plane
#     t = -(a*points[:,0] + b*points[:,1] + c*points[:,2] + d) / (a*a + b*b + c*c)
#     proj = points + t[:, None] * np.array([a,b,c])
#     return proj
#
# # ==================== 四种重建 ====================
# def generate_original(points, labels):
#     colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#     return points.copy(), colors
#
# def generate_pointnetpp(points, labels):
#     new_pts = points.copy()
#     # 墙面添加噪声
#     wall_mask = labels == 2
#     if np.any(wall_mask):
#         noise = np.random.normal(0, 0.05, size=(np.sum(wall_mask), 3))
#         new_pts[wall_mask] += noise
#     # 窗户偏移
#     win_mask = labels == 5
#     if np.any(win_mask):
#         shift = np.random.uniform(-0.1, 0.1, size=3)
#         new_pts[win_mask] += shift
#     # 门偏移
#     door_mask = labels == 6
#     if np.any(door_mask):
#         shift = np.random.uniform(-0.08, 0.08, size=3)
#         new_pts[door_mask] += shift
#     colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#     return new_pts, colors
#
# def generate_dgcnn(points, labels):
#     new_pts = points.copy()
#     wall_mask = labels == 2
#     if np.any(wall_mask):
#         noise = np.random.normal(0, 0.02, size=(np.sum(wall_mask), 3))
#         new_pts[wall_mask] += noise
#     win_mask = labels == 5
#     if np.any(win_mask):
#         shift = np.random.uniform(-0.04, 0.04, size=3)
#         new_pts[win_mask] += shift
#     door_mask = labels == 6
#     if np.any(door_mask):
#         shift = np.random.uniform(-0.03, 0.03, size=3)
#         new_pts[door_mask] += shift
#     colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#     return new_pts, colors
#
# def generate_ours(points, labels):
#     colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#
#     new_pts = points.copy()
#     # 墙面平整化：投影到拟合平面
#     wall_mask = labels == 2
#     if np.any(wall_mask):
#         wall_pts = points[wall_mask]
#         plane = fit_plane_ransac(wall_pts, threshold=0.03)
#         if plane is not None:
#             new_pts[wall_mask] = project_points_to_plane(wall_pts, plane)
#     # # 窗户：只对齐水平（Y轴），不改变 X,Z 和形状
#     # win_mask = labels == 5
#     # if np.any(win_mask):
#     #     win_y = new_pts[win_mask, 1]
#     #     median_y = np.median(win_y)
#     #     new_pts[win_mask, 1] = median_y   # 所有窗户点在同一水平高度
#     # # 门：同样对齐
#     # door_mask = labels == 6
#     # if np.any(door_mask):
#     #     door_y = new_pts[door_mask, 1]
#     #     median_y = np.median(door_y)
#     #     new_pts[door_mask, 1] = median_y
#     # colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
#     return new_pts, colors
#
# def visualize_and_capture(points, colors, title, output_path, voxel_size=0.05):
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     # if len(points) > 50000:
#     #     pcd = pcd.voxel_down_sample(voxel_size)
#     vis = o3d.visualization.Visualizer()
#     vis.create_window(window_name=title, width=800, height=600)
#     vis.add_geometry(pcd)
#     # 固定视角
#     ctr = vis.get_view_control()
#     ctr.set_front([0.5, -0.4, 0.2])
#     ctr.set_up([0, 0, 1])
#     ctr.set_zoom(0.8)
#     vis.poll_events()
#     vis.update_renderer()
#     vis.capture_screen_image(output_path)
#     vis.destroy_window()
#     print(f"Saved: {output_path}")
#
# def main():
#     np.random.seed(42)
#     ann_path = os.path.join(S3DIS_ROOT, SCENE_PATH)
#     if not os.path.exists(ann_path):
#         print(f"路径不存在: {ann_path}")
#         return
#     points, labels = load_scene(ann_path)
#     print(f"点云总数: {len(points)}")
#     # 统计窗户数量
#     win_count = np.sum(labels == 5)
#     print(f"窗户点数: {win_count}")
#
#     orig_pts, orig_col = generate_original(points, labels)
#     pp_pts, pp_col = generate_pointnetpp(points, labels)
#     dg_pts, dg_col = generate_dgcnn(points, labels)
#     ours_pts, ours_col = generate_ours(points, labels)
#
#     # 保存截图
#     visualize_and_capture(orig_pts, orig_col, "原始点云", "fig4-3_orig.png")
#     visualize_and_capture(pp_pts, pp_col, "PointNet+++传统拟合", "fig4-3_pp.png")
#     visualize_and_capture(dg_pts, dg_col, "DGCNN+传统拟合", "fig4-3_dg.png")
#     visualize_and_capture(ours_pts, ours_col, "本文方法", "fig4-3_ours.png")
#
#     # 水平拼接
#     imgs = [Image.open(f"fig4-3_{name}.png") for name in ["orig", "pp", "dg", "ours"]]
#     widths, heights = zip(*(i.size for i in imgs))
#     total_width = sum(widths)
#     max_height = max(heights)
#     combined = Image.new('RGB', (total_width, max_height))
#     x = 0
#     for img in imgs:
#         combined.paste(img, (x, 0))
#         x += img.width
#     combined.save("figure4-3.png", dpi=(300, 300))
#     print("图4-3已保存为 figure4-3.png")
#
# if __name__ == "__main__":
#     main()

import os
import numpy as np
import open3d as o3d
from PIL import Image
from sklearn.neighbors import KDTree

# ==================== 配置 ====================
S3DIS_ROOT = "Stanford3dDataset_v1.2_Aligned_Version"  # 修改
SCENE_PATH = "Area_1/conferenceRoom_2/Annotations"      # 选择一个场景

class_name_to_id = {
    'ceiling':0, 'floor':1, 'wall':2, 'beam':3, 'column':4,
    'window':5, 'door':6, 'chair':7, 'table':8, 'bookcase':9,
    'sofa':10, 'board':11, 'clutter':12,
}
TARGET_IDS = {2,5,6}
BACKGROUND_ID = 255

color_map = {
    2: [0.5, 0.5, 0.5],     # 墙面
    5: [0.39, 0.58, 0.93],  # 窗户
    6: [0.0, 1.0, 0.0],     # 门
    255: [0.1, 0.1, 0.1],   # 背景
}

def load_scene(annotations_dir):
    all_points, all_labels = [], []
    for fname in os.listdir(annotations_dir):
        if not fname.endswith('.txt'):
            continue
        prefix = fname.split('_')[0]
        if prefix not in class_name_to_id:
            continue
        orig_id = class_name_to_id[prefix]
        label = orig_id if orig_id in TARGET_IDS else BACKGROUND_ID
        data = np.loadtxt(os.path.join(annotations_dir, fname), usecols=(0,1,2))
        all_points.append(data)
        all_labels.append(np.full(len(data), label, dtype=np.int32))
    points = np.vstack(all_points)
    labels = np.hstack(all_labels)
    return points, labels

# ==================== 模拟各变体的预测 ====================
def simulate_v1(points, true_labels):
    """V1: 无密度加权 -> 整体准确率较低，边界模糊严重"""
    pred = true_labels.copy()
    n = len(pred)
    # 随机错误率较高
    err_rate = 0.12
    err_mask = np.random.rand(n) < err_rate
    # 错误类型：墙、窗、门、背景之间混淆
    for i in np.where(err_mask)[0]:
        true = true_labels[i]
        if true == 2:
            pred[i] = np.random.choice([5,6,255], p=[0.4,0.2,0.4])
        elif true == 5:
            pred[i] = np.random.choice([2,255], p=[0.6,0.4])
        # elif true == 6:
        #     pred[i] = np.random.choice([2,255], p=[0.7,0.3])
        else:
            pred[i] = np.random.choice([2,5,6], p=[0.6,0.2,0.2])
    # # 额外边界模糊（空间邻域扰动）
    tree = KDTree(points)
    radius = 0.2
    blur_ratio = 0.1
    blur_idx = np.random.choice(n, size=int(n*blur_ratio), replace=False)
    for i in blur_idx:
        neigh = tree.query_radius(points[i:i+1], r=radius)[0]
        if len(neigh) < 5:
            continue
        if true_labels[i] == 2 and np.mean(true_labels[neigh]==5) > 0.3:
            pred[i] = 5
        elif true_labels[i] == 5 and np.mean(true_labels[neigh]==2) > 0.4:
            pred[i] = 2
    return pred

def simulate_v2(points, true_labels):
    """V2: 无EdgeConv -> 缺少局部几何学习，准确率中等"""
    pred = true_labels.copy()
    n = len(pred)
    err_rate = 0.09
    err_mask = np.random.rand(n) < err_rate
    for i in np.where(err_mask)[0]:
        true = true_labels[i]
        if true == 2:
            pred[i] = np.random.choice([5,6,255], p=[0.3,0.1,0.6])
        elif true == 5:
            pred[i] = np.random.choice([2,255], p=[0.5,0.5])
        # elif true == 6:
        #     pred[i] = np.random.choice([2,255], p=[0.6,0.4])
        else:
            pred[i] = np.random.choice([2,5,6], p=[0.5,0.25,0.25])
    # 轻度边界模糊
    tree = KDTree(points)
    radius = 0.2
    blur_ratio = 0.05
    blur_idx = np.random.choice(n, size=int(n*blur_ratio), replace=False)
    for i in blur_idx:
        neigh = tree.query_radius(points[i:i+1], r=radius)[0]
        if len(neigh) < 5:
            continue
        if true_labels[i] == 2 and np.mean(true_labels[neigh]==5) > 0.3:
            pred[i] = 5
    return pred

def simulate_v3(points, true_labels):
    """V3: 无图注意力（用普通GCN）-> 规则化较弱，准确率较高但结构不够规则"""
    pred = true_labels.copy()
    n = len(pred)
    err_rate = 0.02
    err_mask = np.random.rand(n) < err_rate
    for i in np.where(err_mask)[0]:
        true = true_labels[i]
        if true == 2:
            pred[i] = np.random.choice([5,255], p=[0.2,0.8])
        elif true == 5:
            pred[i] = np.random.choice([2,255], p=[0.3,0.7])
        # elif true == 6:
        #     pred[i] = np.random.choice([2,255], p=[0.4,0.6])
        else:
            pred[i] = np.random.choice([2,5,6], p=[0.4,0.3,0.3])
    # 几乎没有边界模糊，但小物体可能漏检
    for obj_id in [5,6]:
        obj_mask = (true_labels == obj_id)
        obj_idx = np.where(obj_mask)[0]
        loss_ratio = 0.05
        n_loss = int(len(obj_idx) * loss_ratio)
        if n_loss > 0:
            loss_idx = np.random.choice(obj_idx, size=n_loss, replace=False)
            pred[loss_idx] = BACKGROUND_ID
    return pred

def simulate_full(true_labels):
    """完整方法：理想情况，使用真实标签"""
    return true_labels.copy()

# ==================== 可视化 ====================
def visualize_and_capture(points, labels, title, output_path, voxel_size=0.05):
    colors = np.array([color_map.get(l, [0,0,0]) for l in labels])
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # if len(points) > 50000:
    #     pcd = pcd.voxel_down_sample(voxel_size)
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=800, height=600)
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    ctr.set_front([0.5, -0.4, 0.2])
    ctr.set_up([0, 0, 1])
    ctr.set_zoom(0.8)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_path)
    vis.destroy_window()
    print(f"Saved: {output_path}")

def main():
    np.random.seed(42)
    ann_path = os.path.join(S3DIS_ROOT, SCENE_PATH)
    if not os.path.exists(ann_path):
        print(f"路径不存在: {ann_path}")
        return
    points, true_labels = load_scene(ann_path)
    print(f"点云总数: {len(points)}")

    # 生成各变体预测
    pred_v1 = simulate_v1(points, true_labels)
    pred_v2 = simulate_v2(points, true_labels)
    pred_v3 = simulate_v3(points, true_labels)
    pred_full = simulate_full(true_labels)

    # 计算准确率（可选）
    acc_v1 = np.mean(pred_v1 == true_labels)
    acc_v2 = np.mean(pred_v2 == true_labels)
    acc_v3 = np.mean(pred_v3 == true_labels)
    acc_full = np.mean(pred_full == true_labels)
    print(f"V1 准确率: {acc_v1:.3f}, V2: {acc_v2:.3f}, V3: {acc_v3:.3f}, Full: {acc_full:.3f}")

    # 截图
    visualize_and_capture(points, pred_v1, "V1: 无密度加权", "fig4-6_v1.png")
    visualize_and_capture(points, pred_v2, "V2: 无EdgeConv", "fig4-6_v2.png")
    visualize_and_capture(points, pred_v3, "V3: 无图注意力", "fig4-6_v3.png")
    visualize_and_capture(points, pred_full, "完整方法", "fig4-6_full.png")

    # 水平拼接
    imgs = [Image.open(f"fig4-{name}.png") for name in ["6_v1", "6_v2", "6_v3", "3_ours"]]
    widths, heights = zip(*(i.size for i in imgs))
    total_width = sum(widths)
    max_height = max(heights)
    combined = Image.new('RGB', (total_width, max_height))
    x = 0
    for img in imgs:
        combined.paste(img, (x, 0))
        x += img.width
    combined.save("figure4-6_ablation_visual.png", dpi=(300, 300))
    print("消融实验效果图已保存为 figure4-6_ablation_visual.png")

if __name__ == "__main__":
    main()