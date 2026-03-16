import re
import numpy as np
import cv2
import matplotlib.pyplot as plt
import open3d as o3d
import argparse
import os


def read_pfm(filename):
    """读取 PFM 文件，返回 (depth, scale)，depth 为 h×w float32 数组。"""
    with open(filename, 'rb') as f:
        header = f.readline().rstrip()
        if header not in (b'PF', b'Pf'):
            raise ValueError('Not a PFM file.')
        dims = f.readline().decode('utf-8').strip().split()
        width, height = map(int, dims)
        scale = float(f.readline().rstrip())
        data = np.fromfile(f,
                           dtype='<f4' if scale < 0 else '>f4',
                           count=width * height)
        depth = data.reshape((height, width))
        return depth, abs(scale)


def visualize_depth(depth, out_prefix):
    """
    生成伪彩色深度图和深度直方图，并保存到 out_prefix 相关文件。
    depth 已在 main 中做过上下翻转。
    """
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)
    # 1. 伪彩色
    mask = depth > 0
    valid = depth[mask]
    vmin, vmax = np.percentile(valid, (2, 98))
    norm = np.clip((depth - vmin) / (vmax - vmin), 0, 1)
    cmap = plt.get_cmap('turbo')
    colored = (cmap(norm)[..., :3] * 255).astype(np.uint8)
    colored_bgr = cv2.cvtColor(colored, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"{out_prefix}_colormap.png", colored_bgr)
    print(f"Saved colored depth: {out_prefix}_colormap.png")

    # 2. 直方图
    plt.figure(figsize=(6, 4))
    plt.hist(valid.flatten(), bins=200, color='steelblue', edgecolor='black')
    plt.xlabel('Depth value')
    plt.ylabel('Count')
    plt.title('Depth Histogram')
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_hist.png", dpi=200)
    plt.close()
    print(f"Saved histogram: {out_prefix}_hist.png")


def visualize_pointcloud_interactive(ply_file):
    """
    打开一个可交互窗口，用户可通过鼠标旋转/缩放/平移点云。
    """
    pcd = o3d.io.read_point_cloud(ply_file)
    o3d.visualization.draw_geometries(
        [pcd],
        window_name='Interactive PointCloud',
        width=1024, height=768,
        left=50, top=50,
        point_show_normal=False,
        mesh_show_wireframe=False,
        mesh_show_back_face=False
    )
    # 鼠标操作：左键旋转、右键平移、滚轮缩放 :contentReference[oaicite:0]{index=0}


def visualize_pointcloud_snapshot(ply_file, out_image):
    """
    离屏渲染并截图，适合批量生成论文用图。
    """
    os.makedirs(os.path.dirname(out_image), exist_ok=True)
    pcd = o3d.io.read_point_cloud(ply_file)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=768, visible=False)
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.point_size = 1.5
    # 设置相机视角
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    ctr.set_up([0, -1, 0])
    ctr.set_lookat(pcd.get_center())
    ctr.set_zoom(0.7)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(out_image)
    vis.destroy_window()
    print(f"Saved point cloud snapshot: {out_image}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize depth (.pfm) and point cloud (.ply) with correct orientation & interaction')
    # 移除 required=True，保留默认值，或在命令行中指定
    parser.add_argument('--pfm', type=str,
                        default='outputs/scan10/depth_est/00000001.pfm',
                        help='path to input depth .pfm file')
    parser.add_argument('--ply', type=str,
                        default='outputs/mvsnet010_l3.ply',
                        help='path to input point cloud .ply file')
    parser.add_argument('--outdir', type=str, default='vis_fixed10',
                        help='directory to save output images')
    args = parser.parse_args()

    # 1. 读取并翻转深度图
    depth, scale = read_pfm(args.pfm)
    print(f"Read PFM: shape={depth.shape}, scale={scale}")
    depth = np.flipud(depth)  # 上下翻转 :contentReference[oaicite:1]{index=1}

    # 2. 深度可视化
    prefix = os.path.join(args.outdir, 'depth_vis')
    visualize_depth(depth, prefix)

    # 3. 交互式点云查看
    print("Launching interactive viewer...")
    visualize_pointcloud_interactive(args.ply)

    # 4. 离屏渲染截图
    snapshot = os.path.join(args.outdir, 'pcd_snapshot.png')
    visualize_pointcloud_snapshot(args.ply, snapshot)

    print("All done. Outputs in:", args.outdir)


if __name__ == '__main__':
    main()
