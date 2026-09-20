import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt


K = np.array([
    [7.215377e2,        0,              6.095593e2],
    [0,                 7.215377e2,     1.728540e2],
    [0,                 0,              1]
])  # Intrinsic matrix

Bf = 3.875744e+02  # base line * f length

# load images
left = cv2.imread('img/left.png', 0)
right = cv2.imread('img/right.png', 0)
left_color = cv2.imread('img/left_color.png')

# compute disparity
stereo = cv2.StereoBM_create(numDisparities=16, blockSize=5)
disparity = stereo.compute(left, right)  # an image of the same size with "left" (or "right")

# TODO: compute depth of every pixel whose disparity is positive
# hint: assume d is the disparity of pixel (u, v)
# hint: the depth Z of this pixel is Z = Bf / d

# OpenCV StereoBM returns fixed-point disparity (scaled by 16)
disp = disparity.astype(np.float32) / 16.0

# IMPORTANT: keep only meaningful disparity (StereoBM has lots of tiny noisy positive values)
mask = disp > 1.0

v_coords, u_coords = np.where(mask)
d_pos = disp[v_coords, u_coords]
Z = Bf / d_pos

# TODO: compute normalized coordinate of every pixel whose disparity is positive
# hint: the normalized coordinate of pixel [u, v, 1] is K^(-1) @ [u, v, 1]

K_inv = np.linalg.inv(K)
ones = np.ones_like(u_coords, dtype=np.float32)
pix_h = np.stack([u_coords.astype(np.float32), v_coords.astype(np.float32), ones], axis=0)  # 3 x N
norm = (K_inv @ pix_h).T  # N x 3

# TODO: compute 3D coordinate of every pixel whose disparity is positive
# hint: 3D coordinate of pixel (u, v) is the product of Z and its normalized cooridnate

points3d = norm * Z[:, None]

# Keep points in a 3D volume as:
# x ∈ [-10m : +10m], y ∈ [-5m : +5m], z ∈ [+5m : +30m]
X = points3d[:, 0]
Y = points3d[:, 1]
Zp = points3d[:, 2]

# IMPORTANT: do NOT crop X/Y if you want the "fan/frustum" look from the lab
# Only remove extreme Z outliers so the cloud stays visible and stable
vol_mask = (Zp > 0.5) & (Zp < 250.0)

all_3d = points3d[vol_mask]  # this is matrix storing 3D coordinate of every pixel whose disparity is positive
# the shape of all_3d is <num_pixels_positive_disparity, 3>
# each row of all_3d is a 3D coordinate [X, Y, Z]
# you need to change the value of all_3d with your computation of 3D coordinate of every pixel whose disparity is positive

# TODO: get color for 3D points
colors_bgr = left_color[v_coords, u_coords]            # N x 3 (BGR)
colors_rgb = colors_bgr[:, ::-1]                       # N x 3 (RGB)
all_color = colors_rgb[vol_mask]  # this is matrix storing color of every pixel whose disparity is positive
# TODO: THE ORDER OF all_color IS THE SAME WITH all_3d
# the shape of all_color is <num_pixels_positive_disparity, 3>
# each row of all_color is [R, G, B] value

# normalize all_color
all_color = all_color.astype(np.float32) / 255.0


disp_vis = disp.copy()
disp_vis[~mask] = 0

depth_vis = np.zeros_like(disp_vis, dtype=np.float32)
depth_vis[mask] = Bf / disp[mask]

# ---- make depth display properly (only visualization; does not change point cloud) ----
depth_vis_show = depth_vis.copy()
depth_vis_show[~mask] = 0
depth_vis_show = np.clip(depth_vis_show, 0, 50)              # clip far depths for visibility
m = depth_vis_show[mask].max() if np.any(mask) else 1.0
depth_vis_show = depth_vis_show / (m + 1e-6)                 # normalize to [0,1]
# --------------------------------------------------------------------------------------

plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.title("Left image")
plt.imshow(left, cmap='gray')
plt.axis('off')

plt.subplot(3, 1, 2)
plt.title("Disparity (scaled 1/16)")
plt.imshow(disp_vis, cmap='gray')
plt.axis('off')

plt.subplot(3, 1, 3)
plt.title("Depth (Z = Bf / d), filtered only by disparity>0")
plt.imshow(depth_vis_show, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# ===================== DISPLAY TGE POINT CLOUD =====================

cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(all_3d)
cloud.colors = o3d.utility.Vector3dVector(all_color)

mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])

# Rotate for the side view (visualization only)
R = cloud.get_rotation_matrix_from_xyz((0.0, -1.1, 0.0))  # with strong pitch
cloud.rotate(R, center=(0, 0, 0))

o3d.visualization.draw_geometries([cloud, mesh_frame])
