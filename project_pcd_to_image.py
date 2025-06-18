import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import cv2

# --- 1. Load PCD File ---
pcd = o3d.io.read_point_cloud('data/pcds/output_cloud.pcd')  # Ersetze durch deinen Pfad

# --- 2. Kamera-Parameter aus ROS camera_info ---
image_width = 640
image_height = 480

fx = 554.3827128226441
fy = 554.3827128226441
cx = 320.5
cy = 240.5

intrinsic = np.array([
    [fx,  0, cx],
    [0,  fy, cy],
    [0,   0,  1]
])

extrinsic = np.eye(4)  # Falls die Punktwolke im Kamerasystem vorliegt

# --- 3. Projektion ---
def project_points_to_image(pcd, intrinsic, extrinsic, image_size):
    points = np.asarray(pcd.points)
    points_h = np.hstack((points, np.ones((points.shape[0], 1))))
    cam_points = (extrinsic @ points_h.T)[:3, :]  # 3xN

    u = (intrinsic[0, 0] * cam_points[0] / cam_points[2]) + intrinsic[0, 2]
    v = (intrinsic[1, 1] * cam_points[1] / cam_points[2]) + intrinsic[1, 2]

    mask = (cam_points[2] > 0) & (u >= 0) & (v >= 0) & (u < image_size[0]) & (v < image_size[1])
    u_valid = u[mask].astype(np.int32)
    v_valid = v[mask].astype(np.int32)
    cam_points_valid = cam_points[:, mask].T  # Nx3

    return u_valid, v_valid, cam_points_valid, mask

u_img, v_img, cam_pts, mask = project_points_to_image(pcd, intrinsic, extrinsic, (image_width, image_height))

# --- 4. Visualisierung mit echten Farben ---
proj_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    colors = (colors * 255).astype(np.uint8)  # [0–1] → [0–255]
    valid_colors = colors[mask]
    for u, v, color in zip(u_img, v_img, valid_colors):
        proj_image[v, u] = color  # RGB
else:
    # Fallback: alles grün
    proj_image[v_img, u_img] = [0, 255, 0]

plt.imshow(proj_image)
plt.title("3D → 2D Projektion mit Farbinformation")
plt.axis('off')
plt.show()

# --- 5. Mapping speichern (für Matching-Rückprojektion) ---
uv_to_3d = {(int(u), int(v)): pt for u, v, pt in zip(u_img, v_img, cam_pts)}
np.save("uv_to_3d_map.npy", uv_to_3d)

