import open3d as o3d
import numpy as np

# --- 1. PCD laden ---
pcd = o3d.io.read_point_cloud("data/pcds/output_cloud.pcd")

# Optional: Farbkanäle korrigieren
if pcd.has_colors():
    colors = np.asarray(pcd.colors)
    colors = colors[:, [2, 1, 0]]
    pcd.colors = o3d.utility.Vector3dVector(colors)

# --- 2. Alle Ebenen schrittweise entfernen ---
remaining_pcd = pcd
all_planes = []
max_planes = 2  # z. B. max. 5 Ebenen entfernen
plane_threshold = 0.01  # Distanzschwelle in Metern

for i in range(max_planes):
    plane_model, inliers = remaining_pcd.segment_plane(
        distance_threshold=plane_threshold,
        ransac_n=3,
        num_iterations=1000
    )
    if len(inliers) < 100:  # abbrechen, wenn Ebene zu klein
        break
    print(f"[{i}] Ebene gefunden mit {len(inliers)} Punkten")
    inlier_cloud = remaining_pcd.select_by_index(inliers)
    all_planes.append(inlier_cloud)
    remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

# --- 3. Ergebnis anzeigen: ohne Planflächen ---
o3d.visualization.draw_geometries([remaining_pcd], window_name="Ohne Planflächen")

# Optional: Ergebnis speichern
o3d.io.write_point_cloud("data/pcds/no_planes.pcd", remaining_pcd)
