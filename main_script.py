import warnings
warnings.filterwarnings("ignore")


import numpy as np
from PIL import Image
import torch
import open3d as o3d
import matplotlib.pyplot as plt

from segment_object import grounded_sam_bounding_box
from match_and_get_3dpoint import (
    ViTExtractor,
    visualize_3d_points_in_cloud,
)

def visualize_2d_matches(img_ref_path, img_scene_path, ref_pts, tgt_pts, matches):
    img1 = np.array(Image.open(img_ref_path).convert("RGB"))
    img2 = np.array(Image.open(img_scene_path).convert("RGB"))
    # h1, w1 und h2, w2 aus den beiden Bildern
    h1, w1, _ = img1.shape
    h2, w2, _ = img2.shape
    # keine Assertion mehr!
    # bilder nebeneinander anordnen
    # pad Höhe auf max(h1,h2)
    H = max(h1, h2)
    canvas = np.zeros((H, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1
    canvas[:h2, w1:w1 + w2] = img2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(canvas)
    colors = ['red', 'green', 'blue', 'yellow', 'magenta']
    for (i_ref, i_tgt), color in zip(matches, colors):
        x1, y1 = ref_pts[i_ref]
        x2, y2 = tgt_pts[i_tgt]
        # Ref-Punkt
        ax.plot(x1, y1, 'o', color=color)
        # Zielpunkt – horizontal um w1 verschoben
        ax.plot(x2 + w1, y2, 'o', color=color)
        # Linie dazwischen
        ax.plot([x1, x2 + w1], [y1, y2], color=color, linewidth=1.5)
    ax.axis('off')
    plt.title("2D-Matches deiner gewählten Punkte")
    plt.show()


def estimate_local_normal(xyz_path, grip_point, radius=0.05, min_neighbors=10):
    xyz = np.loadtxt(xyz_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    k, idx, _ = pcd_tree.search_radius_vector_3d(grip_point, radius)
    if k < min_neighbors:
        raise RuntimeError(f"Zu wenig Punkte ({k}) im Radius {radius}")
    neighbors = np.asarray(pcd.points)[idx, :]
    centroid = neighbors.mean(axis=0)
    centered = neighbors - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1] / np.linalg.norm(vh[-1])
    return normal

def filter_features_by_box(points, features, box):
    x0, y0, x1, y1 = box
    fp, ff = [], []
    for (x, y), feat in zip(points, features):
        if x0 <= x <= x1 and y0 <= y <= y1:
            fp.append((x, y)); ff.append(feat)
    return np.array(fp), np.array(ff)

def match_features(ref_feats, tgt_feats):
    ref = torch.tensor(ref_feats, dtype=torch.float32)
    tgt = torch.tensor(tgt_feats, dtype=torch.float32)
    ref = ref / ref.norm(dim=1, keepdim=True)
    tgt = tgt / tgt.norm(dim=1, keepdim=True)
    sim = ref @ tgt.T
    idx = sim.argmax(dim=1).tolist()
    return list(zip(range(len(ref)), idx))

def manual_point_selection(image_path, num_points=5):
    pts = []
    img = np.array(Image.open(image_path).convert("RGB"))
    def onclick(evt):
        if evt.inaxes and len(pts) < num_points:
            x, y = int(evt.xdata), int(evt.ydata)
            pts.append((x, y))
            plt.plot(x, y, 'ro'); plt.draw()
            if len(pts) == num_points:
                plt.close()
    fig, ax = plt.subplots(); ax.imshow(img)
    plt.title(f"Wähle {num_points} Punkte")
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return np.array(pts)

def find_closest_patch_features(selected_pts, all_pts, all_feats):
    sel = []
    for s in selected_pts:
        d = np.linalg.norm(all_pts - s, axis=1)
        sel.append(all_feats[d.argmin()])
    return np.array(sel)

def get_3d_points_from_coords(xyz_path, coords, img_shape):
    xyz = np.loadtxt(xyz_path).reshape((img_shape[0], img_shape[1], 3))
    pts3d = []
    H, W = img_shape
    for x, y in coords:
        ix, iy = int(x), int(y)
        if 0 <= ix < W and 0 <= iy < H:
            p = xyz[iy, ix]
            pts3d.append(p if np.isfinite(p).all() and p[2]>0 else None)
        else:
            pts3d.append(None)
    return pts3d

def scale_box_to_feature_space(box, img_size, feat_size=(224, 224)):
    x0, y0, x1, y1 = box; w,h=img_size; fx,fy=feat_size
    sx, sy = fx/w, fy/h
    return [int(x0*sx), int(y0*sy), int(x1*sx), int(y1*sy)]

def main(ref_image, target_image, xyz_path, object_name):
    ref_pts2d = manual_point_selection(ref_image)
    extractor = ViTExtractor(device='cpu')
    
    all_ref2d, all_ref_feats = extractor.extract_features(Image.open(ref_image))
    
    ref_feats = find_closest_patch_features(ref_pts2d, all_ref2d, all_ref_feats)

    
    box = grounded_sam_bounding_box(target_image, object_name)
    
    img = Image.open(target_image); W,H = img.size
    box_scaled = scale_box_to_feature_space(box, (W,H))
    

    
    tgt2d, tgt_feats = extractor.extract_features(img)
    
    f2d, ffeats = filter_features_by_box(tgt2d, tgt_feats, box_scaled)
    if len(ffeats)==0: raise RuntimeError("Keine Features in der Box!")

    # Unscale aufs Original
    unscaled_2d = [(x*(W/224), y*(H/224)) for x,y in f2d]
    unscaled_2d = np.array(unscaled_2d)

    
    matches = match_features(ref_feats, ffeats)
    # 2D-Visualisierung
    visualize_2d_matches(ref_image, target_image, ref_pts2d, unscaled_2d, matches)

    # in 3D mappen
    pts3d = get_3d_points_from_coords(xyz_path, unscaled_2d, (H,W))
    matched3d = [pts3d[j] for (_,j) in matches if pts3d[j] is not None]

    grip, normal = None, None
    if len(matched3d) >= 3:
        grip = matched3d[0]
        try:
            normal = estimate_local_normal(xyz_path, grip, radius=0.05)
        except RuntimeError as e:
            print(f"[WARN] {e}")
    else:
        print("[WARN] Nicht genug Punkte für Normalen!")

    print("Greifpunkt:", grip)
    print("Normale:", normal)

    if grip is not None and normal is not None:
        visualize_3d_points_in_cloud(xyz_path, target_image, matched3d, grip, normal)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ref_image",    required=True)
    p.add_argument("--target_image", required=True)
    p.add_argument("--xyz",          required=True)
    p.add_argument("--object",       required=True)
    args = p.parse_args()
    main(args.ref_image, args.target_image, args.xyz, args.object)






