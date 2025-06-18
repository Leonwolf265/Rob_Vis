import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import open3d as o3d


def compute_surface_normal(points):
    pts = np.array(points)
    if pts.shape[0] < 3:
        raise ValueError("Mindestens 3 Punkte nötig zur Berechnung einer Fläche")
    centroid = np.mean(pts, axis=0)
    centered = pts - centroid
    _, _, vh = np.linalg.svd(centered)
    normal = vh[-1]
    return normal / np.linalg.norm(normal)


class ViTExtractor:
    def __init__(self, model_type='dino_vits16', device='cpu'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dino:main', model_type)
        self.model.eval().to(self.device)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, img: Image.Image):
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            x = self.model.patch_embed(x)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.model.pos_embed
            x = self.model.pos_drop(x)
            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)

            features = x[:, 1:, :].squeeze(0)  # ohne CLS Token, (num_patches, dim)

        num_patches = features.shape[0]
        grid_size = int(num_patches ** 0.5)
        patch_size = 224 // grid_size

        coords = []
        for y in range(grid_size):
            for x in range(grid_size):
                cx = x * patch_size + patch_size // 2
                cy = y * patch_size + patch_size // 2
                coords.append([cx, cy])
        coords = torch.tensor(coords).to(self.device)

        return coords.cpu().numpy(), features.cpu().numpy()


def visualize_3d_points_in_cloud(xyz_path, color_path, xyz_points, main_point=None, normal=None):
    """
    Liest die Punktwolke aus xyz_path direkt ein, lädt die RGB-Farben aus color_path
    und visualisiert:
      - die komplette Punktwolke,
      - die matched xyz_points als bunte Kugeln,
      - den Greifpunkt als grüne Kugel,
      - die Normale als roten Pfeil.
    """

    # 1) Punktwolke laden
    #    XYZ-Datei hat Format H*W Zeilen mit je 3 floats
    xyz = np.loadtxt(xyz_path)     # shape = (H*W,3)
    color = np.array(Image.open(color_path).convert("RGB"))
    H,W,_ = color.shape
    if xyz.shape[0] != H*W:
        raise ValueError("XYZ und Bildgröße passen nicht zusammen")

    # Farben als float 0..1
    colors = (color.reshape(-1,3).astype(np.float32) / 255.0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # 2) Kugeln für correspondences
    sphere_colors = [[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1]]
    spheres = []
    for i,pt in enumerate(xyz_points):
        if pt is not None:
            sph = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
            sph.paint_uniform_color(sphere_colors[i % len(sphere_colors)])
            sph.translate(pt)
            spheres.append(sph)

    # 3) Greifpunkt & Normale einzeichnen
    extras = []
    if main_point is not None and normal is not None:
        # 3a) grüne Kugel für Greifpunkt
        grip = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
        grip.paint_uniform_color([0,1,0])
        grip.translate(main_point)
        extras.append(grip)

        # 3b) roter Pfeil für Normale
        arrow = o3d.geometry.LineSet()
        start = main_point
        end   = main_point + normal * 0.1
        arrow.points = o3d.utility.Vector3dVector([start,end])
        arrow.lines  = o3d.utility.Vector2iVector([[0,1]])
        arrow.colors = o3d.utility.Vector3dVector([[1,0,0]])
        extras.append(arrow)

    # 4) alles zusammen anzeigen
    o3d.visualization.draw_geometries([pcd, *spheres, *extras],
                                      window_name="Punktwolke + Matches + Normale")





