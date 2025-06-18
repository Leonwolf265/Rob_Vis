import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Feste Pfade zu den Bildern und Feature-Dateien
features_path_a = "data/outputs/cup2_features.pth"
features_path_b = "data/outputs/cup2_features.pth"
image_path_a = "data/images/cup2.jpeg"
image_path_b = "data/images/cup2.jpeg"
num_points = 20

# Lade Feature-Tensoren
feat_a = torch.load(features_path_a).squeeze(0)[0]  # [N, D]
feat_b = torch.load(features_path_b).squeeze(0)[0]

print(f"feat A: {feat_a.shape}, feat B: {feat_b.shape}")

# Normalisieren
feat_a = torch.nn.functional.normalize(feat_a, dim=1)
feat_b = torch.nn.functional.normalize(feat_b, dim=1)

# Ähnlichkeitsmatrix berechnen
sim_matrix = feat_a @ feat_b.T  # [N, N]

# Zufällige Indizes
patch_indices_a = np.random.choice(feat_a.shape[0], num_points, replace=False)
patch_indices_b = sim_matrix[patch_indices_a].argmax(dim=1).numpy()

# Patchgröße bestimmen (bei 224x224 und z.B. 53x53 Patches)
grid_size = int(np.sqrt(feat_a.shape[0]))
patch_size = 224 // grid_size

def idx_to_coord(idx, grid_w, patch_size):
    y = idx // grid_w
    x = idx % grid_w
    return (x * patch_size + patch_size // 2, y * patch_size + patch_size // 2)

points_a = [idx_to_coord(i, grid_size, patch_size) for i in patch_indices_a]
points_b = [idx_to_coord(i, grid_size, patch_size) for i in patch_indices_b]

# Bilder laden
img_a = Image.open(image_path_a).convert("RGB").resize((224, 224))
img_b = Image.open(image_path_b).convert("RGB").resize((224, 224))

# Kombiniertes Bild anzeigen
fig, ax = plt.subplots(figsize=(10, 5))
canvas = np.hstack([np.array(img_a), np.array(img_b)])
ax.imshow(canvas)

for (xa, ya), (xb, yb) in zip(points_a, points_b):
    ax.plot([xa, xb + 224], [ya, yb], 'r-', linewidth=1)
    ax.plot(xa, ya, 'bo')
    ax.plot(xb + 224, yb, 'go')

ax.set_title("Semantisch korrespondierende Punkte zwischen cat0 und cat1")
ax.axis("off")
plt.tight_layout()
plt.show()
