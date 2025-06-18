import torch
import numpy as np
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from extractor_dino import ViTExtractor 
import matplotlib.pyplot as plt

# --- 1. Initialisierung ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
extractor = ViTExtractor(device=device)

# --- 2. Bilder laden ---
img_ref = Image.open("reference.jpg").convert("RGB")
img_scene = Image.open("scene.jpg").convert("RGB")

# --- 3. Feature-Extraktion ---
feat_ref = extractor.extract_features(img_ref)[0].cpu().numpy()  # [num_patches, feat_dim]
feat_scene = extractor.extract_features(img_scene)[0].cpu().numpy()

# --- 4. Patch-Koordinaten berechnen (ViT: 14x14) ---
def get_patch_coords(image_size, patch_size=16):
    w, h = image_size
    grid_w = w // patch_size
    grid_h = h // patch_size
    coords = [(x * patch_size + patch_size // 2, y * patch_size + patch_size // 2)
              for y in range(grid_h) for x in range(grid_w)]
    return coords

coords_ref = get_patch_coords(img_ref.size)
coords_scene = get_patch_coords(img_scene.size)

# --- 5. Benutzer wählt einen Punkt im Referenzbild ---
# --- Interaktive Punktauswahl im Referenzbild ---
def select_point(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.set_title("Klicke auf einen Punkt im Referenzbild")
    clicked = []

    def onclick(event):
        ix, iy = int(event.xdata), int(event.ydata)
        clicked.append((ix, iy))
        print(f"Gewählter Punkt: ({ix}, {iy})")
        plt.close()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    return clicked[0] if clicked else None

selected_uv = select_point(img_ref)



# Finde nächsten Patch im Referenzbild
def find_nearest_patch(coords, uv):
    distances = [np.linalg.norm(np.array(uv) - np.array(c)) for c in coords]
    return np.argmin(distances)

idx_ref_patch = find_nearest_patch(coords_ref, selected_uv)
ref_vec = feat_ref[idx_ref_patch:idx_ref_patch+1]  # [1, D]

# --- 6. Nearest Neighbor Matching im Szenenbild ---
sims = cosine_similarity(ref_vec, feat_scene)  # [1, N]
idx_best = np.argmax(sims)

matched_uv = coords_scene[idx_best]
print(f"Matched Punkt im Szenenbild: {matched_uv}")

# --- 7. Lade 2D→3D-Mapping und gebe 3D-Koordinate zurück ---
uv_to_3d = np.load("uv_to_3d_map.npy", allow_pickle=True).item()
matched_uv_int = (int(round(matched_uv[0])), int(round(matched_uv[1])))

if matched_uv_int in uv_to_3d:
    point_3d = uv_to_3d[matched_uv_int]
    print("Korrespondierender 3D-Punkt:", point_3d)
else:
    print("Kein 3D-Punkt für Match gefunden.")

# --- 8. Visualisierung (optional) ---
def visualize_match(img1, uv1, img2, uv2):
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].imshow(img1)
    axs[0].scatter(*uv1, color='red')
    axs[0].set_title("Referenzbild")
    axs[1].imshow(img2)
    axs[1].scatter(*uv2, color='green')
    axs[1].set_title("Szenenbild")
    plt.show()

visualize_match(img_ref, selected_uv, img_scene, matched_uv)
