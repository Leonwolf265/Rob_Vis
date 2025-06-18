import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import timm
import torchvision.transforms as T


# ------------------------------
# DINO Feature Extraktion Klasse
# ------------------------------
class ViTExtractor:
    def __init__(self, model_type='dino_vits16', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dino:main', model_type)
        self.model.eval().to(device)
        self.patch_size = self.model.patch_embed.patch_size

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, img: Image.Image):
        x = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        with torch.no_grad():
            x = self.model.patch_embed(x)
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.model.pos_embed
            x = self.model.pos_drop(x)
            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)
            return x[:, 1:, :]  # ohne cls token


# ------------------------------
# Manuelle Auswahl und Visualisierung
# ------------------------------
if __name__ == '__main__':
    img1_path = 'data/images/cat0.jpg'
    img2_path = 'data/images/cat1.jpg'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(device=device)

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    img1_resized = img1.resize((224, 224))
    img2_resized = img2.resize((224, 224))

    feat1 = extractor.extract_features(img1).squeeze(0)  # [N, D]
    feat2 = extractor.extract_features(img2).squeeze(0)  # [N, D]

    feat1_norm = torch.nn.functional.normalize(feat1, dim=-1)
    feat2_norm = torch.nn.functional.normalize(feat2, dim=-1)
    sim = torch.matmul(feat1_norm, feat2_norm.T)  # [N, N]

    grid_size = int(np.sqrt(feat1.shape[0]))
    patch_size = 224 // grid_size

    def coord_to_index(x, y):
        gx = int(x) // patch_size
        gy = int(y) // patch_size
        return gy * grid_size + gx

    def idx_to_coord(idx):
        y = idx // grid_size
        x = idx % grid_size
        return (x * patch_size + patch_size // 2, y * patch_size + patch_size // 2)

    selected_coords = []

    def onclick(event):
        if event.inaxes:
            x, y = int(event.xdata), int(event.ydata)
            selected_coords.append((x, y))
            ax.plot(x, y, 'ro')
            fig.canvas.draw()
            if len(selected_coords) == 5:
                plt.close()

    fig, ax = plt.subplots()
    ax.imshow(img1_resized)
    ax.set_title("Klicke auf 5 beliebige Punkte im Bild")
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    # Matching und Visualisierung
    coords1 = selected_coords
    coords2 = []

    for x, y in coords1:
        idx = coord_to_index(x, y)
        best_match = torch.argmax(sim[idx]).item()
        coords2.append(idx_to_coord(best_match))

    img1_np = np.array(img1_resized)
    img2_np = np.array(img2_resized)
    # Kombiniere beide Bilder horizontal
    combined = np.concatenate((img1_np, img2_np), axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(combined)

    for (x1, y1), (x2, y2) in zip(coords1, coords2):
        ax.plot(x1, y1, 'ro')  # Punkt in Bild 1
        ax.plot(x2 + 224, y2, 'ro')  # Punkt in Bild 2 (verschoben)
        ax.plot([x1, x2 + 224], [y1, y2], color='lime', linewidth=1.5)  # Verbindungslinie

    ax.axis('off')
    plt.title("Manuelle Semantic Correspondences")
    plt.tight_layout()
    plt.show()




