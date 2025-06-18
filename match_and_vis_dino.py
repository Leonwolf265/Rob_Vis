import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import argparse
import timm
import torchvision.transforms as T


# ------------------------------
# DINO Feature Extraktion Klasse
# ------------------------------
class ViTExtractor:
    def __init__(self, model_type='dino_vits16', stride=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = torch.hub.load('facebookresearch/dino:main', model_type)
        self.model.eval().to(device)
        self.patch_size = self.model.patch_embed.patch_size  # int
        self.feat_dim = self.model.embed_dim

        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_features(self, img: Image.Image):
        x = self.transform(img).unsqueeze(0).to(self.device)  # [1, 3, 224, 224]
        with torch.no_grad():
            x = self.model.patch_embed(x)  # Patchify
            cls_token = self.model.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
            x = x + self.model.pos_embed
            x = self.model.pos_drop(x)
            for blk in self.model.blocks:
                x = blk(x)
            x = self.model.norm(x)
            return x[:, 1:, :]  # ohne cls token


# ------------------------------
# Semantisches Matching Skript
# ------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True)
    parser.add_argument('--img2', type=str, required=True)
    parser.add_argument('--topk', type=int, default=5)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = ViTExtractor(device=device)

    # Bilder laden
    img1 = Image.open(args.img1).convert("RGB")
    img2 = Image.open(args.img2).convert("RGB")

    # Features extrahieren
    feat1 = extractor.extract_features(img1).squeeze(0)  # [N, D]
    feat2 = extractor.extract_features(img2).squeeze(0)  # [N, D]

    print(f"Features 1: {feat1.shape} | Features 2: {feat2.shape}")

    # Ähnlichkeitsmatrix
    feat1_norm = torch.nn.functional.normalize(feat1, dim=-1)
    feat2_norm = torch.nn.functional.normalize(feat2, dim=-1)
    sim = torch.matmul(feat1_norm, feat2_norm.T)  # [N, N]

    # Top-k Indizes: höchste Einzelübereinstimmung
    max_scores = sim.max(dim=1).values  # beste Übereinstimmung pro Punkt
    topk = torch.topk(max_scores, args.topk).indices


    grid_size = int(np.sqrt(feat1.shape[0]))
    patch_size = 224 // grid_size

    def idx_to_coord(idx):
        y = idx // grid_size
        x = idx % grid_size
        return (x * patch_size + patch_size // 2, y * patch_size + patch_size // 2)

    # Korrespondenzen berechnen
    coords1, coords2 = [], []
    for idx in topk:
        best_match = torch.argmax(sim[idx]).item()
        coords1.append(idx_to_coord(idx))
        coords2.append(idx_to_coord(best_match))

    # Visualisierung
    img1_np = np.array(img1.resize((224, 224)))
    img2_np = np.array(img2.resize((224, 224)))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(img1_np)
    ax[1].imshow(img2_np)

    for (x1, y1), (x2, y2) in zip(coords1, coords2):
        ax[0].plot(x1, y1, 'ro')
        ax[1].plot(x2, y2, 'ro')
        con = plt.Line2D([x1, x2 + 224], [y1, y2], c='lime')
        fig.add_artist(con)

    for a in ax:
        a.axis('off')
    plt.suptitle("Top-k Semantic Correspondences")
    plt.tight_layout()
    plt.show()

