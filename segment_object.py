import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from groundingdino.util.inference import load_model, predict
from segment_anything import sam_model_registry, SamPredictor

# DEVICE wählen
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# SAM initialisieren
sam = sam_model_registry["vit_b"](checkpoint="weights/sam_vit_b_01ec64.pth").to(DEVICE)
predictor = SamPredictor(sam)

# GroundingDINO initialisieren
dino_model = load_model(
    "GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py",
    "weights/groundingdino_swinb_cogcoor.pth"
).to(DEVICE)

def grounded_sam_bounding_box(image_path: str, text_prompt: str,
                              box_threshold: float = 0.3,
                              text_threshold: float = 0.1) -> np.ndarray:
    # PIL laden und Originalmaße bestimmen
    pil_img = Image.open(image_path).convert("RGB")
    orig_w, orig_h = pil_img.size

    # PIL→NumPy→Tensor (3×H×W) und normieren
    arr = np.array(pil_img)
    tensor_img = (
        torch.from_numpy(arr)
             .permute(2, 0, 1)
             .float()
             .to(DEVICE)
        / 255.0
    )

    # GroundingDINO erwartet hier einen 3D-Tensor, macht image[None] intern
    boxes, scores, phrases = predict(
        model=dino_model,
        image=tensor_img,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        device=DEVICE
    )

    if len(boxes) == 0:
        raise RuntimeError(f"Kein Objekt '{text_prompt}' erkannt.")

    # Erste Box abrufen und zurückskalieren
    cx, cy, w, h = boxes[0].cpu().numpy()
    x0 = int((cx - w/2) * orig_w)
    y0 = int((cy - h/2) * orig_h)
    x1 = int((cx + w/2) * orig_w)
    y1 = int((cy + h/2) * orig_h)
    x0, x1 = max(0, x0), min(orig_w-1, x1)
    y0, y1 = max(0, y0), min(orig_h-1, y1)
    box = np.array([x0, y0, x1, y1])

    # Visualisierung
    plt.figure(figsize=(8,6))
    plt.imshow(pil_img)
    plt.gca().add_patch(
        plt.Rectangle((x0, y0), x1-x0, y1-y0,
                      edgecolor='red', facecolor='none', linewidth=2)
    )
    plt.title(f"Bounding Box für '{text_prompt}'", color='red')
    plt.axis('off')
    plt.show()

    return box

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True, help="Pfad zum Bild")
    p.add_argument("--text",  required=True, help="Objektname")
    args = p.parse_args()

    print("BBox:", grounded_sam_bounding_box(args.image, args.text))














