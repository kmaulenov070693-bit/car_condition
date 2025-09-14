# infer.py
import argparse
from pathlib import Path
import torch
from PIL import Image
import torchvision.transforms as T

from models import MultiHeadCarNet

def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = MultiHeadCarNet(backbone=ckpt.get("backbone", "resnet18"), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt.get("img_size", 224)

def make_tfms(size):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def predict_on_image(model, img_path, tfm, device):
    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)  # [1,2]
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [2]
    return {"dirty_prob": float(probs[0]), "damaged_prob": float(probs[1])}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="путь к чекпоинту .pt")
    ap.add_argument("--path", required=True, help="изображение или папка")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, size = load_model(args.ckpt)
    model.to(device)
    tfm = make_tfms(size)

    p = Path(args.path)
    files = [p] if p.is_file() else [f for f in p.iterdir() if f.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]

    for f in files:
        out = predict_on_image(model, f, tfm, device)
        dirty = "грязный" if out["dirty_prob"] >= 0.5 else "чистый"
        dmg = "битый" if out["damaged_prob"] >= 0.5 else "целый"
        print(f"{f.name}: dirty={out['dirty_prob']:.2f} ({dirty}), damaged={out['damaged_prob']:.2f} ({dmg})")

if __name__ == "__main__":
    main()
