# dataset.py
from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torch
import torchvision.transforms as T
import random

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class CarConditionDataset(Dataset):
    def __init__(self, root=".", split="train", labels_csv="labels.csv", img_size=224, aug=True):
        self.root = Path(root)
        self.split = split
        self.dir = self.root / split

        # собираем список файлов в папке
        self.files = sorted([p for p in self.dir.iterdir() if p.suffix.lower() in IMG_EXT])

        # метки (если есть)
        labels_path = self.root / labels_csv
        self.labels_df = None
        if labels_path.exists():
            df = pd.read_csv(labels_path)
            # нормализуем имена (иногда в csv путь относительный)
            df["filename_norm"] = df["filename"].apply(lambda s: Path(s).name)
            self.map_labels = df[df["split"] == split].set_index("filename_norm")[["dirty", "damaged"]]
            self.labels_df = self.map_labels
        else:
            self.map_labels = None

        # аугментации/препроцессинг
        self.train_tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.RandomApply([T.GaussianBlur(3)], p=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.val_tfms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])
        self.aug = aug

    def __len__(self):
        return len(self.files)

    def _get_labels(self, fname):
        # Возвращаем тензоры: target (2,), mask (2,)
        # mask[i]=0 → нет метки; mask[i]=1 → есть метка
        if self.labels_df is None:
            # меток нет: все маски 0
            target = torch.zeros(2, dtype=torch.float32)
            mask = torch.zeros(2, dtype=torch.float32)
            return target, mask

        name = fname.name
        if name not in self.labels_df.index:
            target = torch.zeros(2, dtype=torch.float32)
            mask = torch.zeros(2, dtype=torch.float32)
            return target, mask

        row = self.labels_df.loc[name]
        t = []
        m = []
        # dirty
        if pd.isna(row["dirty"]):
            t.append(0.0)
            m.append(0.0)
        else:
            t.append(float(row["dirty"]))
            m.append(1.0)
        # damaged
        if pd.isna(row["damaged"]):
            t.append(0.0)
            m.append(0.0)
        else:
            t.append(float(row["damaged"]))
            m.append(1.0)

        return torch.tensor(t, dtype=torch.float32), torch.tensor(m, dtype=torch.float32)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert("RGB")
        tfm = self.train_tfms if (self.split == "train" and self.aug) else self.val_tfms
        x = tfm(img)

        y, m = self._get_labels(img_path)

        return {
            "image": x,
            "target": y,   # [dirty, damaged]
            "mask": m,     # [0/1, 0/1]
            "path": str(img_path)
        }
