# labels_build.py
import argparse
from pathlib import Path
import pandas as pd

DAMAGE_CLASSES = {
    # частые варианты классов в датасетах из описания
    "scratch", "scratches", "dent", "dent_and_scratch", "rust", "crack", "damage", "damaged"
}
DIRTY_CLASSES = {
    # если в разметке есть «грязные» классы; может и не быть
    "dirty", "mud", "dust", "soot", "stain", "dirty_car"
}

def build_labels(dir_path: Path):
    ann_path = dir_path / "_annotations.csv"
    if not ann_path.exists():
        print(f"[WARN] {ann_path} не найден. Пропускаю.")
        return None

    df = pd.read_csv(ann_path)

    # Roboflow формат часто имеет столбцы: filename,width,height,class,xmin,ymin,xmax,ymax
    # Нормализуем к нижнему регистру для классов
    if "class" not in df.columns:
        raise ValueError(f"{ann_path} не содержит столбца 'class'.")

    df["class_norm"] = df["class"].astype(str).str.lower().str.strip()

    # агрегируем по файлу
    g = df.groupby("filename")["class_norm"].apply(list).reset_index()

    def has_any(lst, vocab):
        return any(c in vocab for c in lst)

    g["damaged"] = g["class_norm"].apply(lambda x: 1 if has_any(x, DAMAGE_CLASSES) else 0)
    # dirty может отсутствовать в целом — тогда останется 0, но дальше дадим опцию делать NaN
    g["dirty"] = g["class_norm"].apply(lambda x: 1 if has_any(x, DIRTY_CLASSES) else None)

    # оставим только нужное
    out = g[["filename", "dirty", "damaged"]]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default=".", help="Корень проекта (где лежат train/ и test/)")
    ap.add_argument("--force_dirty_nan_if_absent", action="store_true",
                    help="Если в разметке нет ни одного dirty-класса, проставить dirty=NaN, иначе 0/1.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out_all = []

    for split in ["train", "test"]:
        labels = build_labels(root / split)
        if labels is not None:
            # Если dirty нигде не встретился в split — делаем NaN (маска для обучения)
            if args.force_dirty_nan_if_absent and labels["dirty"].notna().sum() == 0:
                labels["dirty"] = None
            labels["split"] = split
            out_all.append(labels)

    if not out_all:
        print("[ERROR] Не удалось собрать метки: _annotations.csv не найден.")
        return

    all_df = pd.concat(out_all, ignore_index=True)
    out_csv = root / "labels.csv"
    all_df.to_csv(out_csv, index=False)
    print(f"[OK] Сохранено: {out_csv} (rows={len(all_df)})")

if __name__ == "__main__":
    main()
