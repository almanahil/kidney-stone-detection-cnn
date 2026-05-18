import random
import shutil
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXTS

def copy_files(files, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(str(f), str(dst_dir / f.name))

def split_imagefolder(
    raw_root: str,
    out_root: str,
    train=0.7,
    val=0.15,
    test=0.15,
    seed=42,
):
    raw_root = Path(raw_root)
    out_root = Path(out_root)

    assert abs((train + val + test) - 1.0) < 1e-9, "train+val+test must equal 1.0"
    if not raw_root.exists():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")

    random.seed(seed)

    class_dirs = [d for d in raw_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise ValueError(
            "No class folders found. Expected: raw_root/classA/*, raw_root/classB/*"
        )

    # Always create split dirs
    for split_name in ["train", "val", "test"]:
        (out_root / split_name).mkdir(parents=True, exist_ok=True)

    for cdir in class_dirs:
        images = [p for p in cdir.rglob("*") if p.is_file() and is_image(p)]
        if not images:
            print(f"[WARN] No images found in: {cdir.name}")
            continue

        random.shuffle(images)
        n = len(images)
        n_train = int(n * train)
        n_val = int(n * val)

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        copy_files(train_files, out_root / "train" / cdir.name)
        copy_files(val_files, out_root / "val" / cdir.name)
        copy_files(test_files, out_root / "test" / cdir.name)

        print(f"{cdir.name}: total={n} train={len(train_files)} val={len(val_files)} test={len(test_files)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python src/split_dataset.py <raw_root> <out_root>")
        raise SystemExit(1)

    split_imagefolder(sys.argv[1], sys.argv[2])
