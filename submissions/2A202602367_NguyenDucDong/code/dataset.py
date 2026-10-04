"""dataset.py - đọc DeepWeeds, kiểm tra chia dữ liệu, transform, DataLoader.

Hoàn thiện từ starter/dataset.py. Giao diện giữ nguyên:
    load_split(labels_dir, fold=0)            -> (train_df, val_df, test_df)
    check_split(train_df, val_df, test_df, images_dir) -> dict  (số liệu để ghi báo cáo)
    build_transforms(train, img_size, aug)    -> torchvision transform
    DeepWeedsDataset[i]                       -> (image_tensor, label:int, filename:str)
    make_loader(df, images_dir, transform, batch_size, train, sampler, num_workers)

Lựa chọn đã ghi rõ (đưa vào báo cáo):
  - Val/test: Resize(img_size * 256/224) rồi CenterCrop(img_size). Với img_size=224, ảnh gốc 256x256
    giữ nguyên rồi cắt giữa 224.
  - KHÔNG dùng lật dọc mặc định: ảnh chụp từ robot hướng xuống mặt đất nên lật dọc về lý thuyết hợp lệ,
    nhưng để công thức nền giống slide, lật dọc chỉ bật ở aug="basic_vflip" (một giá trị của trục B).
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TOTAL_IMAGES = 17509
AUG_CHOICES = ("basic", "basic_vflip", "color", "trivial", "randaug")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Đọc train/val/test_subset{fold}.csv nguyên bản (S1). Không sửa, lọc hay chia lại."""
    labels_dir = Path(labels_dir)
    out = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(labels_dir / f"{split}_subset{fold}.csv")
        missing = {"Filename", "Label"} - set(df.columns)
        if missing:
            raise ValueError(f"{split}_subset{fold}.csv thiếu cột {missing}")
        out.append(df)
    return tuple(out)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path, verbose: bool = True, expected_total: int = TOTAL_IMAGES) -> dict:
    """Các kiểm tra bắt buộc (README mục 2.1). Lỗi -> AssertionError để dừng ngay."""
    sets = {"train": train_df, "val": val_df, "test": test_df}
    n = {k: int(len(v)) for k, v in sets.items()}
    total = sum(n.values())
    frac = {k: v / total for k, v in n.items()}
    per_class = {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).astype(int).tolist()
                 for k, v in sets.items()}
    names = {k: set(v["Filename"]) for k, v in sets.items()}
    for k, v in sets.items():
        assert v["Filename"].is_unique, f"{k}: có Filename trùng lặp trong cùng một tập"
    overlap = {
        "train&val": len(names["train"] & names["val"]),
        "train&test": len(names["train"] & names["test"]),
        "val&test": len(names["val"] & names["test"]),
    }
    union = len(names["train"] | names["val"] | names["test"])
    assert all(v == 0 for v in overlap.values()), f"Giao giữa các tập khác rỗng: {overlap}"
    if expected_total is not None:
        assert union == expected_total, f"Hợp ba tập = {union}, kỳ vọng {expected_total}"
    for k, f in zip(("train", "val", "test"), (0.6, 0.2, 0.2)):
        if abs(frac[k] - f) > 0.01:
            print(f"[CẢNH BÁO] tỉ lệ {k} = {frac[k]:.3f} lệch > 1 điểm % so với {f}: báo giảng viên")
    images_dir = Path(images_dir)
    on_disk = {p.name for p in images_dir.iterdir()} if images_dir.exists() else set()
    all_names = names["train"] | names["val"] | names["test"]
    missing = sorted(all_names - on_disk)
    assert not missing, f"{len(missing)} file trong CSV không có trong {images_dir}, ví dụ {missing[:3]}"
    for k, v in sets.items():
        assert v["Label"].between(0, NUM_CLASSES - 1).all(), f"{k}: nhãn ngoài [0, 8]"
    info = {"n": n, "fraction": frac, "union": union, "overlap": overlap,
            "per_class": per_class, "missing_files": len(missing)}
    if verbose:
        print(f"Số ảnh: {n} | tỉ lệ " + " / ".join(f"{frac[k]*100:.1f}%" for k in n) + f" | hợp = {union}")
        print(f"Giao theo Filename: {overlap} | file thiếu trên đĩa: {len(missing)}")
        tab = pd.DataFrame(per_class, index=CLASS_NAMES)
        tab["total"] = tab.sum(1)
        print(tab.to_string())
    return info


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic",
                     mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Tạo transform. aug ∈ AUG_CHOICES (trục B). Mixup/CutMix nằm ở losses.py."""
    import torchvision.transforms as T

    norm = [T.ToTensor(), T.Normalize(mean, std)]
    if not train:
        resize = int(round(img_size * 256 / 224))
        return T.Compose([T.Resize(resize), T.CenterCrop(img_size), *norm])
    if aug not in AUG_CHOICES:
        raise ValueError(f"aug={aug!r} không hợp lệ, chọn trong {AUG_CHOICES}")
    ops = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "basic_vflip":
        ops.append(T.RandomVerticalFlip())
    elif aug == "color":
        ops.append(T.ColorJitter(0.3, 0.3, 0.3, 0.05))
    elif aug == "trivial":
        ops.append(T.TrivialAugmentWide())
    elif aug == "randaug":
        ops.append(T.RandAugment(num_ops=2, magnitude=9))
    return T.Compose([*ops, *norm])


try:
    from torch.utils.data import Dataset as _TorchDataset
except Exception:  # cho phép import module mà không có torch (test không GPU)
    _TorchDataset = object


class DeepWeedsDataset(_TorchDataset):
    """__getitem__(i) -> (ảnh đã transform, nhãn int, tên file str)."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.filenames = self.df["Filename"].astype(str).tolist()
        self.labels = self.df["Label"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        from PIL import Image

        with Image.open(self.images_dir / self.filenames[i]) as im:
            img = im.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, int(self.labels[i]), self.filenames[i]


def seed_worker(worker_id: int) -> None:
    """worker_init_fn: mỗi worker có seed suy ra từ seed gốc của torch (tái lập augmentation)."""
    import torch

    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def balanced_weights(labels) -> np.ndarray:
    """Trọng số mẫu 1/(số ảnh của lớp) cho WeightedRandomSampler (trục D)."""
    labels = np.asarray(labels)
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    return 1.0 / counts[labels]


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2, seed: int = 0):
    """DataLoader. train=False giữ đúng thứ tự df (để ghép logit với Filename)."""
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    ds = DeepWeedsDataset(df, images_dir, transform)
    g = torch.Generator()
    g.manual_seed(seed)
    smp = None
    if train and sampler == "balanced":
        w = torch.as_tensor(balanced_weights(ds.labels), dtype=torch.double)
        smp = WeightedRandomSampler(w, num_samples=len(ds), replacement=True, generator=g)
    elif sampler not in (None, "none", "balanced"):
        raise ValueError(f"sampler={sampler!r} không hợp lệ")
    return DataLoader(
        ds, batch_size=batch_size,
        shuffle=(train and smp is None), sampler=smp,
        drop_last=train, num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker, generator=g,
        persistent_workers=num_workers > 0,
    )


def denormalize(x, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    """Giải chuẩn hoá tensor (C,H,W) hoặc (N,C,H,W) về [0,1] để vẽ ảnh sau augmentation."""
    import torch

    m = torch.tensor(mean).view(-1, 1, 1)
    s = torch.tensor(std).view(-1, 1, 1)
    return (x.cpu() * s + m).clamp(0, 1)
