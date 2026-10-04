"""train.py - vòng huấn luyện dùng chung cho mọi thí nghiệm (B, T, F).

MỘT hàm run(cfg) cho mọi cấu hình (RUBRIC mục H): đổi thí nghiệm chỉ bằng cách đổi Config.

    python train.py --set exp_id=B01 backbone=resnet50 seed=0

Mỗi lần chạy ghi vào runs/<exp_id>/seed<k>/:
    config.json, log.txt, history.csv, lr_steps.npy, best.pt (checkpoint tốt nhất theo macro-F1 val),
    val_logits.npy, val_labels.npy, val_files.txt, [test_logits.npy...], summary.json
và curves/<exp_id>_<mota>.png. Một dòng tóm tắt được nối vào runs/all_runs.csv (nguồn cho results.xlsx).

Chọn checkpoint bằng eval.compute_metrics (cùng định nghĩa với lúc chấm). Test chỉ được đánh giá khi
cfg.save_test_predictions=True (Bước 4), đúng MỘT lần với checkpoint đã chọn trên val.

Mức tái lập: seed cố định cho random/numpy/torch/DataLoader worker; cudnn.benchmark=True để nhanh nên
kết quả GPU có thể lệch nhỏ giữa hai lần chạy cùng seed (ghi trong báo cáo).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import math
import os
import random
import sys
import time
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _import_eval():
    """Import eval.py gốc của repo (không sửa). Tìm ở sys.path, rồi ở các thư mục cha của file này."""
    try:
        import eval as ev  # noqa: F401
        if hasattr(ev, "compute_metrics"):
            return ev
    except ImportError:
        pass
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "eval.py").exists():
            sys.path.insert(0, str(parent))
            sys.modules.pop("eval", None)
            import eval as ev
            return ev
    raise ImportError("Không tìm thấy eval.py của repo bài lab; thêm thư mục gốc repo vào sys.path")


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    desc: str = ""                    # mô tả ngắn cho tên ảnh curves/<exp_id>_<desc>.png
    pred_tag: str | None = None       # tên dùng cho file predictions (mặc định = exp_id)
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | basic_vflip | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền, GUIDE.md mục 1.4) ---
    epochs: int = 12
    batch_size: int = 64
    optimizer: str = "adamw"          # adamw | sgd
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    grad_clip: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- tiện ích ---
    measure_latency: bool = True      # đo độ trễ sơ bộ batch 1 (Bước 1); Bước 3 đo kỹ bằng benchmark.py
    skip_if_done: bool = True         # Colab bị ngắt: chạy lại sẽ bỏ qua lần chạy đã xong
    save_checkpoint: bool = True
    max_steps_per_epoch: int | None = None  # chỉ để smoke-test; None khi chạy thật
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST. Mặc định TẮT (quy tắc S4). ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả của một lần chạy: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """<pred_dir>/<tag>_seed<k>_<split>.csv (tag = pred_tag hoặc exp_id)."""
    tag = cfg.pred_tag or cfg.exp_id
    return Path(cfg.pred_dir) / f"{tag}_seed{cfg.seed}_{split}.csv"


def curve_path(cfg: Config) -> Path:
    desc = cfg.desc or cfg.backbone.split("_patch")[0]
    suffix = f"_seed{cfg.seed}" if cfg.seed != 0 else ""
    return Path(cfg.curves_dir) / f"{cfg.exp_id}_{desc}{suffix}.png"


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_optimizer(model, cfg: Config):
    import torch
    from model import param_groups

    groups = param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    for g in groups:
        g["initial_lr"] = g["lr"]
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(groups, betas=(0.9, 0.999))
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(groups, momentum=0.9, nesterov=True)
    raise ValueError(f"optimizer={cfg.optimizer!r}")


def lr_factor(step: int, total: int, warmup: int) -> float:
    """Warmup tuyến tính (từ 1/warmup) rồi cosine về 0. Cập nhật THEO BƯỚC."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    import torch

    total = cfg.epochs * steps_per_epoch
    warmup = int(round(cfg.warmup_epochs * steps_per_epoch))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_factor(s, total, warmup))


class EMA:
    """W_ema <- d*W_ema + (1-d)*W sau mỗi bước tối ưu (slide trang 56).

    Bản sao riêng (self.module) để đánh giá. Tham số float được trung bình; buffer (running_mean/var
    của BN, num_batches_tracked) được CHÉP thẳng từ model (cách của timm ModelEmaV2 khi không EMA buffer).
    Decay khởi động: d_t = min(d, (1+t)/(10+t)) để EMA không bị kéo về trọng số ban đầu quá lâu.
    """

    def __init__(self, model, decay: float):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.num_updates = 0

    def update(self, model) -> None:
        import torch

        self.num_updates += 1
        d = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for e, m in zip(self.module.parameters(), model.parameters()):
                e.mul_(d).add_(m.detach(), alpha=1.0 - d)
            for e, m in zip(self.module.buffers(), model.buffers()):
                e.copy_(m)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None, lr_log: list | None = None) -> dict:
    import torch
    from losses import mix_batch, mixed_loss
    from model import set_train_mode

    set_train_mode(model)  # giữ BN đóng băng ở eval nếu init == "frozen"
    use_amp = cfg.amp and device.type == "cuda"
    total, n = 0.0, 0
    for step, (x, y, _) in enumerate(loader):
        if cfg.max_steps_per_epoch and step >= cfg.max_steps_per_epoch:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        targets = y
        if cfg.mix:
            x, targets = mix_batch(x, y, cfg.mix_alpha, cfg.mix)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        loss = mixed_loss(criterion, logits.float(), targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"loss = {loss.item()} (không hữu hạn) ở bước {step}")
        scaler.scale(loss).backward()
        if cfg.grad_clip:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        if ema is not None:
            ema.update(model)
        if lr_log is not None:
            lr_log.append([g["lr"] for g in optimizer.param_groups])
        total += loss.item() * x.size(0)
        n += x.size(0)
    return {"train_loss": total / max(1, n), "lr": optimizer.param_groups[0]["lr"]}


def evaluate(model, loader, criterion, device, amp: bool = True, max_steps: int | None = None):
    """eval + inference_mode; trả về (filenames, y_true, logits[N,9], loss) đúng thứ tự loader."""
    import torch

    model.eval()
    use_amp = amp and device.type == "cuda"
    names, ys, outs, total = [], [], [], 0.0
    with torch.inference_mode():
        for step, (x, y, f) in enumerate(loader):
            if max_steps and step >= max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
            logits = logits.float()
            total += criterion(logits, y).item() * x.size(0)
            names.extend(f)
            ys.append(y.cpu().numpy())
            outs.append(logits.cpu().numpy())
    y_true = np.concatenate(ys)
    logits = np.concatenate(outs)
    return names, y_true, logits, total / max(1, len(y_true))


def softmax_np(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str, lr_steps=None) -> None:
    """Loss train/val, macro-F1 & top-1 val theo epoch, và LR theo bước (warmup + cosine)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = pd.DataFrame(history)
    ncols = 3 if lr_steps is not None and len(lr_steps) else 2
    fig, ax = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4))
    ax[0].plot(h.epoch, h.train_loss, "o-", label="train loss")
    ax[0].plot(h.epoch, h.val_loss, "s-", label="val loss")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Loss")
    ax[1].plot(h.epoch, h.val_macro_f1, "o-", color="tab:green", label="val macro-F1")
    ax[1].plot(h.epoch, h.val_top1, "s--", color="tab:purple", label="val top-1")
    if "val_macro_f1_raw" in h and h.val_macro_f1_raw.notna().any():
        ax[1].plot(h.epoch, h.val_macro_f1_raw, "^:", color="tab:gray", label="val macro-F1 (không EMA)")
    best = h.loc[h.val_macro_f1.idxmax()]
    ax[1].axvline(best.epoch, color="tab:red", lw=0.8, ls=":", label=f"best ep {int(best.epoch)}: {best.val_macro_f1:.4f}")
    ax[1].set(xlabel="epoch", ylabel="metric", title="Val metric")
    if ncols == 3:
        lr = np.asarray(lr_steps)
        ax[2].plot(lr[:, 0], label="LR backbone")
        if lr.shape[1] > 1:
            ax[2].plot(lr[:, -1], label="LR head")
        ax[2].set(xlabel="bước (iteration)", ylabel="LR", title="Lịch LR", yscale="log")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


class _Logger:
    def __init__(self, path: Path):
        self.f = open(path, "a", encoding="utf-8")

    def __call__(self, *msg):
        s = " ".join(str(m) for m in msg)
        print(s, flush=True)
        self.f.write(s + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def _env_info(device) -> dict:
    import timm
    import torch

    return {
        "python": sys.version.split()[0], "torch": torch.__version__, "timm": timm.__version__,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }


def _test_predictions(cfg: Config, model, device, criterion, test_df, log, ev, rd: Path) -> dict:
    """Đánh giá TEST đúng một lần với checkpoint đã chọn (chỉ khi save_test_predictions)."""
    from dataset import build_transforms, make_loader

    loader = make_loader(test_df, cfg.images_dir, build_transforms(False, cfg.img_size), cfg.batch_size,
                         train=False, num_workers=cfg.num_workers, seed=cfg.seed)
    names, y, logits, _ = evaluate(model, loader, criterion, device, cfg.amp)
    probs = softmax_np(logits)
    np.save(rd / "test_logits.npy", logits)
    np.save(rd / "test_labels.npy", y)
    (rd / "test_files.txt").write_text("\n".join(names))
    ev.save_predictions(pred_path(cfg, "test"), names, y, probs)
    m = ev.compute_metrics(y, probs.argmax(1), probs)
    log(f"[TEST - chạy 1 lần] macro-F1={m['macro_f1']:.4f} top-1={m['top1']:.4f} -> {pred_path(cfg, 'test')}")
    return {"test_macro_f1": m["macro_f1"], "test_top1": m["top1"], "test_ece": m["ece"]}


def load_trained(cfg: Config, device="cpu"):
    """Nạp lại model + checkpoint tốt nhất của một lần chạy (dùng ở Bước 3 / 4)."""
    import torch
    from model import build_model

    model = build_model(cfg.backbone, pretrained=False, num_classes=9, drop_rate=cfg.drop_rate,
                        init="finetune", drop_path_rate=cfg.drop_path_rate)
    sd = torch.load(run_dir(cfg) / "best.pt", map_location="cpu")
    model.load_state_dict(sd)
    return model.to(device).eval()


def run(cfg: Config) -> dict:
    import torch
    from dataset import NUM_CLASSES, build_transforms, check_split, load_split, make_loader
    from losses import build_criterion, class_weights
    from model import build_model, count_gmacs, count_params, weight_tag

    ev = _import_eval()
    rd = run_dir(cfg)
    rd.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary_file = rd / "summary.json"

    # 0. Đã chạy xong trước đó (Colab bị ngắt) -> dùng lại, chỉ bổ sung test nếu Bước 4 cần.
    if cfg.skip_if_done and summary_file.exists():
        summary = json.loads(summary_file.read_text())
        need_test = cfg.save_test_predictions and not pred_path(cfg, "test").exists()
        if not need_test:
            print(f"[skip] {cfg.exp_id} seed{cfg.seed} đã có kết quả: val macro-F1={summary['val_macro_f1']:.4f}")
            return summary
        log = _Logger(rd / "log.txt")
        log(f"[resume] {cfg.exp_id} seed{cfg.seed}: nạp best.pt để chạy test một lần")
        _, _, test_df = load_split(cfg.labels_dir, cfg.fold)
        model = load_trained(cfg, device)
        summary.update(_test_predictions(cfg, model, device, torch.nn.CrossEntropyLoss(), test_df, log, ev, rd))
        val_csv = pred_path(cfg, "val")
        if not val_csv.exists() and (rd / "val_logits.npy").exists():
            ev.save_predictions(val_csv, (rd / "val_files.txt").read_text().split("\n"),
                                np.load(rd / "val_labels.npy"), softmax_np(np.load(rd / "val_logits.npy")))
        summary_file.write_text(json.dumps(summary, indent=2, default=float))
        log.close()
        return summary

    # 1. seed, config, log
    set_seed(cfg.seed)
    (rd / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
    log = _Logger(rd / "log.txt")
    env = _env_info(device)
    log(f"=== {cfg.exp_id} seed{cfg.seed} | {cfg.backbone} | {env} ===")
    log("config:", json.dumps(dataclasses.asdict(cfg)))

    # 2. split + kiểm tra S1-S6
    train_df, val_df, test_df = load_split(cfg.labels_dir, cfg.fold)
    check_split(train_df, val_df, test_df, cfg.images_dir, verbose=False,
                expected_total=17509 if cfg.max_steps_per_epoch is None else None)

    # 3. loaders (test KHÔNG tạo ở đây)
    tr_loader = make_loader(train_df, cfg.images_dir, build_transforms(True, cfg.img_size, cfg.aug),
                            cfg.batch_size, train=True, sampler=cfg.sampler,
                            num_workers=cfg.num_workers, seed=cfg.seed)
    va_loader = make_loader(val_df, cfg.images_dir, build_transforms(False, cfg.img_size), cfg.batch_size,
                            train=False, num_workers=cfg.num_workers, seed=cfg.seed)

    # 4. model, loss, optimizer, scheduler, scaler, EMA
    model = build_model(cfg.backbone, True, NUM_CLASSES, cfg.drop_rate, cfg.init, cfg.drop_path_rate)
    tag = weight_tag(model)
    n_params = count_params(model)
    try:
        gmacs = count_gmacs(model, cfg.img_size)
    except Exception as e:  # không chặn huấn luyện nếu bộ đếm lỗi
        log(f"[cảnh báo] count_gmacs lỗi: {e}")
        gmacs = float("nan")
    model = model.to(device)
    weight = None
    if cfg.loss == "ce_weighted" or cfg.class_weight_beta is not None:
        counts = np.bincount(train_df["Label"], minlength=NUM_CLASSES)  # CHỈ từ train
        weight = class_weights(counts, cfg.class_weight_beta or 0.0)
        log("class weights (từ train):", np.round(weight.numpy(), 3).tolist())
    criterion = build_criterion(cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma,
                                weight=weight).to(device)
    eval_criterion = torch.nn.CrossEntropyLoss()  # val loss luôn là CE thường để so được giữa các run
    steps = len(tr_loader) if not cfg.max_steps_per_epoch else min(len(tr_loader), cfg.max_steps_per_epoch)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, steps)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay else None
    log(f"tag={tag} | params={n_params:.2f}M | GMAC={gmacs:.3f} | steps/epoch={steps} | device={device}")

    # 5. vòng epoch, chọn checkpoint theo macro-F1 val (hòa -> epoch sớm hơn: dùng '>')
    history, lr_log = [], []
    best = {"f1": -1.0, "epoch": -1, "state": None, "val": None}
    vmax = cfg.max_steps_per_epoch
    for ep in range(1, cfg.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        tr = train_one_epoch(model, tr_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema, lr_log)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.perf_counter() - t0
        eval_model = ema.module if ema is not None else model
        names, y, logits, vloss = evaluate(eval_model, va_loader, eval_criterion, device, cfg.amp, vmax)
        probs = softmax_np(logits)
        m = ev.compute_metrics(y, probs.argmax(1), probs)
        row = {"epoch": ep, "train_loss": tr["train_loss"], "val_loss": vloss,
               "val_macro_f1": m["macro_f1"], "val_top1": m["top1"], "val_ece": m["ece"],
               "lr_backbone_end": tr["lr"], "train_time_s": t_train}
        if ema is not None:  # để so sánh EMA với trọng số thường (I06)
            _, y2, lg2, _ = evaluate(model, va_loader, eval_criterion, device, cfg.amp, vmax)
            p2 = softmax_np(lg2)
            row["val_macro_f1_raw"] = ev.compute_metrics(y2, p2.argmax(1), p2)["macro_f1"]
        history.append(row)
        flag = ""
        if m["macro_f1"] > best["f1"]:
            best = {"f1": m["macro_f1"], "epoch": ep,
                    "state": {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()},
                    "val": (names, y, logits, m)}
            flag = " *"
        log(f"ep {ep:02d}/{cfg.epochs} | train_loss {tr['train_loss']:.4f} | val_loss {vloss:.4f} | "
            f"val macro-F1 {m['macro_f1']:.4f} | top-1 {m['top1']:.4f} | {t_train:.0f}s{flag}")

    # 6. nạp checkpoint tốt nhất, lưu val logits + predictions val
    model.load_state_dict(best["state"])
    model.eval()
    if cfg.save_checkpoint:
        torch.save(best["state"], rd / "best.pt")
    names, y, logits, m = best["val"]
    np.save(rd / "val_logits.npy", logits)
    np.save(rd / "val_labels.npy", y)
    (rd / "val_files.txt").write_text("\n".join(names))
    val_csv = pred_path(cfg, "val") if cfg.save_test_predictions else rd / f"{cfg.exp_id}_seed{cfg.seed}_val.csv"
    ev.save_predictions(val_csv, names, y, softmax_np(logits))
    pd.DataFrame(history).to_csv(rd / "history.csv", index=False)
    np.save(rd / "lr_steps.npy", np.asarray(lr_log))

    # độ trễ sơ bộ batch 1 (Bước 1). Bước 3 đo kỹ bằng benchmark.py
    lat = {}
    if cfg.measure_latency and device.type == "cuda":
        from benchmark import latency_report
        lat = latency_report(model, 1, cfg.img_size, "fp32", "cuda", warmup=10, iters=50)
        log(f"độ trễ sơ bộ batch1 fp32: p50={lat['p50']:.2f}ms p95={lat['p95']:.2f}ms")

    epoch_times = [h["train_time_s"] for h in history]
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "backbone": cfg.backbone, "weight_tag": tag,
        "params_M": n_params, "gmacs": gmacs, "img_size": cfg.img_size, "epochs": cfg.epochs,
        "best_epoch": best["epoch"], "val_macro_f1": m["macro_f1"], "val_top1": m["top1"],
        "val_ece": m["ece"], "val_balanced_acc": m["balanced_acc"],
        "val_f1_per_class": [float(v) for v in m["f1"]],
        "train_time_per_epoch_s": float(np.mean(epoch_times)),
        "latency_b1_p50_ms": lat.get("p50"), "latency_b1_p95_ms": lat.get("p95"),
        "curve": str(curve_path(cfg)), **env,
        "config": dataclasses.asdict(cfg),
    }
    # 7. TEST: chỉ ở Bước 4, đúng một lần
    if cfg.save_test_predictions:
        _, _, test_df = load_split(cfg.labels_dir, cfg.fold)
        summary.update(_test_predictions(cfg, model, device, eval_criterion, test_df, log, ev, rd))

    # 8. biểu đồ + tóm tắt
    plot_curves(history, curve_path(cfg),
                f"{cfg.exp_id} | {cfg.backbone} | seed {cfg.seed} | best val macro-F1 {m['macro_f1']:.4f} (ep {best['epoch']})",
                lr_log)
    summary_file.write_text(json.dumps(summary, indent=2, default=float))
    flat = {k: v for k, v in summary.items() if k not in ("config", "val_f1_per_class")}
    flat.update({f"cfg_{k}": v for k, v in dataclasses.asdict(cfg).items()})
    flat.update({f"val_f1_c{i}": v for i, v in enumerate(summary["val_f1_per_class"])})
    all_runs = Path(cfg.out_dir) / "all_runs.csv"
    pd.DataFrame([flat]).to_csv(all_runs, mode="a", header=not all_runs.exists(), index=False)
    log(f"XONG {cfg.exp_id} seed{cfg.seed}: best ep {best['epoch']} | val macro-F1 {m['macro_f1']:.4f} | "
        f"top-1 {m['top1']:.4f} | {np.mean(epoch_times):.0f}s/epoch | curve {curve_path(cfg)}")
    log.close()
    del model, ema, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return summary


def _cast(value: str, typ):
    v = value.strip()
    if v.lower() in ("none", "null"):
        return None
    if isinstance(typ, str):
        typ_s = typ
    else:
        typ_s = getattr(typ, "__name__", str(typ))
    if "bool" in typ_s:
        if v.lower() in ("1", "true", "yes", "y"):
            return True
        if v.lower() in ("0", "false", "no", "n"):
            return False
        raise ValueError(f"không đọc được bool từ {value!r}")
    if "int" in typ_s and "float" not in typ_s:
        return int(v)
    if "float" in typ_s:
        return float(v)
    return v


def parse_overrides(pairs: list[str]) -> dict:
    """['seed=1', 'loss=focal', 'ema_decay=none'] -> dict đã ép kiểu theo field của Config."""
    fields = {f.name: f.type for f in dataclasses.fields(Config)}
    out = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"{pair!r} phải có dạng KEY=VALUE")
        k, v = pair.split("=", 1)
        k = k.strip()
        if k not in fields:
            raise KeyError(f"Config không có trường {k!r}. Các trường: {sorted(fields)}")
        out[k] = _cast(v, fields[k])
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Huấn luyện một cấu hình DeepWeeds")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args(argv)
    cfg = Config(**parse_overrides(args.set))
    res = run(cfg)
    print(json.dumps({k: v for k, v in res.items() if k != "config"}, indent=2, default=float))


if __name__ == "__main__":
    main()
