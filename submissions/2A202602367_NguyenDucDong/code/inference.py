"""inference.py - các phương pháp suy luận (Bước 3).

Mọi hàm chạy ở eval, không gradient. Chọn phương pháp CHỈ trên val; T khớp trên VAL rồi áp sang test.

Giao diện:
    predict_logits(model, loader, device, view=None) -> (filenames, y_true, logits[N, 9])
    predict_views(model, loader, device, views_fn)   -> (filenames, y_true, [logits_view1, ...])
    aggregate_views(list_of_logits, space)           -> probs[N, 9]
    fit_temperature(val_logits, val_labels)          -> float T
    apply_temperature(logits, T)                     -> probs
    ensemble_probs(list_of_probs)                    -> probs
    fuse_conv_bn(model)                              -> model (BN đã gộp vào conv)

Với TTA multi-crop/multi-scale, loader nên cho ảnh CHƯA cắt (eval_transform_full(256)): 5-crop lấy
các crop 224 từ ảnh 256; multi-scale resize từ ảnh đầy đủ.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def eval_transform_full(size: int = 256, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Resize ảnh về size x size, KHÔNG crop (đầu vào cho TTA multi-crop / multi-scale / dò độ phân giải)."""
    import torchvision.transforms as T

    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])


def _forward(model, x, amp: bool):
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp and x.device.type == "cuda"):
        return model(x).float()


def predict_logits(model, loader, device, view=None, amp: bool = False):
    """Chạy model trên loader, giữ đúng thứ tự file. view: hàm biến đổi batch hoặc None."""
    names, ys, outs = [], [], []
    model.eval()
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            if view is not None:
                x = view(x)
            outs.append(_forward(model, x, amp).cpu().numpy())
            ys.append(np.asarray(y))
            names.extend(f)
    return names, np.concatenate(ys), np.concatenate(outs)


def predict_views(model, loader, device, views_fn, amp: bool = False):
    """Một lượt qua loader, mỗi batch sinh K view (views_fn(x) -> list); trả về list K mảng logit."""
    names, ys, outs = [], [], None
    model.eval()
    with torch.inference_mode():
        for x, y, f in loader:
            x = x.to(device, non_blocking=True)
            vs = views_fn(x)
            res = [_forward(model, v, amp).cpu().numpy() for v in vs]
            outs = [[r] for r in res] if outs is None else [o + [r] for o, r in zip(outs, res)]
            ys.append(np.asarray(y))
            names.extend(f)
    return names, np.concatenate(ys), [np.concatenate(o) for o in outs]


def view_identity(x):
    return x


def view_hflip(x):
    """Lật ngang batch (N, C, H, W): đảo chiều rộng."""
    return torch.flip(x, dims=[3])


def view_center_crop(x, crop: int = 224):
    h, w = x.shape[-2:]
    t, l = (h - crop) // 2, (w - crop) // 2
    return x[..., t:t + crop, l:l + crop]


def views_multicrop(x, crop: int = 224, flip: bool = False):
    """5 crop (4 góc + giữa) cỡ `crop` (+ bản lật nếu flip=True -> 10 view)."""
    h, w = x.shape[-2:]
    crops = [x[..., :crop, :crop], x[..., :crop, w - crop:], x[..., h - crop:, :crop],
             x[..., h - crop:, w - crop:], view_center_crop(x, crop)]
    if flip:
        crops += [view_hflip(c) for c in crops]
    return crops


def views_multiscale(x, sizes=(224, 256, 288)):
    """Resize batch về từng kích thước. CNN có global pooling nhận mọi cỡ; ViT/Swin cố định cỡ
    vị trí/cửa sổ nên KHÔNG dùng được trực tiếp (ghi rõ trong báo cáo)."""
    return [x if x.shape[-1] == s else F.interpolate(x, size=(s, s), mode="bilinear",
                                                       align_corners=False, antialias=True) for s in sizes]


def views_hflip_pair(x):
    return [x, view_hflip(x)]


def softmax_np(z):
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    """Gộp K view: "prob" = trung bình softmax; "logit" = trung bình logit rồi softmax."""
    arr = np.stack([np.asarray(l, dtype=np.float64) for l in logits_per_view])
    if space == "prob":
        p = softmax_np(arr).mean(0)
    elif space == "logit":
        p = softmax_np(arr.mean(0))
    else:
        raise ValueError("space phải là 'prob' hoặc 'logit'")
    return p / p.sum(1, keepdims=True)


def aggregate_logits(logits_per_view):
    """Logit đại diện của TTA (trung bình logit) để khớp temperature sau TTA."""
    return np.mean(np.stack(logits_per_view), 0)


def ensemble_probs(list_of_probs):
    """Trung bình xác suất của nhiều mô hình (cùng tập ảnh, cùng thứ tự file)."""
    arr = np.stack([np.asarray(p, dtype=np.float64) for p in list_of_probs])
    p = arr.mean(0)
    return p / p.sum(1, keepdims=True)


def fit_temperature(val_logits, val_labels, max_iter: int = 200) -> float:
    """T > 0 cực tiểu NLL trên VAL (LBFGS trên log T, khởi đầu sau lưới thô). KHÔNG khớp trên test."""
    z = torch.as_tensor(np.asarray(val_logits), dtype=torch.float64)
    y = torch.as_tensor(np.asarray(val_labels), dtype=torch.long)
    grid = np.exp(np.linspace(np.log(0.05), np.log(20), 120))
    nll = [F.cross_entropy(z / t, y).item() for t in grid]
    log_t = torch.tensor([np.log(grid[int(np.argmin(nll))])], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(z / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits, T: float):
    """softmax(logits / T)."""
    return softmax_np(np.asarray(logits, dtype=np.float64) / T)


def _fuse_pair(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """w' = gamma*w/sqrt(var+eps); b' = beta + gamma*(b-mean)/sqrt(var+eps)."""
    fused = copy.deepcopy(conv)
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64, device=w.device)
    gamma = bn.weight.detach().double() if bn.weight is not None else torch.ones_like(b)
    beta = bn.bias.detach().double() if bn.bias is not None else torch.zeros_like(b)
    scale = gamma / torch.sqrt(bn.running_var.double() + bn.eps)
    fused.weight = nn.Parameter((w * scale.view(-1, 1, 1, 1)).to(conv.weight.dtype))
    fused.bias = nn.Parameter((beta + (b - bn.running_mean.double()) * scale).to(conv.weight.dtype))
    return fused


def _bn_replacement(bn) -> nn.Module:
    """BN thường -> Identity. BatchNormAct2d của timm (BN + drop + act) -> giữ phần drop + act."""
    act = getattr(bn, "act", None)
    drop = getattr(bn, "drop", None)
    if act is None and drop is None:
        return nn.Identity()
    return nn.Sequential(drop if drop is not None else nn.Identity(), act if act is not None else nn.Identity())


def fuse_conv_bn(model, check_input=None, verbose: bool = True):
    """Gộp mọi cặp (Conv2d, BatchNorm2d) đăng ký liền nhau trong cùng module cha.

    Trả về bản sao đã gộp (model gốc không đổi). Nếu `check_input` được truyền, in sai số lớn nhất
    giữa đầu ra trước/sau gộp (kỳ vọng <= ~1e-5 ở FP32). Kiến trúc chỉ có LayerNorm (ViT/Swin/ConvNeXt)
    trả về 0 cặp gộp: không áp dụng.
    """
    model = model.eval()
    fused = copy.deepcopy(model).eval()
    n_fused = 0
    for parent in list(fused.modules()):
        names = list(parent._modules.keys())
        for a, b in zip(names[:-1], names[1:]):
            conv, bn = parent._modules[a], parent._modules[b]
            if (isinstance(conv, nn.Conv2d) and isinstance(bn, nn.modules.batchnorm._BatchNorm)
                    and bn.num_features == conv.out_channels and bn.track_running_stats):
                parent._modules[a] = _fuse_pair(conv, bn)
                parent._modules[b] = _bn_replacement(bn)
                n_fused += 1
    fused.n_fused = n_fused
    if check_input is not None:
        with torch.inference_mode():
            diff = (model(check_input).float() - fused(check_input).float()).abs().max().item()
        fused.fuse_max_abs_diff = diff
        if verbose:
            print(f"fuse_conv_bn: gộp {n_fused} cặp conv+BN | sai số lớn nhất = {diff:.2e}")
    elif verbose:
        print(f"fuse_conv_bn: gộp {n_fused} cặp conv+BN")
    return fused


def ece(probs, y, bins: int = 15) -> float:
    """ECE 15 bin; dùng đúng định nghĩa của eval.ece_score nếu import được."""
    try:
        from eval import ece_score
        return float(ece_score(np.asarray(probs), np.asarray(y), bins))
    except Exception:
        probs, y = np.asarray(probs), np.asarray(y)
        conf, pred = probs.max(1), probs.argmax(1)
        edges = np.linspace(0, 1, bins + 1)
        e = 0.0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.any():
                e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
        return float(e)
