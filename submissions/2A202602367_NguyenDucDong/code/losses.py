"""losses.py - các hàm loss và trộn mẫu (Mixup, CutMix).

Giao diện:
    build_criterion(kind, **kw)                 -> callable(logits, target) -> loss scalar
    class_weights(counts, beta)                 -> tensor trọng số lớp
    mix_batch(x, y, alpha, mode)                -> (x_mixed, (y_a, y_b, lam))
    mixed_loss(criterion, logits, targets)      -> loss scalar
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOSS_CHOICES = ("ce", "ls", "focal", "ce_weighted")


class LabelSmoothingCE(nn.Module):
    """CE với label smoothing, TỰ CÀI ĐẶT: q'(k) = (1-eps)*1[k==y] + eps/K.

    Kiểm tra trong tests_code.py: eps=0 trùng F.cross_entropy, và khớp
    nn.CrossEntropyLoss(label_smoothing=eps).
    """

    def __init__(self, smoothing: float = 0.1, weight=None):
        super().__init__()
        self.eps = float(smoothing)
        self.register_buffer("weight", None if weight is None else torch.as_tensor(weight, dtype=torch.float))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        k = logits.size(-1)
        q = torch.full_like(logp, self.eps / k)
        q.scatter_(1, target.view(-1, 1), 1.0 - self.eps + self.eps / k)
        if self.weight is None:
            return -(q * logp).sum(-1).mean()
        w = self.weight.to(logp.device)
        loss = -(q * logp * w.view(1, -1)).sum(-1)
        return loss.sum() / w[target].sum()


class FocalLoss(nn.Module):
    """FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t), trung bình theo batch.

    gamma=0 và alpha=None phải cho đúng cross-entropy (kiểm tra trong tests_code.py).
    """

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("alpha", None if alpha is None else torch.as_tensor(alpha, dtype=torch.float))

    def forward(self, logits, target):
        logp = F.log_softmax(logits.float(), dim=-1)
        logp_t = logp.gather(1, target.view(-1, 1)).squeeze(1)
        p_t = logp_t.exp()
        loss = -((1.0 - p_t).clamp_min(0) ** self.gamma) * logp_t
        if self.alpha is not None:
            loss = loss * self.alpha.to(loss.device)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0):
    """Trọng số lớp từ số ảnh mỗi lớp của tập TRAIN.

    beta=0: w_c ∝ 1/n_c, chuẩn hoá trung bình = 1.
    beta>0: class-balanced (Cui et al.): w_c = (1-beta)/(1-beta^n_c), chuẩn hoá tổng = K.
    """
    n = np.asarray(counts, dtype=np.float64)
    if beta and beta > 0:
        w = (1.0 - beta) / (1.0 - np.power(beta, n))
        w = w / w.sum() * len(n)
    else:
        w = 1.0 / n
        w = w / w.mean()
    return torch.tensor(w, dtype=torch.float)


def build_criterion(kind: str = "ce", smoothing: float = 0.1, gamma: float = 2.0,
                    alpha=None, weight=None):
    """kind: "ce" | "ls" | "focal" | "ce_weighted" (cần weight = class_weights(...))."""
    if kind == "ce":
        return nn.CrossEntropyLoss()
    if kind == "ls":
        return LabelSmoothingCE(smoothing)
    if kind == "focal":
        return FocalLoss(gamma, alpha)
    if kind == "ce_weighted":
        if weight is None:
            raise ValueError("ce_weighted cần `weight` (class_weights từ số liệu train)")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float))
    raise ValueError(f"loss={kind!r} không hợp lệ, chọn trong {LOSS_CHOICES}")


def rand_bbox(h: int, w: int, lam: float, rng: np.random.Generator | None = None):
    """Hộp CutMix có diện tích ~ (1-lam)*H*W, tâm ngẫu nhiên, bị cắt ở biên ảnh."""
    rng = rng or np.random
    cut = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut), int(w * cut)
    cy, cx = rng.randint(h), rng.randint(w)
    y1, y2 = np.clip(cy - ch // 2, 0, h), np.clip(cy + ch // 2, 0, h)
    x1, x2 = np.clip(cx - cw // 2, 0, w), np.clip(cx + cw // 2, 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def mix_batch(x, y, alpha: float = 1.0, mode: str = "cutmix"):
    """Trộn batch ảnh và nhãn; trả về (x_mix, (y_a, y_b, lam)).

    CutMix: lam được tính LẠI theo diện tích thật của hộp sau khi bị cắt ở biên.
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(x.size(0), device=x.device)
    if mode == "mixup":
        x_mix = lam * x + (1.0 - lam) * x[perm]
    elif mode == "cutmix":
        h, w = x.shape[-2], x.shape[-1]
        y1, y2, x1, x2 = rand_bbox(h, w, lam)
        x_mix = x.clone()
        x_mix[..., y1:y2, x1:x2] = x[perm][..., y1:y2, x1:x2]
        lam = 1.0 - ((y2 - y1) * (x2 - x1)) / float(h * w)
    else:
        raise ValueError(f"mode={mode!r} phải là 'mixup' hoặc 'cutmix'")
    return x_mix, (y, y[perm], lam)


def mixed_loss(criterion, logits, targets):
    """lam*crit(logits, y_a) + (1-lam)*crit(logits, y_b). Nhận cả nhãn thường (tensor)."""
    if isinstance(targets, tuple):
        y_a, y_b, lam = targets
        return lam * criterion(logits, y_a) + (1.0 - lam) * criterion(logits, y_b)
    return criterion(logits, targets)
