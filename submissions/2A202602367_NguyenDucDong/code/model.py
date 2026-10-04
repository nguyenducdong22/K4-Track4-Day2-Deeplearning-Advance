"""model.py - tạo backbone, đóng băng, nhóm tham số, đếm params/GMAC.

Giao diện:
    build_model(name, pretrained, num_classes, drop_rate, init, drop_path_rate) -> nn.Module
    freeze_backbone(model)                                        -> None
    set_train_mode(model)                                         -> None (giữ BN đóng băng ở eval)
    param_groups(model, lr_backbone, lr_head, weight_decay)       -> list[dict] cho optimizer
    count_params(model) -> float (triệu)     count_gmacs(model, img_size) -> float
    weight_tag(model) -> str
"""
from __future__ import annotations

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}
INIT_CHOICES = ("scratch", "frozen", "finetune")


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune", drop_path_rate: float = 0.0):
    """Tạo model 9 lớp qua timm; timm tự thay head mới khởi tạo ngẫu nhiên.

    init: "scratch" (không tiền huấn luyện) | "frozen" (đóng băng backbone) | "finetune" (train toàn bộ).
    """
    import timm

    if init not in INIT_CHOICES:
        raise ValueError(f"init={init!r} không hợp lệ, chọn trong {INIT_CHOICES}")
    kw = dict(pretrained=(pretrained and init != "scratch"), num_classes=num_classes, drop_rate=drop_rate)
    if drop_path_rate:
        kw["drop_path_rate"] = drop_path_rate
    model = timm.create_model(name, **kw)
    model.init_mode = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def weight_tag(model) -> str:
    """Tag trọng số timm đã tải (ghi vào results.xlsx), hoặc 'scratch'."""
    if getattr(model, "init_mode", "finetune") == "scratch":
        return "scratch (không tiền huấn luyện)"
    cfg = getattr(model, "pretrained_cfg", None) or {}
    arch = cfg.get("architecture", "?")
    tag = cfg.get("tag", "")
    return f"{arch}.{tag}" if tag else arch


def _head_param_ids(model) -> set[int]:
    head = model.get_classifier()
    if hasattr(head, "parameters"):
        return {id(p) for p in head.parameters()}
    return set()


def freeze_backbone(model) -> None:
    """requires_grad=False cho mọi tham số trừ head; BN của backbone chuyển sang eval.

    Train loop phải gọi set_train_mode(model) thay vì model.train() để BN đóng băng không bị đưa
    lại về train mode (GUIDE mục 3.2).
    """
    head_ids = _head_param_ids(model)
    for p in model.parameters():
        p.requires_grad = id(p) in head_ids
    model.frozen_backbone = True
    _bn_to_eval(model)


def _bn_to_eval(model) -> None:
    import torch.nn as nn

    head = model.get_classifier()
    head_modules = set(head.modules()) if hasattr(head, "modules") else set()
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and m not in head_modules:
            m.eval()


def set_train_mode(model) -> None:
    """model.train(), nhưng nếu backbone bị đóng băng thì giữ BatchNorm của backbone ở eval."""
    model.train()
    if getattr(model, "frozen_backbone", False):
        _bn_to_eval(model)


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    """3 nhóm như slide trang 52.

    Head được xác định CHÍNH XÁC bằng model.get_classifier() (không đoán theo tên "fc"/"head",
    vì cách đoán tên làm nhầm các lớp mlp.fc1/fc2 của ConvNeXt/Swin/ViT và conv_head của MobileNetV3
    thành head mới).
    """
    head_ids = _head_param_ids(model)
    decay, no_decay, head = [], [], []
    for _, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if id(p) in head_ids:
            head.append(p)
        elif p.ndim <= 1:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "lr": lr_backbone, "weight_decay": weight_decay, "name": "backbone_decay"},
        {"params": no_decay, "lr": lr_backbone, "weight_decay": 0.0, "name": "backbone_no_decay"},
        {"params": head, "lr": lr_head, "weight_decay": weight_decay, "name": "head"},
    ]
    return [g for g in groups if g["params"]]


def count_params(model) -> float:
    """Số tham số (triệu), đếm cả tham số đóng băng."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """GMAC cho 1 ảnh 3 x img_size x img_size.

    Công cụ: torch.utils.flop_counter.FlopCounterMode (đếm FLOPs = 2 x MAC cho matmul/conv), chia 2.
    Đếm trên CPU, FP32, model.eval(). Số có thể lệch vài % so với fvcore/ptflops.
    """
    import copy

    import torch
    from torch.utils.flop_counter import FlopCounterMode

    m = copy.deepcopy(model).cpu().float().eval()
    x = torch.zeros(1, 3, img_size, img_size)
    counter = FlopCounterMode(display=False)
    with counter, torch.no_grad():
        m(x)
    return counter.get_total_flops() / 2 / 1e9
