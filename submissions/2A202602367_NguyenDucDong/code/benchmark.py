"""benchmark.py - đo độ trễ suy luận đúng cách.

Quy tắc: warmup >= 10 lần (bỏ), torch.cuda.synchronize() TRƯỚC và SAU đoạn đo, >= 50 lần đo,
báo cáo p50/p95/p99. Ghi GPU, dtype, batch, độ phân giải, gộp BN, phiên bản torch.
Đo KHÔNG tính tiền xử lý (decode JPEG/resize): đầu vào là tensor ngẫu nhiên đã nằm sẵn trên GPU.
"""
from __future__ import annotations

import time

import numpy as np


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo fn() (không tham số), trả về mili-giây."""
    sync = sync or (lambda: None)
    for _ in range(warmup):
        fn()
    sync()
    times = []
    for _ in range(iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1000.0)
    t = np.asarray(times)
    return {"p50": float(np.percentile(t, 50)), "p95": float(np.percentile(t, 95)),
            "p99": float(np.percentile(t, 99)), "mean": float(t.mean()), "std": float(t.std()),
            "n": iters, "warmup": warmup}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100, fused_bn: bool = False, views: int = 1,
                   channels_last: bool = False) -> dict:
    """Độ trễ forward với đầu vào ngẫu nhiên (batch, 3, img, img). dtype: fp32 | amp | fp16.

    views > 1: chạy K lượt forward liên tiếp (TTA K view), đo tổng thời gian.
    """
    import copy

    import torch

    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    m = copy.deepcopy(model).to(dev).eval()
    x = torch.randn(batch_size, 3, img_size, img_size, device=dev)
    if dtype == "fp16":
        m = m.half()
        x = x.half()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
        x = x.to(memory_format=torch.channels_last)
    use_amp = dtype == "amp" and dev.type == "cuda"
    sync = torch.cuda.synchronize if dev.type == "cuda" else None

    def fn():
        with torch.inference_mode(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
            for _ in range(views):
                m(x)

    r = bench(fn, warmup=warmup, iters=iters, sync=sync)
    r.update({
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
        "dtype": dtype, "batch": batch_size, "img_size": img_size, "views": views,
        "fused_bn": fused_bn, "images_per_s": batch_size / (r["p50"] / 1000.0),
        "torch": torch.__version__, "preprocessing": False, "channels_last": channels_last,
    })
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return r


def tta_latency(model, k_views: int, **kw) -> dict:
    """Độ trễ TTA K view (đo thật) và so với K * p50 của 1 view."""
    one = latency_report(model, views=1, **kw)
    k = latency_report(model, views=k_views, **kw)
    k["ratio_vs_1view"] = k["p50"] / one["p50"]
    k["expected_linear"] = k_views
    return k
