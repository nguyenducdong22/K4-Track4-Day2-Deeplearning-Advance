"""Kiểm tra tự viết cho các phần dễ sai (RUBRIC mục H). Không cần GPU, không cần dữ liệu thật.

    cd code && python -m unittest tests_code -v
"""
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import inference as inf  # noqa: E402
import losses  # noqa: E402
import model as mdl  # noqa: E402
import train  # noqa: E402


class TestLosses(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.z = torch.randn(32, 9) * 3
        self.y = torch.randint(0, 9, (32,))

    def test_focal_gamma0_equals_ce(self):
        fl = losses.FocalLoss(gamma=0.0)(self.z, self.y)
        ce = F.cross_entropy(self.z, self.y)
        self.assertLess(abs(fl.item() - ce.item()), 1e-6)

    def test_focal_downweights_easy(self):
        self.assertLess(losses.FocalLoss(2.0)(self.z, self.y).item(), F.cross_entropy(self.z, self.y).item())

    def test_label_smoothing(self):
        self.assertLess(abs(losses.LabelSmoothingCE(0.0)(self.z, self.y).item()
                            - F.cross_entropy(self.z, self.y).item()), 1e-6)
        ref = nn.CrossEntropyLoss(label_smoothing=0.1)(self.z, self.y)
        self.assertLess(abs(losses.LabelSmoothingCE(0.1)(self.z, self.y).item() - ref.item()), 1e-5)

    def test_class_weights(self):
        counts = [675, 637, 618, 613, 637, 605, 644, 609, 5463]
        w = losses.class_weights(counts, 0.0)
        self.assertAlmostEqual(w.mean().item(), 1.0, places=5)
        self.assertLess(w[8].item(), w[0].item())
        wb = losses.class_weights(counts, 0.999)
        self.assertAlmostEqual(wb.sum().item(), 9.0, places=4)

    def test_cutmix_area_and_labels(self):
        np.random.seed(1)
        torch.manual_seed(1)
        x = torch.zeros(8, 3, 32, 32)
        for i in range(8):
            x[i] = i  # mỗi ảnh một giá trị hằng để đếm diện tích bị dán
        y = torch.arange(8)
        for _ in range(20):
            xm, (ya, yb, lam) = losses.mix_batch(x, y, 1.0, "cutmix")
            self.assertTrue(torch.equal(ya, y))
            # diện tích pixel giữ nguyên ảnh gốc = lam (tính theo hộp đã bị cắt ở biên)
            for i in range(8):
                if yb[i] == ya[i]:
                    continue
                kept = (xm[i, 0] == i).float().mean().item()
                self.assertAlmostEqual(kept, lam, places=6)
            perm_vals = {int(v) for v in yb}
            self.assertEqual(perm_vals, set(range(8)))

    def test_mixup_mixes_images_and_labels(self):
        x = torch.randn(4, 3, 8, 8)
        y = torch.arange(4)
        xm, (ya, yb, lam) = losses.mix_batch(x, y, 0.4, "mixup")
        idx = [int((y == v).nonzero()) for v in yb]
        self.assertTrue(torch.allclose(xm, lam * x + (1 - lam) * x[idx], atol=1e-6))
        z = torch.randn(4, 9)
        crit = nn.CrossEntropyLoss()
        self.assertAlmostEqual(losses.mixed_loss(crit, z, (ya, yb, lam)).item(),
                               (lam * crit(z, ya) + (1 - lam) * crit(z, yb)).item(), places=6)


class TestModel(unittest.TestCase):
    def test_param_groups_head_exact(self):
        for name in ("convnext_atto", "resnet18", "mobilenetv3_small_050"):
            m = mdl.build_model(name, pretrained=False, num_classes=9)
            groups = {g["name"]: g for g in mdl.param_groups(m, 1e-4, 1e-3, 0.05)}
            head_ids = {id(p) for p in m.get_classifier().parameters()}
            self.assertEqual({id(p) for p in groups["head"]["params"]}, head_ids, name)
            self.assertTrue(all(p.ndim <= 1 for p in groups["backbone_no_decay"]["params"]))
            self.assertEqual(groups["backbone_no_decay"]["weight_decay"], 0.0)
            n = sum(len(g["params"]) for g in groups.values())
            self.assertEqual(n, len(list(m.parameters())))

    def test_freeze_keeps_bn_eval(self):
        m = mdl.build_model("resnet18", pretrained=False, num_classes=9, init="frozen")
        mdl.set_train_mode(m)
        bns = [x for x in m.modules() if isinstance(x, nn.BatchNorm2d)]
        self.assertTrue(all(not b.training for b in bns))
        trainable = [p for p in m.parameters() if p.requires_grad]
        self.assertEqual(len(trainable), len(list(m.get_classifier().parameters())))
        groups = mdl.param_groups(m, 1e-4, 1e-3, 0.05)
        self.assertEqual([g["name"] for g in groups], ["head"])

    def test_initial_loss_close_to_ln9(self):
        torch.manual_seed(0)
        m = mdl.build_model("resnet18", pretrained=False, num_classes=9).eval()
        with torch.no_grad():
            loss = F.cross_entropy(m(torch.randn(16, 3, 64, 64)), torch.randint(0, 9, (16,)))
        self.assertLess(abs(loss.item() - math.log(9)), 0.5)


class TestInference(unittest.TestCase):
    def test_fuse_conv_bn_resnet(self):
        torch.manual_seed(0)
        m = mdl.build_model("resnet18", pretrained=False, num_classes=9)
        # cho BN thống kê khác mặc định
        m.train()
        with torch.no_grad():
            for _ in range(3):
                m(torch.randn(8, 3, 64, 64))
        m.eval()
        x = torch.randn(2, 3, 64, 64)
        fused = inf.fuse_conv_bn(m, check_input=x, verbose=False)
        self.assertGreater(fused.n_fused, 10)
        self.assertLess(fused.fuse_max_abs_diff, 1e-4)
        self.assertFalse(any(isinstance(z, nn.BatchNorm2d) for z in fused.modules()))

    def test_fuse_conv_bn_act_timm(self):
        torch.manual_seed(0)
        m = mdl.build_model("mobilenetv3_small_050", pretrained=False, num_classes=9)
        m.train()
        with torch.no_grad():
            m(torch.randn(8, 3, 64, 64))
        m.eval()
        x = torch.randn(2, 3, 64, 64)
        fused = inf.fuse_conv_bn(m, check_input=x, verbose=False)
        self.assertGreater(fused.n_fused, 5)
        self.assertLess(fused.fuse_max_abs_diff, 1e-4)

    def test_temperature_recovers_T(self):
        rng = np.random.default_rng(0)
        true = rng.normal(size=(4000, 9)) * 2
        y = np.array([rng.choice(9, p=p) for p in inf.softmax_np(true)])
        T = inf.fit_temperature(true * 3.0, y)  # logit quá tự tin gấp 3 -> T ~ 3
        self.assertLess(abs(T - 3.0), 0.3)
        p = inf.apply_temperature(true * 3.0, T)
        np.testing.assert_array_equal(p.argmax(1), (true * 3).argmax(1))

    def test_aggregate_and_views(self):
        x = torch.arange(2 * 3 * 256 * 256, dtype=torch.float).view(2, 3, 256, 256)
        crops = inf.views_multicrop(x, 224)
        self.assertEqual(len(crops), 5)
        self.assertTrue(all(c.shape[-1] == 224 for c in crops))
        self.assertTrue(torch.equal(inf.view_hflip(inf.view_hflip(x)), x))
        l = [np.random.randn(5, 9) for _ in range(3)]
        for s in ("prob", "logit"):
            np.testing.assert_allclose(inf.aggregate_views(l, s).sum(1), 1, atol=1e-9)


class TestTrainHelpers(unittest.TestCase):
    def test_parse_overrides(self):
        d = train.parse_overrides(["seed=1", "loss=focal", "ema_decay=none", "lr_head=0.002",
                                   "amp=false", "sampler=balanced", "max_steps_per_epoch=3"])
        self.assertEqual(d, {"seed": 1, "loss": "focal", "ema_decay": None, "lr_head": 0.002,
                             "amp": False, "sampler": "balanced", "max_steps_per_epoch": 3})
        with self.assertRaises(KeyError):
            train.parse_overrides(["khong_co=1"])

    def test_lr_schedule_shape(self):
        total, warm = 100, 10
        f = [train.lr_factor(s, total, warm) for s in range(total)]
        self.assertAlmostEqual(f[warm - 1], 1.0)
        self.assertTrue(all(a <= b for a, b in zip(f[:warm - 1], f[1:warm])))
        self.assertTrue(all(a >= b for a, b in zip(f[warm:-1], f[warm + 1:])))
        self.assertLess(f[-1], 0.01)

    def test_ema_tracks(self):
        m = nn.Linear(3, 2)
        e = train.EMA(m, 0.9)
        with torch.no_grad():
            m.weight.add_(1.0)
        for _ in range(200):
            e.update(m)
        self.assertTrue(torch.allclose(e.module.weight, m.weight, atol=1e-4))


if __name__ == "__main__":
    unittest.main()
