# Lab Day 2 — DeepWeeds · 2A202602367 · Nguyễn Đức Đông

Bài làm Lab Day 2 (Track 4): so sánh backbone, công thức huấn luyện và phương pháp suy luận trên DeepWeeds (fold 0).

## Chạy lại

- **Notebook Colab:** https://colab.research.google.com/drive/1RLjfbas_taFNdvxUwlCkJqVHuQFb4eB-
  (bản sao trong repo: [`code/lab_day2.ipynb`](code/lab_day2.ipynb))
- **Phần cứng / thư viện (in ở ô đầu notebook):** Colab Tesla T4, Python 3.13.15, PyTorch 2.11.0+cu130, timm 1.0.29.
- **Thứ tự chạy:** mở notebook trên Colab (GPU T4) → *Runtime › Run all*. Notebook tự:
  1. mount Google Drive, clone repo bài lab để dùng `eval.py` **gốc** (không sửa), thêm `code/` vào `sys.path`;
  2. tải `images.zip` (Zenodo trả 403 cho IP Colab nên dùng bản của tác giả trên Google Drive, **MD5 khớp `b7b30f96d466fba86016aa5a26606e0f`**) và 4 file nhãn fold 0 từ `github.com/AlexOlsen/DeepWeeds`;
  3. chạy test tự viết (`code/tests_code.py`) và test của repo;
  4. Bước 0 → 5 theo GUIDE. Mọi kết quả ghi vào Google Drive (`runs/`, `curves/`, `predictions/`, `eval_out/`, `results.xlsx`).
     Colab bị ngắt thì *Run all* lại: lần chạy đã có `summary.json` được bỏ qua.
- **Chạy một thí nghiệm từ dòng lệnh:**
  ```bash
  cd code
  python train.py --set exp_id=B01 backbone=resnet50 seed=0 images_dir=<thư mục ảnh> labels_dir=<thư mục nhãn>
  ```
- **Seed:** seed 0 cho Bước 1–2; seed 0, 1, 2 cho chung kết `F01` và mốc `T00` (Bước 4).
- **Test tự viết** (không cần GPU): `cd code && python -m unittest tests_code -v`
  (focal γ=0 ≡ CE, label smoothing, CutMix trộn ảnh+nhãn đúng diện tích, Mixup, param_groups, đóng băng giữ BN ở eval,
  gộp Conv+BN sai số < 1e-4, temperature scaling, lịch LR warmup+cosine, EMA, parse_overrides).

## Cấu trúc `code/`

| File | Nội dung |
|---|---|
| `dataset.py` | `load_split`, `check_split` (giao rỗng, hợp 17.509, đủ file), transform/augmentation (`basic`, `basic_vflip`, `color`, `trivial`, `randaug`), Dataset, DataLoader (sampler cân bằng, seed worker) |
| `model.py` | `build_model` qua timm (`scratch`/`frozen`/`finetune`), `freeze_backbone` + `set_train_mode` (BN đóng băng ở eval), `param_groups` 3 nhóm (head xác định bằng `get_classifier()`), đếm params/GMAC |
| `losses.py` | CE, label smoothing (tự cài), focal, CE có trọng số / class-balanced, Mixup/CutMix (lam theo diện tích thật) |
| `train.py` | Một hàm `run(Config)` cho mọi thí nghiệm: AMP, AdamW/SGD, warmup + cosine theo bước, EMA, chọn checkpoint theo macro-F1 val bằng `eval.compute_metrics`, lưu logit/predictions, vẽ đường cong, test chỉ khi `save_test_predictions=True` |
| `inference.py` | TTA lật / 5-crop / 10-crop / đa tỉ lệ, gộp xác suất vs logit, ensemble, temperature scaling, gộp Conv+BN |
| `benchmark.py` | Độ trễ p50/p95/p99: warmup 10 lần, `cuda.synchronize` trước/sau, ≥ 50 lần đo; FP32/AMP/FP16, batch 1 và 32 |
| `lab_steps.py` | Tổng hợp bảng Backbones/Training/Inference, chung kết (T khớp trên val, test một lần mỗi seed), ghi `results.xlsx` |
| `lab_day2.ipynb` | Notebook chạy toàn bộ Bước 0 → 5 |
| `tests_code.py` | Kiểm tra tự viết cho các phần dễ sai |

## Trạng thái

Đang huấn luyện trên Colab. `results.xlsx`, `report.md`, `curves/`, `predictions/` sẽ được bổ sung sau khi chạy xong
(mọi số liệu lấy từ log chạy thật, truy ngược được theo `exp_id`). Không commit dataset và checkpoint.
