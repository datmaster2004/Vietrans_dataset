# VieTrans Evaluation

Bộ công cụ độc lập để chuẩn bị và chấm ba thành phần:

- OCR: Total-Text test.
- Inpainting: SCUT-EnsText official test và OTR_easy.
- Dịch Anh–Việt: FLORES-200 devtest và MASSIVE test.

Bắt đầu tại [`00_Doc/HUONG_DAN_DANH_GIA.md`](00_Doc/HUONG_DAN_DANH_GIA.md).

Lệnh kiểm tra nhanh:

```powershell
Set-Location D:\VieTrans_Evaluation
.\bootstrap.ps1
.\.venv\Scripts\python.exe .\tools\validate_workspace.py
```

Không dùng IIMT30k_Vi làm final test vì dữ liệu đó đã tham gia train/tuning. Chạy và
chốt từng thành phần trước, sau đó mới đánh giá toàn pipeline.

## Chạy trên server nội bộ

Runner không phụ thuộc Hugging Face. Khởi động Space/API trên cùng server, rồi gọi
endpoint nội bộ qua `VIETRANS_SERVER_URL` (mặc định `http://127.0.0.1:7860`). Gắn
dataset tại một đường dẫn riêng và chạy smoke IIMT đa nền tảng:

```powershell
$env:VIETRANS_SERVER_URL = "http://127.0.0.1:7860"
$env:VIETRANS_SERVER_ID = "staging-vietrans"
$env:VIETRANS_BUILD_ID = "<GIT_COMMIT_OR_IMAGE_TAG>"
python .\tools\run_iimt_smoke.py --dataset-root D:\0_Dataset --limit 1
```

Lệnh trên gọi `/eval_ocr`, `/eval_inpaint`, `/eval_translate` trên server nội bộ và
chấm lại output bằng evaluator local. Dùng `--limit 0` chỉ khi muốn chạy đủ 1.500
mẫu mỗi component. Các cờ cũ `--space-url`, `--space-id` vẫn được nhận để tương
thích với deployment Hugging Face.
