# VieTrans A100 — lệnh chạy copy–paste

> Mục tiêu: lấy **code Space** từ Hugging Face, chạy API nội bộ trên A100; dataset lấy từ Drive. Không upload dataset hoặc script test lên Space.

## 1. Tải và chạy code Space

```bash
# Tải source Space từ Hugging Face (làm một lần).
export APP=/opt/VieTrans
git clone https://huggingface.co/spaces/masterdzzzz/VieTrans-ModelSpace "$APP"

# Nếu repository dùng Git LFS thì lấy các file LFS (bỏ qua nếu không dùng).
git -C "$APP" lfs pull

# Tạo môi trường cho API và cài dependencies của Space.
python -m venv /opt/vietrans-app-venv
/opt/vietrans-app-venv/bin/pip install --upgrade pip
/opt/vietrans-app-venv/bin/pip install -r "$APP/requirements.txt"

# Khởi động API local trên port 7860.
# Nếu repo không dùng app.py, thay `app.py` bằng entrypoint ghi trong README của repo.
cd "$APP"
nohup env GRADIO_SERVER_NAME=127.0.0.1 GRADIO_SERVER_PORT=7860 \
  /opt/vietrans-app-venv/bin/python app.py \
  > /opt/vietrans-server.log 2>&1 &
```

API chạy đúng khi có: `/eval_ocr`, `/eval_inpaint`, `/eval_translate`. Nếu smoke test ở bước 3 báo thiếu endpoint, source Space đang clone chưa phải phiên bản có API đánh giá.

## 2. Chuẩn bị script test và dataset

```bash
# Chép thư mục VieTrans_TestTools từ máy local vào /opt/VieTrans_TestTools trước.
# Dataset đã tải từ Drive phải nằm đúng bốn thư mục dưới đây.
export TEST=/opt/VieTrans_TestTools
export DATA=/data/vietrans-data
export OUT=/data/vietrans-results/run_$(date +%Y%m%d_%H%M%S)

# $DATA/IIMT30K
# $DATA/OCR_TotalText
# $DATA/Inpainting_SCUT
# $DATA/Translation_FLORES

# Tạo môi trường chạy test.
python -m venv /opt/vietrans-test-venv
/opt/vietrans-test-venv/bin/pip install --upgrade pip
/opt/vietrans-test-venv/bin/pip install -r "$TEST/requirements.txt"

# Khai báo server và phiên bản code đang đánh giá.
export PYTHON=/opt/vietrans-test-venv/bin/python
export VIETRANS_SERVER_URL=http://127.0.0.1:7860
export VIETRANS_SERVER_ID=a100-local
export VIETRANS_BUILD_ID="$(git -C "$APP" rev-parse --short HEAD)"
export WORKERS=1
```

## 3. Smoke test trước

```bash
# Một mẫu cho mỗi bước: OCR, inpainting, dịch.
"$PYTHON" "$TEST/tools/run_iimt_smoke.py" \
  --dataset-root "$DATA" \
  --output-dir "$OUT/iimt_smoke" \
  --limit 1
```

## 4. Chạy đầy đủ và chấm điểm

```bash
# Hàm: chạy model rồi chấm điểm một manifest. Không có --limit = chạy toàn bộ bộ dữ liệu.
run_and_score () {
  local suite="$1"
  local name="$2"
  local manifest="$3"
  local root="$OUT/$suite/$name"

  "$PYTHON" "$TEST/tools/run_component_server.py" "$suite" \
    --server-url "$VIETRANS_SERVER_URL" \
    --server-id "$VIETRANS_SERVER_ID" \
    --build-id "$VIETRANS_BUILD_ID" \
    --manifest "$manifest" \
    --output-dir "$root/inference" \
    --workers "$WORKERS"

  "$PYTHON" "$TEST/tools/evaluate.py" "$suite" \
    --manifest "$root/inference/selected_manifest.jsonl" \
    --predictions "$root/inference/predictions.jsonl" \
    --output-dir "$root/score"
}

# OCR: IIMT30K + Total-Text.
run_and_score ocr iimt30k \
  "$DATA/IIMT30K/manifests/ocr_iimt30k_en_arial/manifest.jsonl"
run_and_score ocr totaltext \
  "$DATA/OCR_TotalText/manifests/ocr_totaltext/manifest.jsonl"

# Inpainting: IIMT30K + SCUT-EnsText.
run_and_score inpainting iimt30k \
  "$DATA/IIMT30K/manifests/inpainting_iimt30k_en_arial/manifest.jsonl"
run_and_score inpainting scut_enstext \
  "$DATA/Inpainting_SCUT/manifests/inpainting_scut_enstext/manifest.jsonl"

# Dịch: IIMT30K + FLORES-200.
run_and_score translation iimt30k \
  "$DATA/IIMT30K/manifests/translation_iimt30k_en_vi/manifest.jsonl"
run_and_score translation flores200 \
  "$DATA/Translation_FLORES/manifests/translation_flores200_en_vi/manifest.jsonl"
```

## 5. Lấy kết quả

```bash
# Đọc điểm tổng của tất cả test.
find "$OUT" -path '*/score/summary.json' -print -exec cat {} \;

# Sau khi xong chỉ upload $OUT lên Drive.
# prediction: <test>/inference/predictions.jsonl
# điểm tổng:  <test>/score/summary.json
# chi tiết:   <test>/score/per_case.jsonl
# ảnh xóa chữ: <test>/inference/artifacts/ (chỉ inpainting)
```
