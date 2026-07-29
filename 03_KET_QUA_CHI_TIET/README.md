# Kết quả chi tiết và cách đối chiếu

## Nơi xem từng loại kết quả

| Phần | Ý nghĩa |
|---|---|
| `01_KET_QUA_CHINH/01_OCR_TOTALTEXT/` | Run OCR 300 ảnh Total-Text |
| `01_KET_QUA_CHINH/02_DICH_FLORES200/` | Run dịch 1.012 câu FLORES-200 |
| `01_KET_QUA_CHINH/03_INPAINTING_SCUT_ENSTEXT_PARTIAL/` | Lượt chạy inpainting SCUT-EnsText: 812/813 output và điểm tổng hợp |
| `02_KET_QUA_BO_SUNG_IIMT30K/` | Ba run 1.500 mẫu/bộ trên test split IIMT30K, tách từ tập gốc theo train/valid/test |
| `03_SMOKE_TEST_1_MAU/` | Kiểm tra kỹ thuật 1 case/bộ, không dùng để kết luận chất lượng |
| `04_KET_QUA_KY_THUAT_LICH_SU/` | Hai lần đối chiếu pipeline IIMT30K, metric riêng |

Mỗi benchmark có hai thư mục giống nhau:

```text
inference/
  selected_manifest.jsonl  case thực sự được đưa vào run
  predictions.jsonl        prediction theo case_id
  raw_log.jsonl            trạng thái từng invocation
  inference_summary.json   coverage, endpoint và số lỗi
  server_probe.json        provenance endpoint (nếu có)
  artifacts/               ảnh input/output khi benchmark sinh artifact
score/
  summary.json             chỉ số tổng hợp, SHA-256, command và môi trường
  per_case.jsonl           chỉ số chi tiết từng case
```

## Quy trình kiểm tra một case

1. Chọn một `case_id` trong `score/per_case.jsonl`.
2. Tìm cùng `case_id` trong `inference/selected_manifest.jsonl` để xem input/reference.
3. Đối chiếu với `inference/predictions.jsonl` để xem output mô hình.
4. Với inpainting, mở file tương ứng trong `inference/artifacts/`; với OCR/dịch, đối chiếu text trực tiếp trong JSONL.
5. Kiểm tra `score/summary.json` để thấy SHA-256 của hai tệp đầu vào, coverage và command tạo điểm.

`per_case.jsonl` được để ở định dạng máy đọc nhằm không làm mất trường dữ liệu. Có thể mở bằng VS Code, Notepad++ hoặc dùng tìm kiếm `case_id` trong trình soạn thảo. Không chỉnh sửa các JSON/JSONL này nếu muốn giữ bằng chứng tái lập.

Riêng SCUT-EnsText có raw inference cho 812 case; `score_partial/BAO_CAO_TONG_HOP.md` và `metrics.json` ghi điểm tổng hợp. Chưa có `score/summary.json` cho đủ 813 mẫu; `raw_log.jsonl` ghi lỗi ở `inp_scut_img_324`.
