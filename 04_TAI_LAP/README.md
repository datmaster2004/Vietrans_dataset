# Tái lập và phương pháp đánh giá

`VieTrans_TestTools/` là bản sao công cụ/tài liệu đã dùng để chuẩn bị manifest và chấm component. Tài liệu nên đọc theo thứ tự:

1. `VieTrans_TestTools/README.md` - cấu trúc công cụ.
2. `VieTrans_TestTools/00_Doc/HUONG_DAN_DANH_GIA.md` - protocol, metric, điều kiện coverage và cách tách component khỏi end-to-end.
3. `VieTrans_TestTools/HUONG_DAN_CHAY_A100.md` - cách chạy/probe trên A100.

Để kiểm tra một kết quả đã công bố, dùng chính `selected_manifest.jsonl` và `predictions.jsonl` trong thư mục kết quả tương ứng, rồi chạy evaluator theo tài liệu. `summary.json` hiện hữu là bản evidence gốc: nó chứa command, Python/platform và SHA-256 manifest/prediction.

Không dùng smoke test 1 mẫu để thay cho đánh giá đầy đủ. IIMT30K được chấm riêng trên test split nội miền (tách từ tập gốc theo train/valid/test); tài liệu protocol yêu cầu tách OCR, translation, inpainting component và end-to-end, đồng thời yêu cầu coverage 100%/không lỗi trước khi công bố.

Lượt SCUT-EnsText hiện có 812/813 output. Để hoàn tất, dùng lại đúng manifest SCUT, khắc phục mask của `inp_scut_img_324`, sinh prediction còn thiếu và chạy chấm để tạo `summary.json` cùng `per_case.jsonl` cho 813/813.
