# Đối chiếu kỹ thuật inpainting IIMT30K

Hai thư mục con lưu output pipeline. Mỗi bản có `summary.json`, `per_image_metrics.csv` và ảnh so sánh.

Các chỉ số như `pipeline_outside_mae_mean`, `fixed_context_psnr_mean`, mask IoU/precision/recall là metric chẩn đoán để so sánh biến thể pipeline hoặc residual fix. Chúng không cùng protocol với `score/summary.json` component, vì vậy không đưa vào bảng headline và không so sánh trực tiếp với SCUT/Total-Text/FLORES.
