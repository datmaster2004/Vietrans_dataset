# SCUT-EnsText - kết quả lượt chạy 812/813

`inference/` giữ nguyên 812 output ảnh, metadata/artifact, prediction, selected manifest, log và server probe của lượt chạy ngày 27/07/2026. `score_partial/` chứa báo cáo cùng JSON chỉ số tổng hợp ngày 28/07/2026.

| Thuộc tính | Giá trị |
|---|---|
| Benchmark | SCUT-EnsText official test, inpainting real-world |
| Coverage | 812/813 (99,877%) |
| Mẫu thiếu | `inp_scut_img_324` |
| Nguyên nhân | `mask_image contains no selected pixels` |
| Full PSNR / SSIM | 34,67 / 0,963 |
| Inside PSNR / SSIM | 25,60 / 0,755 |
| Inside MAE | 11,76 |
| Error reduction trong mask | 66,42% |
| Boundary MAE / SSIM | 6,31 / 0,793 |

Chưa có `score/summary.json` và `per_case.jsonl` cho đủ 813 mẫu vì còn thiếu một output. Không điền giá trị cho ảnh thiếu và không lấy trung bình với IIMT30K. Khi hoàn tất `inp_scut_img_324`, cần chạy lại toàn bộ 813 case để có kết quả đầy đủ.
