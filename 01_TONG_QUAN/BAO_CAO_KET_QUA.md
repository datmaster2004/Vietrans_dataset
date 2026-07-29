# Báo cáo kết quả VieTrans

Số liệu được lấy từ các tệp kết quả của từng lượt chạy; các tệp gốc được giữ nguyên.

## 1. Kết quả chính

| Thành phần | Benchmark | Mẫu chấm | Coverage | Lỗi | Chỉ số chính |
|---|---|---:|---:|---:|---|
| OCR | Total-Text official test | 300 | 100% | 0 | OCR accuracy 40,97; Detection F1 38,76; Text-spotting F1 27,91 |
| Dịch Anh - Việt | FLORES-200 `eng_Latn -> vie_Latn` devtest | 1.012 | 100% | 0 | Corpus BLEU 39,18; corpus chrF 57,54; corpus chrF++ 57,52 |

### OCR - Total-Text

| Chỉ số | Giá trị |
|---|---:|
| OCR CER | 0,6505 |
| OCR WER | 1,0127 |
| OCR accuracy score | 40,97 |
| OCR exact match | 4,33% |
| Detection precision / recall / F1 | 47,34 / 35,89 / 38,76 |
| Matched-region CER | 0,1640 |
| Region recognition exact rate | 53,76% |
| Text-spotting precision / recall / F1 | 33,76 / 26,01 / 27,91 |

Phân rã theo loại ảnh: `curved` 73 mẫu (Detection F1 29,29; Text-spotting F1 19,73), `mixed_orientation` 226 mẫu (41,81; 30,51), và `multi_oriented` 1 mẫu. Xem toàn bộ tại `03_KET_QUA_CHI_TIET/01_KET_QUA_CHINH/01_OCR_TOTALTEXT/score/`.

### Dịch - FLORES-200

| Chỉ số | Giá trị |
|---|---:|
| Sentence BLEU trung bình | 36,86 |
| Sentence chrF trung bình | 57,46 |
| Sentence chrF++ trung bình | 57,46 |
| Edit similarity trung bình | 72,78 |
| Corpus BLEU | 39,18 |
| Corpus chrF | 57,54 |
| Corpus chrF++ | 57,52 |

Phân rã theo miền: `wikibooks` 351 câu, `wikinews` 341 câu và `wikivoyage` 320 câu. SacreBLEU signatures được lưu nguyên trong `score/summary.json`.

## 2. Kết quả IIMT30K

IIMT30K trong gói là test split nội miền, được tách từ tập gốc theo train/valid/test. Ba dataset card trong `02_DU_LIEU_SU_DUNG/01_IIMT30K/manifests/` ghi `role: in_domain_held_out_test` và `split: test`.

| Thành phần | Mẫu | Chỉ số đáng chú ý |
|---|---:|---|
| OCR IIMT30K | 1.500 | Accuracy 97,47; Detection F1 71,66; Text-spotting F1 53,96 |
| Inpainting IIMT30K | 1.500 | Inside PSNR 18,16; inside SSIM 0,407; outside MAE 0,0146; outside preservation 99,994 |
| Dịch IIMT30K | 1.500 | Corpus BLEU 34,00; corpus chrF++ 52,21 |

Cả ba lượt chạy đều đạt coverage 100%, không có prediction thừa và không có lỗi. Thư mục `03_SMOKE_TEST_1_MAU/` chỉ dùng để kiểm tra đường chạy cho 1 mẫu/mỗi thành phần.

## 3. Inpainting SCUT-EnsText

Báo cáo tổng hợp nằm ở `01_TONG_QUAN/02_KET_QUA_GHI_NHAN/`. Ảnh đầu vào, ảnh sạch, mask và artifact đầu ra nằm trong `03_KET_QUA_CHI_TIET/01_KET_QUA_CHINH/03_INPAINTING_SCUT_ENSTEXT_PARTIAL/`.

| Benchmark | Coverage | Trạng thái | Full PSNR | Full SSIM | Inside PSNR | Inside SSIM | Boundary SSIM |
|---|---:|---|---:|---:|---:|---:|---:|
| SCUT-EnsText official test | 812/813 (99,877%) | Còn 1 ảnh chưa chạy xong | 34,67 | 0,963 | 25,60 | 0,755 | 0,793 |

Điểm trên được tổng hợp từ dataset gốc và 812 output PNG. Thư mục `inference` lưu `selected_manifest.jsonl`, `predictions.jsonl`, `raw_log.jsonl`, `inference_summary.json` và 812 artifact. Chưa có `score/summary.json` và `per_case.jsonl` cho đủ 813 mẫu. Ảnh còn lại, `inp_scut_img_324`, dừng vì `mask_image contains no selected pixels`.

Thư mục `04_KET_QUA_KY_THUAT_LICH_SU/` giữ thêm hai lần đối chiếu pipeline inpainting IIMT30K, gồm summary, CSV theo ảnh và ảnh so sánh. Các chỉ số ở đây dùng để so sánh biến thể pipeline, không trộn với kết quả component ở trên.

## 4. Phạm vi và giới hạn cần công khai

- OCR Total-Text và dịch FLORES-200 đã chạy đủ mẫu. SCUT-EnsText còn một ảnh chưa chạy xong; không dùng IIMT30K hoặc smoke test để thay thế ảnh này.
- Báo cáo ghi nhận còn lưu một lượt OCR Total-Text cũ chỉ đạt 176/300. Lượt chạy Total-Text 300/300 trong gói chính là nguồn điểm OCR hiện hành; báo cáo cũ được giữ để đối chiếu lịch sử.
- IIMT30K là test split nội miền từ cách chia train/valid/test của tập gốc. Dataset card, manifest và báo cáo tổng hợp trong gói dùng cùng cách phân loại này.
- Các metric OCR ở đây được tính bởi evaluator kèm gói (polygon IoU 0,5 theo tài liệu). Không tự so sánh trực tiếp với bảng xếp hạng Total-Text nếu protocol khác.
- `server_probe.json` của đợt Total-Text ghi `server_id: a100-local`, `server_build_id: ddaef3e`, `space_id: a100-local`, `space_sha: null`. Vì thiếu SHA của Space remote, diễn giải đúng là đánh giá thành phần trên A100 local, không phải chứng cứ deploy immutable.

## 5. Thông tin tái lập đã ghi nhận

`summary.json` của mỗi run lưu sẵn: SHA-256 của manifest, SHA-256 của prediction, command chấm, phiên bản Python và platform. `inference_summary.json` lưu coverage/endpoint; `server_probe.json` ở run Total-Text lưu provenance thành phần:

- OCR: PP-OCRv5, ngưỡng confidence 0,5.
- Translation: `masterdzzzz/mt-nllb-1p3b-en-vi`, revision `1ac5819590c0fb3691f1a105f23ad365bf6ff584`, 5 beams, tối đa 384 token mới.
- Inpainting: `inference._inpaint_text`, chế độ background phẳng và Telea production.

Mở `04_TAI_LAP/README.md` để xem công cụ và trình tự kiểm tra. Các dòng JSONL và artifact được giữ nguyên để hội đồng có thể lấy mẫu bất kỳ, đối chiếu `case_id` xuyên suốt manifest - prediction - per-case score.
