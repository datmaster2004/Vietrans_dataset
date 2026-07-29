# Báo cáo tổng hợp đánh giá VieTrans

**Môi trường suy luận:** `a100-local`, build `ddaef3e`  
**Phạm vi:** OCR, dịch Anh–Việt và inpainting.

## 1. Kết luận điều hành

* Dịch trên FLORES-200 đã hoàn tất và là kết quả external hoàn chỉnh hiện có: corpus BLEU **39,18**, chrF++ **57,52**.
* IIMT30k được phân loại là **in-domain held-out test** theo quy trình chia train/valid/test trước khi huấn luyện do chủ dự án xác nhận. Ba kết quả IIMT30k vì thế có giá trị đánh giá nội miền, với điều kiện lưu lại manifest/seed chia dữ liệu và xác minh không rò rỉ dữ liệu giữa các split.
* Inpainting SCUT-EnsText đã được chấm lại từ 812 output PNG cùng dataset gốc: chất lượng tốt, nhưng chỉ đạt **812/813** nên là **partial score**, chưa phải kết quả final.
* OCR Total-Text không có điểm chất lượng do chỉ sinh được **176/300** output. Lỗi chính là timeout khi gọi `/eval_ocr`.
* Không cộng các component hoặc dataset khác loại thành một điểm duy nhất.

## 2. Tình trạng coverage

| Component | Dataset | Vai trò | Case | Output/score | Trạng thái |
|---|---|---|---:|---:|---|
| Translation | FLORES-200 en→vi | External | 1.012 | 1.012 / 1.012 | Hoàn tất |
| Translation | IIMT30k | In-domain held-out | 1.500 | 1.500 / 1.500 | Hoàn tất |
| OCR | IIMT30k | In-domain held-out | 1.500 | 1.500 / 1.500 | Hoàn tất |
| OCR | Total-Text | External | 300 | 176 / 300 | Không đủ coverage |
| Inpainting | IIMT30k | In-domain held-out | 1.500 | 1.500 / 1.500 | Hoàn tất |
| Inpainting | SCUT-EnsText | External real-world | 813 | 812 / 813 | Partial; thiếu 1 output |
| Smoke | IIMT smoke | Kiểm tra endpoint | 3 | 3 / 3 | Qua, không đại diện chất lượng |

## 3. Dịch Anh–Việt

| Dataset | Sentence BLEU | Corpus BLEU | Sentence chrF++ | Corpus chrF++ | Edit similarity | Exact match |
|---|---:|---:|---:|---:|---:|---:|
| FLORES-200 | 36,86 | **39,18** | 57,46 | **57,52** | 72,78 | 0,0% |
| IIMT30k | 31,72 | **34,00** | 52,08 | **52,21** | 71,05 | 1,4% |

### Đánh giá

* FLORES-200 có kết quả ổn định trên ba nhóm nội dung: wikibooks (chrF++ 57,34), wikinews (58,15) và wikivoyage (56,86).
* Exact match thấp không phải là lỗi độc lập: dịch có thể đúng nghĩa nhưng dùng cách diễn đạt khác câu tham chiếu. Nên bổ sung review thủ công theo ba tiêu chí: bảo toàn nghĩa, thuật ngữ và văn phong.
* IIMT30k thấp hơn FLORES-200 theo BLEU/chrF++, phù hợp với tính chất câu phụ đề ngắn, nhiều hội thoại và cách diễn đạt tham chiếu đa dạng.

## 4. OCR

### IIMT30k held-out test

| Metric | Giá trị |
|---|---:|
| CER | **3,41%** |
| WER | **5,13%** |
| OCR accuracy score | 97,47 |
| OCR exact match | 75,67% |
| Detection precision / recall / F1 | 71,56% / 71,87% / **71,66%** |
| Matched-region CER | 3,28% |
| Region recognition exact rate | 54,00% |
| Text spotting precision / recall / F1 | 53,93% / 54,00% / **53,96%** |

**Đánh giá:** nhận dạng ký tự tốt khi vùng chữ đã được ghép đúng, nhưng phát hiện/định vị vùng chữ là nút thắt chính. Điều này thể hiện qua CER thấp nhưng text-spotting F1 chỉ 53,96%.

### Total-Text external test

| Hạng mục | Giá trị |
|---|---:|
| Tổng ảnh | 300 |
| Output sinh thành công | 176 (58,67%) |
| Lỗi | 124 (41,33%) |
| Timeout sau 2 lần thử | 115 |
| Lỗi không có thông điệp gốc | 9 |
| Điểm OCR | Chưa có; không hợp lệ khi coverage chưa đủ |

115 lỗi có dạng `Space request /eval_ocr failed after 2 attempt(s): timed out`; có các cụm lỗi liên tiếp, ví dụ `img572–img606`. Chín lỗi còn lại xảy ra tại `img10`, `img95`, `img542`, `img1092`, `img1094`, `img1192`, `img1292`, `img1348`, `img1548`, nhưng log client không có traceback gốc.

## 5. Inpainting

| Metric | IIMT30k (1.500/1.500) | SCUT-EnsText (812/813) |
|---|---:|---:|
| Full PSNR | 20,09 | **34,67** |
| Full SSIM | 0,645 | **0,963** |
| Inside MAE (thấp hơn tốt hơn) | 27,14 | **11,76** |
| Inside PSNR | 18,16 | **25,60** |
| Inside SSIM | 0,407 | **0,755** |
| Inside similarity score | 89,36 | **95,39** |
| Error reduction trong mask | 32,91% | **66,42%** |
| Outside MAE (thấp hơn tốt hơn) | **0,015** | 0,053 |
| Outside preservation score | **99,994** | 99,979 |
| Boundary MAE (thấp hơn tốt hơn) | 20,14 | **6,31** |
| Boundary SSIM | 0,643 | **0,793** |

### Đánh giá

* IIMT30k: nền ngoài vùng xóa được giữ gần như nguyên vẹn, nhưng tái tạo trong vùng có chữ còn ở mức trung bình; đây là hướng cải thiện chính cho dữ liệu nội miền.
* SCUT-EnsText: 812 output được chấm trực tiếp bằng manifest/dataset gốc và artifact đã sinh. Kết quả cho thấy tái tạo vùng xóa và vùng biên tốt hơn đáng kể so với IIMT30k; tuy nhiên hai dataset khác miền và khác phân bố mask nên không dùng chênh lệch này để suy ra chất lượng tuyệt đối giữa các bộ.
* Mẫu thiếu là `inp_scut_img_324`: request không chạy vì `mask_image contains no selected pixels`. Cần tái tạo mask hợp lệ, chạy lại một ảnh này và chấm strict 813/813.

## 6. Smoke test

| Component | Coverage | Kết quả |
|---|---:|---|
| OCR | 1/1 | Toàn bộ metric OCR = 100% |
| Translation | 1/1 | BLEU 37,22; chrF++ 58,97; edit similarity 78,91 |
| Inpainting | 1/1 | Inside similarity 95,00; error reduction 78,42%; preservation ngoài mask 100,00 |

Smoke test chỉ xác nhận endpoint hoạt động và không thay thế benchmark nhiều mẫu.

## 7. Hiệu năng request quan sát được

| Dataset/component | Trung bình | Trung vị | P95 |
|---|---:|---:|---:|
| Translation FLORES-200 | 0,558 s | 0,547 s | 0,728 s |
| Translation IIMT30k | 0,238 s | 0,235 s | 0,308 s |
| OCR IIMT30k | 0,335 s | 0,333 s | 0,406 s |
| Inpainting IIMT30k | 4,683 s | 4,600 s | 6,216 s |
| Inpainting SCUT-EnsText | 0,652 s | 0,628 s | 0,991 s |
| OCR Total-Text | 47,187 s | 16,412 s | 186,083 s |

Các số liệu này là thời gian request end-to-end của lượt chạy, không phải benchmark latency của model thuần.

## 8. Provenance và giới hạn

* Probe xác nhận các endpoint component sẵn sàng; toàn bộ lượt chạy dùng server `a100-local`, build `ddaef3e`.
* Translation báo cáo model `masterdzzzz/mt-nllb-1p3b-en-vi`, revision `1ac5819590c0fb3691f1a105f23ad365bf6ff584`, beam size 5.
* Kết quả không có Space SHA công khai hoặc run-lock đầy đủ trong bundle được cung cấp; cần lưu các thông tin này cho một lần công bố tái lập hoàn toàn.
* Phân loại IIMT30k là held-out test dựa trên xác nhận về quy trình chia dữ liệu trước huấn luyện. Trước khi công bố, cần đính kèm seed/quy tắc split, hash manifest và kiểm tra trùng lặp ảnh/scene/câu nguồn giữa các split.

## 9. Việc cần làm để hoàn tất bộ đánh giá

1. Sửa mask rỗng của `inp_scut_img_324`, chạy lại đúng một ảnh và chấm SCUT strict 813/813.
2. Chạy lại 124 ảnh Total-Text thiếu bằng mức song song thấp hơn; lưu traceback server cho 9 lỗi không có message; chỉ chấm chính thức khi đạt 300/300.
3. Bổ sung benchmark external còn chưa chạy: MASSIVE cho dịch câu ngắn và OTR_easy cho inpainting.
4. Lưu run lock, commit/SHA triển khai, manifest hash và cấu hình suy luận cho lượt đánh giá công bố.
