# Hướng dẫn đánh giá VieTrans

## 1. Kết luận và phạm vi

Bộ đánh giá này được tổ chức để trả lời hai câu hỏi khác nhau:

1. Từng thành phần OCR, inpainting và dịch tốt đến đâu?
2. Khi ghép ba thành phần, lỗi tổng thể xuất phát từ đâu?

Phải chấm riêng từng thành phần trước. Điểm end-to-end chỉ chạy sau khi model, code
và tham số của ba thành phần đã khóa; nó không thay thế các điểm thành phần.

Các bộ được phép dùng:

| Thành phần | Dataset | Số mẫu | Vai trò |
|---|---|---:|---|
| OCR | Total-Text official test | 300 ảnh, khoảng 2.204 vùng | External benchmark phụ |
| Inpainting | SCUT-EnsText official test | 813 cặp | External real-world paired chính |
| Inpainting | OTR_easy | 5.538 cặp | External paired bổ sung |
| Dịch | FLORES-200 `eng_Latn→vie_Latn` devtest | 1.012 câu | External MT chính |
| Dịch | MASSIVE `en-US→vi-VN` test | 2.974 câu | Bổ sung cho câu lệnh ngắn |

IIMT30k_Vi không được dùng làm bằng chứng final vì đã tham gia train hoặc tuning.
Nếu muốn báo cáo khả năng trong miền VieTrans, phải tạo một holdout mới, không
trùng ảnh/scene/source text với train và không được xem trước khi khóa hệ thống.

Total-Text là dữ liệu công khai và khác miền sản phẩm. Có thể báo cáo, nhưng không
nên là bằng chứng OCR duy nhất. Cần thêm một holdout ảnh đúng miền VieTrans với
polygon và transcription được người kiểm duyệt.

## 2. Trạng thái dữ liệu đã kiểm kê

- `01_OCR/ToTalText/totaltext.zip`: đúng bản 432.596.071 byte, chứa 300 ảnh test.
- `01_OCR/ToTalText/total-text-groundtruth-text.zip`: nhãn polygon/text chính thức.
- `02_Inpainting/SCUT_EnsText/test_set/test`: 813 `all_images`, 813 `all_labels`
  nền sạch và 813 file polygon; ID khớp nhau.
- `02_Inpainting/OTR_easy/data`: đủ 12 shard Parquet, tổng 5.538 hàng.
- `03_Trans/FLORES200`: archive gốc, checksum đã biết.
- `03_Trans/MASSIVE_1.0`: archive gốc, checksum đã biết.
- `02_Inpainting/IIMT30K`: đang trống và không nằm trong final-test protocol.

Kết quả kiểm tra máy đọc được nằm tại `00_Doc/dataset_inventory.json` sau khi chạy
validator.

## 3. Cài môi trường

Mở PowerShell:

```powershell
Set-Location D:\VieTrans_Evaluation
Set-ExecutionPolicy -Scope Process Bypass
.\bootstrap.ps1
```

Mọi dependency được cài trong `D:\VieTrans_Evaluation\.venv`, không dùng Python hệ
thống để chấm chính thức.

Kiểm tra file, số lượng, schema và checksum:

```powershell
.\.venv\Scripts\python.exe .\tools\validate_workspace.py
```

Lần đầu và trước khi công bố kết quả, chạy kiểm tra sâu 12 shard OTR:

```powershell
.\.venv\Scripts\python.exe .\tools\validate_workspace.py --deep-hash
```

Validator trả exit code khác 0 nếu có lỗi dữ liệu. Warning “manifest chưa được
prepare” không phải hỏng dữ liệu; thực hiện bước tiếp theo rồi chạy lại.

## 4. Chuẩn bị manifest

### 4.1 OCR

```powershell
.\.venv\Scripts\python.exe .\tools\prepare_benchmarks.py --suite ocr
```

Đầu ra:

```text
prepared/ocr_totaltext/Images/Test/
prepared/ocr_totaltext/Groundtruth/Test/
manifests/ocr_totaltext/manifest.jsonl
manifests/ocr_totaltext/dataset_card.json
```

### 4.2 Inpainting SCUT-EnsText

```powershell
.\.venv\Scripts\python.exe .\tools\prepare_benchmarks.py --suite scut
```

Mask là hợp của polygon chính thức trong `all_gts`, không suy ra từ output model.
`all_images` là input có chữ; `all_labels` là ground-truth nền sạch.

### 4.3 Inpainting OTR_easy

Thử luồng với 10 mẫu; manifest smoke được ghi riêng, không thể nhầm với bộ full:

```powershell
.\.venv\Scripts\python.exe .\tools\prepare_benchmarks.py --suite otr --limit-otr 10
```

Sau khi smoke chạy đúng, materialize đủ 5.538 mẫu:

```powershell
.\.venv\Scripts\python.exe .\tools\prepare_benchmarks.py --suite otr
```

OTR lưu input, clean target và mask ở `prepared/inpainting_otr_easy`. Mask là hợp
các `word_bboxes` chính thức sau khi chuẩn hóa thứ tự tọa độ.

### 4.4 Dịch

```powershell
.\.venv\Scripts\python.exe .\tools\prepare_benchmarks.py --suite translation
```

Script chỉ giải nén các file `eng_Latn`, `vie_Latn`, metadata của FLORES và hai
locale `en-US`, `vi-VN` của MASSIVE. Alignment MASSIVE theo stable ID và chỉ lấy
official test partition.

Có thể chạy tất cả bằng:

```powershell
.\.venv\Scripts\python.exe .\tools\prepare_benchmarks.py --suite all
```

Lưu ý: lệnh này materialize toàn bộ OTR nên tốn thêm dung lượng và thời gian.

## 5. Khóa hệ thống trước final test

Sao chép và điền đầy đủ:

```powershell
Copy-Item .\config\model_run.example.json .\config\model_run.json
notepad .\config\model_run.json
```

Không còn `FILL_ME`, sau đó tạo run lock:

```powershell
.\.venv\Scripts\python.exe .\tools\freeze_run.py `
  --run-id vietrans_v1 `
  --model-id "masterdzzzz/mt-nllb-1p3b-en-vi" `
  --model-revision "1ac5819590c0fb3691f1a105f23ad365bf6ff584" `
  --code-commit "<GIT_COMMIT>" `
  --config .\config\model_run.json
```

Thay model ID/revision bằng model thực tế nếu khác. `run_lock.json` ghi hash config,
mọi manifest, Python và toàn bộ package version. Nếu thay checkpoint, threshold,
beam, prompt, mask dilation hoặc post-processing sau khi xem điểm final test, lần
chạy đó không còn là final độc lập.

## 6. Sinh prediction trực tiếp từ đúng Hugging Face Space

### 6.1 Nguyên tắc bắt buộc

Evaluator local chỉ chấm file prediction; nó không tự chứng minh prediction đến từ model nào.
Vì vậy, kết quả chính thức phải được sinh bởi `run_server_component.ps1`, gọi thẳng Space
đang dùng trong sản phẩm. Mỗi lần chạy sẽ lưu:

- URL và ID của Space;
- commit SHA của Space lấy từ Hugging Face Hub;
- endpoint đã gọi;
- manifest con đúng các mẫu đã gửi;
- prediction, artifact ảnh, thời gian chạy và log lỗi;
- coverage trong `inference_summary.json`.

Không được thay prediction bằng output oracle, model local khác, Google Translate hoặc
checkpoint tải về không trùng revision đang triển khai.

Probe server trước mọi đợt đánh giá:

```powershell
.\run_server_component.ps1 `
  -Mode probe `
  -OutputDir .\results\vietrans_v1\server_probe
```

Chỉ chạy final component test khi `server_probe.json` có:

```json
{
  "component_endpoints_ready": true,
  "deployed_pipeline_endpoint_ready": true
}
```

### 6.2 Trạng thái Space đã xác minh ngày 23/07/2026

Space `masterdzzzz/VieTrans-ModelSpace` tại thời điểm kiểm tra đang ở commit:

```text
aad1105866294852b9b95e48df9b82c6d28d24e4
```

Các endpoint live khi kiểm tra là:

```text
/lambda
/translate
/translate_no_qa
/translate_stream
```

Do chưa có `/eval_ocr`, `/eval_inpaint` và `/eval_translate`, bản đang live chỉ cho phép:

- lấy OCR thật của pipeline từ debug `/translate_no_qa`;
- lấy ảnh inpaint thật nhưng mask/vùng xóa do OCR của pipeline quyết định;
- chạy toàn pipeline ảnh.

Nó chưa thể chấm độc lập translation trên FLORES/MASSIVE và chưa thể chấm độc lập
inpainting bằng ground-truth mask. Không được gọi hai kết quả phụ thuộc pipeline này là
“điểm component độc lập”.

Mã local `Space/app.py` đã chuẩn bị bốn endpoint ẩn:

- `/eval_ocr`: gọi đúng OCR production;
- `/eval_inpaint`: gọi đúng inpainting production với mask chuẩn của benchmark;
- `/eval_translate`: gọi đúng đường dịch NLLB production trên source text benchmark;
- `/eval_info`: trả provenance/config của các thành phần.

Các endpoint chỉ có hiệu lực sau khi phiên bản `Space/app.py` này được commit và deploy lên
đúng Space. Sau deploy phải probe lại; không suy đoán endpoint đã live từ mã local.

### 6.3 Bật endpoint component trên Space

Snapshot commit live đã được tải về và đối chiếu. Để vẫn đánh giá đúng pipeline hiện tại,
chỉ upload `Space/app.py`, không upload toàn bộ thư mục `Space`: local
`Space/inference.py` và `Space/vietrans_space_inference/model_loader.py` đang có thay đổi
pin revision chưa nằm trong commit live. Upload cả thư mục sẽ làm thay đổi hệ thống cần
đánh giá và có thể đưa cả `__pycache__` lên repository.

Trong thư mục source VieTrans:

```powershell
Set-Location <THU_MUC_SOURCE_VIETRANS>
python -m py_compile .\Space\app.py
git diff -- .\Space\app.py

# Đăng nhập một lần; nhập token ở prompt, không đặt token trên command line.
D:\VieTrans_Evaluation\.venv\Scripts\hf.exe auth login

# Chỉ thêm API đánh giá, giữ nguyên inference.py/model_loader.py đang live.
D:\VieTrans_Evaluation\.venv\Scripts\hf.exe upload `
  masterdzzzz/VieTrans-ModelSpace `
  .\Space\app.py `
  app.py `
  --repo-type space `
  --commit-message "Add reproducible component evaluation endpoints"
```

Không ghi token vào tài liệu, command history hoặc file kết quả. Không dùng
`upload_space.py` cho lần bật endpoint này vì script đó upload toàn bộ thư mục.

Source live hiện chưa pin model revision. Tại thời điểm xác minh, HEAD của
`masterdzzzz/mt-nllb-1p3b-en-vi` là
`1ac5819590c0fb3691f1a105f23ad365bf6ff584`. `/eval_info` sẽ đọc revision thực sự
được Transformers load và ghi nó vào provenance; không chỉ dựa vào giá trị mặc định local.

Upload sẽ làm Space rebuild. Khi Space chạy lại, quay về thư mục đánh giá và probe:

```powershell
Set-Location D:\VieTrans_Evaluation
.\run_server_component.ps1 `
  -Mode probe `
  -OutputDir .\results\vietrans_v1\server_probe_after_deploy
```

Chỉ tiếp tục khi danh sách endpoint có đủ `/eval_ocr`, `/eval_inpaint`,
`/eval_translate`, `/eval_info` và `component_endpoints_ready` là `true`. Ghi lại
`space_sha` mới; SHA này phải đi cùng toàn bộ báo cáo của lần chạy.

### 6.4 Smoke test với bản đang live

Hai chế độ sau dùng chính `/translate_no_qa`, thích hợp để kiểm tra kết nối và chẩn đoán
stage trong pipeline hiện tại:

```powershell
# OCR do pipeline live sinh
.\run_server_component.ps1 `
  -Mode deployed-ocr `
  -Manifest .\manifests\ocr_totaltext\manifest.jsonl `
  -OutputDir .\results\vietrans_v1\deployed_ocr_smoke `
  -Limit 1 `
  -Workers 1

# Inpainting do pipeline live sinh, phụ thuộc OCR/mask của pipeline
.\run_server_component.ps1 `
  -Mode deployed-inpainting `
  -Manifest .\manifests\inpainting_otr_easy_smoke\manifest.jsonl `
  -OutputDir .\results\vietrans_v1\deployed_inpainting_smoke `
  -Limit 1 `
  -Workers 1
```

Chấm chính output vừa sinh, không dùng manifest/prediction khác:

```powershell
.\run_component.ps1 `
  -Suite ocr `
  -Manifest .\results\vietrans_v1\deployed_ocr_smoke\selected_manifest.jsonl `
  -Predictions .\results\vietrans_v1\deployed_ocr_smoke\predictions.jsonl `
  -OutputDir .\results\vietrans_v1\deployed_ocr_smoke\score

.\run_component.ps1 `
  -Suite inpainting `
  -Manifest .\results\vietrans_v1\deployed_inpainting_smoke\selected_manifest.jsonl `
  -Predictions .\results\vietrans_v1\deployed_inpainting_smoke\predictions.jsonl `
  -OutputDir .\results\vietrans_v1\deployed_inpainting_smoke\score
```

`-Limit 1` chỉ kiểm tra đường chạy. Không dùng điểm một mẫu làm kết luận chất lượng.

### 6.5 Final test từng thành phần sau khi deploy endpoint đánh giá

OCR production trên toàn bộ Total-Text:

```powershell
.\run_server_component.ps1 `
  -Mode ocr `
  -Manifest .\manifests\ocr_totaltext\manifest.jsonl `
  -OutputDir .\results\vietrans_v1\ocr_totaltext\inference `
  -Workers 3
```

Inpainting production với ground-truth mask:

```powershell
.\run_server_component.ps1 `
  -Mode inpainting `
  -Manifest .\manifests\inpainting_scut_enstext\manifest.jsonl `
  -OutputDir .\results\vietrans_v1\inpainting_scut\inference `
  -Workers 3

.\run_server_component.ps1 `
  -Mode inpainting `
  -Manifest .\manifests\inpainting_otr_easy\manifest.jsonl `
  -OutputDir .\results\vietrans_v1\inpainting_otr_easy\inference `
  -Workers 3
```

Translation production trên source text official:

```powershell
.\run_server_component.ps1 `
  -Mode translation `
  -Manifest .\manifests\translation_flores200_en_vi\manifest.jsonl `
  -OutputDir .\results\vietrans_v1\flores200\inference

.\run_server_component.ps1 `
  -Mode translation `
  -Manifest .\manifests\translation_massive_en_vi\manifest.jsonl `
  -OutputDir .\results\vietrans_v1\massive\inference
```

Nếu bị ngắt giữa chừng, chạy lại đúng lệnh và thêm `-Resume`. Chỉ chấm khi
`inference_summary.json` có coverage `1.0` và log không có lỗi chưa được chạy lại.

Ở bước chấm trong mục 8, dùng:

```text
Manifest    = <OutputDir inference>\selected_manifest.jsonl
Predictions = <OutputDir inference>\predictions.jsonl
```

### 6.6 Phân biệt hai loại kết quả

| Loại | Đầu vào stage | Dùng để kết luận |
|---|---|---|
| Component OCR | ảnh benchmark | chất lượng OCR đang triển khai |
| Component translation | source text official | chất lượng model/logic dịch đang triển khai |
| Component inpainting | ảnh + ground-truth mask | chất lượng thuật toán xóa chữ, không lẫn lỗi OCR |
| Pipeline-stage diagnostic | ảnh, OCR/mask do server tự sinh | lỗi tích lũy của hệ thống đang live |
| End-to-end | ảnh holdout mới + nhãn/human rating | chất lượng sản phẩm hoàn chỉnh |

Phải báo cáo component và end-to-end riêng; không lấy một loại thay cho loại còn lại.

## 7. Định dạng prediction và adapter tùy chọn

Script tạo skeleton có đúng toàn bộ `case_id`:

```powershell
# OCR
.\.venv\Scripts\python.exe .\tools\make_prediction_template.py ocr `
  --manifest .\manifests\ocr_totaltext\manifest.jsonl `
  --output .\predictions\ocr_totaltext.jsonl

# SCUT
.\.venv\Scripts\python.exe .\tools\make_prediction_template.py inpainting `
  --manifest .\manifests\inpainting_scut_enstext\manifest.jsonl `
  --output .\predictions\inpainting_scut.jsonl

# FLORES
.\.venv\Scripts\python.exe .\tools\make_prediction_template.py translation `
  --manifest .\manifests\translation_flores200_en_vi\manifest.jsonl `
  --output .\predictions\flores200.jsonl
```

Phần này chỉ dùng khi cần tích hợp một backend khác. Với Hugging Face Space của VieTrans,
ưu tiên runner ở mục 6 để tự sinh đúng định dạng. Không đổi
`case_id`, không bỏ mẫu lỗi và không tự sắp xếp lại ground truth.

### OCR JSONL

```json
{"case_id":"ocr_totaltext_img1","regions":[{"polygon":[[206,633],[251,811],[386,931]],"text":"PETROSAINS"}],"text":"PETROSAINS"}
```

`polygon` dùng tọa độ pixel trên ảnh gốc. `text` toàn ảnh là metric chẩn đoán;
headline OCR dùng ghép region theo polygon.

### Inpainting JSONL

```json
{"case_id":"inp_scut_0001","output_image":"D:/outputs/scut/1.png"}
```

Ảnh output phải cùng width/height với input. Đường dẫn có thể tuyệt đối hoặc tương
đối so với file prediction.

### Dịch JSONL

```json
{"case_id":"mt_flores200_0001","translation":"Bản dịch tiếng Việt của mô hình."}
```

Mỗi source chỉ có đúng một hypothesis. Không dùng Google Translate để tạo
“reference”; reference là bản dịch người của dataset.

## 8. Chấm riêng từng thành phần

### OCR

```powershell
.\run_component.ps1 `
  -Suite ocr `
  -Manifest .\results\vietrans_v1\ocr_totaltext\inference\selected_manifest.jsonl `
  -Predictions .\results\vietrans_v1\ocr_totaltext\inference\predictions.jsonl `
  -OutputDir .\results\vietrans_v1\ocr_totaltext\score `
  -IouThreshold 0.5
```

Headline:

- Detection precision/recall/F1 với polygon IoU 0,5.
- CER trên các region đã ghép.
- Text-spotting F1: polygon ghép đúng và transcription exact sau chuẩn hóa.

Không cần “đánh giá lại PP-OCRv5 gốc” để chứng minh paper. Ta đánh giá đúng
checkpoint, config và miền dữ liệu đang dùng trong VieTrans; đặc biệt cần nếu có
fine-tune, đổi detector threshold, crop, orientation hoặc post-processing.

### Inpainting SCUT

```powershell
.\run_component.ps1 `
  -Suite inpainting `
  -Manifest .\results\vietrans_v1\inpainting_scut\inference\selected_manifest.jsonl `
  -Predictions .\results\vietrans_v1\inpainting_scut\inference\predictions.jsonl `
  -OutputDir .\results\vietrans_v1\inpainting_scut\score
```

### Inpainting OTR

```powershell
.\run_component.ps1 `
  -Suite inpainting `
  -Manifest .\results\vietrans_v1\inpainting_otr_easy\inference\selected_manifest.jsonl `
  -Predictions .\results\vietrans_v1\inpainting_otr_easy\inference\predictions.jsonl `
  -OutputDir .\results\vietrans_v1\inpainting_otr_easy\score
```

Headline inpainting:

- `inside_psnr`, `inside_ssim`, `inside_mae`: độ đúng trong vùng chữ.
- `outside_mae`, `outside_ssim`: mức phá hủy vùng không cần sửa, so với input.
- `boundary_mae`, `boundary_ssim`: seam/halo ở vòng biên ngoài mask.
- `full_psnr`, `full_ssim`: chỉ báo toàn ảnh; không dùng một mình vì vùng ngoài mask
  thường chiếm đa số.

PSNR cao hơn, SSIM gần 1 hơn và MAE thấp hơn là tốt hơn. Báo cáo SCUT và OTR riêng,
không lấy trung bình hai dataset.

### Dịch FLORES

```powershell
.\run_component.ps1 `
  -Suite translation `
  -Manifest .\results\vietrans_v1\flores200\inference\selected_manifest.jsonl `
  -Predictions .\results\vietrans_v1\flores200\inference\predictions.jsonl `
  -OutputDir .\results\vietrans_v1\flores200\score
```

### Dịch MASSIVE

```powershell
.\run_component.ps1 `
  -Suite translation `
  -Manifest .\results\vietrans_v1\massive\inference\selected_manifest.jsonl `
  -Predictions .\results\vietrans_v1\massive\inference\predictions.jsonl `
  -OutputDir .\results\vietrans_v1\massive\score
```

Headline dịch:

- FLORES: corpus chrF++ chính, SacreBLEU phụ.
- MASSIVE: chrF++, SacreBLEU và terminology accuracy; ghi rõ đây là miền assistant
  utterance ngắn.
- `summary.json` lưu signature SacreBLEU/chrF để tái lập.

## 9. Điều kiện một kết quả được chấp nhận

Trình chấm mặc định trả lỗi nếu:

- coverage nhỏ hơn 100%;
- có mẫu chấm lỗi;
- có prediction thừa;
- ảnh output thiếu hoặc sai kích thước;
- trùng `case_id`.

`--allow-partial` chỉ dành cho debug, không được dùng trong báo cáo. Mỗi
`summary.json` tự ghi SHA-256 manifest, SHA-256 prediction, command, Python và OS.

Gộp các summary thành báo cáo, nhưng giữ điểm từng dataset:

```powershell
.\.venv\Scripts\python.exe .\tools\aggregate_report.py `
  .\results\vietrans_v1\ocr_totaltext\score\summary.json `
  .\results\vietrans_v1\inpainting_scut\score\summary.json `
  .\results\vietrans_v1\inpainting_otr_easy\score\summary.json `
  .\results\vietrans_v1\flores200\score\summary.json `
  .\results\vietrans_v1\massive\score\summary.json `
  --output .\results\vietrans_v1\REPORT.md
```

## 10. Khi nào chạy end-to-end

Chỉ chạy sau khi năm báo cáo component ở trên đã hoàn tất và config đã khóa. Bộ
pipeline test phải là holdout ảnh mới có:

- polygon và transcription OCR;
- bản dịch Việt do người duyệt;
- rating mù cho độ tự nhiên nền, khả năng đọc, layout và nghĩa;
- latency, tỷ lệ thành công và log lỗi từng stage.

Không lấy 282 ảnh từng dùng để phát triển làm final test. Có thể dùng chúng làm
development/qualitative regression. Khi báo cáo pipeline, vẫn phải trình bày điểm
OCR, dịch, inpainting riêng để truy nguyên lỗi.

## 11. Checklist trước khi công bố

- [ ] `validate_workspace.py --deep-hash` trả `PASS`.
- [ ] Model ID/revision và code commit là bất biến.
- [ ] `server_probe.json` ghi đúng Space SHA và có `component_endpoints_ready: true`.
- [ ] Prediction được sinh trực tiếp bởi endpoint production, không phải oracle/model thay thế.
- [ ] Không còn `FILL_ME` trong config.
- [ ] Mỗi prediction có đủ và chỉ đúng các `case_id` trong manifest.
- [ ] Mỗi summary có coverage `1.0`, error `0`, extra prediction `0`.
- [ ] Không dùng IIMT30k_Vi làm final test.
- [ ] FLORES, MASSIVE, SCUT, OTR và Total-Text được báo cáo riêng.
- [ ] Không chỉnh hệ thống sau khi xem điểm final test.
- [ ] Lưu `run_lock.json`, config, prediction, summary và per-case JSONL cùng báo cáo.

## 12. Tự kiểm tra evaluator

`tools/create_oracle_smoke.py` chỉ tạo prediction bằng ground truth trên vài mẫu để
kiểm tra đường dẫn và công thức. Kết quả oracle tuyệt đối không phải kết quả model
và không được đưa vào báo cáo.
