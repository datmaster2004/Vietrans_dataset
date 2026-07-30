# Dữ liệu được lưu trong gói minh chứng

Gói chỉ mang theo dữ liệu cần thiết để quan sát và đối chiếu các run có kết quả. Các archive nguồn và cây giải nén lặp lại đã được loại để giảm dung lượng; dữ liệu/nhãn/manifest được dùng trong đánh giá vẫn được giữ.

| Thư mục | Dataset và vai trò | Nội dung giữ lại |
|---|---|---|
| [`01_IIMT30K/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/01_IIMT30K/) | Tập test nội miền của OCR, inpainting và dịch; tách từ tập gốc theo train/valid/test | 3.000 ảnh, 1.500 ảnh clean, mask, annotation, source/reference text và 3 manifest 1.500 case |
| [`02_TOTALTEXT/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/02_TOTALTEXT/) | Total-Text official test cho OCR | 300 ảnh test, 300 ground-truth, manifest 300 case, README và LICENSE gốc |
| [`03_FLORES200_EN_VI/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/03_FLORES200_EN_VI/) | FLORES-200 `eng_Latn -> vie_Latn` devtest cho dịch | Hai tệp devtest English/Vietnamese, metadata, manifest 1.012 case và README gốc |
| [`04_SCUT_ENSTEXT/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/04_SCUT_ENSTEXT/) | SCUT-EnsText official test cho inpainting | 813 input, 813 clean ground-truth, 813 polygon/mask và manifest; lượt chạy hiện có 812/813 output |

## IIMT30K

Ba dataset card trong [`01_IIMT30K/manifests/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/01_IIMT30K/manifests/) đều ghi `role: in_domain_held_out_test` và `split: test`. Đây là tập test được tách từ IIMT30K gốc theo train/valid/test; mỗi manifest có 1.500 case. Manifest dùng đường dẫn tương đối; mỗi dòng có `case_id` để nối đến dữ liệu và kết quả.

## Total-Text

[`02_TOTALTEXT/prepared/Images/Test/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/02_TOTALTEXT/prepared/Images/Test/) chứa 300 ảnh dùng cho OCR; nhãn polygon và text tương ứng nằm trong [`02_TOTALTEXT/prepared/Groundtruth/Test/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/02_TOTALTEXT/prepared/Groundtruth/Test/). Manifest [`02_TOTALTEXT/manifests/ocr_totaltext/manifest.jsonl`](https://github.com/datmaster2004/Vietrans_dataset/blob/main/02_DU_LIEU_SU_DUNG/02_TOTALTEXT/manifests/ocr_totaltext/manifest.jsonl) trỏ bằng đường dẫn tương đối đến ảnh và chứa annotation dùng để chấm.

License và README chính thức được sao chép ở [`02_TOTALTEXT/NGUON_VA_LICENCE/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/02_TOTALTEXT/NGUON_VA_LICENCE/). Khi công bố lại, phải giữ nguyên attribution/license này.

## FLORES-200

[`03_FLORES200_EN_VI/prepared/devtest/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/03_FLORES200_EN_VI/prepared/devtest/) chứa phần `eng_Latn` và `vie_Latn` theo hàng song song; manifest đã chọn đúng 1.012 case dùng bởi run. Reference trong manifest là bản dịch chuẩn để chấm, không phải bản dịch do hệ thống tạo.

## SCUT-EnsText

[`04_SCUT_ENSTEXT/test_set/test/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/04_SCUT_ENSTEXT/test_set/test/) có 813 ảnh input `all_images`, 813 ảnh clean `all_labels` và polygon `all_gts`; [`04_SCUT_ENSTEXT/prepared/masks/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/02_DU_LIEU_SU_DUNG/04_SCUT_ENSTEXT/prepared/masks/) có 813 mask. Kết quả đã phát hiện nằm ở [`03_KET_QUA_CHI_TIET/01_KET_QUA_CHINH/03_INPAINTING_SCUT_ENSTEXT_PARTIAL/`](https://github.com/datmaster2004/Vietrans_dataset/tree/main/03_KET_QUA_CHI_TIET/01_KET_QUA_CHINH/03_INPAINTING_SCUT_ENSTEXT_PARTIAL/): 812 prediction/artifact và 1 lỗi `inp_scut_img_324` vì mask không có pixel được chọn.



