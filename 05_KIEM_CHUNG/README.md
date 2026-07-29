# Kiểm chứng toàn vẹn

Sau khi hoàn tất đóng gói, tệp `SHA256SUMS.txt` trong thư mục này sẽ liệt kê hash SHA-256 của mọi tệp minh chứng (trừ chính nó, gồm cả `THONG_KE_GOI.txt`). Dùng tệp đó để xác nhận file không bị thay đổi sau khi upload/tải về.

PowerShell mẫu:

```powershell
$root = Resolve-Path .
Get-Content .\05_KIEM_CHUNG\SHA256SUMS.txt | ForEach-Object {
  $hash, $relativePath = $_ -split '  ', 2
  $actual = (Get-FileHash -LiteralPath (Join-Path $root $relativePath) -Algorithm SHA256).Hash.ToLower()
  if ($actual -ne $hash) { "FAIL  $relativePath" } else { "OK    $relativePath" }
}
```

Ngoài hash đóng gói, `score/summary.json` của mỗi run còn có hash của manifest và prediction đã chấm. Đây là kiểm tra quan trọng nhất để bảo đảm `per_case.jsonl` và chỉ số tổng hợp xuất phát từ đúng cặp dữ liệu đầu vào.
