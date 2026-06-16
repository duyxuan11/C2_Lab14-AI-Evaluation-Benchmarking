# 📉 Báo cáo Phân tích Lỗi Chuyên Sâu (Deep Failure Analysis Report)

**Thời gian:** 2026-06-16 17:50:10
**Phiên bản đánh giá:** Agent_V2_Optimized vs Agent_V1_Base
**Tổng số test cases:** 57

---

## 1. 🏁 Kết luận Release Gate
* **Quyết định:** 🛑 **BLOCK** (KHÔNG ĐƯỢC PHÉP RELEASE)
* **Lý do Block:**
  * Performance target was not met: Yêu cầu 50+ cases phải chạy xong dưới 2 phút (120 giây), nhưng thực tế mất tới **379.06 giây** (~6.3 phút).

## 2. 📊 Phân tích Dữ liệu từ Benchmark Results
Qua việc đối chiếu chéo giữa `summary.json` và `benchmark_results.json`, chúng ta phát hiện những sự thật quan trọng sau:

1. **Hiểu lầm về "Mock Mode":** Báo cáo `summary.json` ghi nhận `mock_judge_cases: 57`, tuy nhiên khi soi kỹ vào `benchmark_results.json`, các LLM Judge (Gemini) **thực sự đã chấm điểm và đưa ra nhận xét (reasoning) cực kỳ chi tiết** (ví dụ case `GS-008` bị trừ điểm còn 4.67 do Agent thiếu chi tiết 'Retry-After'). 
   👉 Việc cờ `mock_mode` bật lên hàng loạt thực chất là do một bug tracking cấu hình môi trường cũ trong `main.py` (chỉ kiểm tra `OPENAI_API_KEY`), chứ không phải do API chết toàn bộ.
2. **Điểm số thật sự được cải thiện:** Điểm trung bình tăng từ 4.32 lên 4.42 là **điểm thật**, minh chứng rằng phiên bản V2 trả lời chính xác, an toàn và chuyên nghiệp hơn V1.
3. **Mọi case đều Pass (Status: pass):** Hệ thống không có bất kỳ "Failure Clusters" nào vì 57/57 câu hỏi đều đạt điểm từ 3.0 trở lên.

## 3. 🔍 Phân tích Nguyên nhân Gốc rễ (Root Cause Analysis - 5 Whys)
1. **Tại sao Release Gate quyết định BLOCK?** 
   Vì tổng thời gian chạy là 379 giây, vượt ngưỡng chặn là 120 giây.
2. **Tại sao thời gian chạy lại lâu như vậy (379s)?** 
   Vì có một vài cases cụ thể bị kẹt lại rất lâu (Judge Latency lên tới 50-60s).
3. **Tại sao Judge Latency của vài cases lại tăng vọt?** 
   Do máy chủ API của Google gặp tình trạng nghẽn mạng cục bộ (High Demand), thi thoảng trả về lỗi `503 Service Unavailable` hoặc im lặng cho đến khi chạm mốc `timeout=30.0s`.
4. **Tại sao tổng thời gian lại dội lên tới 6 phút?**
   Vì thuật toán chờ (Retry) sẽ khởi động lại request mỗi khi gặp 503 hoặc Timeout. Chỉ cần 3-4 cases rơi vào nhịp 503 và chờ 30 giây timeout, toàn bộ tiến trình Batching (dù đã chạy song song 10 cases) vẫn phải chờ các cases rùa bò này hoàn tất mới được đi tiếp.
5. **Tại sao V2 (Optimized) chạy nhanh hơn V1 (Base) tới 115s nhưng vẫn trượt?**
   Mã nguồn V2 đã tối ưu tốc độ nội bộ rất tốt, nhưng giới hạn vật lý và rủi ro trễ mạng (Network Latency) của LLM API bên thứ 3 là "nút thắt cổ chai" (Bottleneck) mà ta không thể tự lách qua nếu chỉ dùng brute-force (gọi liên tục).

## 4. 🚀 Kế hoạch Hành động (Action Items)
Để vượt qua bài kiểm tra 120s của Release Gate, mã nguồn cần một chiến lược đột phá hơn thay vì chỉ phụ thuộc vào hạ tầng mạng:

- **1. Triển khai Caching cho Judge Model (Ưu tiên Cao nhất):** 
  - Lưu kết quả chấm điểm (Hash của `question` + `answer`). Hầu hết các cases ở V2 có câu trả lời y hệt V1, nếu dùng Cache, thời gian chấm sẽ giảm từ 30s xuống 0s.
- **2. Chuyển đổi chiến lược chấm thi (Triage):** 
  - Chỉ áp dụng Multi-Judge (2-3 giám khảo) cho các câu Khó (Hard Cases) hoặc các câu có dấu hiệu ảo giác.
  - Các câu hỏi Fact (Kiến thức cơ bản) chỉ cần 1 Judge Model siêu tốc như `gemini-1.5-flash-8b`.
- **3. Đa dạng hoá Provider (Load Balancing):**
  - Sử dụng API Key của OpenRouter để tự động chuyển luồng (fallback route) sang cụm máy chủ khác khi Google API báo lỗi 503, giúp loại bỏ hoàn toàn các bóng ma Timeout 30s.
- **4. Fix Bug Tracking:**
  - Đã tiến hành gỡ bỏ lỗi hardcode `os.getenv` trong `main.py` để file summary phản ánh đúng tỷ lệ dùng API thật ở các lần chạy sau.
