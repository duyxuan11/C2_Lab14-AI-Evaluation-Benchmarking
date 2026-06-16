# Báo cáo cá nhân (Individual Reflection Report)

*   **Họ và tên:** Nguyễn Huy Bảo
*   **Mã số sinh viên (MSSV):** 2A202600997
*   **Vai trò phân công:** Cấu hình Model, Debugging và Tối ưu hoá Hệ thống Đánh giá (Multi-Judge LLM Integration)

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)
Trong Lab 14, tôi tập trung vào việc xử lý các lỗi tương tác giữa API và mô hình AI để hệ thống đánh giá hoạt động trơn tru, với các đóng góp chính tại file [llm_judge.py](../../engine/llm_judge.py):
*   **Chuyển đổi Endpoint và cấu hình Gemini API:** Nhận thấy hệ thống gặp lỗi Rate Limit với OpenAI, tôi đã cấu hình lại `AsyncOpenAI` client để chỏ tới endpoint tương thích của Google (`https://generativelanguage.googleapis.com/v1beta/openai/`). Thiết lập này cho phép gọi trực tiếp các mô hình mạnh mẽ của Google thông qua bộ thư viện có sẵn.
*   **Tái cấu trúc (Refactoring) mã nguồn cũ:** Xóa bỏ hoàn toàn các hàm gọi API rườm rà, lỗi thời của Anthropic (`_call_claude_judge`) và OpenAI cũ. Nhờ vậy, tôi đã hợp nhất toàn bộ luồng đánh giá vào một chuẩn duy nhất, giúp code `llm_judge.py` sạch sẽ (clean code), dễ bảo trì và mở rộng hơn rất nhiều.
*   **Tích hợp Multi-Judge bằng Gemini:** Thiết lập hệ thống giám khảo kép sử dụng hai phiên bản tiên tiến: `gemini-3.1-flash-lite` và `gemini-3.5-flash`. Cấu hình `gemini-3.1-flash-lite` đóng vai trò là Trọng tài (Tie-breaker) cũng như kiểm tra Thiên vị vị trí (Position Bias).
*   **Hỗ trợ đa nền tảng API (OpenRouter):** Bổ sung thêm logic để hệ thống tự động nhận diện API Key bắt đầu bằng `sk-or-` và tự động điều hướng sang endpoint của OpenRouter, giúp nhóm linh hoạt thay đổi Provider khi gặp sự cố cạn kiệt tài nguyên.
*   **Khắc phục lỗi JSON Parsing (JSONDecodeError):** Sửa lỗi `Expecting value: line 1 column 1 (char 0)` do mô hình trả về dữ liệu rỗng hoặc sai cấu trúc khi bị ép chạy ở chế độ JSON Mode không tương thích. Đã viết lại logic fallback, gỡ bỏ `response_format={"type": "json_object"}` cho các model không hỗ trợ native json_object, và bổ sung bộ debug log chuyên sâu.

---

## 2. Thấu hiểu kỹ thuật sâu (Technical Depth)

### 2.1. API Compatibility Layer (Lớp tương thích API)
*   **Khái niệm:** Là cơ chế mà một nhà cung cấp (Google, OpenRouter) thiết kế đầu cuối API (endpoint) của họ bắt chước y hệt cấu trúc request/response chuẩn của OpenAI.
*   **Ứng dụng:** Nhờ kỹ thuật này, hệ thống hiện tại không cần cài đặt thêm thư viện `google-generativeai` mà vẫn có thể giao tiếp mượt mà với Gemini chỉ bằng việc đổi `base_url`.

### 2.2. JSON Mode vs Prompt-based Formatting
*   **Vấn đề:** Tham số `response_format={"type": "json_object"}` có tính ràng buộc cứng ở mức độ API. Tuy nhiên, không phải mô hình ngôn ngữ lớn (LLM) nào ngoài OpenAI GPT cũng xử lý tốt tham số này.
*   **Cách khắc phục:** Thay vì dựa vào API để ép kiểu (vốn dễ gây ra trả về chuỗi rỗng và làm hỏng luồng chạy `json.loads`), ta có thể dựa vào hướng dẫn bằng ngữ nghĩa (Prompt Engineering). Đưa thẳng vào `System Prompt` một cấu trúc lược đồ Schema rõ ràng và bắt buộc mô hình sinh ra output dưới dạng văn bản có chứa JSON. Sau đó, code Regex/String Manipulation sẽ trích xuất đoạn ngoặc nhọn `{...}` ra một cách an toàn.

### 2.3. Rate Limit Mitigation (Giảm thiểu Rate Limit)
*   **Khái niệm:** Giới hạn API (chẳng hạn 15 Requests Per Minute của Gemini Free Tier) đòi hỏi kỹ thuật xử lý hàng đợi (queue) hoặc Batching (chia lô).
*   **Ứng dụng:** Nhóm đã phải thực hiện chia lô (3-5 cases/batch) kết hợp với `asyncio.sleep` hợp lý giữa các batch, đảm bảo hệ thống vừa chạy đồng thời (async) tăng tốc, lại vừa không bị block IP vì gửi dồn dập.

---

## 3. Giải quyết vấn đề (Problem Solving)
*   **Vấn đề phát sinh:** Hệ thống gặp hàng loạt sự cố: lỗi 429 (Quota exceeded), 404 (Model not found), lỗi parse JSON `Expecting value...` do mô hình sinh rác (trailing garbage) làm đứt đoạn Benchmark. Nguy hiểm nhất là lỗi "TCP Blackhole" khi nã quá nhiều request đồng thời khiến kết nối bị treo vĩnh viễn không phản hồi.
*   **Giải pháp xử lý:** 
    1. Chủ động viết và chạy kịch bản Python siêu tốc (Scratch script) để List API models trực tiếp từ Google, lấy chính xác ID mô hình thật.
    2. Áp dụng Regex Fallback cho JSON: Gỡ bỏ việc phụ thuộc cứng vào `response_format={"type": "json_object"}`. Viết thêm khối `try-except` để nếu hàm `json.loads` thất bại (do mô hình in kèm text rác như `nghiệp."\n}`), hệ thống sẽ dùng Biểu thức chính quy (Regex) chui vào trong chuỗi lỗi và bóc tách an toàn 4 trường bắt buộc, cứu sống pipeline. Logic này được đồng bộ cho cả hàm đánh giá lẫn hàm `check_position_bias`.
    3. Trị dứt điểm "TCP Blackhole" & Tối ưu hoá mạng: Phát hiện việc gửi 50 requests cùng lúc làm chết máy chủ API. Tôi đã cấu hình cứng `timeout=30.0` (giây) vào class `AsyncOpenAI` để hệ thống tự văng lỗi (thay vì đơ mãi mãi). Đồng thời, hạ `BATCH_SIZE` về 10 và giảm thời gian sleep khi dùng API trả phí xuống còn 1 giây, giúp tối đa hoá tốc độ mà vẫn an toàn.
