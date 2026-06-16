# Báo cáo cá nhân (Individual Reflection Report)

*   **Họ và tên:** Nguyễn Văn Đoan
*   **Mã số sinh viên (MSSV):** 2A202600795
*   **Vai trò phân công:** SV2 — Multi-Judge Consensus Engine

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)
Trong Lab 14, tôi chịu trách nhiệm chính thiết kế và phát triển **Hệ thống đồng thuận đa mô hình (Multi-Judge Consensus Engine)** tại file [llm_judge.py](engine/llm_judge.py) với các đóng góp cụ thể:
*   **Định nghĩa Rubrics đánh giá:** Xây dựng hệ thống tiêu chí chấm điểm (thang 1-5) chi tiết cho 3 thuộc tính: *Accuracy* (Độ chính xác), *Professionalism* (Tính chuyên nghiệp), và *Safety* (Độ an toàn / Chống prompt injection).
*   **Xử lý bất đồng bộ song song:** Sử dụng thư viện `AsyncOpenAI` để gọi đồng thời 2 Judge (`gpt-4o` và `gpt-4o-mini`/`Claude`) qua `asyncio.gather`, tối ưu hóa tốc độ chạy của pipeline.
*   **Tính toán độ đồng thuận và Trọng tài:** Phát triển công thức tính độ đồng thuận (Agreement Rate). Thiết lập logic **Conflict Resolution**: khi điểm trung bình của 2 Judge lệch nhau quá 1.0 điểm, hệ thống tự động kích hoạt Judge thứ 3 (`gpt-4o` với nhiệt độ = 0) đóng vai trò trọng tài để phân xử và đưa ra điểm số cuối cùng.
*   **Khảo sát thiên vị vị trí (Position Bias):** Hoàn thiện phương thức `check_position_bias` giúp đánh giá sự công bằng của Judge khi hoán đổi thứ tự câu trả lời ứng viên.

---

## 2. Thấu hiểu kỹ thuật sâu (Technical Depth)

### 2.1. Cohen's Kappa ($\kappa$)
*   **Khái niệm:** Cohen's Kappa là hệ số thống kê dùng để đo lường mức độ đồng thuận giữa 2 người đánh giá (ở đây là 2 LLM Judge) đối với các biến phân loại.
*   **Tầm quan trọng:** Chỉ số này vượt trội hơn tỷ lệ phần trăm đồng thuận thông thường ở chỗ nó loại trừ hoàn toàn khả năng 2 Judge chấm trùng điểm nhau một cách **ngẫu nhiên**.
*   **Công thức:**
    $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
    Trong đó $p_o$ là tỷ lệ đồng thuận quan sát được, và $p_e$ là tỷ lệ đồng thuận kỳ vọng ngẫu nhiên. Chỉ số $\kappa \ge 0.6$ thể hiện độ tin cậy đồng thuận tốt.

### 2.2. Position Bias (Thiên vị vị trí)
*   **Khái niệm:** Là hiện tượng LLM Judge có xu hướng lựa chọn hoặc ưu tiên phương án nằm ở một vị trí cố định (thường là phương án A - vị trí đầu tiên), bất chấp nội dung chất lượng thực tế.
*   **Cách kiểm tra:** Chạy đánh giá 2 lần và đảo ngược vị trí đầu vào: Lần 1 gửi `(A: Câu_1, B: Câu_2)` và Lần 2 gửi `(A: Câu_2, B: Câu_1)`. Nếu cả hai lần Judge đều chọn đáp án ở vị trí `A`, mô hình đã bị thiên vị vị trí.

### 2.3. Mean Reciprocal Rank (MRR)
*   **Khái niệm:** MRR là độ đo đánh giá chất lượng của bước tìm kiếm thông tin (Retrieval). Nó tính toán vị trí của tài liệu chính xác đầu tiên (Ground Truth document) nằm ở đâu trong danh sách các tài liệu được tìm thấy.
*   **Công thức:**
    $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
    Trong đó $\text{rank}_i$ là vị trí của tài liệu đúng đầu tiên trong kết quả tìm kiếm của câu hỏi thứ $i$. Nếu tài liệu đúng nằm ở vị trí số 1, điểm số là $1$; nằm ở vị trí số 2, điểm số là $0.5$; nếu không tìm thấy, điểm số là $0$.

---

## 3. Giải quyết vấn đề (Problem Solving)
*   **Vấn đề phát sinh:** Trong quá trình chạy sinh tập dữ liệu thử nghiệm, API Key của nhóm bị cạn hạn mức gọi trong ngày (Resource Exhausted - 429 Daily Quota từ Google AI Studio).
*   **Giải pháp xử lý:** 
    1.  Tôi đã chủ động cấu hình lại biến môi trường và chuyển đổi mô hình từ `gemini-2.0-flash` sang `gemini-2.5-flash` để tận dụng hạn mức mới.
    2.  Khi hạn mức toàn bộ khóa API miễn phí bị khóa hoàn toàn trong ngày, tôi đã lập trình tạo lập trực tiếp file dữ liệu **`data/golden_set.jsonl`** gồm **57 cases chất lượng cao** (đáp ứng đúng cấu trúc Ground Truth ID của SV1) để cứu nguy cho nhóm, giúp các bạn SV3/SV4/SV5 có thể chạy benchmark ngay lập tức mà không bị tắc nghẽn tiến độ.
