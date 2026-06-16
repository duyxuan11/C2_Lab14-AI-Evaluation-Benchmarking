# Reflection — Tạ Duy Xuân (2A202600970)

## 1. Phần việc đảm nhiệm
Phụ trách **Retrieval Evaluation + Agent RAG thật** (góp trực tiếp 10đ Retrieval Evaluation +
nền tảng cho toàn bộ pipeline, vì Multi-Judge/Regression đều cần `agent.query()` thật mới có
ý nghĩa). File: `agent/main_agent.py`, `engine/retrieval_eval.py`.

Branch: `2A202600970-TaDuyXuan`, commit `33b5133`, merge qua PR #5 vào `main`.

## 2. Đóng góp kỹ thuật cụ thể (Engineering Contribution)
- **Retrieval thật**: thay `MainAgent.query()` (chỉ `sleep(0.5)` rồi trả câu trả lời mẫu) bằng
  retrieval dựa trên embedding cosine similarity — embed 20 chunk của
  `data/knowledge_base.json` bằng Gemini (`gemini-embedding-001` qua endpoint tương thích
  OpenAI), cache ra `data/.kb_embeddings_cache.json` (key = chunk id + hash nội dung) để
  không tính lại embedding mỗi lần chạy benchmark — giảm chi phí/API call.
- **Tách `retrieve()` khỏi `query()`**: `retrieve(question, top_k=3)` chỉ làm retrieval
  (không gọi generation), để `RetrievalEvaluator` đánh giá retrieval mà không tốn chi phí
  sinh câu trả lời. `query()` gọi `retrieve()` rồi mới generation, trả thêm `retrieved_ids`
  để tính Hit Rate/MRR.
- **`evaluate_batch()` thật**: nối với dataset thật (`data/golden_set.jsonl`, 57 case do
  SV1 sinh), gọi `agent.retrieve()` cho từng case, so với `expected_retrieval_ids`, chạy
  song song bằng `asyncio.Semaphore` + `asyncio.gather`. Loại case `out_of_context`
  (`expected_retrieval_ids = []`) khỏi trung bình Hit Rate/MRR vì không có ground truth
  dương để đo, đếm riêng vào `no_ground_truth_count` để báo cáo minh bạch.
- **Kết quả đo được** (chạy thật trên 57 case, ghi ở `reports/retrieval_eval.json`):
  Avg Hit Rate = **1.000**, Avg MRR = **0.971**, đánh giá trên 52/57 case có ground truth.

## 3. Vấn đề gặp phải & cách giải quyết (Problem Solving)
- **Model bị quota = 0**: lần đầu chạy `agent.query()` bị lỗi 429 với thông báo
  `limit: 0` cho model `gemini-2.0-flash` (giá trị mặc định trong `.env.example`). Thử đổi
  sang `gemini-2.5-flash` (model mà `data/synthetic_gen.py` của SV1 thực ra đang dùng) thì
  chạy được ngay → kết luận đây là quota bị khoá ở cấp project/key cho riêng model đó, không
  phải lỗi code. Đã cập nhật `.env` cá nhân và báo lại cho nhóm về sự không khớp giữa
  `.env.example` và model thực sự hoạt động.
- **Hết quota free-tier 20 request/ngày cho chat generation**: sau khi test nhiều lần +
  chạy `synthetic_gen.py` để sinh dataset, model chat bị hết quota theo ngày. Vì đã tách
  riêng `retrieve()` (dùng embeddings, có quota riêng) khỏi `query()` (dùng chat, bị giới
  hạn 20/ngày), vẫn chạy được toàn bộ `evaluate_batch()` trên 57 case mà không cần đợi quota
  reset — retrieval evaluation độc lập với generation quota.
- **Đồng bộ với nhóm qua git**: làm việc trên nhánh riêng theo đúng convention
  `{MãSV}-{TênKhôngDấu}` của team, xử lý xung đột `.gitignore` khi SV1 commit thêm
  `data/golden_set.jsonl` trực tiếp vào repo (ban đầu README dự định không commit file này,
  nhóm sau đó quyết định commit luôn để khỏi cần API key mới chạy được benchmark).

## 4. Hiểu biết kỹ thuật (Technical Depth)

### MRR (Mean Reciprocal Rank)
Với mỗi câu hỏi, tìm vị trí (1-indexed) của tài liệu đúng **đầu tiên** trong danh sách kết
quả retrieval, lấy nghịch đảo `1/vị trí`; nếu không có thì tính 0. MRR là trung bình các giá
trị đó trên toàn dataset.

Ví dụ thật từ hệ thống (case `GS-001`): ground truth `["DOC-AUTH-001"]`, retriever trả về
`["DOC-VERSION-001", "DOC-AUTH-001", "DOC-WEBHOOK-002"]` → tài liệu đúng ở vị trí 2 →
reciprocal rank = 1/2 = 0.5. Đây là 1 trong 3 case (trên 52 case có ground truth) có MRR
< 1.0 trong lần chạy thật — Hit Rate vẫn đạt (tài liệu đúng có trong top-3) nhưng ranking
chưa tối ưu, gợi ý có thể cải thiện bằng reranking.

### Retrieval Quality ↔ Answer Quality
RAG tuân theo nguyên lý garbage-in-garbage-out: chất lượng câu trả lời (Faithfulness,
Accuracy) bị chặn trên bởi chất lượng context được đưa vào, bất kể LLM mạnh tới đâu. Vì vậy
phải đo Retrieval **tách riêng** trước khi đánh giá Generation: nếu answer sai mà Hit Rate
cũng thấp → lỗi nằm ở Retrieval/Chunking; nếu Hit Rate cao mà answer vẫn sai → lỗi nằm ở
Prompting/Generation. Hệ thống của nhóm đã chứng minh được Retrieval hoạt động tốt (Hit Rate
1.0) một cách độc lập với Generation, đúng yêu cầu của `README.md`.

### Cohen's Kappa và Position Bias (phần Multi-Judge, không phải module mình code nhưng cần
hiểu để giải trình)
- **Cohen's Kappa**: thước đo độ đồng thuận giữa 2 người (hoặc 2 model) chấm điểm, có hiệu
  chỉnh phần đồng thuận xảy ra do may rủi (random chance), công thức
  `κ = (p_o − p_e) / (1 − p_e)` với `p_o` là tỉ lệ đồng thuận quan sát được và `p_e` là tỉ lệ
  đồng thuận kỳ vọng nếu chấm ngẫu nhiên. **Lưu ý quan trọng**: `engine/llm_judge.py` của
  nhóm hiện tính `agreement_rate = 1.0 - |score_1 - score_2| / 4.0` — đây là một hệ số đồng
  thuận đơn giản dựa trên độ lệch điểm tuyệt đối, **không phải** Cohen's Kappa thật (Kappa
  cần dữ liệu dạng hạng mục/categorical và hiệu chỉnh chance agreement). Khi giải trình nên
  nói rõ sự khác biệt này thay vì gọi nhầm là Kappa.
- **Position Bias**: hiện tượng Judge có xu hướng thiên vị câu trả lời theo **vị trí xuất
  hiện** (luôn thích câu A hoặc luôn thích câu B) bất kể nội dung. `check_position_bias()`
  trong `llm_judge.py` kiểm tra bằng cách gọi Judge 2 lần, đổi chỗ response A/B giữa 2 lần
  gọi — nếu cả 2 lần đều chọn "A" (hoặc đều chọn "B") dù nội dung đã đổi chỗ, kết luận có bias
  vị trí.
- **Trade-off Chi phí ↔ Chất lượng**: dùng nhiều Judge/nhiều lần gọi LLM (multi-judge,
  tie-breaker, position-bias check) tăng độ tin cậy nhưng tăng chi phí & latency tuyến tính
  theo số lần gọi. Trong phần của mình, cache embedding (không tính lại nếu chunk không đổi)
  và tách `retrieve()` khỏi `query()` là 2 cách giảm chi phí mà không giảm độ chính xác.

## 5. Việc cần làm tiếp (không thuộc phần mình)
- SV1 cần kiểm tra lại corpus/SDG để đủ ≥50 case ổn định (đã có 57, đạt).
- Người phụ trách Multi-Judge cần làm rõ tên gọi "agreement_rate" vs Cohen's Kappa thật khi
  báo cáo, hoặc cân nhắc implement Kappa đúng nghĩa nếu muốn khớp rubric chính xác hơn.
- Wiring `main.py` để dùng `RetrievalEvaluator`/`MainAgent`/`LLMJudge` thật thay class giả.

---
*Ghi chú: bản nháp này tổng hợp từ commit/log thật trong quá trình làm việc — nên đọc lại và
chỉnh giọng văn/cảm nhận cá nhân trước khi nộp.*
