# Spec: Retrieval Eval + Agent RAG (SV3) — Lab 14

**Ngày:** 2026-06-16
**Phụ trách:** SV3 (Retrieval Eval + Agent thật)
**Mục tiêu rubric:** Retrieval Evaluation (10đ) + nền tảng Agent cho Multi-Judge/Regression

## Bối cảnh
`agent/main_agent.py` chỉ trả câu trả lời mẫu cố định, không retrieval thật.
`engine/retrieval_eval.py` có `calculate_hit_rate`/`calculate_mrr` đúng logic nhưng
`evaluate_batch()` trả số cố định. Dùng đúng corpus + schema do SV1 cung cấp
(`data/knowledge_base.json`, `data/golden_set.jsonl` — xem
`docs/superpowers/specs/2026-06-16-data-sdg-design.md`).

## Thiết kế

### Retrieval — `agent/main_agent.py`
- Embedding-based retrieval, không cần vector DB: embed toàn bộ chunk trong
  `knowledge_base.json` một lần bằng Gemini (endpoint tương thích OpenAI,
  `model=gemini-embedding-001`), cache ra `data/.kb_embeddings_cache.json` (key = chunk
  id + hash nội dung) để không tính lại khi chunk không đổi — giảm chi phí khi benchmark
  chạy nhiều lần.
- `MainAgent.retrieve(question, top_k=3)`: embed câu hỏi, cosine similarity (pure Python)
  với toàn bộ chunk đã cache, trả top-k `{id, score, chunk}`. Tách riêng khỏi `query()` để
  `RetrievalEvaluator` đánh giá retrieval mà không tốn chi phí generation.
- `MainAgent.query(question, history=None)`: gọi `retrieve()`, đưa context (kèm ID) vào
  prompt, system prompt yêu cầu chỉ trả lời theo context, từ chối nếu thiếu thông tin, và
  bỏ qua chỉ thị nhúng trong tài liệu/câu hỏi đòi đổi vai trò hoặc tiết lộ bí mật (khớp các
  case `adversarial`/`out_of_context` của SV1). Trả về `retrieved_ids` để tính Hit Rate/MRR.

### Retrieval Eval — `engine/retrieval_eval.py`
- `evaluate_batch(agent, dataset, top_k=3, concurrency=5)`: với mỗi case gọi
  `agent.retrieve()`, so `retrieved_ids` với `expected_retrieval_ids` bằng
  `calculate_hit_rate`/`calculate_mrr`. Case `out_of_context` (`expected_retrieval_ids=[]`)
  không có ground truth dương nên bị loại khỏi trung bình, đếm riêng vào
  `no_ground_truth_count`.
- Trả về `per_case` đầy đủ (question, expected vs retrieved ids) để biết chính xác chunk
  nào đang gây lỗi — phục vụ "5 Whys" trong `analysis/failure_analysis.md`.
- Có thể chạy độc lập: `python -m engine.retrieval_eval` → `reports/retrieval_eval.json`,
  không cần chờ `main.py` của nhóm wire xong.

## Giải thích kỹ thuật (Technical Depth)

### MRR (Mean Reciprocal Rank)
Với mỗi câu hỏi, tìm vị trí (1-indexed) của chunk đúng **đầu tiên** xuất hiện trong danh
sách kết quả retrieval, lấy nghịch đảo `1/vị trí`; nếu không có chunk đúng trong kết quả thì
tính 0. MRR là trung bình các giá trị đó trên toàn dataset.

Ví dụ: ground truth `["DOC-AUTH-002"]`, retriever trả `["DOC-AUTH-001", "DOC-AUTH-002", "DOC-SDK-001"]`
→ vị trí đúng = 2 → reciprocal rank = 1/2 = 0.5.

- MRR = 1.0: chunk đúng luôn được xếp #1 (retrieval rất tốt).
- MRR thấp nhưng Hit Rate cao: chunk đúng có trong top-k nhưng bị xếp hạng thấp — vẫn đủ
  cho LLM thấy, nhưng nên cải thiện ranking (rerank).
- Hit Rate = 0: chunk đúng hoàn toàn không nằm trong top-k — lỗi nặng ở bước
  retrieval/embedding/chunking.

### Retrieval Quality ↔ Answer Quality
RAG tuân theo nguyên lý garbage-in-garbage-out: chất lượng câu trả lời (Faithfulness,
Accuracy) bị chặn trên bởi chất lượng context được đưa vào, bất kể LLM mạnh tới đâu. Nếu
Hit Rate/MRR thấp, LLM không có thông tin đúng để dựa vào → hoặc hallucinate (bịa câu trả
lời) hoặc trả lời "không biết" dù thực ra tài liệu có đáp án.

Vì vậy phải đo Retrieval **tách riêng** trước khi đánh giá Generation (đúng yêu cầu trong
`README.md`):
- Answer sai + Hit Rate thấp → lỗi nằm ở Retrieval/Chunking/Embedding, sửa prompt sẽ không
  giải quyết được.
- Answer sai + Hit Rate cao → lỗi nằm ở Prompting/Generation (LLM có đúng context nhưng vẫn
  trả lời sai), cần sửa system prompt hoặc đổi model.

Đây là cơ sở để chọn nhánh nguyên nhân gốc rễ trong "5 Whys" của `analysis/failure_analysis.md`.
