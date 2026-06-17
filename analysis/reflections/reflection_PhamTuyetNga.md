# Báo cáo cá nhân (Individual Reflection Report)

*   **Họ và tên:** Phạm Tuyết Nga
*   **Mã số sinh viên (MSSV):** [Điền MSSV]
*   **Vai trò phân công:** SV1 — Data & SDG (Synthetic Data Generation & Golden Dataset)

---

## 1. Đóng góp kỹ thuật (Engineering Contribution)

Trong Lab 14, tôi chịu trách nhiệm chính cho **nền tảng dữ liệu** của toàn bộ pipeline đánh giá — corpus nguồn và Golden Dataset mà tất cả các module khác (runner, retrieval_eval, judge, regression) đều phụ thuộc vào. Đóng góp được chứng minh qua commit `feat: commit sv1: Data & SDG` (`3fb1cfb`):

*   **Xây dựng corpus ground truth — [knowledge_base.json](data/knowledge_base.json):** Soạn **20 chunk tài liệu kỹ thuật** (tiếng Việt) trải rộng **10 category** (`auth`, `ratelimit`, `webhook`, `sdk`, `errors`, `pagination`, `versioning`, `payment`, `security`, `environment`). Mỗi chunk có **ID ổn định** (vd `DOC-AUTH-001`) để làm ground truth dùng chung — SV3 build retriever trên đúng corpus này nên Hit Rate / MRR mới đo được.

*   **Thiết kế SDG pipeline — [synthetic_gen.py](data/synthetic_gen.py):** Tái cấu trúc file từ placeholder (1 case giả) thành pipeline sinh dữ liệu hoàn chỉnh:
    *   `generate_grounded_qa` gọi OpenAI ở **JSON mode** sinh cặp QA *grounded* cho từng chunk và gán `expected_retrieval_ids` = ID chunk nguồn — đảm bảo mọi câu hỏi đều truy vết được về tài liệu thật.
    *   `generate_multi_hop` sinh câu hỏi cần ghép **2 chunk** (multi-hop) để test khả năng tổng hợp.
    *   Chạy bất đồng bộ qua `AsyncOpenAI` + `asyncio.gather` theo batch, có **retry/backoff** (`RETRY_BASE_DELAY`) để chịu lỗi 429 của free-tier.
    *   `validate_and_finalize` validate schema, **dedupe** theo câu hỏi chuẩn hoá, kiểm tra `expected_retrieval_ids` tồn tại trong corpus (trừ `out_of_context`), rồi gán ID `GS-xxx`.

*   **Bộ Red Teaming curated — `curated_hard_cases()`:** Tự tay thiết kế các case tấn công/biên *grounded vào ID chunk thật* theo [HARD_CASES_GUIDE.md](data/HARD_CASES_GUIDE.md): **prompt injection / goal hijacking** (3 adversarial), **out-of-context** buộc agent từ chối thay vì bịa (4 case, `expected_retrieval_ids: []`), **ambiguous** (2), **conflicting** (2) và **multi-turn** (3) — đây chính là bộ phá hệ thống mà rubric yêu cầu.

*   **Bàn giao [golden_set.jsonl](data/golden_set.jsonl):** **57 cases** (vượt mốc ≥50), phân bổ theo độ khó `easy 30 / medium 17 / hard 10` và theo loại `fact 42 / out_of_context 4 / adversarial 3 / multi_turn 3 / ambiguous 2 / conflicting 2 / multi_hop 1`. Schema được chốt như một **"hợp đồng"** với cả nhóm (`question`, `expected_answer`, `context`, `expected_retrieval_ids`, `metadata`) để khớp `runner.py` và `retrieval_eval.py`.

---

## 2. Thấu hiểu kỹ thuật sâu (Technical Depth)

### 2.1. Hit Rate vs. Mean Reciprocal Rank (MRR)
Đây là hai độ đo cốt lõi đánh giá bước Retrieval mà dataset của tôi cung cấp ground truth.

*   **Hit Rate@k** chỉ trả lời câu hỏi *"có / không"*: tài liệu đúng có nằm trong top-k kết quả không. Trong [retrieval_eval.py](engine/retrieval_eval.py) nó là `1.0 if any(expected in top_k) else 0.0`. Ưu điểm là trực quan, nhược điểm là **không phân biệt thứ hạng** — đúng ở vị trí 1 hay vị trí 3 đều tính như nhau.
*   **MRR** tinh tế hơn vì quan tâm *vị trí* của tài liệu đúng đầu tiên:
    $$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$
    Đúng ở vị trí 1 → điểm $1$; vị trí 2 → $0.5$; vị trí 3 → $0.33$; không tìm thấy → $0$. MRR phạt việc xếp tài liệu đúng xuống dưới, phản ánh sát trải nghiệm thực vì LLM thường chú trọng vào các đoạn context đầu tiên.

### 2.2. Mối liên hệ Retrieval Quality → Answer Quality
Dataset của tôi là cầu nối giữa hai phần điểm này. **Retrieval kém thì câu trả lời chắc chắn kém** ("garbage in, garbage out"): nếu retriever không lấy được `DOC-AUTH-001`, Judge sẽ chấm answer thấp ở tiêu chí *Accuracy*. Vì mỗi golden case của tôi gắn `expected_retrieval_ids` chuẩn, nhóm có thể **phân tách nguyên nhân**: điểm thấp là do retrieval (sai chunk) hay do generation (đúng chunk nhưng trả lời tệ). Riêng case `out_of_context` (`expected_retrieval_ids: []`) kiểm tra điều ngược lại — retriever nên trả về *rỗng* và agent phải từ chối thay vì hallucinate.

### 2.3. Position Bias & Cohen's Kappa (liên kết với module Judge)
*   **Position Bias** là xu hướng Judge ưu tiên phương án ở vị trí cố định (thường là A) bất kể chất lượng. Bộ dữ liệu của tôi hỗ trợ kiểm tra này bằng cách cho phép hoán đổi thứ tự câu trả lời ứng viên rồi so kết quả.
*   **Cohen's Kappa ($\kappa$)** đo độ đồng thuận giữa 2 Judge sau khi *loại trừ phần trùng khớp ngẫu nhiên*: $\kappa = (p_o - p_e)/(1 - p_e)$, với $\kappa \ge 0.6$ là đáng tin cậy. Dataset đa dạng độ khó (easy→hard) của tôi giúp $\kappa$ phản ánh đúng — nếu chỉ toàn câu dễ, hai Judge dễ đồng thuận giả tạo và $\kappa$ bị thổi phồng.

### 2.4. Trade-off Chi phí ↔ Chất lượng trong SDG
Sinh dữ liệu bằng `gpt-4o-mini` (qua `OPENAI_MODEL`) thay vì model lớn là lựa chọn có chủ đích: với QA *grounded* per-chunk, model nhỏ đủ chính xác mà rẻ hơn nhiều lần. Tôi để các hard/adversarial case là **curated thủ công** thay vì sinh tự động — vừa tiết kiệm token, vừa đảm bảo chất lượng tấn công thực sự khó (LLM tự sinh adversarial thường nhạt).

---

## 3. Giải quyết vấn đề (Problem Solving)

*   **Vấn đề 1 — Cạn hạn mức API (429 free-tier) khi sinh dữ liệu hàng loạt:** Gọi OpenAI song song cho 20 chunk dễ vượt rate limit. Tôi xử lý bằng **batch + retry/backoff lũy thừa** (`RETRY_BASE_DELAY * lần_thử`) trong `_call_llm_json`, để các request bị 429 tự chờ và thử lại thay vì làm hỏng cả run.

*   **Vấn đề 2 — Rủi ro dữ liệu rác làm sai lệch metric:** Câu hỏi sinh ra có thể trùng lặp hoặc trỏ tới ID không tồn tại, khiến Hit Rate/MRR vô nghĩa. Tôi viết `validate_and_finalize` để **dedupe** theo câu hỏi đã chuẩn hoá và **chặn** mọi case có `expected_retrieval_ids` không nằm trong corpus (ngoại lệ `out_of_context`), đảm bảo golden set "sạch" trước khi cả nhóm benchmark.

*   **Vấn đề 3 — Hợp đồng schema với cả nhóm:** Vì 4 module khác phụ thuộc cùng một file, mọi thay đổi field đều gây vỡ pipeline. Tôi **chốt schema sớm** trong [spec design](docs/superpowers/specs/2026-06-16-data-sdg-design.md) và đặt tên field khớp đúng với `runner.py` (`question`, `expected_answer`) và `retrieval_eval.py` (`expected_retrieval_ids`), giúp các bạn SV2–SV4 chạy được ngay mà không phải sửa code đọc dữ liệu.

*   **Bảo mật:** `.env` không commit (đã có trong `.gitignore`); tôi cung cấp `.env.example` để cả nhóm cấu hình API key an toàn.
