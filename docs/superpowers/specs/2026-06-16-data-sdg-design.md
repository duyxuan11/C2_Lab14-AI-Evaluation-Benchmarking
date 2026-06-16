# Spec: Data & SDG Module (SV1) — Lab 14

**Ngày:** 2026-06-16
**Phụ trách:** SV1 (Data & SDG)
**Mục tiêu rubric:** Dataset & SDG (10đ) + nền tảng cho Retrieval Eval (10đ)

## Bối cảnh
Repo chưa có corpus tài liệu và `data/synthetic_gen.py` chỉ là placeholder trả về 1 case
giả. Cả nhóm (runner, retrieval_eval, judge, regression) đều phụ thuộc vào
`data/golden_set.jsonl`. SV1 chịu trách nhiệm tạo corpus nguồn + sinh golden dataset.

## Sản phẩm bàn giao

### 1. Corpus nguồn — `data/knowledge_base.json`
- ~20 chunk tài liệu kỹ thuật sản phẩm (tiếng Việt): authentication, API key, rate limit,
  webhook, SDK, error codes, pagination, versioning, v.v.
- Mỗi chunk: `{ "id", "category", "title", "content" }` với **ID ổn định** (vd `DOC-AUTH-001`).
- Là ground truth dùng chung: SV3 build retriever trên đúng corpus này để tính Hit Rate/MRR.

### 2. SDG pipeline — `data/synthetic_gen.py`
- Load corpus → gọi OpenAI (JSON mode) sinh **grounded QA** cho từng chunk, ghi
  `expected_retrieval_ids` = ID chunk nguồn.
- Bổ sung **hard/adversarial cases** curated (grounded vào ID chunk thật) theo
  `HARD_CASES_GUIDE.md`: prompt injection, goal hijacking, out-of-context, ambiguous,
  conflicting, multi-turn.
- Xuất **≥50 cases** ra `data/golden_set.jsonl`. Validate schema + dedupe + retry khi API lỗi.

## Schema golden case (hợp đồng với cả nhóm)
```json
{
  "id": "GS-001",
  "question": "string",
  "expected_answer": "string",
  "context": "string (đoạn text nguồn)",
  "expected_retrieval_ids": ["DOC-AUTH-001"],
  "metadata": {
    "difficulty": "easy|medium|hard",
    "type": "fact|multi_hop|adversarial|out_of_context|ambiguous|conflicting|multi_turn",
    "category": "auth"
  }
}
```
- Khớp `runner.py` (`question`, `expected_answer`) và `retrieval_eval.py` (`expected_retrieval_ids`).
- `out_of_context` → `expected_retrieval_ids: []` và `expected_answer` là câu từ chối/"không có trong tài liệu".

## Phân bổ (~55 cases)
| Loại | SL | Nguồn sinh |
|------|----|-----------|
| fact / grounded | ~30 | OpenAI per-chunk |
| multi_hop | ~6 | OpenAI multi-chunk |
| adversarial (injection/hijacking) | ~6 | Curated |
| out_of_context | ~5 | Curated |
| ambiguous / conflicting / multi_turn | ~6 | Curated |

## Kỹ thuật
- `python-dotenv` load `.env`; model qua env `OPENAI_MODEL` (mặc định `gpt-4o-mini`).
- `AsyncOpenAI` + `asyncio.gather` theo batch; `tqdm` progress; retry/backoff.
- Validate mỗi case: đủ field, `expected_retrieval_ids` tồn tại trong corpus (trừ out_of_context).
- `.env` KHÔNG commit (đã có trong `.gitignore`); cung cấp `.env.example`.
