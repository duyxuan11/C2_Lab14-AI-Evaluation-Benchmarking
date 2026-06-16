import asyncio
import json
import os
from typing import Any, Dict, List


class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(
        self,
        agent,
        dataset: List[Dict[str, Any]],
        top_k: int = 3,
        concurrency: int = 5,
    ) -> Dict[str, Any]:
        """
        Chạy retrieval-only (không generation) cho toàn bộ dataset để tính Hit Rate/MRR.

        Dataset cần có trường 'expected_retrieval_ids'; agent cần có method
        `retrieve(question, top_k)` trả về list [{"id": ..., "score": ..., "chunk": ...}].

        Case có 'expected_retrieval_ids' rỗng (vd out_of_context) không có ground truth
        dương nên bị loại khỏi trung bình Hit Rate/MRR, nhưng được đếm riêng vào
        'no_ground_truth_count' để báo cáo minh bạch.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _eval_one(case: Dict[str, Any]) -> Dict[str, Any]:
            expected = case.get("expected_retrieval_ids", [])
            async with sem:
                retrieved = await agent.retrieve(case["question"], top_k=top_k)
            retrieved_ids = [r["id"] for r in retrieved]
            has_ground_truth = len(expected) > 0
            return {
                "id": case.get("id"),
                "question": case["question"],
                "expected_retrieval_ids": expected,
                "retrieved_ids": retrieved_ids,
                "hit_rate": self.calculate_hit_rate(expected, retrieved_ids, top_k) if has_ground_truth else None,
                "mrr": self.calculate_mrr(expected, retrieved_ids) if has_ground_truth else None,
            }

        per_case = await asyncio.gather(*[_eval_one(case) for case in dataset])

        scored_cases = [c for c in per_case if c["hit_rate"] is not None]
        no_ground_truth_count = len(per_case) - len(scored_cases)

        avg_hit_rate = sum(c["hit_rate"] for c in scored_cases) / len(scored_cases) if scored_cases else 0.0
        avg_mrr = sum(c["mrr"] for c in scored_cases) / len(scored_cases) if scored_cases else 0.0

        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_mrr": avg_mrr,
            "evaluated_count": len(scored_cases),
            "no_ground_truth_count": no_ground_truth_count,
            "per_case": list(per_case),
        }


if __name__ == "__main__":
    from agent.main_agent import MainAgent

    GOLDEN_SET_PATH = "data/golden_set.jsonl"
    OUTPUT_PATH = "reports/retrieval_eval.json"

    async def main():
        if not os.path.exists(GOLDEN_SET_PATH):
            print(f"❌ Thiếu {GOLDEN_SET_PATH}. Hãy chạy 'python data/synthetic_gen.py' trước.")
            return

        with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
            dataset = [json.loads(line) for line in f if line.strip()]

        print(f"📚 Đã load {len(dataset)} test case. Đang chạy Retrieval Eval...")

        agent = MainAgent()
        evaluator = RetrievalEvaluator()
        result = await evaluator.evaluate_batch(agent, dataset)

        print(f"\n✅ Avg Hit Rate: {result['avg_hit_rate']:.3f}")
        print(f"✅ Avg MRR:      {result['avg_mrr']:.3f}")
        print(f"   Đánh giá trên {result['evaluated_count']} case có ground truth "
              f"({result['no_ground_truth_count']} case out_of_context bị loại khỏi trung bình)")

        os.makedirs("reports", exist_ok=True)
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n📄 Đã ghi chi tiết ra {OUTPUT_PATH}")

    asyncio.run(main())
