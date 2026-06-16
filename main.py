import asyncio
import json
import os
import time
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator


GOLDEN_SET_PATH = Path("data/golden_set.jsonl")
REPORTS_DIR = Path("reports")
SUMMARY_PATH = REPORTS_DIR / "summary.json"
RESULTS_PATH = REPORTS_DIR / "benchmark_results.json"

BATCH_SIZE = int(os.getenv("BENCHMARK_BATCH_SIZE", "10"))
PERFORMANCE_TARGET_SECONDS = 120.0


def load_dataset(path: Path = GOLDEN_SET_PATH) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Thieu {path}. Hay chay 'python data/synthetic_gen.py' truoc."
        )

    dataset: List[Dict[str, Any]] = []
    required_fields = {
        "question",
        "expected_answer",
        "context",
        "expected_retrieval_ids",
        "metadata",
    }

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            case = json.loads(line)
            missing = sorted(required_fields - set(case))
            if missing:
                raise ValueError(f"Case dong {line_no} thieu field: {missing}")
            dataset.append(case)

    if not dataset:
        raise ValueError(f"{path} rong.")

    return dataset


def average(values: List[Optional[float]]) -> float:
    valid = [float(v) for v in values if v is not None]
    return round(mean(valid), 4) if valid else 0.0


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return round(ordered[index], 4)


def text_overlap_score(answer: str, reference: str) -> float:
    answer_words = {w.lower() for w in answer.split() if len(w) > 2}
    reference_words = {w.lower() for w in reference.split() if len(w) > 2}
    if not reference_words:
        return 0.0
    return len(answer_words & reference_words) / len(reference_words)


class EvaluationAdapter:
    """Adapter de main.py co the tinh retrieval metrics theo tung response."""

    def __init__(self, top_k: int = 3):
        self.top_k = top_k
        self.retrieval = RetrievalEvaluator()

    async def score(self, case: Dict[str, Any], response: Dict[str, Any]) -> Dict[str, Any]:
        expected_ids = case.get("expected_retrieval_ids", [])
        retrieved_ids = response.get("retrieved_ids", [])
        has_ground_truth = bool(expected_ids)

        hit_rate = (
            self.retrieval.calculate_hit_rate(expected_ids, retrieved_ids, self.top_k)
            if has_ground_truth
            else None
        )
        mrr = (
            self.retrieval.calculate_mrr(expected_ids, retrieved_ids)
            if has_ground_truth
            else None
        )

        answer = response.get("answer", "")
        expected_answer = case.get("expected_answer", "")
        contexts = response.get("contexts", [])

        relevancy = min(
            1.0,
            0.7 * text_overlap_score(answer, expected_answer)
            + 0.3 * text_overlap_score(answer, case.get("question", "")),
        )
        faithfulness = 1.0 if contexts and answer else 0.0
        if case.get("metadata", {}).get("type") == "out_of_context":
            says_unknown = any(
                phrase in answer.lower()
                for phrase in [
                    "khong co thong tin",
                    "không có thông tin",
                    "khong tim thay",
                    "không tìm thấy",
                    "tai lieu khong",
                    "tài liệu không",
                ]
            )
            faithfulness = 1.0 if says_unknown else 0.5

        return {
            "faithfulness": round(faithfulness, 4),
            "relevancy": round(relevancy, 4),
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "top_k": self.top_k,
                "has_ground_truth": has_ground_truth,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
        }


class SafeJudge:
    """Wrapper de loi judge khong lam sap ca benchmark."""

    def __init__(self):
        self.judge = LLMJudge()

    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        try:
            result = await self.judge.evaluate_multi_judge(question, answer, ground_truth)
        except Exception as exc:  # noqa: BLE001 - benchmark can tiep tuc de tao report
            result = {
                "final_score": 1.0,
                "agreement_rate": 0.0,
                "individual_scores": {},
                "is_resolved": False,
                "mock_mode": True,
                "error": str(exc),
                "reasoning": f"Judge failed: {exc}",
            }

        result.setdefault("tokens_used", 0)
        result.setdefault("cost_usd", 0.0)
        result.setdefault(
            "mock_mode",
            not any(os.getenv(k) for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"]),
        )
        return result


async def query_agent(agent: MainAgent, case: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return await agent.query(case["question"], history=case.get("history"))
    except TypeError:
        return await agent.query(case["question"])


async def run_single_case(
    agent: MainAgent,
    evaluator: EvaluationAdapter,
    judge: SafeJudge,
    case: Dict[str, Any],
) -> Dict[str, Any]:
    started_at = time.perf_counter()
    agent_latency = 0.0

    try:
        response = await query_agent(agent, case)
        agent_latency = time.perf_counter() - started_at
        ragas_scores = await evaluator.score(case, response)
        judge_result = await judge.evaluate_multi_judge(
            case["question"],
            response.get("answer", ""),
            case["expected_answer"],
        )
        total_latency = time.perf_counter() - started_at
        status = "pass" if judge_result.get("final_score", 0.0) >= 3.0 else "fail"

        return {
            "id": case.get("id"),
            "test_case": case["question"],
            "expected_answer": case["expected_answer"],
            "expected_retrieval_ids": case.get("expected_retrieval_ids", []),
            "metadata": case.get("metadata", {}),
            "history": case.get("history", []),
            "agent_response": response.get("answer", ""),
            "agent": {
                "contexts": response.get("contexts", []),
                "retrieved_ids": response.get("retrieved_ids", []),
                "metadata": response.get("metadata", {}),
            },
            "latency": round(total_latency, 4),
            "agent_latency": round(agent_latency, 4),
            "judge_latency": round(max(total_latency - agent_latency, 0.0), 4),
            "ragas": ragas_scores,
            "judge": judge_result,
            "status": status,
        }
    except Exception as exc:  # noqa: BLE001 - ghi loi theo case thay vi dung ca batch
        total_latency = time.perf_counter() - started_at
        return {
            "id": case.get("id"),
            "test_case": case.get("question"),
            "expected_answer": case.get("expected_answer"),
            "expected_retrieval_ids": case.get("expected_retrieval_ids", []),
            "metadata": case.get("metadata", {}),
            "history": case.get("history", []),
            "agent_response": "",
            "agent": {"contexts": [], "retrieved_ids": [], "metadata": {}},
            "latency": round(total_latency, 4),
            "agent_latency": round(agent_latency, 4),
            "judge_latency": 0.0,
            "ragas": {
                "faithfulness": 0.0,
                "relevancy": 0.0,
                "retrieval": {
                    "hit_rate": 0.0 if case.get("expected_retrieval_ids") else None,
                    "mrr": 0.0 if case.get("expected_retrieval_ids") else None,
                    "top_k": evaluator.top_k,
                    "has_ground_truth": bool(case.get("expected_retrieval_ids")),
                    "expected_ids": case.get("expected_retrieval_ids", []),
                    "retrieved_ids": [],
                },
            },
            "judge": {
                "final_score": 1.0,
                "agreement_rate": 0.0,
                "individual_scores": {},
                "tokens_used": 0,
                "cost_usd": 0.0,
                "mock_mode": True,
                "error": str(exc),
                "reasoning": f"Case failed before judging: {exc}",
            },
            "status": "fail",
            "error": str(exc),
        }


async def run_all_cases(
    dataset: List[Dict[str, Any]],
    agent: MainAgent,
    evaluator: EvaluationAdapter,
    judge: SafeJudge,
    batch_size: int = BATCH_SIZE,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        tasks = [run_single_case(agent, evaluator, judge, case) for case in batch]
        batch_results = await asyncio.gather(*tasks)
        results.extend(batch_results)
        print(f"   Da chay {len(results)}/{len(dataset)} cases")
    return results


def extract_agent_tokens(result: Dict[str, Any]) -> int:
    tokens = result.get("agent", {}).get("metadata", {}).get("tokens_used")
    return int(tokens or 0)


def extract_agent_cost(result: Dict[str, Any]) -> float:
    cost = result.get("agent", {}).get("metadata", {}).get("cost_usd")
    return float(cost or 0.0)


def build_summary(
    agent_version: str,
    results: List[Dict[str, Any]],
    runtime_sec: float,
) -> Dict[str, Any]:
    total = len(results)
    status_counts = Counter(r.get("status") for r in results)
    failure_types = Counter(
        r.get("metadata", {}).get("type", "unknown")
        for r in results
        if r.get("status") == "fail"
    )

    agent_tokens = sum(extract_agent_tokens(r) for r in results)
    judge_tokens = sum(int(r.get("judge", {}).get("tokens_used") or 0) for r in results)
    agent_cost = sum(extract_agent_cost(r) for r in results)
    judge_cost = sum(float(r.get("judge", {}).get("cost_usd") or 0.0) for r in results)
    total_cost = agent_cost + judge_cost

    agent_cost_complete = all(
        "cost_usd" in r.get("agent", {}).get("metadata", {}) for r in results
    )
    judge_cost_complete = all("cost_usd" in r.get("judge", {}) for r in results)

    metrics = {
        "avg_score": average([r.get("judge", {}).get("final_score") for r in results]),
        "pass_rate": round(status_counts.get("pass", 0) / total, 4) if total else 0.0,
        "hit_rate": average(
            [r.get("ragas", {}).get("retrieval", {}).get("hit_rate") for r in results]
        ),
        "mrr": average(
            [r.get("ragas", {}).get("retrieval", {}).get("mrr") for r in results]
        ),
        "faithfulness": average([r.get("ragas", {}).get("faithfulness") for r in results]),
        "relevancy": average([r.get("ragas", {}).get("relevancy") for r in results]),
        "agreement_rate": average(
            [r.get("judge", {}).get("agreement_rate") for r in results]
        ),
        "avg_latency_sec": average([r.get("latency") for r in results]),
        "p95_latency_sec": percentile([float(r.get("latency", 0.0)) for r in results], 0.95),
        "avg_agent_latency_sec": average([r.get("agent_latency") for r in results]),
        "avg_judge_latency_sec": average([r.get("judge_latency") for r in results]),
        "total_tokens": agent_tokens + judge_tokens,
        "agent_tokens": agent_tokens,
        "judge_tokens": judge_tokens,
        "total_cost_usd": round(total_cost, 6),
        "agent_cost_usd": round(agent_cost, 6),
        "judge_cost_usd": round(judge_cost, 6),
        "cost_per_case_usd": round(total_cost / total, 6) if total else 0.0,
        "cost_tracking_complete": agent_cost_complete and judge_cost_complete,
        "total_runtime_sec": round(runtime_sec, 4),
        "performance_target_met": total >= 50 and runtime_sec <= PERFORMANCE_TARGET_SECONDS,
        "errors": sum(1 for r in results if r.get("error")),
        "mock_judge_cases": sum(1 for r in results if r.get("judge", {}).get("mock_mode")),
    }

    return {
        "metadata": {
            "version": agent_version,
            "total": total,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "batch_size": BATCH_SIZE,
            "performance_target_seconds": PERFORMANCE_TARGET_SECONDS,
        },
        "metrics": metrics,
        "failure_clusters": dict(failure_types),
        "status_counts": dict(status_counts),
    }


def metric_delta(v1: Dict[str, Any], v2: Dict[str, Any]) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for key, v2_value in v2.items():
        v1_value = v1.get(key)
        if isinstance(v1_value, (int, float)) and isinstance(v2_value, (int, float)):
            deltas[key] = round(float(v2_value) - float(v1_value), 6)
    return deltas


def regression_gate(
    v1_summary: Dict[str, Any],
    v2_summary: Dict[str, Any],
) -> Dict[str, Any]:
    v1 = v1_summary["metrics"]
    v2 = v2_summary["metrics"]
    reasons: List[str] = []

    if v2["avg_score"] < v1["avg_score"] - 0.1:
        reasons.append("Average judge score dropped by more than 0.10.")
    if v2["hit_rate"] < v1["hit_rate"] - 0.05:
        reasons.append("Retrieval Hit Rate dropped by more than 5 percentage points.")
    if v2["mrr"] < v1["mrr"] - 0.05:
        reasons.append("Retrieval MRR dropped by more than 5 percentage points.")
    if v2["agreement_rate"] < 0.65:
        reasons.append("Multi-judge agreement rate is below 0.65.")
    if v1["avg_latency_sec"] > 0 and v2["avg_latency_sec"] > v1["avg_latency_sec"] * 1.3:
        reasons.append("Average latency increased by more than 30%.")
    if (
        v1["cost_per_case_usd"] > 0
        and v2["cost_per_case_usd"] > v1["cost_per_case_usd"] * 1.3
        and v2["avg_score"] <= v1["avg_score"] + 0.1
    ):
        reasons.append("Cost per case increased by more than 30% without quality gain.")
    if not v2["performance_target_met"]:
        reasons.append("Performance target was not met: 50+ cases must run within 2 minutes.")

    return {
        "decision": "BLOCK" if reasons else "APPROVE",
        "reasons": reasons or ["No quality, retrieval, cost, or latency regression detected."],
        "thresholds": {
            "max_avg_score_drop": 0.1,
            "max_hit_rate_drop": 0.05,
            "max_mrr_drop": 0.05,
            "min_agreement_rate": 0.65,
            "max_latency_increase_ratio": 1.3,
            "max_cost_increase_ratio": 1.3,
            "performance_target_seconds": PERFORMANCE_TARGET_SECONDS,
        },
    }


def cost_optimization_plan() -> List[str]:
    return [
        "Cache judge results by a hash of question, answer, and expected_answer.",
        "Use a cheaper judge model for easy fact cases and reserve two-judge consensus for hard or borderline cases.",
        "Run the second judge only when the first score is below 4.0 or the case type is adversarial, ambiguous, conflicting, or multi_turn.",
        "Cap judge max_tokens and require compact JSON reasoning to reduce output tokens.",
        "Reuse retrieval embedding cache across V1/V2 benchmark runs.",
    ]


async def run_benchmark_with_results(agent_version: str) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    print(f"Khoi dong Benchmark cho {agent_version}...")
    dataset = load_dataset()
    agent = MainAgent()
    evaluator = EvaluationAdapter()
    judge = SafeJudge()

    started_at = time.perf_counter()
    results = await run_all_cases(dataset, agent, evaluator, judge, BATCH_SIZE)
    runtime_sec = time.perf_counter() - started_at
    summary = build_summary(agent_version, results, runtime_sec)
    return results, summary


async def main():
    try:
        v1_results, v1_summary = await run_benchmark_with_results("Agent_V1_Base")
        v2_results, v2_summary = await run_benchmark_with_results("Agent_V2_Optimized")
    except Exception as exc:  # noqa: BLE001
        print(f"Khong the chay Benchmark: {exc}")
        return

    delta = metric_delta(v1_summary["metrics"], v2_summary["metrics"])
    gate = regression_gate(v1_summary, v2_summary)

    final_summary = {
        **v2_summary,
        "regression": {
            "baseline": v1_summary,
            "candidate": v2_summary,
            "delta": delta,
        },
        "release_gate": gate,
        "cost_optimization_plan": cost_optimization_plan(),
    }

    benchmark_results = {
        "baseline_version": v1_summary["metadata"]["version"],
        "candidate_version": v2_summary["metadata"]["version"],
        "baseline_results": v1_results,
        "candidate_results": v2_results,
    }

    REPORTS_DIR.mkdir(exist_ok=True)
    with SUMMARY_PATH.open("w", encoding="utf-8") as f:
        json.dump(final_summary, f, ensure_ascii=False, indent=2)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, ensure_ascii=False, indent=2)

    print("\n--- KET QUA SO SANH (REGRESSION) ---")
    print(f"V1 Score: {v1_summary['metrics']['avg_score']:.2f}")
    print(f"V2 Score: {v2_summary['metrics']['avg_score']:.2f}")
    print(f"Delta Score: {delta.get('avg_score', 0.0):+.2f}")
    print(f"Hit Rate V2: {v2_summary['metrics']['hit_rate'] * 100:.1f}%")
    print(f"MRR V2: {v2_summary['metrics']['mrr']:.3f}")
    print(f"Runtime V2: {v2_summary['metrics']['total_runtime_sec']:.2f}s")
    print(f"Cost/Case V2: ${v2_summary['metrics']['cost_per_case_usd']:.6f}")
    print(f"QUYET DINH: {gate['decision']}")
    for reason in gate["reasons"]:
        print(f"- {reason}")
    print(f"\nDa ghi {SUMMARY_PATH} va {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
