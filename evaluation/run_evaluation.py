"""Run a reproducible local evaluation against the running RAG API.

The script keeps deterministic retrieval metrics independent from optional RAGAS
versions, then uses RAGAS for answer faithfulness and relevancy when installed.
No hosted model or paid API is required; RAGAS receives the API's local answers
and evidence as its evaluation dataset.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    reference: str
    relevant_chunk_ids: tuple[str, ...]


def load_cases(path: Path) -> list[EvaluationCase]:
    """Load JSON cases with stable, explicit relevance labels."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        EvaluationCase(
            query=item["query"],
            reference=item.get("reference", ""),
            relevant_chunk_ids=tuple(item.get("relevant_chunk_ids", [])),
        )
        for item in payload
    ]


def retrieval_precision_recall(retrieved: list[str], relevant: tuple[str, ...]) -> tuple[float, float]:
    """Calculate precision/recall@k from chunk IDs and gold labels."""
    retrieved_set = set(retrieved)
    relevant_set = set(relevant)
    hits = len(retrieved_set & relevant_set)
    precision = hits / len(retrieved_set) if retrieved_set else 0.0
    recall = hits / len(relevant_set) if relevant_set else 0.0
    return precision, recall


def request_case(client: httpx.Client, base_url: str, case: EvaluationCase) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.post(
        f"{base_url.rstrip('/')}/verification/answer",
        json={"query": case.query, "top_k": 5},
    )
    latency_ms = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    payload = response.json()
    retrieved = [item["chunk_id"] for item in payload.get("evidence", [])]
    precision, recall = retrieval_precision_recall(retrieved, case.relevant_chunk_ids)
    return {
        "query": case.query,
        "reference": case.reference,
        "answer": payload.get("answer", ""),
        "contexts": [item.get("text", "") for item in payload.get("evidence", [])],
        "retrieved_chunk_ids": retrieved,
        "retrieval_precision": precision,
        "retrieval_recall": recall,
        "faithfulness": payload.get("grounding_score", 0.0),
        "answer_relevancy": payload.get("confidence_score", 0.0),
        "latency_ms": latency_ms,
    }


def ragas_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    """Evaluate faithfulness/relevancy with local RAGAS, if installed."""
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, faithfulness
    except ImportError:
        return {"ragas_faithfulness": 0.0, "ragas_answer_relevancy": 0.0, "ragas_available": 0.0}

    dataset = Dataset.from_list(
        [
            {
                "question": row["query"],
                "answer": row["answer"],
                "contexts": row["contexts"],
                "ground_truth": row["reference"],
            }
            for row in rows
        ]
    )
    try:
        result = evaluate(dataset=dataset, metrics=[faithfulness, answer_relevancy])
    except Exception as exc:  # RAGAS providers are optional and may need local model setup.
        return {
            "ragas_faithfulness": 0.0,
            "ragas_answer_relevancy": 0.0,
            "ragas_available": 0.0,
            "ragas_error": str(exc),
        }
    values = result.to_pandas().mean(numeric_only=True).to_dict()
    return {
        "ragas_faithfulness": float(values.get("faithfulness", 0.0)),
        "ragas_answer_relevancy": float(values.get("answer_relevancy", 0.0)),
        "ragas_available": 1.0,
    }


def summarize(rows: list[dict[str, Any]], ragas: dict[str, float]) -> dict[str, Any]:
    latencies = [row["latency_ms"] for row in rows]
    return {
        "cases": len(rows),
        "retrieval_precision": statistics.fmean(row["retrieval_precision"] for row in rows),
        "retrieval_recall": statistics.fmean(row["retrieval_recall"] for row in rows),
        "answer_faithfulness": statistics.fmean(row["faithfulness"] for row in rows),
        "answer_relevancy": statistics.fmean(row["answer_relevancy"] for row in rows),
        "latency_ms_mean": statistics.fmean(latencies),
        "latency_ms_p95": sorted(latencies)[max(0, int(len(latencies) * 0.95) - 1)],
        "throughput_requests_per_second": len(rows) / (sum(latencies) / 1000),
        **ragas,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evaluation/datasets/sample_questions.json"))
    parser.add_argument("--base-url", default="http://localhost:8000/api/v1")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("evaluation/reports/latest.json"))
    args = parser.parse_args()
    cases = load_cases(args.dataset)
    with httpx.Client(timeout=180) as client:
        if args.workers == 1:
            rows = [request_case(client, args.base_url, case) for case in cases]
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                rows = list(pool.map(lambda case: request_case(client, args.base_url, case), cases))
    report = {"summary": summarize(rows, ragas_scores(rows)), "results": rows, "configuration": vars(args) | {"dataset": str(args.dataset), "output": str(args.output)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
