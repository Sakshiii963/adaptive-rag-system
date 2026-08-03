# Evaluation guide

The evaluation harness in `evaluation/run_evaluation.py` is intentionally local and reproducible. It calls the existing verification endpoint, records evidence IDs and timing, calculates retrieval precision/recall from explicit gold labels, and records the endpoint's grounding/confidence proxies. If the optional `ragas` and `datasets` dependencies are installed, it additionally computes RAGAS faithfulness and answer relevancy over the returned answer/context pairs.

## Dataset format

```json
{
  "query": "What is the retention period?",
  "reference": "The policy retains records for seven years.",
  "relevant_chunk_ids": ["document-id:page:chunk"]
}
```

Use chunk IDs from retrieval responses for gold labels. Empty labels are useful as smoke tests but produce zero retrieval scores by design.

## Run

```bash
pip install -e '.[evaluation]'
python evaluation/run_evaluation.py --workers 2
```

The JSON report includes per-case results and aggregate retrieval precision, retrieval recall, answer faithfulness, answer relevancy, mean/p95 latency, and requests per second. `--workers` exercises concurrent request throughput; keep it within the capacity of the local machine.

RAGAS integrations are optional because model-backed metrics can require additional local model configuration. The core benchmark never calls a paid API and remains runnable without RAGAS installed.
