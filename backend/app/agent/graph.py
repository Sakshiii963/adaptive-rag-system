"""LangGraph state machine for bounded adaptive retrieval."""

from datetime import UTC, datetime
from time import perf_counter

from langgraph.graph import END, START, StateGraph

from backend.app.agent.confidence import RetrievalConfidenceEvaluator
from backend.app.agent.rewrite import HeuristicQueryRewriter, QueryRewriter
from backend.app.agent.state import AgentState, TraceStep
from backend.app.core.logging import get_logger
from backend.app.domain.entities import RetrievalFilters
from backend.app.services.reranking import RerankingService
from backend.app.services.retrieval import HybridRetrievalEngine

logger = get_logger(__name__)


class AdaptiveRetrievalAgent:
    """Orchestrates retrieval, reranking, evaluation, and bounded rewrite retries only."""

    def __init__(
        self,
        retrieval_engine: HybridRetrievalEngine,
        reranking_service: RerankingService,
        confidence_evaluator: RetrievalConfidenceEvaluator | None = None,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        self.retrieval_engine = retrieval_engine
        self.reranking_service = reranking_service
        self.confidence_evaluator = confidence_evaluator or RetrievalConfidenceEvaluator()
        self.query_rewriter = query_rewriter or HeuristicQueryRewriter()
        self.graph = self._build_graph()

    def run(
        self,
        query: str,
        filters: RetrievalFilters,
        top_k: int,
        confidence_threshold: float,
        max_retries: int,
    ) -> AgentState:
        """Execute the graph and return evidence, trace, confidence, and reasoning only."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Query must not be empty.")
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1.")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative.")
        initial: AgentState = {
            "original_query": cleaned_query,
            "current_query": cleaned_query,
            "rewritten_query": None,
            "attempt": 0,
            "max_retries": max_retries,
            "top_k": top_k,
            "confidence_threshold": confidence_threshold,
            "filters": filters,
            "status": "running",
            "stop_reason": None,
            "seen_queries": [cleaned_query],
            "trace": [],
            "reasoning_steps": [],
        }
        # Each attempt can traverse retrieval, reranking, evaluation, rewrite, and retry.
        return self.graph.invoke(initial, config={"recursion_limit": 10 + (max_retries * 5)})

    def _build_graph(self):
        """Build the explicit START → retrieve → rerank → decide loop once per agent."""
        builder = StateGraph(AgentState)
        builder.add_node("hybrid_retrieval", self._hybrid_retrieval_node)
        builder.add_node("cross_encoder_reranking", self._reranking_node)
        builder.add_node("evaluate_confidence", self._evaluate_confidence_node)
        builder.add_node("rewrite_query", self._rewrite_query_node)
        builder.add_node("retry_retrieval", self._retry_retrieval_node)
        builder.add_edge(START, "hybrid_retrieval")
        builder.add_edge("hybrid_retrieval", "cross_encoder_reranking")
        builder.add_edge("cross_encoder_reranking", "evaluate_confidence")
        builder.add_conditional_edges(
            "evaluate_confidence",
            self._route_decision,
            {
                "return_evidence": END,
                "rewrite_query": "rewrite_query",
                "insufficient_evidence": END,
            },
        )
        builder.add_conditional_edges(
            "rewrite_query",
            self._route_rewrite,
            {"retry_retrieval": "retry_retrieval", "insufficient_evidence": END},
        )
        builder.add_edge("retry_retrieval", "hybrid_retrieval")
        return builder.compile()

    def _hybrid_retrieval_node(self, state: AgentState) -> dict:
        started_at = perf_counter()
        result = self.retrieval_engine.retrieve(
            state["current_query"], state["filters"], state["top_k"]
        )
        return self._record(
            state,
            "hybrid_retrieval",
            started_at,
            {
                "query": state["current_query"],
                "result_count": len(result.candidates),
                "candidates": [
                    {
                        "chunk_id": item.chunk.id,
                        "document_id": item.chunk.document_id,
                        "filename": item.chunk.filename,
                        "page": item.chunk.page_number,
                        "semantic_score": item.semantic_score,
                        "keyword_score": item.keyword_score,
                        "rrf_score": item.rrf_score,
                        "normalized_score": item.normalized_score,
                    }
                    for item in result.candidates
                ],
            },
            retrieval_result=result,
            reasoning=f"Attempt {state['attempt']}: hybrid retrieval returned {len(result.candidates)} candidates.",
        )

    def _reranking_node(self, state: AgentState) -> dict:
        started_at = perf_counter()
        result = self.reranking_service.rerank(
            state["current_query"], state["retrieval_result"], state["top_k"]
        )
        return self._record(
            state,
            "cross_encoder_reranking",
            started_at,
            {
                "query": state["current_query"],
                "result_count": len(result.candidates),
                "candidates": [
                    {
                        "chunk_id": item.candidate.chunk.id,
                        "rank": item.rank,
                        "reranker_score": item.reranker_score,
                        "normalized_reranker_score": item.normalized_reranker_score,
                    }
                    for item in result.candidates
                ],
            },
            reranking_result=result,
            reasoning=f"Cross-encoder reranked {len(result.candidates)} candidates.",
        )

    def _evaluate_confidence_node(self, state: AgentState) -> dict:
        started_at = perf_counter()
        confidence = self.confidence_evaluator.evaluate(state["reranking_result"])
        accepted = confidence >= state["confidence_threshold"]
        exhausted = (not accepted) and state["attempt"] >= state["max_retries"]
        details = {
            "confidence": confidence,
            "threshold": state["confidence_threshold"],
            "accepted": accepted,
            "attempt": state["attempt"],
        }
        return self._record(
            state,
            "evaluate_confidence",
            started_at,
            details,
            confidence=confidence,
            status="evidence"
            if accepted
            else ("insufficient_evidence" if exhausted else "running"),
            stop_reason="retry_budget_exhausted" if exhausted else None,
            reasoning=(
                f"Confidence {confidence:.3f} meets threshold."
                if accepted
                else f"Confidence {confidence:.3f} is below threshold; evaluating retry budget."
            ),
        )

    def _retry_retrieval_node(self, state: AgentState) -> dict:
        """Record the explicit bounded retry transition before starting retrieval again."""
        started_at = perf_counter()
        return self._record(
            state,
            "retry_retrieval",
            started_at,
            {"attempt": state["attempt"], "query": state["current_query"]},
            reasoning=f"Starting retrieval retry {state['attempt']} within the configured budget.",
        )

    def _rewrite_query_node(self, state: AgentState) -> dict:
        started_at = perf_counter()
        next_attempt = state["attempt"] + 1
        rewritten = self.query_rewriter.rewrite(state["current_query"], next_attempt).strip()
        blocked = not rewritten or rewritten in state["seen_queries"]
        if blocked:
            return self._record(
                state,
                "rewrite_query",
                started_at,
                {"blocked": True, "query": rewritten, "attempt": next_attempt},
                status="insufficient_evidence",
                stop_reason="query_rewrite_repeated",
                next_route="insufficient_evidence",
                reasoning="Query rewrite repeated a prior query; loop prevented.",
            )
        return self._record(
            state,
            "rewrite_query",
            started_at,
            {"blocked": False, "query": rewritten, "attempt": next_attempt},
            current_query=rewritten,
            rewritten_query=rewritten,
            attempt=next_attempt,
            seen_queries=[*state["seen_queries"], rewritten],
            next_route="retry_retrieval",
            reasoning=f"Rewrote the query for retry {next_attempt}: {rewritten}",
        )

    @staticmethod
    def _route_decision(state: AgentState) -> str:
        if state.get("status") == "evidence":
            return "return_evidence"
        if state["attempt"] >= state["max_retries"]:
            return "insufficient_evidence"
        return "rewrite_query"

    @staticmethod
    def _route_rewrite(state: AgentState) -> str:
        return state.get("next_route", "insufficient_evidence")

    @staticmethod
    def _record(state: AgentState, node: str, started_at: float, details: dict, **updates) -> dict:
        """Append a structured trace event and an operator-readable reasoning step."""
        reasoning = updates.pop("reasoning", None)
        trace: TraceStep = {
            "node": node,
            "attempt": state["attempt"],
            "timestamp": datetime.now(UTC).isoformat(),
            "duration_ms": round((perf_counter() - started_at) * 1000, 2),
            "details": details,
        }
        updates["trace"] = [*state["trace"], trace]
        updates["reasoning_steps"] = [
            *state["reasoning_steps"],
            *([reasoning] if reasoning else []),
        ]
        logger.info(
            "agent_node_completed",
            extra={"node": node, "attempt": state["attempt"], **details},
        )
        return updates
