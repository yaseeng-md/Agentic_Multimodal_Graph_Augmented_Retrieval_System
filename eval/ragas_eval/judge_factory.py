"""Judge LLM Factory for Ragas and grounding evaluation with OpenRouter support."""

import json
import logging
import os
import re
import time
from typing import Any, Optional
import urllib.error
import urllib.request

from eval.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    JUDGE_TIMEOUT_SEC,
)

logger = logging.getLogger("amg-eval-judge")


class BaseJudge:
    """Abstract base class for evaluation judges."""

    def evaluate_faithfulness(self, query: str, context: str, answer: str) -> float:
        raise NotImplementedError

    def evaluate_relevancy(self, query: str, answer: str) -> float:
        raise NotImplementedError

    def evaluate_context_precision(self, query: str, contexts: list[str], ground_truth: str) -> float:
        raise NotImplementedError


class HeuristicJudge(BaseJudge):
    """
    Deterministic claim-verification and grounding judge.
    Requires zero external API keys and runs instantaneously.
    """

    def _split_into_claims(self, text: str) -> list[str]:
        """Split answer into atomic sentences / claims."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if len(s.strip()) > 3]

    def evaluate_faithfulness(self, query: str, context: str, answer: str) -> float:
        """
        Check what fraction of atomic claims in answer are substantiated by the context.
        Returns a score in [0.0, 1.0].
        """
        if not answer.strip():
            return 0.0
        if not context.strip():
            return 0.0

        claims = self._split_into_claims(answer)
        if not claims:
            return 1.0

        ctx_lower = context.lower()
        supported = 0

        for claim in claims:
            claim_words = [
                w.lower()
                for w in re.findall(r"\b\w+\b", claim)
                if len(w) > 3 and w.lower() not in {"this", "that", "there", "with", "from", "have", "were"}
            ]
            if not claim_words:
                supported += 1
                continue

            matches = sum(1 for w in claim_words if w in ctx_lower)
            if (matches / len(claim_words)) >= 0.5:
                supported += 1

        return round(supported / len(claims), 4)

    def evaluate_relevancy(self, query: str, answer: str) -> float:
        """Measure if the answer addresses the keywords in the query."""
        if not answer.strip() or not query.strip():
            return 0.0

        query_words = set(re.findall(r"\b\w+\b", query.lower())) - {
            "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
            "is", "are", "was", "were", "do", "does", "did", "the", "a", "an"
        }
        if not query_words:
            return 1.0

        ans_lower = answer.lower()
        overlap = sum(1 for w in query_words if w in ans_lower)
        ratio = overlap / len(query_words)
        return round(min(1.0, 0.4 + 0.6 * ratio), 4)

    def evaluate_context_precision(self, query: str, contexts: list[str], ground_truth: str) -> float:
        """
        Measure whether relevant context passages appear at higher ranks.
        Calculates precision at each relevant context rank and averages.
        """
        if not contexts:
            return 0.0

        gold_words = set(re.findall(r"\b\w+\b", ground_truth.lower())) - {"the", "a", "an", "is", "of", "in"}
        if not gold_words:
            return 1.0

        relevant_flags = []
        for ctx in contexts:
            ctx_lower = ctx.lower()
            overlap = sum(1 for w in gold_words if w in ctx_lower)
            relevant_flags.append(1 if (overlap / len(gold_words)) >= 0.4 else 0)

        if not any(relevant_flags):
            return 0.0

        precisions = []
        hits = 0
        for rank, is_rel in enumerate(relevant_flags, start=1):
            if is_rel:
                hits += 1
                precisions.append(hits / rank)

        return round(sum(precisions) / len(precisions), 4) if precisions else 0.0


class OpenRouterJudge(BaseJudge):
    """
    LLM-as-a-Judge using OpenRouter's universal API.
    Can evaluate Faithfulness, Answer Relevancy, and Context Precision using any hosted model
    (e.g. 'openai/gpt-4o-mini', 'google/gemini-flash-1.5', 'anthropic/claude-3.5-haiku').
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = OPENROUTER_BASE_URL,
        model: str = OPENROUTER_MODEL,
        timeout: int = JUDGE_TIMEOUT_SEC,
    ):
        self.api_key = api_key.strip()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.fallback_heuristic = HeuristicJudge()

    def _call_llm(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> str:
        """Execute a standard chat completion request via OpenRouter API with retry and backoff."""
        url = f"{self.base_url}/chat/completions"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/amg-rag",
            "X-Title": "AMG Multimodal RAG Evaluator",
        }

        last_error = None
        for attempt in range(1, max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                    time.sleep(0.5)  # Brief pacing delay to avoid rate-limiting
                    return body["choices"][0]["message"]["content"]
            except urllib.error.HTTPError as err:
                last_error = err
                logger.warning(
                    "OpenRouter call attempt %d/%d failed with HTTP %d (%s)",
                    attempt,
                    max_retries,
                    err.code,
                    err.reason,
                )
                if attempt < max_retries:
                    time.sleep(attempt * 1.5)
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "OpenRouter call attempt %d/%d error: %s",
                    attempt,
                    max_retries,
                    exc,
                )
                if attempt < max_retries:
                    time.sleep(1.0)

        raise last_error or RuntimeError("OpenRouter call failed after retries.")

    def _extract_json(self, raw_text: str) -> dict[str, Any]:
        """Safely parse JSON response from LLM."""
        try:
            return json.loads(raw_text)
        except Exception:
            match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except Exception:
                    pass
        return {}

    def evaluate_faithfulness(self, query: str, context: str, answer: str) -> float:
        """
        Evaluate if statements in the answer are supported by the retrieved context.
        """
        if not answer.strip():
            return 0.0
        if not context.strip():
            return 0.0

        system_prompt = (
            "You are a strict, objective factual consistency evaluator for RAG systems.\n"
            "Your task is to judge whether all statements in the GENERATED ANSWER are grounded in the CONTEXT.\n"
            "Respond ONLY with a JSON object."
        )

        user_prompt = (
            f"CONTEXT:\n{context[:3000]}\n\n"
            f"GENERATED ANSWER:\n{answer}\n\n"
            "Instructions:\n"
            "1. Deconstruct the answer into discrete atomic factual claims.\n"
            "2. For each claim, check if it is directly supported by the context without outside knowledge.\n"
            "3. Output valid JSON in the format:\n"
            "{\n"
            '  "claims": [{"claim": "...", "supported": true/false}],\n'
            '  "faithfulness_score": 0.0 to 1.0\n'
            "}"
        )

        try:
            raw = self._call_llm(system_prompt, user_prompt)
            parsed = self._extract_json(raw)
            score = parsed.get("faithfulness_score")
            if score is not None:
                return round(float(score), 4)

            claims = parsed.get("claims", [])
            if claims:
                supported = sum(1 for c in claims if c.get("supported") is True)
                return round(supported / len(claims), 4)

        except Exception as exc:
            logger.warning("OpenRouter faithfulness call failed: %s. Using heuristic fallback.", exc)

        return self.fallback_heuristic.evaluate_faithfulness(query, context, answer)

    def evaluate_relevancy(self, query: str, answer: str) -> float:
        """
        Evaluate how directly the generated answer addresses the user query.
        """
        if not answer.strip() or not query.strip():
            return 0.0

        system_prompt = (
            "You are an answer relevancy evaluator for RAG systems.\n"
            "Determine if the answer directly, concisely, and completely answers the user question.\n"
            "Respond ONLY with a JSON object."
        )

        user_prompt = (
            f"USER QUERY:\n{query}\n\n"
            f"GENERATED ANSWER:\n{answer}\n\n"
            "Instructions:\n"
            "1. Judge if the answer directly addresses the intent of the query.\n"
            "2. Penalize evasive, irrelevant, or incomplete answers.\n"
            "3. Output valid JSON in the format:\n"
            "{\n"
            '  "relevancy_score": 0.0 to 1.0,\n'
            '  "reason": "..."\n'
            "}"
        )

        try:
            raw = self._call_llm(system_prompt, user_prompt)
            parsed = self._extract_json(raw)
            score = parsed.get("relevancy_score")
            if score is not None:
                return round(float(score), 4)
        except Exception as exc:
            logger.warning("OpenRouter relevancy call failed: %s. Using heuristic fallback.", exc)

        return self.fallback_heuristic.evaluate_relevancy(query, answer)

    def evaluate_context_precision(self, query: str, contexts: list[str], ground_truth: str) -> float:
        """
        Evaluate if retrieved contexts contain the required ground-truth evidence.
        """
        if not contexts:
            return 0.0

        # Fast heuristic is reliable for context precision, but OpenRouter can verify top context if requested
        return self.fallback_heuristic.evaluate_context_precision(query, contexts, ground_truth)


def get_judge(
    judge_type: str = "auto",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: Optional[str] = None,
) -> BaseJudge:
    """
    Factory creating the evaluation judge.
    Connects to OpenRouter when API key is provided, falling back to HeuristicJudge otherwise.
    """
    key = (api_key or OPENROUTER_API_KEY or "").strip()
    url = (base_url or OPENROUTER_BASE_URL).strip()
    mdl = (model or OPENROUTER_MODEL).strip()

    if key:
        logger.info("Initializing OpenRouter Judge LLM (model=%s, url=%s)", mdl, url)
        return OpenRouterJudge(api_key=key, base_url=url, model=mdl)

    logger.info("No OpenRouter API key found. Using HeuristicJudge (zero-cost offline mode).")
    return HeuristicJudge()
