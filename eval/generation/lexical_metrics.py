"""Deterministic generation evaluation metrics (Exact Match, Token F1, ROUGE-L, BLEU-4)."""

from collections import Counter
import math
import re
import string
from typing import Sequence, Union


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, ground_truth: str) -> float:
    """Return 1.0 if normalized strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score between prediction and ground truth."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def _lcs_length(s1: list[str], s2: list[str]) -> int:
    """Compute length of Longest Common Subsequence between two token sequences."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]


def rouge_l_score(prediction: str, ground_truth: str) -> float:
    """Compute ROUGE-L F1 score based on Longest Common Subsequence."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    lcs = _lcs_length(pred_tokens, gold_tokens)
    if lcs == 0:
        return 0.0

    r_lcs = lcs / len(gold_tokens)
    p_lcs = lcs / len(pred_tokens)
    return (2 * p_lcs * r_lcs) / (p_lcs + r_lcs)


def bleu_score(prediction: str, ground_truth: str, max_n: int = 4) -> float:
    """Compute sentence-level BLEU-4 with brevity penalty and smoothing."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)

    effective_n = min(max_n, len(pred_tokens), len(gold_tokens))
    if effective_n < 1:
        return 0.0

    # Brevity penalty
    c = len(pred_tokens)
    r = len(gold_tokens)
    bp = 1.0 if c >= r else math.exp(1 - r / c)

    precisions = []
    for n in range(1, effective_n + 1):
        pred_ngrams = Counter(tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1))
        gold_ngrams = Counter(tuple(gold_tokens[i : i + n]) for i in range(len(gold_tokens) - n + 1))

        clipped_matches = sum((pred_ngrams & gold_ngrams).values())
        total_ngrams = sum(pred_ngrams.values())

        if clipped_matches == 0:
            # Smoothing for zero matches on higher-order n-grams
            precisions.append(1.0 / (2 ** n * max(total_ngrams, 1)))
        else:
            precisions.append(clipped_matches / total_ngrams)

    # Geometric mean
    log_sum = sum(math.log(max(p, 1e-9)) for p in precisions) / effective_n
    return bp * math.exp(log_sum)


def compute_generation_lexical_metrics(
    predictions: Sequence[str],
    ground_truths: Sequence[Union[str, Sequence[str]]],
) -> dict[str, float]:
    """
    Compute aggregate EM, F1, ROUGE-L, and BLEU-4.
    Supports single ground truth string or list of alternative valid answers.
    """
    if not predictions or not ground_truths:
        return {}

    em_scores: list[float] = []
    f1_scores: list[float] = []
    rouge_scores: list[float] = []
    bleu_scores: list[float] = []

    for pred, gold in zip(predictions, ground_truths):
        candidates = [gold] if isinstance(gold, str) else list(gold)
        if not candidates:
            candidates = [""]

        # Max over multiple acceptable ground truth answers
        best_em = max(exact_match_score(pred, c) for c in candidates)
        best_f1 = max(f1_score(pred, c) for c in candidates)
        best_rouge = max(rouge_l_score(pred, c) for c in candidates)
        best_bleu = max(bleu_score(pred, c) for c in candidates)

        em_scores.append(best_em)
        f1_scores.append(best_f1)
        rouge_scores.append(best_rouge)
        bleu_scores.append(best_bleu)

    n = len(predictions)
    return {
        "ExactMatch": round(sum(em_scores) / n, 4),
        "F1": round(sum(f1_scores) / n, 4),
        "ROUGE-L": round(sum(rouge_scores) / n, 4),
        "BLEU-4": round(sum(bleu_scores) / n, 4),
    }
