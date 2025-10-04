from __future__ import annotations
from typing import Dict, List
import math, json, re, yaml

NUM_RE = re.compile(r"\b\d+(?:\.\d+)?\b")  # non-capturing group
DOC_RE = re.compile(r"\bDOC-\d+\b")

def load_policy(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def running_schema_penalty(partial_text: str) -> float:
    # crude JSON-ish balance check
    open_braces = partial_text.count("{") - partial_text.count("}")
    open_brackets = partial_text.count("[") - partial_text.count("]")
    quotes = partial_text.count('"')
    quote_imbalance = quotes % 2  # odd -> unbalanced
    penalty = 0.3 * max(0, open_braces) + 0.2 * max(0, open_brackets) + 0.5 * quote_imbalance
    return penalty

def evidence_score(window_text: str, evidence_sentences: List[str]) -> float:
    wtoks = set([t.lower() for t in re.findall(r"[A-Za-z0-9.%°]+", window_text)])
    scores = []
    for s in evidence_sentences:
        stoks = set([t.lower() for t in re.findall(r"[A-Za-z0-9.%°]+", s)])
        overlap = len(wtoks & stoks) / (len(wtoks) + 1e-6)
        # number agreement bonus
        nums_w = set(NUM_RE.findall(window_text))
        nums_s = set(NUM_RE.findall(s))
        num_bonus = 0.2 if (nums_w & nums_s) else 0.0
        scores.append(overlap + num_bonus)
    return max(scores) if scores else 0.0

def build_logit_bias(tokenizer, evidence_texts: List[str], boost: float = 2.0) -> Dict[int, float]:
    joined = " ".join(evidence_texts)
    ids = tokenizer.encode(joined, add_special_tokens=False)
    return {tid: boost for tid in set(ids)}

def compose_score(logprob: float, evid: float, schema_pen: float, cost: float, w: dict) -> float:
    return w["logprob"] * logprob + w["evidence"] * evid - w["schema_penalty"] * schema_pen - w["cost"] * cost

def proxy_unsupported_claim_rate(text: str, evidence_texts: List[str]) -> float:
    ev = " ".join(evidence_texts)
    nums = set(NUM_RE.findall(text))
    if not nums: 
        return 0.0
    ok = sum(1 for n in nums if n in ev)
    return 1.0 - ok / max(1, len(nums))
