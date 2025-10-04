from __future__ import annotations
import torch, math
from typing import List
from .critic import running_schema_penalty, evidence_score, build_logit_bias, compose_score

def decode_greedy(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float=0.7):
    inp = tokenizer(prompt, return_tensors="pt")
    generated = inp["input_ids"]
    for _ in range(max_new_tokens):
        logits = model.next_logits(generated)
        logits = logits / max(1e-6, temperature)
        next_id = torch.argmax(logits, dim=-1)
        generated = torch.cat([generated, next_id.unsqueeze(0)], dim=1)
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        if text.strip().endswith("}"):
            break
    return tokenizer.decode(generated[0], skip_special_tokens=True)

def decode_with_companion(model, tokenizer, prompt: str, evidence_texts: List[str], cfg: dict):
    w = cfg["critic"]["weights"]; thr = cfg["critic"]["thresholds"]; dec = cfg["decode"]
    max_new, top_k, temp = dec["max_new_tokens"], dec["top_k"], dec["temperature"]

    inp = tokenizer(prompt, return_tensors="pt")
    generated = inp["input_ids"]
    bias_map = build_logit_bias(tokenizer, evidence_texts, boost=2.0)

    for _ in range(max_new):
        logits = model.next_logits(generated)[0]  # [vocab]

        # evidence-aware logit bias
        for tid, boost in bias_map.items():
            if tid < logits.shape[-1]:
                logits[tid] += boost

        probs = torch.softmax(logits / max(1e-6, temp), dim=-1)
        topk = torch.topk(probs, k=top_k)
        cand_ids = topk.indices.tolist()
        cand_ps  = topk.values.tolist()

        partial = tokenizer.decode(generated[0], skip_special_tokens=True)
        best_score, best_id = -1e9, cand_ids[0]
        for tid, p in zip(cand_ids, cand_ps):
            token_str = tokenizer.decode([tid])
            hypo = (partial + token_str)[-200:]  # local window
            logprob = math.log(max(p, 1e-12))
            evid = evidence_score(hypo, evidence_texts)
            schema_pen = running_schema_penalty(hypo)
            cost = 1.0
            score = compose_score(logprob, evid, schema_pen, cost, w)
            if score > best_score:
                best_score, best_id = score, tid

        # very small gating heuristic when score is too low
        if best_score < thr["accept_score"]:
            for tok_piece in ["[", " DOC-"]:
                enc = tokenizer.encode(tok_piece, add_special_tokens=False)
                if enc:
                    best_id = enc[0]
                    break

        next_id = torch.tensor([[best_id]])
        generated = torch.cat([generated, next_id], dim=1)

        txt = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
        if txt.endswith("}"):
            break

    return tokenizer.decode(generated[0], skip_special_tokens=True)
