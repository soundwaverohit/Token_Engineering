import os, json, yaml, random
from rich import print
from pathlib import Path
from datetime import datetime

from .base_llm import DraftLM
from .retriever import Retriever
from .critic import load_policy, proxy_unsupported_claim_rate
from .token_controller import decode_greedy, decode_with_companion
from .metrics import parse_json_safe, schema_valid, count_tokens, first_pass_accept, summarize_kpis

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RUBRICS = ROOT / "rubrics"

def seed_everything(s=42):
    random.seed(s)
    try:
        import torch
        torch.manual_seed(s)
    except Exception:
        pass

def build_prompt(instruction: str) -> str:
    return (
        "You are generating a strict JSON object with keys: drug, change, rationale, citations.\n"
        "Rules: citations is an array of DOC-ids (e.g., \"DOC-1\"). Do not include extra keys.\n"
        "Return only the JSON, nothing else.\n\n"
        f"Task: {instruction}\n\n"
        "JSON: {"
    )

def run():
    seed_everything()
    policy = load_policy(str(RUBRICS / "policy.yaml"))
    retr = Retriever(str(DATA / "vault_docs.jsonl"))
    model = DraftLM()
    tok = model.tokenizer

    tasks = [json.loads(l) for l in open(DATA / "tasks.jsonl")]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = ROOT / "runs" / ts
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for t in tasks:
        prompt = build_prompt(t["instruction"])
        evidence_texts = retr.evidence_sentences(t["target_drug"], k=4)

        # Baseline
        baseline = decode_greedy(model, tok, prompt, policy["decode"]["max_new_tokens"], policy["decode"]["temperature"])
        ok_b, j_b = parse_json_safe(baseline)
        schema_b = ok_b and schema_valid(j_b, policy)
        unsupported_b = proxy_unsupported_claim_rate(baseline, evidence_texts)
        row_b = {
            "task_id": t["task_id"], "variant": "baseline", "text": baseline,
            "schema_valid": bool(schema_b),
            "unsupported_rate": float(unsupported_b),
            "tokens": count_tokens(baseline),
            "first_pass_accept": first_pass_accept(bool(schema_b), float(unsupported_b)),
        }

        # Companion
        companion = decode_with_companion(model, tok, prompt, evidence_texts, policy)
        ok_c, j_c = parse_json_safe(companion)
        schema_c = ok_c and schema_valid(j_c, policy)
        unsupported_c = proxy_unsupported_claim_rate(companion, evidence_texts)
        row_c = {
            "task_id": t["task_id"], "variant": "companion", "text": companion,
            "schema_valid": bool(schema_c),
            "unsupported_rate": float(unsupported_c),
            "tokens": count_tokens(companion),
            "first_pass_accept": first_pass_accept(bool(schema_c), float(unsupported_c)),
        }

        rows.extend([row_b, row_c])

        print(f"\n[bold]Task {t['task_id']}[/bold]")
        print("[cyan]Baseline[/cyan]:", baseline)
        print("[green]Companion[/green]:", companion)

    with open(outdir / "results.jsonl", "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    base = [r for r in rows if r["variant"]=="baseline"]
    comp = [r for r in rows if r["variant"]=="companion"]

    bsum = summarize_kpis(base); csum = summarize_kpis(comp)
    print("\n[bold yellow]KPI Summary[/bold yellow]")
    print("Baseline:", bsum)
    print("Companion:", csum)

    proof = {
        "unsupported_claim_rate_improvement": bsum["unsupported_rate_avg"] - csum["unsupported_rate_avg"],
        "schema_validity_delta": csum["schema_valid_rate"] - bsum["schema_valid_rate"],
        "first_pass_accept_delta": csum["first_pass_accept_rate"] - bsum["first_pass_accept_rate"],
        "tokens_delta": bsum["tokens_avg"] - csum["tokens_avg"]
    }
    with open(outdir / "proof.json", "w") as f:
        json.dump(proof, f, indent=2)

    print("\n[bold green]Proof deltas[/bold green]:", proof)
    print(f"\nArtifacts saved to: {outdir}")

if __name__ == "__main__":
    run()
