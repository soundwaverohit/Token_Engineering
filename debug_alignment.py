#!/usr/bin/env python3
"""
Debug script to show evidence alignment in detail
"""
import json
from src.base_llm import DraftLM
from src.retriever import Retriever
from src.critic import load_policy, evidence_score, build_logit_bias
from src.token_controller import decode_greedy, decode_with_companion
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUBRICS = ROOT / "rubrics"

def debug_evidence_alignment():
    print("🔍 DEBUGGING EVIDENCE ALIGNMENT")
    print("=" * 50)
    
    # Load components
    policy = load_policy(str(RUBRICS / "policy.yaml"))
    retriever = Retriever(str(DATA / "vault_docs.jsonl"))
    model = DraftLM()
    
    # Test with CardiCure task
    task = {"task_id": "T1", "instruction": "Draft a labeling change memo for CardiCure reflecting a new 20 mg once-daily dose. Include one-sentence justification and cite relevant documents by doc_id.", "target_drug": "CardiCure"}
    
    print(f"📋 Task: {task['instruction']}")
    print(f"🎯 Target Drug: {task['target_drug']}")
    print()
    
    # Get evidence documents
    evidence_texts = retriever.evidence_sentences(task["target_drug"], k=4)
    print("📚 Retrieved Evidence Documents:")
    for i, evidence in enumerate(evidence_texts, 1):
        print(f"  {i}. {evidence}")
    print()
    
    # Show logit bias tokens
    bias_map = build_logit_bias(model.tokenizer, evidence_texts, boost=2.0)
    print("🎯 Evidence Tokens Being Boosted:")
    evidence_tokens = []
    for token_id, boost in list(bias_map.items())[:20]:  # Show first 20
        token_text = model.tokenizer.decode([token_id])
        evidence_tokens.append(token_text)
        print(f"  Token ID {token_id}: '{token_text}' (boost: +{boost})")
    print(f"  ... and {len(bias_map) - 20} more tokens")
    print()
    
    # Test evidence scoring on sample text
    print("🧪 Testing Evidence Scoring:")
    test_texts = [
        "CardiCure 20 mg daily dose",
        "The approved dose is now 20 mg",
        "Phase III Study CC-301",
        "random unrelated text about cats"
    ]
    
    for text in test_texts:
        score = evidence_score(text, evidence_texts)
        print(f"  '{text}' -> Evidence Score: {score:.3f}")
    print()
    
    # Show token-level differences
    prompt = f"You are generating a strict JSON object with keys: drug, change, rationale, citations.\nRules: citations is an array of DOC-ids (e.g., \"DOC-1\"). Do not include extra keys.\nReturn only the JSON, nothing else.\n\nTask: {task['instruction']}\n\nJSON: {{"
    
    print("🔄 Generating with Baseline (Greedy):")
    baseline = decode_greedy(model, model.tokenizer, prompt, 50, 0.7)
    print(f"Output: {baseline[-100:]}")  # Last 100 chars
    print()
    
    print("🔄 Generating with Companion (Evidence-Biased):")
    companion = decode_with_companion(model, model.tokenizer, prompt, evidence_texts, policy)
    print(f"Output: {companion[-100:]}")  # Last 100 chars
    print()
    
    # Analyze token differences
    print("📊 Token Analysis:")
    baseline_tokens = baseline.split()
    companion_tokens = companion.split()
    
    print(f"Baseline tokens: {len(baseline_tokens)}")
    print(f"Companion tokens: {len(companion_tokens)}")
    print(f"Token reduction: {len(baseline_tokens) - len(companion_tokens)}")
    print()
    
    # Check for evidence alignment
    print("🎯 Evidence Alignment Check:")
    baseline_evidence_score = evidence_score(baseline, evidence_texts)
    companion_evidence_score = evidence_score(companion, evidence_texts)
    
    print(f"Baseline evidence score: {baseline_evidence_score:.3f}")
    print(f"Companion evidence score: {companion_evidence_score:.3f}")
    print(f"Evidence improvement: {companion_evidence_score - baseline_evidence_score:.3f}")
    
    # Check for specific evidence terms
    print("\n🔍 Evidence Term Analysis:")
    evidence_terms = ["CardiCure", "20", "mg", "daily", "dose", "CC-301", "Phase", "III"]
    for term in evidence_terms:
        baseline_count = baseline.lower().count(term.lower())
        companion_count = companion.lower().count(term.lower())
        print(f"  '{term}': Baseline={baseline_count}, Companion={companion_count}")

if __name__ == "__main__":
    debug_evidence_alignment()
