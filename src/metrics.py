from __future__ import annotations
import json, re

def parse_json_safe(txt: str):
    try:
        start = txt.index("{")
        end = txt.rindex("}")
        return True, json.loads(txt[start:end+1])
    except Exception:
        return False, None

def schema_valid(j: dict, policy: dict) -> bool:
    req = policy["output_schema"]["required"]
    return all(k in j for k in req)

def count_tokens(text: str) -> int:
    return len(re.findall(r"\S+", text))

def first_pass_accept(valid_schema: bool, unsupported_rate: float) -> bool:
    return valid_schema and (unsupported_rate <= 0.0)

def summarize_kpis(outputs):
    return {
        "unsupported_rate_avg": sum(o["unsupported_rate"] for o in outputs)/len(outputs),
        "schema_valid_rate": sum(1 for o in outputs if o["schema_valid"])/len(outputs),
        "tokens_avg": sum(o["tokens"] for o in outputs)/len(outputs),
        "first_pass_accept_rate": sum(1 for o in outputs if o["first_pass_accept"])/len(outputs),
    }
