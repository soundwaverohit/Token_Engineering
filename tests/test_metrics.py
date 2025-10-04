import pathlib, yaml, sys
sys.path.append(str(pathlib.Path(__file__).parents[1]))
from src.metrics import parse_json_safe, schema_valid

def test_schema():
    policy = yaml.safe_load(open(pathlib.Path(__file__).parents[1]/"rubrics"/"policy.yaml"))
    ok, j = parse_json_safe('{"drug": "X", "change": "Y", "rationale": "Z", "citations": ["DOC-1"]}')
    assert ok and schema_valid(j, policy)
