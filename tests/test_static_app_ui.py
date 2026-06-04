import json
import subprocess
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _node_eval(expression: str):
    expression_json = json.dumps(expression)
    script = textwrap.dedent(
        f"""
        const fs = require('fs');
        const vm = require('vm');
        const code = fs.readFileSync('static/app.js', 'utf8');
        const noop = () => undefined;
        const element = () => ({{
            classList: {{ add: noop, remove: noop, toggle: noop }},
            style: {{}},
            dataset: {{}},
            addEventListener: noop,
            querySelector: () => element(),
            querySelectorAll: () => [],
        }});
        const context = {{
            console,
            document: {{
                addEventListener: noop,
                getElementById: () => element(),
                querySelector: () => element(),
                querySelectorAll: () => [],
                createElement: () => element(),
            }},
            window: {{}},
            fetch: async () => ({{ ok: true, json: async () => ({{}}) }}),
            setTimeout,
            clearTimeout,
        }};
        vm.createContext(context);
        vm.runInContext(code, context);
        const result = vm.runInContext({expression_json}, context);
        console.log(JSON.stringify(result));
        """
    )
    out = subprocess.check_output(["node", "-e", script], cwd=ROOT, text=True)
    return json.loads(out)


def test_evidence_strength_badge_hides_raw_percent_for_users():
    """참고 문서 score는 정확도처럼 보이는 퍼센트가 아니라 사용자 친화 배지로 표시한다."""
    badge = _node_eval("formatEvidenceStrengthBadge(0.70)")

    assert badge["label"] == "근거 강함"
    assert "%" not in badge["label"]
    assert "70" not in badge["label"]
    assert "정확도" not in badge["title"]
    assert "내부 검색 점수" in badge["title"]


def test_evidence_strength_badge_classifies_mid_and_low_scores():
    mid = _node_eval("formatEvidenceStrengthBadge(0.50)")
    low = _node_eval("formatEvidenceStrengthBadge(0.30)")

    assert mid["label"] == "근거 적합"
    assert low["label"] == "참고 후보"


def test_v4_support_badge_labels_exact_related_and_none():
    exact = _node_eval("formatV4SupportBadge('exact')")
    related = _node_eval("formatV4SupportBadge('related')")
    none = _node_eval("formatV4SupportBadge('none')")

    assert exact["label"] == "근거 직접 확인"
    assert related["label"] == "관련 근거만"
    assert none["label"] == "근거 없음"
    assert "정확도" not in exact["title"]


def test_v4_metadata_panel_warns_for_deterministic_fallback_and_lists_trace():
    html = _node_eval("renderV4MetadataPanel({support_level:'related', runtime_mode:'deterministic_fallback', required_citations:['BRAKE-DESC'], forbidden_claims:['unsupported requested procedure'], ontology_trace:{rdf_related_dmcs:['BRAKE-DESC'], graph_paths:['SPARQL selected BRAKE-DESC']}})")

    assert "관련 근거만" in html
    assert "직접 절차 근거 없음" in html
    assert "BRAKE-DESC" in html
    assert "SPARQL selected BRAKE-DESC" in html
    assert "unsupported requested procedure" in html
    assert "v4-metadata" in html


def test_v4_metadata_panel_is_empty_without_metadata():
    assert _node_eval("renderV4MetadataPanel(null)") == ""
    assert _node_eval("renderV4MetadataPanel({})") == ""
