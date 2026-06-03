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
