"""Standalone, escaped HTML export of the completed debate page."""
from html import escape
import json

from markdown_it import MarkdownIt
from core.llm.search import search_suggestions_frame, search_summary

_MARKDOWN = MarkdownIt("commonmark", {"html": False, "linkify": False}).enable("table").disable("image")
_STYLE = """
:root{color-scheme:light}body{font:16px/1.65 system-ui,sans-serif;color:#24332c;background:#f4f6f3;margin:0}
main{max-width:1050px;margin:auto;padding:32px 22px}h1,h2,h3{line-height:1.25;color:#244c39}
h1{font-family:Georgia,serif;font-size:2.2rem}.topic{border-left:5px solid #2e5945;padding:16px 24px;background:#e8efe8}
section,details{background:white;border:1px solid #dbe3dc;border-radius:8px;padding:18px 22px;margin:18px 0}
summary{cursor:pointer;font-weight:650;color:#244c39}table{border-collapse:collapse;width:100%;font-size:.92rem}
th,td{padding:9px;text-align:left;border-bottom:1px solid #dbe3dc;vertical-align:top;overflow-wrap:anywhere}
th{background:#edf2ed}.table{overflow-x:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f5f3;padding:14px;font-size:.85rem}
blockquote,.reasoning{border-left:3px solid #87a28c;margin:14px 0;padding:10px 18px;background:#f4f7f3;font-style:italic}
.warning{border-left:4px solid #ac6520;padding:10px 16px;background:#fff4e5}small{color:#57685e}article{margin:24px 0}
@media print{body{background:white}main{max-width:none;padding:0}section{break-inside:auto}h2,h3{break-after:avoid}}
"""


def _text(value):
    return escape(str(value if value is not None else "Not available"))


def _md(value):
    return _MARKDOWN.render(str(value or ""))


def _table(headers, rows):
    return '<div class="table"><table><thead><tr>' + ''.join(
        '<th>' + _text(h) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + _text(v) + '</td>' for v in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def _json(value):
    return '<pre>' + _text(json.dumps(value, indent=2, ensure_ascii=False, default=str)) + '</pre>'


def _search_table(agents, searches):
    rows = search_summary(agents, searches)
    return _table(['Agent', 'Count', 'Unit'], [[r['Agent'], r['Count'], r['Unit']] for r in rows]) + (
        '<details><summary>Search queries</summary>' + ''.join('<h4>' + _text(r['Agent']) + '</h4><pre>' +
        _text('\n'.join(r['Queries']) or 'No queries reported') + '</pre>' for r in rows) + '</details>')


def run_to_html(result, cfg, save_error=None):
    """Export all run-page content, including every collapsed contribution and API call.

    UI navigation/download controls are not part of the saved report. No scripts,
    external fonts or stylesheets are needed to read it offline.
    """
    topic = cfg.get('task', {}).get('description', '')
    az = cfg.get('agent_zero', {})
    design = result.get('agent_zero_design', {})
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width,initial-scale=1">',
             '<title>Debate: ' + _text(topic) + '</title><style>' + _STYLE + '</style></head><body><main>',
             '<header><small>OMAS · Debate transcript</small><h1>' + _text(topic) + '</h1></header>',
             '<div class="topic">' + _table(['Moderator', 'Model', 'Max rounds', 'Max panel size'],
                [[az.get('provider'), az.get('model'), az.get('max_rounds'), az.get('max_agents')]]) + '</div>']
    parts.append('<p>Web search for panel contributions: ' + ('enabled' if cfg.get('web_search_enabled') else 'disabled') + '</p>')
    documents = result.get("shared_documents", cfg.get("shared_documents", []))
    if documents:
        parts.append('<section><h2>Shared source documents</h2>')
        for document in documents:
            parts.append('<details><summary>' + _text(document['name']) + '</summary><small>SHA-256: ' +
                         _text(document.get('sha256', '')) + '</small><pre>' + _text(document['text']) + '</pre></details>')
        parts.append('</section>')
    if save_error:
        parts.append('<p class="warning">' + _text(save_error) + '</p>')
    parts.append('<section><h2>Panel and rules</h2>' + _table(['Agent', 'Provider / model', 'Role'],
        [[a.get('name'), f"{a.get('provider')} / {a.get('model')}", a.get('role')]
         for a in design.get('agents', [])]))
    parts.append('<h3>Shared instructions</h3>' + _md(design.get('base_instructions')))
    if design.get('guideline_notes'):
        parts.append('<h3>Shared context</h3>' + _md(design['guideline_notes']))
    if az.get('web_search_enabled'):
        parts.append('<h3>Superagent setup search</h3>' + _search_table(['Agent0'], {'Agent0': design.get('setup_search', {})})
                     + search_suggestions_frame(design.get('setup_search', {})))
        if design.get('setup_search_error'):
            parts.append('<p class="warning">Setup search failed: ' + _text(design['setup_search_error']) + '</p>')
    parts.append('<h3>Scoring criteria</h3>'  + _table(['Dimension', 'Rubric', 'ECU weight', 'SW weight'],
        [[d.get('label', d.get('name')), d.get('rubric'), d.get('weight'), d.get('sw_weight')]
         for d in design.get('ecu_dimensions', [])]))
    parts.append(_table(['Visibility', 'Review depth', 'ECU visibility', 'Coalition threshold'],
        [[design.get(k) for k in ('visibility_mode', 'review_depth', 'info_condition', 'coalition_threshold')]]))
    parts.append('<div class="reasoning">' + _md(design.get('reasoning')) + '</div></section>')
    for r in result.get('rounds', []):
        parts.append(f'<section><h2>Round {r["cycle"] + 1}</h2>')
        parts.append('<p><strong>Panel:</strong> ' + _text(', '.join(a['name'] for a in r.get('roster', []))) + '</p>')
        parts.append('<p><strong>Social welfare:</strong> ' + _text(r.get('social_welfare')) + '</p>')
        for issue in r.get('call_errors', []):
            parts.append('<p class="warning">' + _text(f"{issue.get('agent')} · {issue.get('phase')}: {issue.get('error')}") + '</p>')
        for name, contribution in r.get('contributions', {}).items():
            parts.append('<article><h3>' + _text(name) + '</h3>' + _md(contribution) + search_suggestions_frame(r.get('web_search', {}).get(name, {})) + '</article>')
        if r.get('web_search_enabled') or r.get('web_search'):
            parts.append('<h3>Web searches this round</h3>' + _search_table(r.get('contributions', {}), r.get('web_search', {})))
        coverage = r.get('review_coverage', {})
        if coverage:
            parts.append('<h3>Valid reviews / expected reviews</h3>' + _table(
                ['Agent', 'Dimension', 'Coverage'],
                [[name, dim, f"{count}/{entry['expected']}"] for name, entry in coverage.items()
                 for dim, count in entry['valid_by_dimension'].items()]))
            parts.append('<p>Missing scores are unavailable. Incomplete dimension coverage prevents an ECU reward; balances remain unchanged. Social welfare uses available scores and may have unequal coverage.</p>')
        scores = r.get('dimension_scores', {})
        if scores:
            dims = sorted({d for values in scores.values() for d in values})
            parts.append('<h3>Peer review scores</h3>' + _table(['Agent', *dims, 'ECU earned'],
                [[name, *[values.get(d) for d in dims], r.get('ecu_earned_this_round', {}).get(name)]
                 for name, values in scores.items()]))
        checks = r.get('peer_review_checks', [])
        if checks:
            parts.append('<details><summary>Hallucination and risk peer assessments</summary><p><small>'
                         'Peer judgments, not verified facts. Risk scores assess risk reasoning; higher is better. '
                         'These checks do not affect ECU rewards.</small></p>')
            for c in checks:
                parts.append('<h3>' + _text(f"{c['reviewer']} reviewing {c['agent']}") + '</h3>')
                if c.get('error'):
                    parts.append('<p class="warning">Assessment unavailable: ' + _text(c['error']) + '</p>')
                    continue
                hall, risk = c['hallucination'], c['risk']
                if hall.get('error'):
                    parts.append('<p class="warning">Hallucination assessment unavailable: ' + _text(hall['error']) + '</p>')
                else:
                    parts.append('<p><strong>Hallucination:</strong> ' + _text(hall['status'].replace('_', ' ')) + '</p>' + _md(hall['explanation']))
                    for claim in hall['flagged_claims']:
                        parts.append('<blockquote>' + _text(claim['claim']) + '</blockquote>' + _md(claim['reason']))
                if risk.get('error'):
                    parts.append('<p class="warning">Risk assessment unavailable: ' + _text(risk['error']) + '</p>')
                else:
                    parts.append('<p><strong>Risk assessment quality:</strong> ' + _text(risk['assessment_quality']) + '</p>' + _md(risk['explanation']))
                    parts.append('<ul>' + ''.join('<li>' + _text(x) + '</li>' for x in risk['concerns']) + '</ul>')
            parts.append('</details>')
        parts.append('<h3>ECU weights used this round</h3>' + _table(['Dimension', 'Weight'], r.get('ecu_weights', {}).items()))
        parts.append('<p><strong>Coalition:</strong> ' + _text(', '.join(r.get('coalition', [])) or 'None') + '</p>')
        if r.get('agents_added') or r.get('agents_removed'):
            parts.append('<p><strong>Panel change:</strong> ' + _text(f"Added: {r.get('agents_added', [])}; removed: {r.get('agents_removed', [])}") + '</p>')
        for name, instruction in r.get('agent_instructions_issued', {}).items():
            parts.append('<h3>Note to ' + _text(name) + '</h3>' + _md(instruction))
        parts.append('<h3>Agent 0 reasoning</h3><div class="reasoning">' + _md(r.get('agent_zero_reasoning')) + '</div></section>')
    parts.append('<section><h2>Policy brief</h2>' + _md(result.get('final_brief')) + '<p><small>' + _text(
        f"Ended: {result.get('ended_reason')} · {len(result.get('rounds', []))} rounds · {result.get('num_turns')} turns") + '</small></p></section>')
    parts.append('<section><h2>Final ECU balances</h2>' + _table(['Agent', 'Balance'], result.get('final_ecu_balances', {}).items()) + '</section>')
    calls = result.get('debug', {}).get('prompt_log', [])
    parts.append(f'<details><summary>API call diagnostics ({len(calls)} calls)</summary>')
    for i, c in enumerate(calls, 1):
        turn = f"Round {c['cycle'] + 1}" if c.get('cycle') is not None else 'Setup'
        parts.append('<details><summary>' + _text(f"{i}. {turn} · {c.get('agent', c.get('agent_name'))} · {c.get('phase')} · {c.get('model')} · {c.get('status')}") + '</summary>')
        if c.get('error'):
            parts.append('<p class="warning">' + _text(c['error']) + '</p>')
        parts.append('<p>' + _text(f"Stop reason: {c.get('stop_reason')} · Duration: {c.get('duration_ms')} ms") + '</p>')
        for label, key in [('System prompt', 'system_prompt'), ('User prompt', 'prompt'), ('Response text', 'response')]:
            parts.append('<h3>' + label + '</h3><pre>' + _text(c.get(key) or '(empty)') + '</pre>')
        parts.append(_json({'parameters': c.get('parameters'), 'usage': c.get('usage'),
                            'diagnostics': c.get('diagnostics', {k: c.get(k) for k in ('request_id', 'response_id', 'block_types', 'web_search')})}) + '</details>')
    parts.append('</details></main></body></html>')
    return ''.join(parts)
