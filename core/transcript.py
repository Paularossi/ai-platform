"""Self-contained HTML transcripts with escaped, word-level prompt revisions."""
from difflib import SequenceMatcher
from html import escape
import re

from core.ecu import resolve_dimensions


def prompt_diff_html(original: str | None, prompt: str) -> str:
    if original is None or original == prompt:
        return escape(prompt)
    before = re.findall(r"\s+|\w+|[^\w\s]", original)
    after = re.findall(r"\s+|\w+|[^\w\s]", prompt)
    parts = []
    for tag, a, b, c, d in SequenceMatcher(None, before, after, autojunk=False).get_opcodes():
        if tag == "equal":
            parts.append(escape("".join(after[c:d])))
        else:
            if tag in ("delete", "replace"):
                parts.append("<del>" + escape("".join(before[a:b])) + "</del>")
            if tag in ("insert", "replace"):
                parts.append("<ins>" + escape("".join(after[c:d])) + "</ins>")
    return "".join(parts)


def build_transcript_html(result: dict, cfg: dict) -> str:
    def e(value):
        return escape(str(value))

    def text(value):
        return '<div class="text">' + e(value) + '</div>'

    title = (cfg.get("overview", {}).get("name") or "Deliberation") + " - Transcript"
    parts = ['<!doctype html><html lang="en"><head><meta charset="utf-8">',
             '<meta name="viewport" content="width=device-width, initial-scale=1">',
             '<title>' + e(title) + '</title>',
             """<style>
*{box-sizing:border-box}body{font:16px/1.65 system-ui,sans-serif;max-width:1040px;margin:48px auto;padding:0 28px;color:#243932;background:#f8faf8}
h1{font-size:2rem;line-height:1.2;letter-spacing:-.04em;margin:8px 0 20px}h2{font-size:1.35rem;border-bottom:1px solid #cedbd3;padding-bottom:10px;margin-top:40px}h3{font-size:1rem;margin:0 0 12px}
.eyebrow{font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;font-weight:700;color:#526f60;margin-bottom:12px}
.topic{border-left:5px solid #39765c;background:#e8f1ea;border-radius:0 14px 14px 0;padding:24px 28px;margin:24px 0}.topic .text{font-size:1.3rem;line-height:1.5;font-weight:600}
.meta,.weights,.votes{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0}.badge{display:inline-block;padding:4px 10px;background:#edf2ef;border-radius:6px;font-size:.8rem;color:#405d50}
.dimensions,.balances{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px}.dimension{background:white;border:1px solid #d8e2da;padding:20px;border-radius:12px}.dimension .text{font-size:.9rem}.dimension .weights{margin-bottom:0}
article,details,.review,.outcome,.reflection{border:1px solid #d8e2da;border-radius:12px;padding:22px;margin:16px 0;background:white}article{border-left:3px solid #608d75}
summary{cursor:pointer;font-weight:600;line-height:1.5}.text,pre{white-space:pre-wrap;overflow-wrap:anywhere}
pre{font:13px/1.65 ui-monospace,monospace;background:#f5f7f5;padding:16px;border-radius:8px;max-height:560px;overflow:auto}
ins{background:#d3f4db;color:#124c25;text-decoration:underline}del{background:#ffe0e0;color:#852020}
.muted,small{color:#5c6e63}.table-wrap{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:.88rem}caption{text-align:left;color:#5c6e63;margin-bottom:10px}th,td{text-align:left;padding:12px;border-bottom:1px solid #e0e8e2;vertical-align:top}thead{background:#edf3ef}tbody th{min-width:110px}td.score{font-variant-numeric:tabular-nums;font-weight:600;white-space:nowrap}.justification{font-size:.85rem;color:#52655a}
.balance-value{display:block;font-size:1.6rem;font-weight:650;font-variant-numeric:tabular-nums}.outcome{background:#edf4ef}.reflection{font-family:Georgia,serif;font-size:1.12rem;line-height:1.8;font-style:italic;border-left:3px solid #9bad92}
@media(max-width:600px){body{padding:0 16px;margin:24px auto}h1{font-size:1.65rem}.topic,article,.review{padding:16px}.dimensions{grid-template-columns:1fr}}
@media print{body{background:white;margin:0;max-width:none;font-size:11pt}h2,h3{break-after:avoid}.dimension{break-inside:avoid}pre{max-height:none;overflow:visible}details{break-inside:auto}}
</style></head><body>""", '<header><div class="eyebrow">Deliberation record</div><h1>' + e(title) + '</h1>']
    author = cfg.get('overview', {}).get('author')
    if author:
        parts.append('<p class="muted">Prepared by ' + e(author) + '</p>')
    parts.append('<div class="meta">')
    for label, value in [('Setting', cfg.get('protocol', {}).get('setting', '—')),
                         ('Mode', cfg.get('protocol', {}).get('run_mode', '—')),
                         ('Agents', len(cfg.get('agents', []))),
                         ('Turns', result.get('num_turns', '—'))]:
        parts.append('<span class="badge">' + e(label) + ': ' + e(value) + '</span>')
    parts.extend(['</div></header><section class="topic" aria-label="Deliberation topic">',
                  '<div class="eyebrow">Deliberation topic / question</div>',
                  text(cfg.get('task', {}).get('description', '')), '</section>'])
    dimensions = resolve_dimensions(cfg.get('ecu', {}).get('dimensions'))
    labels = {d['name']: d.get('label', d['name']) for d in dimensions}
    if cfg.get('ecu', {}).get('enabled') and dimensions:
        parts.append('<h2>Quality dimensions</h2><p class="muted">Scores range from 0 to 1. SW weights are fixed; ECU weights below are the starting incentives.</p><div class="dimensions">')
        for dim in dimensions:
            parts.extend(['<section class="dimension"><h3>' + e(dim.get('label', dim['name'])) + '</h3>',
                          text(dim.get('rubric', '')), '<div class="weights">',
                          '<span class="badge">SW weight: ' + e(dim.get('sw_weight', 1.0)) + '</span>',
                          '<span class="badge">Initial ECU weight: ' + e(dim.get('weight', 1.0)) + '</span>', '</div></section>'])
        parts.append('</div>')
    log = result.get('log', [])
    reviews = result.get('peer_review_log', [])
    prompts = result.get('prompt_log', [])
    cycles = sorted({entry['cycle'] for entry in log + reviews + prompts})
    for cycle in cycles:
        parts.append('<h2>Round ' + e(cycle + 1) + '</h2>')
        for out in (o for o in log if o['cycle'] == cycle):
            parts.extend(['<article><h3>' + e(out['agent_name']) + '</h3>',
                          text(out.get('contribution') or '(empty)'), '</article>'])
        for review in (r for r in reviews if r['cycle'] == cycle):
            parts.append('<section class="review"><h3>Peer review by ' + e(review['reviewer_name']) + '</h3>')
            rows = list(review.get('scores', {}).items())
            if review.get('self_scores'):
                rows.append((review['reviewer_name'] + ' (self)', review['self_scores']))
            score_dims = list(dict.fromkeys(d for _, scores in rows for d in scores))
            if rows:
                parts.append('<div class="table-wrap"><table><caption>Quality scores · 0–1 scale</caption><thead><tr><th scope="col">Agent</th>')
                parts.extend('<th scope="col">' + e(labels.get(d, d)) + '</th>' for d in score_dims)
                parts.append('</tr></thead><tbody>')
                for name, scores in rows:
                    parts.append('<tr><th scope="row">' + e(name) + '</th>')
                    parts.extend('<td class="score">' + (e(f'{scores[d]:.2f}') if d in scores else '—') + '</td>' for d in score_dims)
                    parts.append('</tr>')
                    justification = review.get('justifications', {}).get(name)
                    if justification:
                        parts.append('<tr><td class="justification" colspan="' + str(len(score_dims) + 1) + '">' + text(justification) + '</td></tr>')
                parts.append('</tbody></table></div>')
            if review.get('importance_votes'):
                parts.append('<p class="muted">Dimension importance votes (out of 100)</p><div class="votes">')
                parts.extend('<span class="badge">' + e(labels.get(d, d)) + ': ' + e(round(v)) + '</span>' for d, v in review['importance_votes'].items())
                parts.append('</div>')
            parts.append('</section>')
        for entry in (p for p in prompts if p['cycle'] == cycle):
            prompt = entry.get('prompt') or ''
            original = entry.get('original_prompt')
            changed = original is not None and original != prompt
            heading = str(entry.get('agent', entry.get('agent_name', 'Agent'))) + ' — ' + str(entry.get('phase', 'prompt')).replace('_', ' ')
            parts.extend(['<details' + (' open' if changed else '') + '><summary>' + e(heading) + (' — edited' if changed else ' — unchanged') + '</summary>',
                          '<pre>' + prompt_diff_html(original, prompt) + '</pre>'])
            parts.append('</details>')
    if result.get('ecu_balances'):
        parts.append('<h2>ECU balances (final)</h2><div class="balances">')
        for name, balance in result['ecu_balances'].items():
            parts.append('<div class="dimension">' + e(name) + '<span class="balance-value">' + e(f'{balance:.3f}') + ' <small>ECU</small></span></div>')
        parts.append('</div>')
    history = result.get('coalition_history', {}).get('history', [])
    if history and len(history[-1].get('coalition', [])) >= 2:
        parts.append(text('Coalition reached: ' + ', '.join(history[-1]['coalition'])))
    if result.get('outcome'):
        parts.extend(['<h2>Outcome of the debate</h2><section class="outcome"><h3>', e(result['outcome'].get('label', '')),
                      '</h3>', text(result['outcome'].get('notes', '')), '</section>'])
    if result.get('reflection'):
        parts.extend(['<h2>Reflection</h2><section class="reflection">', text(result['reflection']), '</section>'])
    parts.append('</body></html>')
    return '\n'.join(parts)
