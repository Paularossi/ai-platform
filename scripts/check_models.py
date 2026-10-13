"""Opt-in live checks through the same provider wrappers used by the app."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ecu import PeerReviewRound
from core.providers import API_KEY_ENV_VARS, PROVIDER_MODELS, get_provider


def validate_text(text):
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Empty response')
    if text.lstrip().startswith(('[ERROR:', '[BLOCKED:')):
        raise ValueError(text)


def validate_review(raw, targets, dimensions):
    """Validate before the app parser can substitute neutral fallback scores."""
    validate_text(raw)
    text = raw.strip()
    if text.startswith('```') and text.endswith('```'):
        text = '\n'.join(text.splitlines()[1:-1])
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError('Review must be a JSON object')
    for name in targets:
        entry = parsed.get(name)
        if not isinstance(entry, dict):
            raise ValueError(f'Missing review for {name}')
        for dimension in dimensions:
            score = entry.get(dimension)
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f'Invalid or missing score: {name}/{dimension}')
        if not isinstance(entry.get('justification'), str) or not entry['justification'].strip():
            raise ValueError(f'Missing justification: {name}')
    votes = parsed.get('importance_votes', {})
    if not isinstance(votes, dict) or set(votes) != set(dimensions):
        raise ValueError('Missing or unexpected importance-vote dimensions')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in votes.values()):
        raise ValueError('Invalid importance votes')
    if not math.isclose(sum(votes.values()), 100, abs_tol=0.01):
        raise ValueError('Importance votes must total 100')
    return parsed


def check_model(provider_name, model, max_tokens, rounds):
    provider = get_provider(provider_name)
    failures = 0
    contributions = {'Agent 1': 'A pilot could measure benefits before expansion.',
                     'Agent 2': 'Compare costs and accessibility before deciding.',
                     'Agent 3': 'Gather resident feedback and publish results.'}
    reviewer = PeerReviewRound()
    dimensions = [d['name'] for d in reviewer.dimensions]
    # Two free-text requests and one genuine peer-review prompt per chosen round.
    for phase in ['complete', 'stream'] + [f'review-{i + 1}' for i in range(rounds)]:
        try:
            system = 'You are contributing to a classroom policy discussion.'
            prompt = 'In one sentence, suggest how to evaluate a pedestrian-only street pilot.'
            if phase == 'complete':
                raw = provider.complete(model, system, prompt, max_tokens=max_tokens, temperature=0.0)
                validate_text(raw)
            elif phase == 'stream':
                raw = ''.join(provider.stream(model, system, prompt, max_tokens=max_tokens, temperature=0.0))
                validate_text(raw)
            else:
                cycle = int(phase.split('-')[1]) - 1
                prompt = reviewer.build_prompt(
                    'Agent 3', contributions['Agent 3'], contributions, cycle,
                    item_context='Topic: Should the city pilot a pedestrian-only street?',
                    agent_histories={name: [text] * (cycle + 1) for name, text in contributions.items()},
                    collect_importance_votes=True,
                )
                kwargs = {'json_mode': True} if provider_name == 'Google' else {}
                raw = provider.complete(model, 'Evaluate contributions and return the requested JSON.',
                                        prompt, max_tokens=max_tokens, temperature=0.0, **kwargs)
                validate_review(raw, ['Agent 1', 'Agent 2'], dimensions)
            print(f'PASS {provider_name}/{model} {phase} usage={provider.get_last_usage()}', flush=True)
        except Exception as exc:
            failures += 1
            print(f'FAIL {provider_name}/{model} {phase}: {type(exc).__name__}: {exc}', flush=True)
    return failures


def main(argv=None):
    parser = argparse.ArgumentParser(description='Live provider checks (paid requests only with --live).')
    parser.add_argument('--live', action='store_true', help='Authorize paid API requests')
    parser.add_argument('--provider', choices=list(PROVIDER_MODELS))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument('--model', help='Model ID; requires --provider')
    selection.add_argument('--all', action='store_true', help='All menu models, optionally for one provider')
    parser.add_argument('--rounds', type=int, default=3, help='Peer-review requests per model (default: 3)')
    parser.add_argument('--max-tokens', type=int, default=1600, help='Output limit per request (default: 1600)')
    args = parser.parse_args(argv)
    if args.model and not args.provider:
        parser.error('--model requires --provider')
    if args.rounds < 1 or args.max_tokens < 1:
        parser.error('--rounds and --max-tokens must be positive')
    selected = ([(args.provider, args.model)] if args.model else
                [(p, m) for p, models in PROVIDER_MODELS.items()
                 if args.provider is None or args.provider == p for m in models])
    print(f'{len(selected)} model(s), {len(selected) * (2 + args.rounds)} requests; '
          f'up to {args.max_tokens} output tokens per request. SDK retries may add requests.')
    for p, m in selected:
        print(f'  {p}: {m}')
    if not args.live:
        print('Preview only. Add --live to make paid requests.')
        return 0
    from dotenv import load_dotenv
    load_dotenv(ROOT / '.env')
    missing = [API_KEY_ENV_VARS[p] for p in dict.fromkeys(p for p, _ in selected)
               if not os.environ.get(API_KEY_ENV_VARS[p])]
    if missing:
        print('Missing credentials: ' + ', '.join(missing), file=sys.stderr)
        return 2
    failures = sum(check_model(p, m, args.max_tokens, args.rounds) for p, m in selected)
    print(f'Finished: {failures} failed check(s).')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
