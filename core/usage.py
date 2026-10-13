"""Provider usage normalized for reporting, with original fields retained."""
TOKEN_FIELDS = ('input_tokens', 'output_tokens', 'total_tokens',
                'cached_input_tokens', 'cache_creation_input_tokens', 'reasoning_tokens')


def _plain(value):
    if hasattr(value, 'model_dump'):
        return value.model_dump(mode='json')
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, '__dict__'):
        return _plain(vars(value))
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def normalize_usage(provider, usage):
    """Input includes cache reads/writes; output includes reasoning when reported.

    Cache and reasoning counters are subsets, not extra tokens to add to totals.
    Missing usage remains None, never an invented zero-cost call.
    """
    result = dict.fromkeys(TOKEN_FIELDS)
    if usage is None:
        return result
    raw = _plain(usage)
    result['raw'] = raw
    if provider == 'OpenAI':
        input_details = raw.get('prompt_tokens_details') or {}
        output_details = raw.get('completion_tokens_details') or {}
        result.update(input_tokens=raw.get('prompt_tokens'),
                      output_tokens=raw.get('completion_tokens'),
                      total_tokens=raw.get('total_tokens'),
                      cached_input_tokens=input_details.get('cached_tokens'),
                      cache_creation_input_tokens=input_details.get('cache_write_tokens'),
                      reasoning_tokens=output_details.get('reasoning_tokens'))
    elif provider == 'Anthropic':
        cache_read = raw.get('cache_read_input_tokens') or 0
        cache_write = raw.get('cache_creation_input_tokens') or 0
        inp, out = raw.get('input_tokens'), raw.get('output_tokens')
        inp = inp + cache_read + cache_write if inp is not None else None
        result.update(input_tokens=inp, output_tokens=out,
                      total_tokens=inp + out if inp is not None and out is not None else None,
                      cached_input_tokens=cache_read, cache_creation_input_tokens=cache_write)
    elif provider == 'Google':
        candidates = raw.get('candidates_token_count')
        thinking = raw.get('thoughts_token_count')
        result.update(input_tokens=raw.get('prompt_token_count'),
                      output_tokens=((candidates or 0) + (thinking or 0)
                                     if candidates is not None or thinking is not None else None),
                      total_tokens=raw.get('total_token_count'),
                      cached_input_tokens=raw.get('cached_content_token_count'),
                      reasoning_tokens=thinking)
    return result


def aggregate_agent_usage(prompt_log, agent_name):
    """NULL totals if any call lacks that count; retain per-call detail separately."""
    calls = [p.get('usage') or {} for p in prompt_log if p.get('agent') == agent_name]
    return {field: (sum(call[field] for call in calls)
                    if calls and all(call.get(field) is not None for call in calls) else None)
            for field in TOKEN_FIELDS}
