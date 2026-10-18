#!/usr/bin/env python3
"""Check the running server's APIs and time prompt processing.

Checks: health and model list; OpenAI chat with and without streaming, reasoning split out, reasoning effort,
a tool call and a tool round trip; Anthropic Messages with streaming, a tool call and count_tokens; OpenAI
Responses. Then times cold prompts of about 2K, 8K and 32K tokens (fresh text each, so nothing is reused) and a
follow-up turn on the 8K conversation (prefix reuse), with thinking off and up to 64 output tokens.

  .venv/bin/python scripts/api_check.py [--base URL] [--out results/api-check.json]

The server URL and model ID default to config.local.json (or config.example.json).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import httpx

sys.dont_write_bytecode = True
import run

ROOT = run.ROOT
MODEL = 'deepseek-v4.1-flash-1017'  # replaced by the configured model_id in main()
SAMPLING = {'temperature': 1.0, 'top_p': 1.0}
TOOL = {'type': 'function', 'function': {
    'name': 'get_weather', 'description': 'Current weather for a city.',
    'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}}, 'required': ['city']}}}


def corpus():
    """Deterministic filler: this repo's own engine and component sources."""
    files = sorted(p for d in ('engine/src', 'components') for p in (ROOT / d).rglob('*')
                   if p.suffix in ('.cpp', '.cu', '.hpp', '.h', '.cuh'))
    return '\n'.join(f'// file: {p.relative_to(ROOT)}\n{p.read_text(errors="ignore")}' for p in files)


class Checker:
    def __init__(self, base):
        self.base = base.rstrip('/')
        self.client = httpx.Client(timeout=httpx.Timeout(3600, connect=10))
        self.results = []

    def check(self, name, fn):
        start = time.perf_counter()
        try:
            detail = fn()
            ok = True
        except Exception as exc:  # report every failure, keep going
            detail, ok = f'{type(exc).__name__}: {exc}', False
        row = {'check': name, 'ok': ok, 'seconds': round(time.perf_counter() - start, 2), 'detail': detail}
        self.results.append(row)
        print(f'{"PASS" if ok else "FAIL"}  {name:28s} {row["seconds"]:7.2f} s  {json.dumps(detail)[:150]}', flush=True)

    def chat(self, messages, **body):
        r = self.client.post(f'{self.base}/v1/chat/completions',
                             json={'model': MODEL, 'messages': messages, **SAMPLING, **body})
        r.raise_for_status()
        return r.json()

    def stream(self, url, body):
        """Stream a request; returns (seconds to first content event, events)."""
        start, first, events = time.perf_counter(), None, []
        with self.client.stream('POST', f'{self.base}{url}', json=body) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith('data: ') and line != 'data: [DONE]':
                    event = json.loads(line[6:])
                    events.append(event)
                    if first is None and any((c.get('delta') or {}).get(k) for c in event.get('choices', [])
                                             for k in ('content', 'reasoning_content')):
                        first = time.perf_counter() - start
        return first, events


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--base', help='server URL (default: http://127.0.0.1:<configured port>)')
    p.add_argument('--out', type=Path)
    p.add_argument('--skip-timing', action='store_true')
    args = p.parse_args()
    global MODEL
    cfg = run.configuration()
    MODEL = cfg['model_id']
    c = Checker(args.base or f'http://127.0.0.1:{cfg["port"]}')
    question = [{'role': 'user', 'content': 'What is 17 * 23? Reply with the number only.'}]

    def health():
        c.client.get(f'{c.base}/health').raise_for_status()
        models = c.client.get(f'{c.base}/v1/models').json()['data']
        assert [m['id'] for m in models] == [MODEL], models
        return models[0]['id']

    def chat():
        d = c.chat(question, max_tokens=1024, reasoning_effort='high')
        m = d['choices'][0]['message']
        assert '391' in m['content'] and m.get('reasoning_content'), m
        return {'content': m['content'], 'reasoning_chars': len(m['reasoning_content']), 'usage': d['usage']}

    def efforts():
        out = {}
        for effort in ('low', 'max'):
            d = c.chat(question, max_tokens=2048, reasoning_effort=effort)
            assert '391' in d['choices'][0]['message']['content']
            out[effort] = d['usage']['completion_tokens_details']['reasoning_tokens']
        return out

    def no_thinking():
        d = c.chat(question, max_tokens=64, thinking=False)
        m = d['choices'][0]['message']
        assert '391' in m['content'] and not m.get('reasoning_content'), m
        return m['content']

    def streaming():
        first, events = c.stream('/v1/chat/completions', {
            'model': MODEL, 'messages': question, **SAMPLING, 'max_tokens': 1024, 'stream': True,
            'stream_options': {'include_usage': True}})
        text = ''.join((ch.get('delta') or {}).get('content') or '' for e in events for ch in e.get('choices', []))
        assert '391' in text and events[-1].get('usage'), text
        return {'ttft_s': round(first, 2), 'content': text}

    weather = [{'role': 'user', 'content': 'What is the weather in Paris right now? Use the tool.'}]

    def tool_call():
        d = c.chat(weather, tools=[TOOL], max_tokens=2048)
        m = d['choices'][0]['message']
        call = m['tool_calls'][0]
        assert call['function']['name'] == 'get_weather' and 'paris' in call['function']['arguments'].lower(), m
        assert d['choices'][0]['finish_reason'] == 'tool_calls'
        return call['function']

    def tool_round_trip():
        d = c.chat(weather, tools=[TOOL], max_tokens=2048)
        m = d['choices'][0]['message']
        call = m['tool_calls'][0]
        messages = weather + [m, {'role': 'tool', 'tool_call_id': call['id'],
                                  'content': json.dumps({'city': 'Paris', 'temp_c': 17, 'sky': 'light rain'})}]
        final = c.chat(messages, tools=[TOOL], max_tokens=2048)['choices'][0]['message']['content']
        assert '17' in final, final
        return final[:120]

    def anthropic():
        body = {'model': MODEL, 'max_tokens': 4096, 'messages': question, **SAMPLING,
                'thinking': {'type': 'enabled', 'budget_tokens': 4096}}
        r = c.client.post(f'{c.base}/v1/messages', json=body)
        r.raise_for_status()
        d = r.json()
        text = ''.join(b.get('text', '') for b in d['content'] if b['type'] == 'text')
        assert '391' in text and any(b['type'] == 'thinking' for b in d['content']), d
        events = []
        with c.client.stream('POST', f'{c.base}/v1/messages', json={**body, 'stream': True}) as s:
            s.raise_for_status()
            events = [json.loads(l[6:]) for l in s.iter_lines() if l.startswith('data: ')]
        assert events[-1]['type'] == 'message_stop', events[-1]
        return {'text': text, 'stream_events': len(events)}

    def anthropic_tool():
        tool = {'name': 'get_weather', 'description': 'Current weather for a city.',
                'input_schema': TOOL['function']['parameters']}
        r = c.client.post(f'{c.base}/v1/messages', json={'model': MODEL, 'max_tokens': 2048, 'messages': weather,
                                                        'tools': [tool], **SAMPLING})
        r.raise_for_status()
        d = r.json()
        use = [b for b in d['content'] if b['type'] == 'tool_use'][0]
        assert use['name'] == 'get_weather' and d['stop_reason'] == 'tool_use', d
        return use['input']

    def count_tokens():
        r = c.client.post(f'{c.base}/v1/messages/count_tokens', json={'model': MODEL, 'messages': question})
        r.raise_for_status()
        return r.json()

    def responses():
        r = c.client.post(f'{c.base}/v1/responses', json={'model': MODEL, 'input': question[0]['content'],
                                                         'max_output_tokens': 1024, **SAMPLING})
        r.raise_for_status()
        d = r.json()
        text = ''.join(part.get('text', '') for item in d['output'] if item['type'] == 'message'
                       for part in item['content'])
        assert '391' in text, d
        return text

    for name, fn in [('health and models', health), ('chat', chat), ('reasoning effort', efforts),
                     ('thinking off', no_thinking), ('chat streaming', streaming), ('tool call', tool_call),
                     ('tool round trip', tool_round_trip), ('anthropic messages', anthropic),
                     ('anthropic tool use', anthropic_tool), ('anthropic count_tokens', count_tokens),
                     ('responses', responses)]:
        c.check(name, fn)

    if not args.skip_timing:
        text, offset = corpus(), 0

        def tokens(doc):
            r = c.client.post(f'{c.base}/v1/messages/count_tokens', json={'model': MODEL, 'messages': [
                {'role': 'user', 'content': f'Summarize what this code does in three sentences.\n\n{doc}'}]})
            r.raise_for_status()
            return r.json()['input_tokens']

        def prompt(target):
            """A fresh slice of the corpus (never reused) sized to about `target` prompt tokens."""
            nonlocal offset
            chars = target * 3
            for _ in range(3):
                chars = int(chars * target / tokens(text[offset:offset + chars]))
            doc, offset = text[offset:offset + chars], offset + chars
            return [{'role': 'user', 'content': f'Summarize what this code does in three sentences.\n\n{doc}'}]

        def timed(messages):
            first, events = c.stream('/v1/chat/completions', {
                'model': MODEL, 'messages': messages, **SAMPLING, 'max_tokens': 64, 'thinking': False,
                'stream': True, 'stream_options': {'include_usage': True}})
            usage = events[-1]['usage']
            reply = ''.join((ch.get('delta') or {}).get('content') or '' for e in events for ch in e.get('choices', []))
            timing = usage.get('ds41_timing', {})
            return reply, {'prompt_tokens': usage['prompt_tokens'],
                           'cached_tokens': (usage.get('prompt_tokens_details') or {}).get('cached_tokens'),
                           'ttft_s': round(first, 2),
                           'prefill_tok_s': round(usage['prompt_tokens'] / (timing['prefill_ms'] / 1000), 1),
                           'decode_tok_s': timing.get('decode_tok_s')}

        c.check('cold prompt ~2K', lambda: timed(prompt(2048))[1])
        state = {}

        def cold_8k():
            state['messages'] = prompt(8192)
            state['reply'], detail = timed(state['messages'])
            state['tokens'] = detail['prompt_tokens']
            return detail
        c.check('cold prompt ~8K', cold_8k)

        def follow_up():
            # the engine keeps the latest conversation, so this reuses the 8K prompt and its answer
            messages = state['messages'] + [{'role': 'assistant', 'content': state['reply']},
                                            {'role': 'user', 'content': 'Now name the two most important functions.'}]
            detail = timed(messages)[1]
            assert detail['cached_tokens'] and detail['cached_tokens'] > 0.9 * state['tokens'], detail
            return detail
        c.check('follow-up on 8K (reuse)', follow_up)
        c.check('cold prompt ~32K', lambda: timed(prompt(32768))[1])

    failed = [r['check'] for r in c.results if not r['ok']]
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'base': c.base, 'model': MODEL, 'results': c.results}, indent=1) + '\n')
    print(f'{len(c.results) - len(failed)}/{len(c.results)} checks passed' + (f'; failed: {failed}' if failed else ''))
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
