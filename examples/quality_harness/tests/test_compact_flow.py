import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import httpx
from bodhan_eval.cache import RequestCache
from bodhan_eval.cli import main
from bodhan_eval.models import HostedTransport, Models, Unavailable
from bodhan_eval.residual_judge import residual_tokens

ENV = {'LLM_BASE_URL': 'https://compact.invalid/v1', 'LLM_MODEL': 'mock-judge',
       'LLM_API_KEY': 'test-key', 'BODHAN_TRANSLATE_KEY': 'test-key'}
REPORTS = ('summary.md', 'summary.json', 'items.jsonl', 'calibration.json')


class CompactFlowTest(unittest.TestCase):
    def test_graph_records_invalid_ids_and_provider_failure_as_unavailable(self):
        for fail in (False, True):
            with self.subTest(fail=fail), TemporaryDirectory() as temp:
                root = Path(temp)
                data = root / 'input.jsonl'
                data.write_text(json.dumps({'id': 'x', 'lang': 'hi', 'source': 'A 2 kg and B 5 kg',
                                            'translation': 'A 3 kg and B 7 kg'}) + '\n')
                def wire(request, fail=fail):
                    body = json.loads(request.content)
                    if request.url.host == 'api.bodhan.ai':
                        return httpx.Response(200, json={'translation': 'Mock evidence', 'usage': {'completion_tokens': 4}})
                    if 'source_tokens' in json.loads(body['messages'][1]['content']):
                        if fail:
                            return httpx.Response(400)
                        answer = {'pairs': [{'source': 'S99', 'translation': 'T1', 'verdict': 'mismatch', 'reason': 'Differs.'}]}
                    else:
                        answer = {'verdict': 'uncertain', 'explanation': 'Mock uncertainty.', 'findings': []}
                    return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]})
                with httpx.Client(transport=httpx.MockTransport(wire)) as client:
                    self.assertEqual(main(['--dataset', str(data), '--langs', 'hi', '--out', str(root / 'report')],
                                          transport=HostedTransport(ENV, client)), 0)
                item = json.loads((root / 'report/items.jsonl').read_text())
                self.assertEqual(item['checks']['judge']['status'], 'unavailable')
                self.assertEqual(item['flags'], [])

    def test_provider_rejects_invalid_ids_and_failed_responses(self):
        content = {'source': '2 kg', 'translation': '3 kg',
                   'source_tokens': [{'id': 'S1', 'value': '2', 'context': '2 kg'}],
                   'translation_tokens': [{'id': 'T1', 'value': '3', 'context': '3 kg'}]}
        for status, answer in ((200, {'pairs': [{'source': 'S9', 'translation': 'T1', 'verdict': 'mismatch', 'reason': 'Differs.'}]}),
                               (503, {})):
            with self.subTest(status=status), TemporaryDirectory() as temp:
                def wire(request, status=status, answer=answer):
                    return httpx.Response(status, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}]})
                with httpx.Client(transport=httpx.MockTransport(wire)) as client:
                    cache = RequestCache(Path(temp) / 'cache.sqlite3')
                    self.addCleanup(cache.close)
                    models = Models(HostedTransport(ENV, client), cache, None, sleeper=lambda _: None)
                    with self.assertRaises(Unavailable):
                        models.residual_judge(content)

    def test_repeated_values_are_not_guessed(self):
        self.assertIsNone(residual_tokens('4 kg and 4 kg', '4 kg and 5 kg'))
        self.assertIsNone(residual_tokens('1 kg', '१ kg'))

    def test_graph_judge_labels_and_network_blocked_replay(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'input.jsonl'
            row = {'id': 'x', 'lang': 'hi', 'source': 'A 2 kg and B 5 kg',
                   'translation': 'A 3 kg and B 7 kg', 'labels': [{'type': 'number'}],
                   'injection': {'secret': 'never-send'}}
            data.write_text(json.dumps(row) + '\n')
            requests = []

            def wire(request):
                body = json.loads(request.content)
                if request.url.host == 'api.bodhan.ai':
                    return httpx.Response(200, json={'translation': 'Mock back translation', 'usage': {'completion_tokens': 4}})
                user = json.loads(body['messages'][1]['content'])
                requests.append(user)
                if 'source_tokens' in user:
                    answer = {'pairs': [{'source': 'S1', 'translation': 'T1', 'verdict': 'mismatch', 'reason': 'The first quantity differs.'},
                                        {'source': 'S2', 'translation': 'T2', 'verdict': 'uncertain', 'reason': 'The second pairing is unclear.'}]}
                else:
                    answer = {'verdict': 'uncertain', 'explanation': 'Mock uncertainty.', 'findings': []}
                return httpx.Response(200, json={'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(answer)}}],
                                                 'usage': {'prompt_tokens': 5, 'completion_tokens': 5}})

            with httpx.Client(transport=httpx.MockTransport(wire)) as client:
                code = main(['--dataset', str(data), '--langs', 'hi', '--out', str(root / 'report')], transport=HostedTransport(ENV, client))
            self.assertEqual(code, 0)
            item = json.loads((root / 'report/items.jsonl').read_text())
            self.assertEqual(item['checks']['judge']['status'], 'inconclusive')
            self.assertEqual(item['flags'][0]['category'], 'number')
            self.assertEqual(row['translation'][slice(*item['flags'][0]['span'])], '3')
            self.assertNotIn('never-send', json.dumps(requests))
            self.assertNotIn('labels', json.dumps(requests))
            script = ("import runpy,sys; "
                      "sys.addaudithook(lambda e,a: (_ for _ in ()).throw(AssertionError('network')) "
                      "if e in {'socket.connect','socket.getaddrinfo','socket.gethostbyname'} else None); "
                      "sys.argv=['bodhan_eval',*sys.argv[1:]]; runpy.run_module('bodhan_eval',run_name='__main__',alter_sys=True)")
            env = {key: val for key, val in os.environ.items() if key in {'PATH', 'TMPDIR', 'SYSTEMROOT'}}
            env.update(ENV)
            replay = subprocess.run([sys.executable, '-I', '-c', script, '--dataset', str(data), '--langs', 'hi',
                                     '--out', str(root / 'replay'), '--cache-only', '--replay-from', str(root / 'report')],
                                    cwd=root, env=env, capture_output=True, text=True, timeout=60)
            self.assertEqual(replay.returncode, 0, replay.stdout + replay.stderr)
            for name in REPORTS:
                self.assertEqual((root / 'report' / name).read_bytes(), (root / 'replay' / name).read_bytes())
            self.assertEqual(json.loads((root / 'replay.runtime.json').read_text())['provider_attempts'], 0)
