"""Independent read-only oracle for an authentic current inherited frontier.

No admission, learning, finalization, effect execution or root publication.
The exact saved cursor is resumed in its already registered owner and only
the generated token text is inspected. This is a tiny structural fixture,
never language-quality or organic-action evidence.
"""
from __future__ import annotations

import copy
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from omni_core.brain import AdaptiveBrain
from omni_core.connected_execution import ConnectedExecutionQueue, _digest, resolve_execution_engine
from omni_core.connected_execution_owner import discard_new_private_runtime
from omni_core.execution_checkpoint import ExecutionCheckpointStore
from omni_core.persistence import atomic_write_json, read_json
if __package__:
    from .capture_connected_execution_fixture import _inventory
else:
    from capture_connected_execution_fixture import _inventory


def reference_current_frontier(brain, execution_id=None, *, progress=None, cancelled=lambda: False):
    """Independently resume the saved cursor on an ALREADY loaded root.

    A Worker fixture can call this after its first explicit Load and before
    any actual turn continuation. It saves one redundant model Load without
    turning the later Worker response into its own posthoc oracle.
    """
    root_path = Path(brain.storage_path).resolve()
    folder = root_path.parent
    if not folder.name.startswith('omni-connected-current-seal-'):
        raise ValueError('reference requires its NEW owned current-seal fixture')
    capture = read_json(folder / 'capture-receipt.json')
    if capture.get('root') != str(root_path) or capture.get('oldCursorRestored') is not False:
        raise ValueError('reference cannot convert a historical unsupported frontier')
    receipt_path = folder / 'capture-reference.json'
    if receipt_path.exists():
        raise ValueError('the independent current-frontier reference already exists; do not repeat it')
    started = time.perf_counter()
    before_metadata = (root_path / 'engine/brain.json').read_bytes()
    metadata = read_json(root_path / 'engine/brain.json')
    if execution_id is not None and execution_id != capture['executionId']:
        raise ValueError('reference addresses another saved captured frontier')
    entry = next(row for row in metadata['connected_executions']['entries'] if row['id'] == capture['executionId'])
    source = resolve_execution_engine(root_path / 'engine', entry)
    store = ExecutionCheckpointStore(source, entry['parentId'])
    saved = store.require_supported_frontier(store.current(entry['sourceCheckpointId']))
    original_source, original_origin = _inventory(source), _inventory(root_path / 'engine/origin')
    runtime = None
    try:
        with patch.object(AdaptiveBrain, '_select_structured_actions', return_value=({'talk': 1., 'ponder': 0.}, [])):
            checksum, counters = brain.parameter_checksum(), copy.deepcopy(brain.counters)
            queue_before = copy.deepcopy(brain.connected_executions)
            with patch.object(AdaptiveBrain, '__init__', side_effect=AssertionError('reference allocated another model')):
                with ConnectedExecutionQueue(brain).prepare(entry['id']) as session:
                    runtime = session._new_runtime
                    source_counters = copy.deepcopy(session.brain.counters)
                    accepted = copy.deepcopy(session.brain.accepted_chat_inputs)
                    def present_only(context, **_callbacks):
                        if session.brain.counters != source_counters or session.brain.accepted_chat_inputs != accepted:
                            raise AssertionError('read-only oracle mutated input/counters before finalization')
                        tokens = context['generated'][0, context['prompt_ids'].shape[1]:].detach().cpu().tolist()
                        return {'text': session.brain.tokenizer.decode(tokens), 'tokens': tokens,
                            'finalizationSkipped': True, 'turnCommitted': False}
                    session.brain._finish_chat_continuation = present_only
                    observed = []
                    result = session.run(cancel_check=cancelled,
                        stream_callback=lambda kind, value: observed.append((kind, copy.deepcopy(value))))
                    if any(kind == 'action' for kind, _ in observed):
                        raise AssertionError('reference fixture unexpectedly emitted an action; no effect executed')
            if (brain.parameter_checksum() != checksum or brain.counters != counters
                    or brain.connected_executions != queue_before
                    or (root_path / 'engine/brain.json').read_bytes() != before_metadata
                    or _inventory(source) != original_source
                    or _inventory(root_path / 'engine/origin') != original_origin):
                raise AssertionError('reference modified its published learned/source/origin state')
            if saved['presentation']['publishedText'] + ''.join(value['delta'] for kind, value in observed if kind == 'token') != result['text']:
                raise AssertionError('read-only reference stream differs from saved-prefix plus real output')
            receipt = {'format': 'omni-current-native-frontier-reference', 'version': 1,
                'checkpointId': saved['checkpointId'], 'manifestSha256': _digest(saved),
                'runtimeCompatibilitySha256': saved['runtimeCompatibility']['sha256'],
                'text': result['text'], 'publishedText': saved['presentation']['publishedText'],
                'finalizationSkipped': True, 'originalSourceUnchanged': True,
                'isolation': {'learnedStateUnchanged': True, 'rootMetadataUnchanged': True,
                    'rootCountersUnchanged': True, 'rootQueueUnchanged': True,
                    'originUnchanged': True, 'effectsExecuted': 0},
                'elapsedSeconds': time.perf_counter() - started}
            atomic_write_json(receipt_path, receipt)
            if progress:
                progress('current-reference-captured', receipt)
            return receipt
    finally:
        if runtime is not None:
            discard_new_private_runtime(brain, entry['id'], runtime)


def capture_read_only_frontier_reference(root_path, *, progress=None):
    """Optional standalone driver; prefer an existing admitted root owner."""
    root_path = Path(root_path).resolve()
    metadata = read_json(root_path / 'engine/brain.json')
    with nullcontext():
        brain = AdaptiveBrain.load(root_path, expected_brain_id=metadata['brain_id'])
        try:
            return reference_current_frontier(brain, progress=progress)
        finally:
            brain.close()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('usage: capture_connected_frontier_reference.py <NEW-owned-capture-root>')
    torch.set_num_threads(1)
    capture_read_only_frontier_reference(sys.argv[1], progress=lambda kind, data: print(
        json.dumps({'phase': kind, **data}, sort_keys=True), flush=True))
