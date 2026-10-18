"""Fixed short own-native research comparison; never initializes a user brain.

Measures actual normal cached decoder generation versus the correct serial
prefix verifier, including proposal/selection, rejection, private cache cleanup
and round re-prefill overhead. No claimed DFlash2 parallel verifier is faked.
"""
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from omni_core.native_draft_evaluation import (NativeDraftRegion, evaluate_round,
    private_target_metadata, sampling_distribution)
from omni_core.optimizers import PackedMutationSnapshot
from omni_core.persistence import tensor_checksum
from tests.test_native_draft_evaluation import tiny_target


SEEDS = (4129, 4131, 4133)
REPEATS, TOKENS, TRAINING_STEPS = 3, 12, 24
PREFIXES = ((259, 65, 260), (259, 100, 101, 260))


def source_identity():
    root = Path(__file__).resolve().parents[1] / 'omni_core'
    names = ('model.py', 'native_draft_evaluation.py', 'cooperative_generation.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def train_own_auxiliary(target, region, prefixes):
    records = []
    for index, prefix in enumerate(prefixes):
        with torch.no_grad(), private_target_metadata(target):
            hidden = target(prefix)['hidden'][:, -1].detach()
            # This teacher is the same random native decoder, not a foreign
            # model, authored answer, preference label or claimed capability.
            teacher, _ = target.generate(prefix, max_new_tokens=region.block_size,
                seed=901 + index, temperature=.7, top_k=20, noise=0., printable_only=False)
            tokens = teacher[0, prefix.shape[1]:]
        records.append((hidden, prefix[:, -1], tokens))
    region.eval()
    with torch.no_grad():
        before_loss = statistics.mean(float(region.teacher_loss(target, *record)) for record in records)
    before = tensor_checksum(region.state_dict().values())
    snapshot = PackedMutationSnapshot.capture((region,), reserve=lambda _size: None)
    try:
        region.train()
        for module in region.modules():
            if hasattr(module, 'online_learning_rate'):
                module.online_learning_rate = .15
        started = time.perf_counter_ns()
        for step in range(TRAINING_STEPS):
            region.teacher_loss(target, *records[step % len(records)]).backward()
        elapsed = (time.perf_counter_ns() - started) / 1e6
        region.eval()
        with torch.no_grad():
            candidate_loss = statistics.mean(float(region.teacher_loss(target, *record)) for record in records)
        candidate = tensor_checksum(region.state_dict().values())
        retained = candidate_loss <= before_loss
        if not retained:
            snapshot.restore()
        return {'updates': TRAINING_STEPS, 'records': len(records), 'milliseconds': elapsed,
            'beforeLoss': before_loss, 'candidateLoss': candidate_loss,
            'candidateRetained': retained, 'beforeSha256': before, 'candidateSha256': candidate,
            'afterSha256': tensor_checksum(region.state_dict().values()),
            'nativeBytesChanged': candidate != before, 'target': 'same own native fixture decoder'}
    finally:
        snapshot.close()


def serial_speculative(target, region, prefix, seed):
    output = prefix.clone()
    draft = torch.Generator().manual_seed(seed ^ 0x5A5A)
    verifier = torch.Generator().manual_seed(seed)
    totals = {'rounds': 0, 'proposed': 0, 'accepted': 0, 'targetCalls': 0}
    while output.shape[1] - prefix.shape[1] < TOKENS:
        result = evaluate_round(target, region, output, TOKENS - (output.shape[1] - prefix.shape[1]),
            draft_generator=draft, verifier_generator=verifier, temperature=.7, top_k=20)
        output = torch.cat((output, result.tokens[None]), dim=1)
        totals['rounds'] += 1
        totals['proposed'] += result.proposed
        totals['accepted'] += result.accepted
        totals['targetCalls'] += result.target_calls
        if result.terminal:
            break
    return output, totals


def prefix_verification_error(target, prefix):
    suffix = torch.tensor([[66, 67, 68]])
    sequence = torch.cat((prefix, suffix), dim=1)
    with torch.no_grad():
        long = target(sequence)['logits'][0, prefix.shape[1] - 1:]
        actual = torch.stack([target(sequence[:, :end])['logits'][0, -1]
            for end in range(prefix.shape[1], sequence.shape[1] + 1)])
    total_variation = [float((sampling_distribution(left, .7, 20) -
        sampling_distribution(right, .7, 20)).abs().sum() / 2) for left, right in zip(long, actual)]
    return {'maxAbsoluteLogitError': float((long - actual).abs().max()),
        'maxTopKProbabilityTotalVariation': max(total_variation)}


def main():
    torch.set_num_threads(1)
    sources = source_identity()
    rows, training, verification, native_storage = [], [], [], []
    for seed in SEEDS:
        target = tiny_target(seed=seed, experts=True)
        region = NativeDraftRegion(8, 3, 4, 8).eval()
        prefixes = [torch.tensor([values]) for values in PREFIXES]
        target_before = tensor_checksum(target.state_dict().values())
        training.append({'seed': seed, **train_own_auxiliary(target, region, prefixes)})
        if tensor_checksum(target.state_dict().values()) != target_before:
            raise AssertionError('auxiliary training altered shared native target bytes')
        region_before = tensor_checksum(region.state_dict().values())
        native_storage.append({'seed': seed,
            'targetCheckpointTensorBytes': sum(value.numel() * value.element_size() for value in target.state_dict().values()),
            'auxiliaryCheckpointTensorBytes': sum(value.numel() * value.element_size() for value in region.state_dict().values()),
            'floatingLearnedParameterBytes': sum(value.numel() * value.element_size()
                for root in (target, region) for value in root.parameters() if value.is_floating_point()),
            'proposalScratchAdmissionBytes': region.scratch_bytes(target.config.vocab_size, region.block_size)})
        for index, prefix in enumerate(prefixes):
            verification.append({'seed': seed, 'prefix': index, **prefix_verification_error(target, prefix)})
            for repeat in range(REPEATS):
                normal = lambda: target.generate(prefix, seed=100 + repeat, max_new_tokens=TOKENS,
                    temperature=.7, top_k=20, noise=0., printable_only=False, use_cache=True)[0]
                operations = (('normal', normal), ('serialVerifiedDraft', lambda:
                    serial_speculative(target, region, prefix, 100 + repeat)))
                if repeat % 2:
                    operations = tuple(reversed(operations))
                for mode, operation in operations:
                    started = time.perf_counter_ns()
                    result = operation()
                    elapsed = (time.perf_counter_ns() - started) / 1e6
                    tokens, receipt = result if isinstance(result, tuple) else (result, {})
                    count = tokens.shape[1] - prefix.shape[1]
                    rows.append({'seed': seed, 'prefix': index, 'repeat': repeat, 'mode': mode,
                        'milliseconds': elapsed, 'outputTokens': count, 'millisecondsPerToken': elapsed / count,
                        **receipt})
        if tensor_checksum(target.state_dict().values()) != target_before or tensor_checksum(region.state_dict().values()) != region_before:
            raise AssertionError('private research generation altered learned native state')
    if source_identity() != sources:
        raise AssertionError('comparison source changed during the fixed run')
    medians = {mode: statistics.median(row['millisecondsPerToken'] for row in rows if row['mode'] == mode)
        for mode in ('normal', 'serialVerifiedDraft')}
    print(json.dumps({'format': 'omni-native-drafting-comparison-1', 'backend': 'cpu',
        'sourceHashes': sources, 'seeds': SEEDS, 'prefixes': PREFIXES, 'repeats': REPEATS,
        'maxNewTokens': TOKENS, 'ordinaryTemperature': .7, 'ordinaryTopK': 20,
        'noise': 0., 'printableMask': False, 'targetExperts': 2, 'auxiliaryBlock': 3,
        'training': training, 'storage': native_storage, 'parallelVerifierErrors': verification,
        'runs': rows, 'medianMillisecondsPerToken': medians,
        'serialVerifiedSpeedRatio': medians['normal'] / medians['serialVerifiedDraft'],
        'targetUnchanged': True, 'auxiliaryUnchangedDuringInference': True,
        'defaultChangeAuthorized': False, 'runtimeIntegrated': False,
        'decision': 'not enabled: general block verifier and exact cache/cursor contract absent; serial verification pays target cost per token',
        'scope': 'short own-native decoder research; not language quality, whole-app speed, Metal/CUDA throughput or production resume'}, indent=2))


if __name__ == '__main__':
    main()
