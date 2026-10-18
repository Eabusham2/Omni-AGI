import unittest
from types import SimpleNamespace

from omni_core.best_work import exploration_target, run_best_work, select_candidate, temperature_ladder


class BestWorkTests(unittest.TestCase):
    def test_ladder_limits_and_adaptive_budget_keep_normal_branch_separate(self):
        self.assertEqual(exploration_target(.1, 16), 0)
        self.assertEqual(exploration_target(.4, 16), 4)
        self.assertEqual(exploration_target(.6, 16), 8)
        self.assertEqual(exploration_target(.9, 16), 16)
        self.assertEqual(exploration_target(.9, 3), 3)
        ladder = temperature_ladder(16)
        self.assertEqual(ladder[0], .2)
        self.assertEqual(ladder[-1], .95)
        self.assertEqual(ladder, sorted(ladder))
        self.assertGreater(sum(.25 <= value <= .55 for value in ladder), 5)

    def test_search_interleaves_preserves_baseline_and_closes_all_admitted_state(self):
        advances, closed, released, specs = [], [], [], []
        class Session:
            def __init__(self, spec):
                self.spec, self.done, self.steps = spec, False, 0
                self.mean_log_probability = -.1
            def advance(self):
                advances.append(self.spec.index)
                if self.steps == 2:
                    self.done = True
                    return None
                self.steps += 1
                return {'step': self.steps, 'entropy': .4}
            def close(self):
                closed.append(self.spec.index)
        def start(spec):
            specs.append(spec)
            return Session(spec), {}, lambda: released.append(spec.index)
        result = run_best_work(seed=71, ordinary_temperature=.65, normalized_entropy=.5,
            exploration_budget=4, start=start, finish=lambda *_: {'text': 'same answer'},
            cancelled=lambda: False, steered=lambda: False)
        self.assertEqual(advances[:5], [0, 1, 2, 3, 4])
        self.assertEqual(specs[0].temperature, .65)
        self.assertTrue(specs[0].ordinary)
        self.assertEqual(result['admittedBranches'], 5)
        self.assertEqual(result['selected'], 0)
        self.assertEqual(result['candidateHostEffects'], 0)
        self.assertEqual(set(closed), set(range(5)))
        self.assertEqual(set(released), set(range(5)))
        self.assertFalse(result['selection']['correctnessProven'])

    def test_steer_aborts_every_candidate_without_selection_or_effect(self):
        flag = {'steer': False}; closed=[]
        class Session:
            done = False
            def advance(self):
                flag['steer'] = True
                return {'step': 0, 'entropy': 1.}
            def close(self):
                closed.append(1)
        result = run_best_work(seed=0, ordinary_temperature=.65, normalized_entropy=.8,
            exploration_budget=2, start=lambda _: (Session(), {}, lambda: None),
            finish=lambda *_: self.fail('interrupted candidates must not finalize'), cancelled=lambda: False,
            steered=lambda: flag['steer'])
        self.assertEqual(result['interrupted'], 'steered')
        self.assertIsNone(result['selection'])
        self.assertEqual(len(closed), 3)

    def test_invalid_code_is_not_correctness_evidence_and_all_fail_falls_back(self):
        candidates = [{'index': index, 'ordinary': index == 0, 'text': 'bad code',
            'meanLogProbability': -.01, 'verification': {'status': 'failed'}} for index in range(3)]
        index, evidence = select_candidate(candidates)
        self.assertEqual(index, 0)
        self.assertEqual(evidence['reason'], 'ordinary-fallback-all-static-checks-failed')
        self.assertFalse(evidence['calibratedConfidence'])

    def test_additional_resource_refusal_keeps_the_ordinary_branch(self):
        class Session:
            done = False
            mean_log_probability = -.2
            def advance(self): self.done = True
            def close(self): pass
        calls = []
        def start(spec):
            calls.append(spec.index)
            return (Session(), {}, lambda: None) if spec.ordinary else None
        result = run_best_work(seed=71, ordinary_temperature=.65, normalized_entropy=.9,
            exploration_budget=16, start=start, finish=lambda *_: {'text': 'ordinary answer'},
            cancelled=lambda: False, steered=lambda: False)
        self.assertEqual(calls, [0, 1])
        self.assertEqual(result['admittedBranches'], 1)
        self.assertTrue(result['admissionPaused'])
        self.assertTrue(result['candidates'][result['selected']]['ordinary'])

    def test_disagreement_expands_only_to_requested_budget(self):
        class Session:
            done = False
            mean_log_probability = -.2
            def advance(self): self.done = True
            def close(self): pass
        result = run_best_work(seed=71, ordinary_temperature=.65, normalized_entropy=.3,
            exploration_budget=8, start=lambda _: (Session(), {}, lambda: None),
            finish=lambda spec, *_: {'text': 'different answer ' + str(spec.index)},
            cancelled=lambda: False, steered=lambda: False)
        self.assertEqual(result['admittedBranches'], 9)
        self.assertLessEqual(max(candidate['temperature'] for candidate in result['candidates']), .95)
        self.assertTrue(result['candidates'][0]['ordinary'])

    def test_strong_completed_baseline_agreement_can_stop_unfinished_exploration(self):
        class Session:
            mean_log_probability = 0.
            def __init__(self, spec): self.done, self.spec, self.steps = False, spec, 0
            def advance(self):
                self.steps += 1
                self.done = self.spec.index < 2 or self.steps > 10
                return None if self.done else {'step': self.steps, 'entropy': .1}
            def close(self): pass
        progress = []
        result = run_best_work(seed=71, ordinary_temperature=.65, normalized_entropy=.9,
            exploration_budget=8, start=lambda spec: (Session(spec), {}, lambda: None),
            finish=lambda *_: {'text': 'same answer'}, cancelled=lambda: False,
            steered=lambda: False, progress=progress.append)
        self.assertEqual(result['earlyStoppedBranches'], list(range(2, 9)))
        self.assertEqual(result['selected'], 0)
        self.assertFalse(result['selection']['correctnessProven'])
        self.assertFalse(progress[-1]['active'])
        self.assertEqual(progress[-1]['stopReason'], 'agreement-convergence')

    def test_cleanup_failure_does_not_leave_other_branch_leases_held(self):
        releases = []
        class Session:
            done = False
            mean_log_probability = -.2
            def advance(self): self.done = True
            def close(self): pass
        def start(spec):
            def release():
                releases.append(spec.index)
                if spec.index == 1:
                    raise RuntimeError('fixture release failed')
            return Session(), {}, release
        with self.assertRaisesRegex(RuntimeError, 'release failed'):
            run_best_work(seed=71, ordinary_temperature=.65, normalized_entropy=.9,
                exploration_budget=3, start=start, finish=lambda *_: {'text': 'same'},
                cancelled=lambda: False, steered=lambda: False)
        self.assertEqual(set(releases), {0, 1, 2, 3})


if __name__ == '__main__':
    unittest.main()
