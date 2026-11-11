# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Independent byte-prompt evidence cases for the counterfactual input gate.

These use real versioned skills and complete host-context manifests, but no
model, Ray, remote endpoint, external tokenizer, or benchmark execution.
"""

from __future__ import annotations

import copy
import importlib
import json
import sys
import tempfile
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
PACKAGE = "_counterfactual_exact_input_tests"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
context = importlib.import_module(PACKAGE + ".context")
probes = importlib.import_module(PACKAGE + ".probes")
store_module = importlib.import_module(PACKAGE + ".store")
credit = importlib.import_module(PACKAGE + ".credit")


class ByteTokenizer:
    """A reversible fixture tokenizer with real UTF-8 byte IDs and offsets."""

    is_fast = True

    @staticmethod
    def encode(text, **_kwargs):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **_kwargs):
        text = "".join(f"<{row['role']}>\n{row['content']}\n</{row['role']}>\n" for row in messages)
        if add_generation_prompt:
            text += "<assistant>\n"
        return self.encode(text) if tokenize else text

    def __call__(self, text, **_kwargs):
        offsets = [(index, index + 1) for index, character in enumerate(text) for _ in character.encode("utf-8")]
        return {"input_ids": self.encode(text), "offset_mapping": offsets}


class CounterfactualInputGateTest(unittest.TestCase):
    NATIVE_PREFIX = (
        "Native model instructions and tool schemas.\n"
        "<environment_context><cwd>/tmp/controlled-pair</cwd><current_date>2026-11-12</current_date>"
        "</environment_context>\n<user_task>\n"
    )
    NATIVE_SUFFIX = "\n</user_task>\n<assistant>"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.store = store_module.SkillStore(self.root / "store")
        self.renderer = context.TokenizerRenderer(ByteTokenizer())
        self.messages = [
            {"role": "system", "content": "Follow the authoritative environment rules."},
            {"role": "user", "content": (
                "Heat the mug α and put it on the counter.\nInitial state: mug, sink, counter."
            )},
        ]
        self.loading = {
            "enabled": True, "branch": "main", "top_k": 8, "skill_token_budget": 10000,
            "max_prompt_tokens": 20000, "tokenizer_path": "/fixture-only/byte-tokenizer",
            "tokenizer_revision": None, "tokenizer_version": "byte-tokenizer-v1", "render_kwargs": {},
        }

    def artifact(self, name):
        return store_module.SkillArtifact(
            name, "Procedure " + name, "Check preconditions, apply the procedure, and verify its result.",
            "Use when the corresponding environment preconditions hold.", ("heat_family",),
        )

    def publish(self, artifact, parent, branch):
        self.store.create_branch(branch, from_snapshot=parent)
        candidate = self.store.stage_candidate(artifact, branch=branch)
        self.store.admit(candidate.candidate_id)
        return self.store.publish([candidate.candidate_id], branch=branch, expected_parent=parent).snapshot_id

    def pair(self, *, baseline_count=0):
        snapshot = self.store.head()
        existing = []
        for index in range(baseline_count):
            artifact = self.artifact("existing_" + str(index))
            snapshot = self.publish(artifact, snapshot, "baseline_" + str(index))
            existing.append(artifact.ref)
        target = self.artifact("target_skill")
        treated = self.publish(target, snapshot, "treated")
        protocol = credit.CreditProtocol(
            evaluator_version="b" * 64, probe_protocol_version="exact-input-v1",
            task_distribution_version="original-train-v1", environment_version="alfworld-v1",
            baseline_snapshot_id=snapshot, required_exposures=1,
        )
        request = probes.ProbeRequest(
            "probe", "window", "curate", target.ref, treated, snapshot, "b" * 64, protocol, 4,
        )
        captures = {
            "with_skill": self.capture("with_skill", treated),
            "without_skill": self.capture("without_skill", snapshot),
        }
        return request, captures, existing

    def capture(self, name, snapshot, *, ordered_refs=None, messages=None):
        kwargs = {
            "store": self.store, "snapshot_id": snapshot, "task_family": "heat_family",
            "render_prompt": self.renderer.render, "encode_text": self.renderer.encode,
            "max_prompt_tokens": self.loading["max_prompt_tokens"],
            "skill_token_budget": self.loading["skill_token_budget"], "top_k": self.loading["top_k"],
            "query": "heat the mug", "group_id": "group", "tokenizer_version": "byte-tokenizer-v1",
        }
        base_messages = copy.deepcopy(messages or self.messages)
        if ordered_refs is None:
            built = context.build_skill_context(base_messages, **kwargs)
        else:
            # A storage-service ordering change is an input variation, not a
            # forged manifest: the public builder recomputes all ranges/hashes.
            original = store_module.SkillStore.retrieve

            def retrieve(view, *args, **options):
                result = original(view, *args, **options)
                return replace(result, skills=tuple(ordered_refs))

            with patch.object(store_module.SkillStore, "retrieve", retrieve):
                built = context.build_skill_context(base_messages, **kwargs)
        directory = self.root / name
        directory.mkdir(exist_ok=True)
        (directory / "agent-messages.json").write_text(json.dumps(built.messages, ensure_ascii=False))
        (directory / "skill-context.json").write_text(json.dumps(built.manifest, ensure_ascii=False))
        user = built.messages[-1]["content"]
        native = self.NATIVE_PREFIX + user + self.NATIVE_SUFFIX
        return {
            "extra_info": {"episode_output_dir": str(directory), "skill_context": built.manifest},
            "trajectories": [{
                "prompt_ids": ByteTokenizer.encode(native), "response_ids": [79, 75], "response_mask": [1, 1],
                "response_logprobs": [-0.3, -0.4], "extra_fields": {"min_global_steps": 0, "max_global_steps": 0},
            }],
        }

    @staticmethod
    def decoded(capture):
        return bytes(capture["trajectories"][0]["prompt_ids"]).decode("utf-8")

    @staticmethod
    def replace_native(capture, old, new):
        text = CounterfactualInputGateTest.decoded(capture)
        if old not in text:
            raise AssertionError("Fixture replacement did not target the actual native prompt")
        capture["trajectories"][0]["prompt_ids"] = ByteTokenizer.encode(text.replace(old, new, 1))

    def verify(self, captures, request):
        return probes.verify_pair_inputs(
            captures, request=request, store=self.store, loading=self.loading,
            decode_prompt_ids=lambda ids: bytes(ids).decode("utf-8"),
        )

    def assert_unobserved(self, captures, request):
        with self.assertRaises(probes.ProbeNotObserved) as caught:
            self.verify(captures, request)
        self.assertIsInstance(caught.exception.input_audit, dict)
        self.assertEqual(caught.exception.input_audit["status"], "unobserved")

    def test_empty_library_control_and_actual_target_text_are_observed_without_mutation(self):
        request, captures, _ = self.pair()
        original = copy.deepcopy(captures)
        messages = {
            name: (Path(capture["extra_info"]["episode_output_dir"]) / "agent-messages.json").read_bytes()
            for name, capture in captures.items()
        }
        self.assertIn(request.skill_ref.content_hash, self.decoded(captures["with_skill"]))
        self.assertNotIn(request.skill_ref.content_hash, self.decoded(captures["without_skill"]))
        self.assertNotIn(request.with_snapshot_id, self.decoded(captures["with_skill"]))
        self.assertEqual(self.verify(captures, request)["status"], "observed")
        self.assertEqual(captures, original)
        for name, capture in captures.items():
            self.assertEqual(
                (Path(capture["extra_info"]["episode_output_dir"]) / "agent-messages.json").read_bytes(),
                messages[name],
            )

    def test_nonempty_baseline_preserves_the_ordered_other_skills(self):
        request, captures, _ = self.pair(baseline_count=2)
        without = captures["without_skill"]["extra_info"]["skill_context"]["injected_refs"]
        with_refs = captures["with_skill"]["extra_info"]["skill_context"]["injected_refs"]
        self.assertEqual([ref for ref in with_refs if ref != request.skill_ref.to_dict()], without)
        self.assertEqual(self.verify(captures, request)["status"], "observed")

    def test_other_skill_eviction_or_reordering_is_not_a_target_only_intervention(self):
        request, captures, _ = self.pair(baseline_count=2)
        baseline = captures["without_skill"]["extra_info"]["skill_context"]["injected_refs"]
        baseline_refs = [store_module.SkillRef(**ref) for ref in baseline]
        for refs in (
            [request.skill_ref, baseline_refs[0]],
            [baseline_refs[1], request.skill_ref, baseline_refs[0]],
        ):
            with self.subTest(refs=[ref.skill_id for ref in refs]):
                changed = copy.deepcopy(captures)
                changed["with_skill"] = self.capture("with_skill", request.with_snapshot_id, ordered_refs=refs)
                self.assert_unobserved(changed, request)

    def test_rehashed_but_changed_base_task_or_system_message_is_rejected(self):
        request, captures, _ = self.pair()
        for index, replacement in ((-1, "Do a different task."), (0, "Replace the environment rules.")):
            forged = copy.deepcopy(self.messages)
            forged[index]["content"] = replacement
            changed = copy.deepcopy(captures)
            changed["with_skill"] = self.capture("with_skill", request.with_snapshot_id, messages=forged)
            self.assert_unobserved(changed, request)

    def test_actual_native_home_or_time_difference_is_never_normalized_away(self):
        request, captures, _ = self.pair()
        for first, second in (
            ("r0=/tmp/codex-home-" + "a" * 32 + "/skills/.system",
             "r0=/tmp/codex-home-" + "b" * 32 + "/skills/.system"),
            ("<current_time>2026-11-12T00:00:00Z</current_time>",
             "<current_time>2026-11-12T00:00:01Z</current_time>"),
        ):
            changed = copy.deepcopy(captures)
            for name, prefix in (("with_skill", first), ("without_skill", second)):
                self.replace_native(changed[name], self.NATIVE_PREFIX, prefix + "\n" + self.NATIVE_PREFIX)
            self.assert_unobserved(changed, request)

    def test_declared_skill_without_actual_native_text_and_duplicate_user_text_are_rejected(self):
        request, captures, _ = self.pair()
        with_messages = json.loads((self.root / "with_skill/agent-messages.json").read_text())
        without_messages = json.loads((self.root / "without_skill/agent-messages.json").read_text())
        user = with_messages[-1]["content"]
        for actual_user in (without_messages[-1]["content"], user + "\n" + user):
            changed = copy.deepcopy(captures)
            self.replace_native(changed["with_skill"], user, actual_user)
            self.assert_unobserved(changed, request)

    def test_snapshot_membership_without_actual_injected_target_is_unobserved(self):
        request, captures, _ = self.pair()
        captures["with_skill"] = self.capture("with_skill", request.with_snapshot_id, ordered_refs=[])
        self.assert_unobserved(captures, request)

    def test_missing_raw_task_messages_is_unobserved_instead_of_zero_utility(self):
        request, captures, _ = self.pair()
        (self.root / "with_skill/agent-messages.json").unlink()
        self.assert_unobserved(captures, request)


if __name__ == "__main__":
    unittest.main()
