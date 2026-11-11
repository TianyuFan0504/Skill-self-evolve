"""Offline CPU checks of token budgets, group pins, and native TQ preservation."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness"
PACKAGE = "standalone_skill_context"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package


def source_module(name):
    spec = importlib.util.spec_from_file_location(PACKAGE + "." + name, ROOT / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


store_module = source_module("store")
context = source_module("context")


class ByteTokenizer:
    """One actual callback token per UTF-8 byte, with exact character offsets."""

    is_fast = True

    def encode(self, text, **kwargs):
        return list(text.encode("utf-8"))

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, **kwargs):
        text = "".join("<" + message["role"] + ">" + message["content"] + "</turn>" for message in messages)
        if add_generation_prompt:
            text += "<assistant>"
        return self.encode(text) if tokenize else text

    def __call__(self, text, **kwargs):
        offsets = [pair for index, char in enumerate(text) for pair in [(index, index + 1)] * len(char.encode("utf-8"))]
        return {"input_ids": self.encode(text), "offset_mapping": offsets}


class NativeFrameworkStub:
    @classmethod
    def from_config(cls, **kwargs):
        return cls()

    async def _run_prompt_rollouts(self, **kwargs):
        self.native_prompt_call = kwargs
        return [copy.deepcopy(kwargs["sample_fields"]) for _ in range(kwargs["num_sessions"])]

    def _trajectory_to_tq_field_and_tag(self, *, trajectory, **kwargs):
        return {
            "prompts": trajectory.prompt_ids,
            "responses": trajectory.response_ids,
            "rollout_log_probs": trajectory.response_logprobs,
            "response_mask": trajectory.response_mask,
            "extra_fields": {"native_extension": "preserved"},
        }, {"global_steps": kwargs["global_steps"]}


class NativeAdapterStub:
    @classmethod
    def create(cls, **kwargs):
        instance = cls()
        instance.native_create_kwargs = kwargs
        return instance


def framework_module():
    entry = types.ModuleType("uni_agent.framework.entry")
    entry.AgentFrameworkRolloutAdapter = NativeAdapterStub
    framework = types.ModuleType("uni_agent.framework.framework")
    framework.GatewayAgentFramework = NativeFrameworkStub
    with patch.dict(sys.modules, {"uni_agent.framework.entry": entry, "uni_agent.framework.framework": framework}):
        return source_module("framework")


framework = framework_module()


class ContextTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = store_module.SkillStore(self.root / "store")
        self.renderer = context.TokenizerRenderer(ByteTokenizer())
        self.messages = [
            {"role": "system", "content": "Environment commands only."},
            {"role": "user", "content": "Clean and place the mug.\nInitial game state: sink, mug, counter."},
        ]

    def publish(self, skill_id="clean", *, principle="Take the object, clean it, then place it."):
        artifact = store_module.SkillArtifact(
            skill_id,
            "Reusable cleaning",
            principle,
            "When the goal requires a clean object.",
            ("clean",),
        )
        parent = self.store.head()
        candidate = self.store.stage_candidate(artifact)
        self.store.admit(candidate.candidate_id)
        return self.store.publish([candidate.candidate_id], expected_parent=parent)

    def build(self, **kwargs):
        defaults = dict(
            store=self.store,
            snapshot_id=self.store.head(),
            task_family="clean",
            render_prompt=self.renderer.render,
            encode_text=self.renderer.encode,
            max_prompt_tokens=10000,
            skill_token_budget=5000,
            query="clean mug",
            group_id="group-a",
        )
        defaults.update(kwargs)
        return context.build_skill_context(self.messages, **defaults)

    def loading(self):
        return {
            "enabled": True,
            "branch": "main",
            "snapshot_id": None,
            "top_k": 6,
            "skill_token_budget": 5000,
            "max_prompt_tokens": 10000,
            "tokenizer_path": str(self.root / "model"),
            "tokenizer_revision": None,
            "tokenizer_version": "fake-exact-byte-tokenizer-v1",
            "render_kwargs": {},
        }

    def fields(self):
        return {
            "uid": "prompt-a",
            "raw_prompt": self.messages,
            "tools_kwargs": {"task": {"name": "alfworld_skill_training", "metadata": {"task_type": "clean"}}},
        }

    def test_real_token_callback_bounds_complete_render_and_skill_block(self):
        self.publish(principle="清洁物品。" * 100)
        generous = self.build()
        text = generous.messages[-1]["content"][len(self.messages[-1]["content"]) :]
        self.assertGreater(len(text.encode()), len(text))
        limited = self.build(skill_token_budget=len(text))
        self.assertEqual(limited.manifest["injected_refs"], [])
        self.assertEqual(len(limited.manifest["selected_refs"]), 1)
        self.assertIn("skill_token_budget", limited.manifest["budget_excluded"][0]["reasons"])
        exact = self.build(skill_token_budget=len(text.encode()))
        self.assertEqual(exact.manifest["skill_tokens"], len(text.encode()))
        self.assertEqual(exact.manifest["rendered_prompt_tokens"], len(self.renderer.render(exact.messages).token_ids))

    def test_full_render_includes_role_and_generation_overhead(self):
        self.publish()
        source_chars = sum(len(message["content"]) for message in self.messages)
        with self.assertRaises(context.ContextBudgetError):
            self.build(max_prompt_tokens=source_chars)
        base_count = len(self.renderer.render(self.messages).token_ids)
        result = self.build(max_prompt_tokens=base_count)
        self.assertEqual(result.messages, self.messages)
        self.assertEqual(result.manifest["injected_refs"], [])
        self.assertEqual(result.manifest["rendered_prompt_tokens"], base_count)

    def test_injected_refs_hashes_and_ranges_match_actual_user_and_render(self):
        self.publish()
        result = self.build()
        rendered = self.renderer.render(result.messages)
        record = result.manifest["ranges"][0]
        left, right = record["user_char_range"]
        actual = result.messages[-1]["content"][left:right]
        start, end = record["rendered_char_range"]
        self.assertEqual(rendered.text[start:end], actual)
        self.assertEqual(hashlib.sha256(actual.encode()).hexdigest(), record["text_sha256"])
        token_left, token_right = record["rendered_overlap_token_range"]
        self.assertEqual(tuple(actual.encode()), rendered.token_ids[token_left:token_right])
        self.assertEqual(result.manifest["selected_refs"], result.manifest["injected_refs"])
        self.assertFalse(result.manifest["actual_native_request_verified"])
        self.assertIn("native_codex_prefix_and_tools_excluded", result.manifest["budget_scope"])
        self.assertEqual(
            self.messages[-1]["content"], "Clean and place the mug.\nInitial game state: sink, mug, counter."
        )

    def test_whole_skills_skip_oversize_and_preserve_small_later_candidate(self):
        self.publish("a-long", principle="Oversized cleaning notes. " * 300)
        self.publish("b-short")
        result = self.build(skill_token_budget=700)
        self.assertEqual([ref["skill_id"] for ref in result.manifest["selected_refs"]], ["a-long", "b-short"])
        self.assertEqual([ref["skill_id"] for ref in result.manifest["injected_refs"]], ["b-short"])
        self.assertNotIn("Oversized cleaning", result.messages[-1]["content"])

    def test_snapshot_is_read_only_and_cannot_load_newly_published_notes(self):
        old = self.publish()
        self.publish("new")
        before = {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.store.root.rglob("*")
            if path.is_file()
        }
        result = self.build(snapshot_id=old.snapshot_id)
        self.assertEqual([ref["skill_id"] for ref in result.manifest["injected_refs"]], ["clean"])
        after = {
            str(path): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.store.root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)

    def test_disabled_task_hook_opens_no_store_or_tokenizer(self):
        with (
            patch.object(context, "_local_tokenizer", side_effect=AssertionError("No tokenizer in baseline")),
            patch.object(context, "SkillStore", side_effect=AssertionError("No library in baseline")),
        ):
            result = context.enrich_task_messages(self.messages, {})
        self.assertEqual(result.messages, self.messages)
        self.assertFalse(result.manifest["enabled"])

    def test_pin_copies_metadata_and_retains_snapshot_across_n_way_fanout(self):
        old = self.publish()
        fields = self.fields()
        instance = framework.SkillAwareAgentFramework()
        instance._skill_loading, instance._skill_store_root = self.loading(), self.store.root
        outcomes = asyncio.run(
            instance._run_prompt_rollouts(
                sample_fields=fields,
                sample_index=0,
                global_steps=3,
                partition_id="train",
                num_sessions=4,
            )
        )
        self.publish("new")
        metadata = [row["tools_kwargs"]["task"]["metadata"] for row in outcomes]
        self.assertEqual({row["skill_snapshot_id"] for row in metadata}, {old.snapshot_id})
        self.assertEqual(len({row["skill_context_version"] for row in metadata}), 1)
        self.assertEqual(len({row["skill_context_group_id"] for row in metadata}), 1)
        self.assertNotIn("skill_snapshot_id", fields["tools_kwargs"]["task"]["metadata"])
        context_result = context.enrich_task_messages(self.messages, metadata[0], renderer=self.renderer)
        self.assertEqual([ref["skill_id"] for ref in context_result.manifest["injected_refs"]], ["clean"])

    def test_tq_hook_only_adds_provenance_to_native_sampled_data(self):
        self.publish()
        fields = context.pin_prompt_group(
            self.fields(),
            loading=self.loading(),
            store_root=self.store.root,
            global_step=3,
            partition_id="train",
        )
        metadata = fields["tools_kwargs"]["task"]["metadata"]
        built = context.enrich_task_messages(self.messages, metadata, renderer=self.renderer)
        trajectory = SimpleNamespace(
            prompt_ids=[1, 7, 19],
            response_ids=[23, 2, 9],
            response_logprobs=[-0.3, -0.2, -0.4],
            response_mask=[1, 0, 1],
            extra_fields={"runner_reward_info": {"reward_context": {"skill_context": built.manifest}}},
        )
        before = copy.deepcopy(trajectory)
        instance = framework.SkillAwareAgentFramework()
        instance._skill_loading = self.loading()
        field, tag = instance._trajectory_to_tq_field_and_tag(
            trajectory=trajectory,
            sample_fields=fields,
            session_index=0,
            global_steps=3,
            uid="prompt-a",
        )
        self.assertEqual(trajectory.__dict__, before.__dict__)
        for key, attribute in (
            ("prompts", "prompt_ids"),
            ("responses", "response_ids"),
            ("rollout_log_probs", "response_logprobs"),
            ("response_mask", "response_mask"),
        ):
            self.assertIs(field[key], getattr(trajectory, attribute))
        provenance = field["extra_fields"]["skill_prompt_provenance"]
        self.assertEqual(provenance["actual_prompt_tokens"], 3)
        self.assertEqual(provenance["actual_prompt_token_ids_sha256"], context.token_ids_hash([1, 7, 19]))
        self.assertEqual(tag["skill_snapshot_id"], metadata["skill_snapshot_id"])
        self.assertEqual(field["extra_fields"]["native_extension"], "preserved")

    def test_context_pin_mismatch_is_rejected_before_tq_publish(self):
        self.publish()
        fields = context.pin_prompt_group(
            self.fields(),
            loading=self.loading(),
            store_root=self.store.root,
            global_step=3,
            partition_id="train",
        )
        metadata = fields["tools_kwargs"]["task"]["metadata"]
        built = context.enrich_task_messages(self.messages, metadata, renderer=self.renderer)
        built.manifest["snapshot_id"] = "0" * 64
        trajectory = SimpleNamespace(
            prompt_ids=[1],
            response_ids=[2],
            response_logprobs=[-1],
            response_mask=[1],
            extra_fields={"runner_reward_info": {"reward_context": {"skill_context": built.manifest}}},
        )
        instance = framework.SkillAwareAgentFramework()
        instance._skill_loading = self.loading()
        with self.assertRaisesRegex(ValueError, "trusted prompt-group pin"):
            instance._trajectory_to_tq_field_and_tag(
                trajectory=trajectory,
                sample_fields=fields,
                session_index=0,
                global_steps=3,
                uid="prompt-a",
            )

    def test_same_group_retry_reuses_durable_snapshot_after_head_advances(self):
        first = self.publish()
        kwargs = dict(
            loading=self.loading(),
            store_root=self.store.root,
            global_step=3,
            partition_id="train",
            pin_directory=self.root / "context_groups",
        )
        original = context.pin_prompt_group(self.fields(), **kwargs)
        self.publish("new")
        retried = context.pin_prompt_group(self.fields(), **kwargs)
        old_metadata = original["tools_kwargs"]["task"]["metadata"]
        new_metadata = retried["tools_kwargs"]["task"]["metadata"]
        self.assertEqual(old_metadata, new_metadata)
        self.assertEqual(new_metadata["skill_snapshot_id"], first.snapshot_id)
        changed = self.fields()
        changed["raw_prompt"] = [{"role": "user", "content": "A different task"}]
        with self.assertRaisesRegex(ValueError, "UID was reused"):
            context.pin_prompt_group(changed, **kwargs)
        record_path = next((self.root / "context_groups").glob("*.json"))
        record = json.loads(record_path.read_text())
        record["metadata"]["skill_snapshot_id"] = self.store.head()
        record_path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError, "integrity check"):
            context.pin_prompt_group(self.fields(), **kwargs)

    def test_adapter_baseline_delegates_original_config_without_tokenizer(self):
        config = SimpleNamespace(skill_harness={"mode": "solver_baseline", "loading": {"enabled": False}})
        with patch.object(framework, "_tokenizer_version", side_effect=AssertionError("Baseline must not load skills")):
            adapter = framework.SkillAwareRolloutAdapter.create(config=config, llm_client="native-client")
        self.assertIs(adapter.native_create_kwargs["config"], config)
        self.assertEqual(adapter.native_create_kwargs["llm_client"], "native-client")
        config.skill_harness["loading"]["enabled"] = True
        with self.assertRaisesRegex(ValueError, "solver_baseline"):
            framework.SkillAwareRolloutAdapter.create(config=config)

    def test_token_callback_estimates_and_non_integer_ids_are_rejected(self):
        self.publish()
        with self.assertRaises(TypeError):
            self.build(render_prompt=lambda messages: len(json.dumps(messages)))
        with self.assertRaises(ValueError):
            self.build(encode_text=lambda text: [0.5])


if __name__ == "__main__":
    unittest.main()
