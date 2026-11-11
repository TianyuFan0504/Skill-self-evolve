"""CPU contracts using the exact installed vLLM0.24 public abort methods."""

from __future__ import annotations

import ast
import collections
import hashlib
import logging
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
ENGINE_DIR = Path(
    os.environ.get("SKILL_VLLM_ABORT_SOURCE_DIR", "/opt/conda/lib/python3.11/site-packages/vllm/v1/engine")
)
PINNED = {
    "async_llm.py": "2128cf018fe136b698c5e89dc7325bc6dced9d516cc9d772695f044716970b64",
    "output_processor.py": "464c3cda347512cf1090f8e6471d0d3f0f65e286eb9a7b97be52efbc1ffd572a",
}
# This exact external/internal pair was present in the actual failed engine's
# SchedulerOutput and matched its immutable owned request registry.
EXTERNAL = "ed2499f1d4134934b75200a01ab43547"
INTERNAL = EXTERNAL + "-a4718379"
OTHER, OTHER_INTERNAL = "unrelated-owned-external", "unrelated-owned-external-12345678"


def method(path, class_name, method_name, namespace):
    tree = ast.parse(Path(path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    node = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == method_name
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[method_name]


class InstalledPublicAbortTest(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        for name, expected in PINNED.items():
            path = ENGINE_DIR / name
            if not path.is_file():
                raise unittest.SkipTest(
                    "Set SKILL_VLLM_ABORT_SOURCE_DIR to the independently archived installed0.24 sources"
                )
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise AssertionError(
                    "The installed public abort source differs from the reviewed0.24 interface: " + name
                )
        scope = {"FinishReason": SimpleNamespace(ABORT="abort"), "EMPTY_CPU_TENSOR": object()}
        cls.output_abort = method(ENGINE_DIR / "output_processor.py", "OutputProcessor", "abort_requests", scope)
        cls.public_abort = method(
            ENGINE_DIR / "async_llm.py",
            "AsyncLLM",
            "abort",
            {
                "as_list": list,
                "logger": logging.getLogger("cpu-public-abort"),
            },
        )
        cls.core_abort = method(
            ROOT / "verl/workers/rollout/vllm_rollout/vllm_async_server.py",
            "vLLMHttpServer",
            "abort_request",
            {"logger": logging.getLogger("cpu-core-abort")},
        )

    async def asyncSetUp(self):
        self.messages, self.core_calls, self.finished = [], [], []

        def state(external, internal):
            return SimpleNamespace(
                external_req_id=external,
                request_id=internal,
                lora_name=None,
                detokenizer=object(),
                queue=SimpleNamespace(put=lambda value: self.messages.append((external, value))),
                make_request_output=lambda **kwargs: kwargs,
            )

        self.processor = SimpleNamespace(
            external_req_ids=collections.defaultdict(list, {EXTERNAL: [INTERNAL], OTHER: [OTHER_INTERNAL]}),
            request_states={INTERNAL: state(EXTERNAL, INTERNAL), OTHER_INTERNAL: state(OTHER, OTHER_INTERNAL)},
            parent_requests={},
            lora_states=SimpleNamespace(request_finished=lambda *args: self.finished.append(args)),
        )
        self.processor.abort_requests = types.MethodType(type(self).output_abort, self.processor)

        async def abort_core(ids):
            self.core_calls.append(list(ids))

        self.engine = SimpleNamespace(
            output_processor=self.processor,
            engine_core=SimpleNamespace(abort_requests_async=abort_core),
            log_requests=False,
        )
        self.engine.abort = types.MethodType(type(self).public_abort, self.engine)
        self.server = SimpleNamespace(node_rank=0, engine=self.engine)

    async def abort(self, identity=EXTERNAL, **kwargs):
        return await type(self).core_abort(self.server, identity, **kwargs)

    async def test_actual_public_api_maps_only_the_verified_external_request(self):
        result = await self.abort()
        self.assertTrue(result["aborted"])
        self.assertEqual(result["ack_scope"], "submitted_not_gpu_drained")
        self.assertFalse(result["prefix_cache_reset"])
        self.assertEqual(result["accepted_internal_request_ids"], [INTERNAL])
        self.assertEqual(self.core_calls, [[INTERNAL]])
        self.assertNotIn(INTERNAL, self.processor.request_states)
        self.assertNotIn(EXTERNAL, self.processor.external_req_ids)
        self.assertIn(OTHER_INTERNAL, self.processor.request_states)
        self.assertEqual(self.processor.external_req_ids[OTHER], [OTHER_INTERNAL])
        self.assertEqual(len(self.messages), 1)  # Only public OutputProcessor creates the ABORT output.
        self.assertEqual(self.messages[0][1]["finish_reason"], "abort")

    async def test_notfound_has_no_defaultdict_side_effect_and_never_guesses_suffix(self):
        before = dict(self.processor.external_req_ids)
        result = await self.abort("missing-external")
        self.assertFalse(result["aborted"])
        self.assertEqual(self.processor.external_req_ids, before)
        self.assertEqual(self.core_calls, [])
        self.processor.external_req_ids.pop(EXTERNAL)
        result = await self.abort()  # The matching-looking internal key still exists; never infer it.
        self.assertFalse(result["aborted"])
        self.assertIn(INTERNAL, self.processor.request_states)

    async def test_ambiguous_stale_and_mismatched_identity_fail_before_any_abort(self):
        self.processor.external_req_ids[EXTERNAL].append(OTHER_INTERNAL)
        self.assertEqual((await self.abort())["reason"], "ambiguous_external_request_mapping")
        self.processor.external_req_ids[EXTERNAL] = ["stale-internal"]
        self.assertEqual((await self.abort())["reason"], "stale_external_request_mapping")
        self.processor.external_req_ids[EXTERNAL] = [INTERNAL]
        self.processor.request_states[INTERNAL].external_req_id = OTHER
        self.assertEqual((await self.abort())["reason"], "request_identity_mismatch")
        self.assertEqual(self.core_calls, [])
        self.assertEqual(self.messages, [])

    async def test_submission_error_never_acknowledges_gpu_drain(self):
        async def unavailable(_ids):
            raise OSError("CPU injected EngineCore submission failure")

        self.engine.engine_core.abort_requests_async = unavailable
        result = await self.abort()
        self.assertFalse(result["aborted"])
        self.assertEqual(result["ack_scope"], "submission_unconfirmed")
        self.assertEqual(result["error_type"], "OSError")
        self.assertIn(OTHER_INTERNAL, self.processor.request_states)

    async def test_nonprimary_or_global_cache_reset_never_calls_abort(self):
        self.assertEqual(
            (await self.abort(reset_prefix_cache=True))["reason"], "targeted_abort_cannot_reset_shared_prefix_cache"
        )
        self.server.node_rank = 1
        self.assertEqual((await self.abort())["reason"], "not_primary_node")
        self.assertEqual(self.core_calls, [])
        self.assertEqual(self.messages, [])


class TailAndFingerprintTest(unittest.IsolatedAsyncioTestCase):
    async def assert_partial_tail(self, stop_reason):
        path = ROOT / "verl/experimental/skill_harness/alfworld/backend_terminal.py"
        tree = ast.parse(path.read_text())
        tree.body = [
            node
            for node in tree.body
            if not (isinstance(node, ast.ImportFrom) and node.module == "verl.workers.rollout.llm_server")
        ]
        module = types.ModuleType("_actual_terminal_tail_cpu")
        sys.modules[module.__name__] = module
        self.addCleanup(lambda: sys.modules.pop(module.__name__, None))
        original = SimpleNamespace(
            token_ids=[17, 19], log_probs=[-0.25, -0.75], stop_reason=stop_reason, extra_fields={"global_steps": 7}
        )

        class Transport:
            def __init__(self, **_):
                pass

            async def generate(self, *_args, **_kwargs):
                lease = module._CURRENT.get()
                lease.actual_request_id = EXTERNAL
                module.fence_episode(lease.binding, {"done": True, "won": True, "steps": 2, "max_steps": 50})
                return original

        module.__dict__["LLMServerClient"] = Transport
        exec(compile(tree, str(path), "exec"), module.__dict__)
        with tempfile.TemporaryDirectory() as temporary:
            binding = module.bind_episode(
                temporary,
                gateway_base_url="http://127.0.0.1:1/sessions/owned/v1",
                environment_session_url="http://127.0.0.1:2/session/env",
                attempt_id="attempt",
            )
            client = module.TerminalAwareLLMServerClient({}, binding_root=temporary)

            async def nonterminal(*_):
                return {"done": False, "won": False, "steps": 1, "max_steps": 50}

            with patch.object(module, "_observe", side_effect=nonterminal):
                with self.assertRaises(module.TerminalMDPStopped):
                    await client.generate("owned", prompt_ids=[1, 2, 3], sampling_params={"logprobs": True})
            receipt = await module.wait_for_terminal_drain(binding, 0.2)
            tail = module._read(Path(receipt["tail_archive_refs"][0]["path"]))
            self.assertEqual(tail["prompt_ids"], [1, 2, 3])
            self.assertEqual(tail["token_ids"], original.token_ids)
            self.assertEqual(tail["log_probs"], original.log_probs)
            self.assertEqual(tail["raw_stop_reason"], stop_reason)
            self.assertTrue(tail["archive_only"])
            self.assertFalse(tail["behavior_gradient"])
            self.assertFalse(tail["model_eos_invented"])

    async def test_partial_ids_and_mu_remain_archive_only_without_codec_or_model_eos(self):
        await self.assert_partial_tail("aborted")

    async def test_natural_completion_race_retains_its_raw_completed_reason(self):
        await self.assert_partial_tail("completed")

    def test_checkpoint_fingerprint_binds_the_core_server_bytes(self):
        tree = ast.parse((ROOT / "verl/experimental/skill_harness/joint_trainer.py").read_text())
        function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_source_fingerprint"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for node in ast.walk(function):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.startswith(("verl/", "examples/")) and node.value.endswith(
                        (".py", ".sh", ".md", ".txt")
                    ):
                        path = root / node.value
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(b"original")
            scope = {
                "Path": Path,
                "hashlib": hashlib,
                "verl": SimpleNamespace(__file__=str(root / "verl/__init__.py")),
                "actual_adapter_sha256": lambda: "verified-external",
                "_digest": lambda value: repr(value),
            }
            exec(compile(ast.Module(body=[function], type_ignores=[]), "actual_joint_fingerprint", "exec"), scope)
            initial = scope["_source_fingerprint"]()
            core = root / "verl/workers/rollout/vllm_rollout/vllm_async_server.py"
            core.write_bytes(b"changed abort interface")
            self.assertNotEqual(scope["_source_fingerprint"](), initial)


if __name__ == "__main__":
    unittest.main()
