# Copyright 2026 The Skill Self-Evolve Authors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""CPU-only behavioral tests; deliberately independent of Ray, Torch, and verl imports."""

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

MODULE_PATH = Path(__file__).resolve().parents[3] / "verl/experimental/skill_harness/store.py"
SPEC = importlib.util.spec_from_file_location("standalone_skill_store", MODULE_PATH)
store_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = store_module
SPEC.loader.exec_module(store_module)

SkillStore = store_module.SkillStore
SkillArtifact = store_module.SkillArtifact
SnapshotConflict = store_module.SnapshotConflict
AdmissionError = store_module.AdmissionError
IntegrityError = store_module.IntegrityError


class SkillStoreTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "skills"
        self.store = SkillStore(self.root)

    def artifact(self, skill_id="clean-object", *, principle="Find, take, clean, then place the object.",
                 task_families=("clean",), provenance=None):
        return SkillArtifact(
            skill_id=skill_id,
            title="Clean and place",
            principle=principle,
            when_to_apply="When an object must be cleaned before placement.",
            task_families=task_families,
            provenance={"producer_episode_id": "episode-a"} if provenance is None else provenance,
        )

    def admitted(self, artifact=None, **kwargs):
        candidate = self.store.stage_candidate(artifact or self.artifact(), **kwargs)
        self.store.admit(candidate.candidate_id)
        return candidate

    def publish(self, artifact=None, **kwargs):
        parent = self.store.head()
        candidate = self.admitted(artifact, **kwargs)
        return self.store.publish([candidate.candidate_id], expected_parent=parent)

    def fingerprint(self):
        return {
            str(p.relative_to(self.root)): (p.read_bytes(), p.stat().st_mtime_ns, p.stat().st_mode)
            for p in self.root.rglob("*") if p.is_file()
        }

    def test_artifact_identity_and_provenance_are_immutable(self):
        provenance = {"producer_episode_id": "a", "evaluation": {"cases": ["one", "two"]}}
        artifact = self.artifact(provenance=provenance)
        equivalent = self.artifact(provenance=json.loads(json.dumps(provenance)))
        self.assertEqual(artifact.ref, equivalent.ref)
        provenance["evaluation"]["cases"].append("external mutation")
        self.assertEqual(tuple(artifact.provenance["evaluation"]["cases"]), ("one", "two"))
        with self.assertRaises(TypeError):
            artifact.provenance["producer_episode_id"] = "replacement"
        serialized = artifact.to_dict()
        serialized["principle"] = "A different principle"
        with self.assertRaises(IntegrityError):
            SkillArtifact.from_dict(serialized)

    def test_explicit_admission_is_required(self):
        parent = self.store.head()
        candidate = self.store.stage_candidate(self.artifact())
        self.assertEqual(self.store.retrieve("clean").skills, ())
        with self.assertRaises(AdmissionError):
            self.store.publish([candidate.candidate_id], expected_parent=parent)
        self.assertEqual(self.store.head(), parent)
        self.store.admit(candidate.candidate_id, reason="Passed publisher checks")
        snapshot = self.store.publish([candidate.candidate_id], expected_parent=parent,
                                      provenance={"publisher_version": "test-v1"})
        self.assertEqual(snapshot.skills, (candidate.artifact,))
        self.assertEqual(snapshot.parent_id, parent)
        self.assertEqual(snapshot.provenance["metadata"]["publisher_version"], "test-v1")
        self.assertEqual(SkillStore(self.root).head(), snapshot.snapshot_id)

    def test_rejected_candidates_cannot_be_published_or_readmitted(self):
        candidate = self.store.stage_candidate(self.artifact())
        self.store.reject(candidate.candidate_id, reason="No demonstrated transfer benefit")
        with self.assertRaises(AdmissionError):
            self.store.publish([candidate.candidate_id], expected_parent=self.store.head())
        with self.assertRaises(IntegrityError):
            self.store.admit(candidate.candidate_id)

    def test_service_payloads_are_plain_json_serializable(self):
        artifact = self.artifact(provenance={"trace": {"producer": "a", "tests": ["one", "two"]}})
        candidate = self.admitted(artifact)
        snapshot = self.store.publish([candidate.candidate_id], expected_parent=self.store.head())
        retrieved = self.store.retrieve("clean", token_counter=len)
        for value in (artifact, artifact.ref, candidate, snapshot, retrieved):
            payload = value.to_dict()
            self.assertEqual(json.loads(json.dumps(payload)), payload)

    def test_revisions_preserve_old_artifacts_and_snapshot_bytes(self):
        old = self.publish()
        old_bytes = (self.root / "snapshots" / f"{old.snapshot_id}.json").read_bytes()
        revised = self.artifact(principle="Take the object before cleaning it, then confirm the destination.")
        conflict = self.admitted(revised)
        with self.assertRaises(AdmissionError):
            self.store.publish([conflict.candidate_id], expected_parent=old.snapshot_id)
        replacement = self.admitted(revised, replace=True)
        new = self.store.publish([replacement.candidate_id], expected_parent=old.snapshot_id)
        self.assertNotEqual(new.skills[0].version, old.skills[0].version)
        self.assertEqual(self.store.load_artifact(old.skills[0]).principle, self.artifact().principle)
        self.assertEqual((self.root / "snapshots" / f"{old.snapshot_id}.json").read_bytes(), old_bytes)
        self.assertEqual(self.store.get_snapshot(old.snapshot_id), old)

    def test_compare_and_swap_rejects_stale_parent_and_candidate(self):
        parent = self.store.head()
        first = self.admitted(self.artifact("first"))
        second = self.admitted(self.artifact("second"))
        current = self.store.publish([first.candidate_id], expected_parent=parent)
        with self.assertRaises(SnapshotConflict):
            self.store.publish([second.candidate_id], expected_parent=parent)
        # Giving the newer parent does not silently rebase the older candidate.
        with self.assertRaises(SnapshotConflict):
            self.store.publish([second.candidate_id], expected_parent=current.snapshot_id)
        self.assertEqual(self.store.head(), current.snapshot_id)

    def test_publication_is_process_safe(self):
        parent = self.store.head()
        candidates = [self.admitted(self.artifact(name)) for name in ("first", "second")]
        code = """
import importlib.util,sys
spec=importlib.util.spec_from_file_location('child_store',sys.argv[1])
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
store=module.SkillStore(sys.argv[2])
try:
    store.publish([sys.argv[3]],expected_parent=sys.argv[4]);print('published')
except module.SnapshotConflict:
    print('conflict')
"""
        children = [subprocess.Popen(
            [sys.executable, "-c", code, str(MODULE_PATH), str(self.root), c.candidate_id, parent],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        ) for c in candidates]
        results = []
        for child in children:
            out, err = child.communicate(timeout=15)
            self.assertEqual(child.returncode, 0, err)
            results.append(out.strip())
        self.assertEqual(sorted(results), ["conflict", "published"])
        self.assertEqual(len(self.store.get_snapshot().skills), 1)

    def test_failed_head_commit_preserves_the_previous_snapshot(self):
        parent = self.store.head()
        previous_head = (self.root / "refs/main.json").read_bytes()
        candidate = self.admitted()
        original_write = self.store._atomic

        def fail_head_write(path, value):
            if path.parent.name == "refs":
                raise OSError("Simulated publication failure before the head commit")
            return original_write(path, value)

        with mock.patch.object(self.store, "_atomic", side_effect=fail_head_write):
            with self.assertRaises(OSError):
                self.store.publish([candidate.candidate_id], expected_parent=parent)
        self.assertEqual((self.root / "refs/main.json").read_bytes(), previous_head)
        restarted = SkillStore(self.root)
        self.assertEqual(restarted.head(), parent)
        self.assertEqual(restarted.retrieve("clean").skills, ())
        # An orphan immutable manifest is harmless and retryable.
        result = restarted.publish([candidate.candidate_id], expected_parent=parent)
        self.assertEqual(result.skills, (candidate.artifact,))

    def test_candidate_branches_are_isolated(self):
        parent = self.store.head()
        self.store.create_branch("candidate-a", from_snapshot=parent)
        self.store.create_branch("candidate-b", from_snapshot=parent)
        candidate = self.admitted(branch="candidate-a")
        private = self.store.publish([candidate.candidate_id], expected_parent=parent, branch="candidate-a")
        self.assertEqual(len(self.store.retrieve("clean", branch="candidate-a").skills), 1)
        self.assertEqual(self.store.retrieve("clean", branch="candidate-b").skills, ())
        self.assertEqual(self.store.retrieve("clean").skills, ())
        self.assertNotEqual(private.snapshot_id, self.store.head())
        with self.assertRaises(SnapshotConflict):
            self.store.publish([candidate.candidate_id], expected_parent=parent, branch="main")

    def test_read_only_eval_is_pinned_and_has_no_write_side_effects(self):
        first = self.publish()
        evaluation = self.store.read_only_view(first.snapshot_id)
        self.publish(self.artifact("new-skill"))
        baseline = self.fingerprint()
        self.assertEqual(evaluation.head(), first.snapshot_id)
        self.assertEqual(evaluation.get_snapshot(), first)
        self.assertEqual(len(evaluation.retrieve("clean").skills), 1)
        self.assertEqual(evaluation.load_artifact(first.skills[0]).ref, first.skills[0])
        with self.assertRaises(PermissionError):
            evaluation.stage_candidate(self.artifact("forbidden"))
        with self.assertRaises(PermissionError):
            evaluation.get_snapshot(self.store.head())
        with self.assertRaises(PermissionError):
            evaluation.load_artifact(self.store.get_snapshot().skills[-1])
        self.assertEqual(self.fingerprint(), baseline)

    def test_read_only_open_does_not_initialize_a_missing_store(self):
        missing = Path(self.temporary.name) / "missing"
        with self.assertRaises(FileNotFoundError):
            SkillStore(missing, read_only=True)
        self.assertFalse(missing.exists())

    def test_retrieval_is_deterministic_filtered_and_budgeted(self):
        candidates = [self.admitted(artifact) for artifact in (
            self.artifact("a-too-long", principle="Find clean " * 120),
            self.artifact("b-short"),
            self.artifact("c-general", principle="Check action prerequisites.", task_families=("*",)),
            self.artifact("d-unrelated", task_families=("heat",)),
        )]
        snapshot = self.store.publish([c.candidate_id for c in candidates], expected_parent=self.store.head())
        baseline = self.fingerprint()
        result = self.store.retrieve("clean", snapshot_id=snapshot.snapshot_id, query="find clean", token_budget=300,
                                     top_k=2, token_counter=len)
        self.assertEqual(result, self.store.retrieve("clean", snapshot_id=snapshot.snapshot_id, query="find clean",
                                                    token_budget=300, top_k=2, token_counter=len))
        self.assertLessEqual(result.token_count, 300)
        self.assertEqual(result.token_count, len(result.text))
        self.assertFalse(result.token_count_is_estimate)
        self.assertEqual([r.skill_id for r in result.skills], ["b-short"])
        self.assertEqual([r.skill_id for r in result.budget_excluded], ["a-too-long", "c-general"])
        self.assertIn("When to apply:", result.text)
        self.assertNotIn("d-unrelated", result.text)
        self.assertNotIn("Find clean Find clean", result.text)
        empty = self.store.retrieve("clean", token_budget=0, token_counter=len)
        self.assertEqual((empty.skills, empty.text, empty.token_count), ((), "", 0))
        self.assertEqual(self.fingerprint(), baseline)

    def test_whole_rendered_block_counts_against_the_budget(self):
        candidates = [self.admitted(self.artifact(name)) for name in ("one", "two")]
        self.store.publish([c.candidate_id for c in candidates], expected_parent=self.store.head())
        one = self.store.retrieve("clean", token_budget=10000, top_k=1, token_counter=len)
        bounded = self.store.retrieve("clean", token_budget=len(one.text), top_k=2, token_counter=len)
        self.assertEqual(len(bounded.skills), 1)
        self.assertEqual(bounded.token_count, len(one.text))
        self.assertTrue(self.store.retrieve("clean").token_count_is_estimate)

    def test_tampered_artifact_and_snapshot_fail_closed(self):
        snapshot = self.publish()
        blob = self.root / "artifacts" / f"{snapshot.skills[0].content_hash}.json"
        data = json.loads(blob.read_text()); data["principle"] = "Tampered text"
        blob.write_text(json.dumps(data))
        with self.assertRaises(IntegrityError):
            self.store.retrieve("clean")
        manifest = self.root / "snapshots" / f"{snapshot.snapshot_id}.json"
        data = json.loads(manifest.read_text()); data["parent_id"] = None
        manifest.write_text(json.dumps(data))
        with self.assertRaises(IntegrityError):
            self.store.get_snapshot()

    def test_invalid_names_fields_and_token_callback_fail_closed(self):
        with self.assertRaises(ValueError):
            self.artifact("../escape")
        with self.assertRaises(ValueError):
            self.artifact(principle=" ")
        invalid = self.artifact().to_dict()
        invalid["task_families"] = "clean"
        with self.assertRaises(ValueError):
            SkillArtifact.from_dict(invalid)
        with self.assertRaises(ValueError):
            self.store.create_branch("../escape")
        self.publish()
        with self.assertRaises(ValueError):
            self.store.retrieve("clean", token_counter=lambda _: True)
        with self.assertRaises(ValueError):
            self.store.retrieve("clean", token_budget=-1)


if __name__ == "__main__":
    unittest.main()
