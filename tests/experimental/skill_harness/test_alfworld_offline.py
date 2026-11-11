"""Offline contract checks; no Ray, Torch, Uni-Agent, TextWorld or GPU required."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Import the dependency-free subpackage without executing verl.__init__, which
# intentionally loads the training stack. Production imports use the normal FQN.
REPO = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = REPO / "verl/experimental/skill_harness/alfworld"
PACKAGE_NAME = "_alfworld_offline_contract"
if PACKAGE_NAME not in sys.modules:
    package = types.ModuleType(PACKAGE_NAME)
    package.__path__ = [str(PACKAGE_ROOT)]
    sys.modules[PACKAGE_NAME] = package
data = importlib.import_module(PACKAGE_NAME + ".data")
protocol = importlib.import_module(PACKAGE_NAME + ".protocol")


def make_game(root, split, scene, trial, task_type="pick_clean_then_place_in_recep", *, solvable=True):
    folder = root / "data/json_2.1.1" / split / f"{task_type}-Apple-None-DiningTable-{scene}" / trial
    folder.mkdir(parents=True)
    (folder / "game.tw-pddl").write_text(json.dumps({"solvable": solvable, "test_instance": trial}))
    (folder / "traj_data.json").write_text(
        json.dumps(
            {"task_type": task_type, "scene": {"floor_plan": f"FloorPlan{scene}"}, "plan": "EXPERT_SECRET"}
        )
    )
    (folder / "initial_state.pddl").write_text("private environment state")
    return folder / "game.tw-pddl"


class TestAlfworldDataContracts(unittest.TestCase):
    def test_scene_group_split_is_train_only_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for scene in range(12):
                make_game(root, "train", scene, "first")
                make_game(root, "train", scene, "second", "pick_heat_then_place_in_recep")
            make_game(root, "valid_seen", 0, "seen")
            make_game(root, "valid_unseen", 99, "unseen")
            records = data.build_catalog(root)
            parts = data.partition_records(records, dev_fraction=0.25, seed=173)
            repeated = data.partition_records(list(reversed(records)), dev_fraction=0.25, seed=173)
            self.assertEqual(parts, repeated)
            train_scenes = {r.scene_id for r in parts["train"]}
            dev_scenes = {r.scene_id for r in parts["dev"]}
            self.assertTrue(train_scenes and dev_scenes)
            self.assertFalse(train_scenes & dev_scenes)
            self.assertEqual(len(parts["train"]) + len(parts["dev"]), 24)
            self.assertTrue(all(r.split == "train" for r in parts["train"] + parts["dev"]))
            self.assertEqual([r.split for r in parts["test_seen"]], ["valid_seen"])
            self.assertEqual([r.split for r in parts["test_unseen"]], ["valid_unseen"])

    def test_game_identity_survives_catalog_growth_order_and_mount_relocation(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            root, relocated = Path(first), Path(second)
            game = make_game(root, "train", 7, "unchanged")
            make_game(relocated, "train", 7, "unchanged")
            original = data.build_catalog(root)[0]
            self.assertEqual(original, data.build_catalog(relocated)[0])
            make_game(root, "train", 1, "sorts_before_original")
            grown = data.build_catalog(root)
            matching = [r for r in grown if r.relative_game_path == original.relative_game_path]
            self.assertEqual(matching, [original])
            self.assertEqual(data.catalog_digest(grown), data.catalog_digest(list(reversed(grown))))
            game.write_text(json.dumps({"solvable": True, "changed": True}))
            changed = [r for r in data.build_catalog(root) if r.relative_game_path == original.relative_game_path][0]
            self.assertNotEqual(changed.game_id, original.game_id)

    def test_catalog_tampering_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_game(root, "train", 1, "one")
            records = data.build_catalog(root)
            catalog = root / "catalog.json"
            data.write_catalog(records, catalog)
            self.assertEqual(data.load_catalog(catalog)[0], records)
            value = json.loads(catalog.read_text())
            value["games"][0]["scene_id"] = "modified"
            catalog.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                data.load_catalog(catalog)
            for invalid in ("/root/game.tw-pddl", "json_2.1.1/train/../../game.tw-pddl", "json_2.1.1/train/a/b/other"):
                with self.assertRaises(ValueError):
                    data.validate_relative_game_path(invalid)

    def test_unsolvable_games_are_filtered_and_expert_content_never_enters_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            make_game(root, "train", 1, "solvable")
            make_game(root, "train", 1, "unsolvable", solvable=False)
            make_game(root, "valid_unseen", 2, "heldout")
            records = data.build_catalog(root)
            self.assertEqual(len(records), 2)
            train = next(r for r in records if r.split == "train")
            sample = data.task_sample(train, phase="train", digest=data.catalog_digest(records), run_dir=root / "run")
            task = sample["extra_info"]["tools_kwargs"]["task"]
            self.assertEqual(task["name"], "alfworld_skill_training")
            self.assertEqual(task["metadata"]["relative_game_path"], train.relative_game_path)
            self.assertEqual(task["metadata"]["catalog_digest"], data.catalog_digest(records))
            self.assertNotIn("EXPERT_SECRET", json.dumps(sample))
            heldout = next(r for r in records if r.split != "train")
            with self.assertRaisesRegex(ValueError, "evaluation game"):
                data.task_sample(heldout, phase="train", digest="digest", run_dir=root)

    def test_offline_data_cli_help_does_not_import_training_stack(self):
        result = subprocess.run(
            [sys.executable, str(PACKAGE_ROOT / "prepare_data.py"), "--help"], text=True, capture_output=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--dev-fraction", result.stdout)


class TestAlfworldEpisodeContracts(unittest.TestCase):
    def test_repeated_parallel_attempts_cannot_overwrite_an_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            with ThreadPoolExecutor(max_workers=8) as pool:
                attempts = list(pool.map(lambda _: protocol.new_attempt_directory(run, "train_stable"), range(64)))
            self.assertEqual(len({identity for identity, _ in attempts}), 64)
            self.assertEqual(len({path for _, path in attempts}), 64)
            first = attempts[0][1] / "result.json"
            first.write_text("preserved")
            _, later = protocol.new_attempt_directory(run, "train_stable")
            self.assertEqual(first.read_text(), "preserved")
            self.assertFalse((later / "result.json").exists())
            with self.assertRaises(ValueError):
                protocol.new_attempt_directory(run, "../../escape")

    def test_reward_cannot_be_overridden_by_agent_or_inconsistent_server_score(self):
        self.assertEqual(protocol.authoritative_reward({"won": True, "reward": 1, "agent_ok": False}), 1.0)
        self.assertEqual(protocol.authoritative_reward({"won": False, "reward": 0, "agent_ok": True}), 0.0)
        for result in ({"won": False, "reward": 1}, {"won": True, "reward": float("nan")}, {"won": "true"}):
            with self.assertRaises(ValueError):
                protocol.authoritative_reward(result)

    def test_public_observation_excludes_internal_and_expert_fields(self):
        internal = {
            "feedback": "Your task is to clean an apple.",
            "admissible_commands": ["help", "go to sinkbasin 1"],
            "walkthrough": "EXPERT_SECRET",
            "gamefile": "/root/private",
            "pddl_problem": "private state",
        }
        public = protocol.public_state(internal, steps=0, max_steps=50, won=False, done=False)
        self.assertEqual(set(public), {"observation", "admissible_commands", "steps", "max_steps", "won", "done"})
        self.assertNotIn("help", public["admissible_commands"])
        self.assertNotIn("EXPERT_SECRET", json.dumps(public))
        with self.assertRaises(ValueError):
            protocol.public_state({"feedback": {"expert": "secret"}}, steps=0, max_steps=50, won=False, done=False)

    def test_prompt_configuration_preserves_native_codex_message_contract(self):
        state = {"observation": "A room", "admissible_commands": ["look"], "walkthrough": "EXPERT_SECRET"}
        messages = protocol.agent_messages(
            [{"role": "system", "content": "English system instruction"}, {"role": "user", "content": "Original"}],
            state,
            instruction="Recipe-owned instruction",
        )
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertIn("Recipe-owned instruction", messages[1]["content"])
        self.assertNotIn("EXPERT_SECRET", json.dumps(messages))
        self.assertNotIn("Original", messages[1]["content"])

    def test_identity_and_loopback_service_boundaries_are_checked(self):
        row = {"game_id": "train_stable", "relative_game_path": "game", "game_sha256": "digest", "split": "train"}
        metadata = {**row, "catalog_digest": "catalog"}
        protocol.validate_game_reference(metadata, row, "catalog")
        with self.assertRaisesRegex(ValueError, "game_sha256"):
            protocol.validate_game_reference({**metadata, "game_sha256": "changed"}, row, "catalog")
        with self.assertRaises(ValueError):
            protocol.validate_loopback_url("https://external.example")
        self.assertEqual(protocol.validate_loopback_url("http://127.0.0.1:1234/"), "http://127.0.0.1:1234")


if __name__ == "__main__":
    unittest.main()
