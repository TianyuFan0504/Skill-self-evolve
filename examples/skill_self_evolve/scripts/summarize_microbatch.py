#!/usr/bin/env python3
"""Summarize all four ranks without hiding stragglers or failed candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(root):
    result = {
        "schema_version": 1,
        "root": str(root),
        "training_optimizer_updates": 0,
        "scope": "standalone official-engine diagnostic; main-run confirmation still required",
        "roles": {},
    }
    for role in ("actor", "reference"):
        paths = [root / role / f"rank_{i}.json" for i in range(4)]
        if not all(path.is_file() for path in paths):
            raise ValueError("All four rank reports are required: " + role)
        reports = [json.loads(path.read_text()) for path in paths]
        if [report["rank"] for report in reports] != list(range(4)):
            raise ValueError("Rank identity mismatch")
        selected = reports[0]["selected_segments"]
        if any(report["selected_segments"] != selected for report in reports):
            raise ValueError("All ranks must benchmark the exact same global corpus")
        phases = sorted({trial["phase"] for report in reports for trial in report["trials"]})
        rows = []
        for phase in phases:
            trials = [[trial for trial in report["trials"] if trial["phase"] == phase] for report in reports]
            if any(len(trial) != 1 for trial in trials):
                rows.append({"phase": phase, "status": "incomplete"})
                continue
            trials = [trial[0] for trial in trials]
            seconds = max(trial["seconds"] for trial in trials)
            parity = [trial.get("logprob_comparison", trial.get("gradient_comparison")) for trial in trials]
            rows.append(
                {
                    "phase": phase,
                    "status": "measured",
                    "microbatch": trials[0]["microbatch"],
                    "slowest_rank_seconds": seconds,
                    "unpadded_tokens_per_second": sum(row["tokens"] for row in selected) / seconds,
                    "peak_allocated_gib": max(trial["memory"]["peak_allocated_gib"] for trial in trials),
                    "peak_reserved_gib": max(trial["memory"]["peak_reserved_gib"] for trial in trials),
                    "all_rank_parity_passed": all(item["passed"] for item in parity),
                    "rank_parity": parity,
                }
            )
        groups = {}
        for row in rows:
            if row["status"] != "measured":
                continue
            group = row["phase"].rsplit("_repeat", 1)[0]
            groups.setdefault(group, []).append(row)
        comparisons = {}
        for group, trials in groups.items():
            trials.sort(key=lambda row: int(row["phase"].rsplit("_repeat", 1)[1]))
            microbatch = trials[0]["microbatch"]
            dense_slots = sum(
                microbatch * max(item["tokens"] for item in rank_rows[start : start + microbatch])
                for rank in range(4)
                for rank_rows in [selected[rank::4]]
                for start in range(0, len(rank_rows), microbatch)
            )
            comparisons[group] = {
                "microbatch": microbatch,
                "padded_dense_input_slots": dense_slots,
                "dense_padding_multiplier": dense_slots / sum(item["tokens"] for item in selected),
                "measured_repeats": len(trials),
                "median_seconds": statistics.median(row["slowest_rank_seconds"] for row in trials),
                "last_repeat_seconds": trials[-1]["slowest_rank_seconds"],
                "last_repeat_tokens_per_second": trials[-1]["unpadded_tokens_per_second"],
                "peak_allocated_gib": max(row["peak_allocated_gib"] for row in trials),
                "peak_reserved_gib": max(row["peak_reserved_gib"] for row in trials),
                "all_rank_parity_passed": all(row["all_rank_parity_passed"] for row in trials),
            }
        result["roles"][role] = {
            "statuses": [report["status"] for report in reports],
            "all_completed": all(report["status"] == "completed" for report in reports),
            "failures": [
                {key: report.get(key) for key in ("rank", "failed_phase", "error_type", "oom")}
                for report in reports
                if report["status"] == "failed"
            ],
            "source_report_hashes": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "selected_segments": selected,
            "groups": comparisons,
            "trials": rows,
        }
    result["all_completed"] = all(role["all_completed"] for role in result["roles"].values())
    result["all_parity_passed"] = result["all_completed"] and all(
        group["all_rank_parity_passed"] for role in result["roles"].values() for group in role["groups"].values()
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = summarize(args.root)
    output = args.root / "summary.json"
    with output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "summary": str(output),
                "all_completed": result["all_completed"],
                "all_parity_passed": result["all_parity_passed"],
            }
        )
    )


if __name__ == "__main__":
    main()
