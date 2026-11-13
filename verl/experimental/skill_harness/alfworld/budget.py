"""Private wall-clock budget evidence; never turns a Native stop into model EOS.

The environment admits actions before its monotonic deadline. An admitted
engine action completes under the same session lock before a host snapshot or
close can observe its result. Public game observations contain no budget clock.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import tempfile
from pathlib import Path

BUDGET_POLICY_VERSION = "alfworld-wall-budget-v1"
ADMISSION_POLICY = "engine_entry_before_deadline; admitted_action_completes_before_snapshot"
BINDING_KEYS = ("binding_id", "binding_sha256", "attempt_id", "gateway_session_id", "environment_session_id")
ACCOUNT_KEYS = ("lease_id", "uid", "gid", "name", "owner")


class BudgetOutcomeUnknown(ValueError):
    """The evidence does not establish an observed budget failure."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            value.update(chunk)
    return value.hexdigest()


def _need(condition, message):
    if not condition:
        raise BudgetOutcomeUnknown(message)


def _hex(value, length):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{" + str(length) + "}", value) is not None


def read_private_json(path):
    value = json.loads(Path(path).read_text())
    if isinstance(value, dict) and "payload" in value:
        _need(
            value.get("sha256", value.get("payload_hash")) == digest(value["payload"]), "Private proof digest differs"
        )
        return value["payload"]
    return value


def write_private_json(path, value, *, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        _need(read_private_json(path) == value, "Immutable budget proof changed")
        return
    descriptor, temporary = tempfile.mkstemp(prefix=".budget-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(canonical(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def make_registration(payload, *, session_id, run_root, output, server_instance_id, started_ns, budget_seconds=900):
    """Validate a host-admin registration, with no model-supplied clock fields."""
    _need(
        isinstance(payload, dict) and set(payload) == {"schema_version", "budget_seconds", "binding", "account"},
        "Invalid budget registration fields",
    )
    _need(type(payload["schema_version"]) is int and payload["schema_version"] == 1, "Invalid budget schema")
    _need(
        type(payload["budget_seconds"]) is int and payload["budget_seconds"] == budget_seconds > 0,
        "Budget differs from the service's declared protocol",
    )
    binding, account = payload["binding"], payload["account"]
    _need(isinstance(binding, dict) and set(binding) == set(BINDING_KEYS), "Budget binding fields differ")
    _need(isinstance(account, dict) and set(account) == set(ACCOUNT_KEYS), "Budget lease fields differ")
    _need(all(_hex(binding[key], 64) for key in ("binding_id", "binding_sha256")), "Invalid bound backend identity")
    _need(_hex(binding["attempt_id"], 32) and _hex(binding["environment_session_id"], 32), "Invalid attempt/session")
    _need(binding["environment_session_id"] == session_id, "Budget belongs to another environment session")
    _need(
        isinstance(binding["gateway_session_id"], str)
        and re.fullmatch(r"[A-Za-z0-9_.:-]{1,256}", binding["gateway_session_id"]),
        "Invalid Gateway identity",
    )
    _need(Path(output).name == "attempt_" + binding["attempt_id"], "Budget belongs to another output attempt")
    _need(
        _hex(account["lease_id"], 32) and all(type(account[key]) is int and account[key] > 0 for key in ("uid", "gid")),
        "Invalid private lease",
    )
    owner = account["owner"]
    _need(
        isinstance(account["name"], str)
        and bool(account["name"])
        and isinstance(owner, dict)
        and set(owner) == {"kind", "run_root", "scope_id"}
        and owner["kind"] in ("ordinary", "pair")
        and owner["run_root"] == str(Path(run_root).resolve())
        and isinstance(owner["scope_id"], str)
        and bool(owner["scope_id"]),
        "Budget lease owner differs",
    )
    _need(type(started_ns) is int and started_ns > 0 and _hex(server_instance_id, 32), "Invalid service clock")
    value = {
        "schema_version": 1,
        "policy_version": BUDGET_POLICY_VERSION,
        "admission_policy": ADMISSION_POLICY,
        "server_instance_id": server_instance_id,
        "binding": binding,
        "account": account,
        "budget_seconds": budget_seconds,
        "clock": "monotonic_ns",
        "started_ns": started_ns,
        "deadline_ns": started_ns + budget_seconds * 1_000_000_000,
    }
    return {**value, "registration_sha256": digest(value)}


def budget_failure_claimed(task):
    return any(
        task.get(name) is True for name in ("budget_exhausted", "expected_budget_failure", "budget_reward_eligible")
    )


def validate_budget_registration(registration, *, payload, session_id, run_root, output, budget_seconds=900):
    """Validate the ACK before Native launch; a retry never grants a new clock."""
    expected = make_registration(
        payload,
        session_id=session_id,
        run_root=run_root,
        output=output,
        server_instance_id=registration.get("server_instance_id"),
        started_ns=registration.get("started_ns"),
        budget_seconds=budget_seconds,
    )
    _need(registration == expected, "The private budget registration ACK differs")
    return registration


def make_budget_stop_reason(ordered_snapshot):
    """Bind the authenticated host fence to the exact server-ordered deadline view."""
    return "declared_budget:" + ordered_snapshot["registration_sha256"] + ":" + digest(ordered_snapshot)


def _verify_environment(output, registration, engine):
    final = read_private_json(output / "budget-environment.json")
    _need(engine.get("episode_budget") == final, "Engine and private budget receipt differ")
    _need(all(final.get(key) == value for key, value in registration.items()), "Budget registration changed at close")
    _need(
        final.get("server_closed") is True
        and final.get("close_reason") == "agent_finished"
        and type(final.get("closed_ns")) is int
        and final["closed_ns"] >= registration["deadline_ns"]
        and final.get("deadline_reached") is True,
        "A normal deadline-bound engine close is unobserved",
    )
    state = final.get("world_state", {})
    _need(
        set(state) == {"won", "done", "steps", "max_steps"}
        and type(state["won"]) is bool
        and type(state["done"]) is bool
        and all(type(state[key]) is int and state[key] >= 0 for key in ("steps", "max_steps")),
        "Invalid raw world state",
    )
    _need(
        state["won"] is False
        and state["max_steps"] > 0
        and state["steps"] < state["max_steps"]
        and all(engine.get(key) == state[key] for key in state)
        and final.get("world_state_sha256") == digest(state),
        "Budget failure cannot overwrite a win or world state",
    )
    ordered = read_private_json(output / "budget-ordered-state.json")
    _need(
        all(ordered.get(key) == value for key, value in registration.items())
        and ordered.get("server_closed") is False
        and type(ordered.get("now_ns")) is int
        and registration["deadline_ns"] <= ordered["now_ns"] <= final["closed_ns"]
        and ordered.get("deadline_reached") is True
        and ordered.get("world_state") == state
        and state["done"] is False
        and ordered.get("world_state_sha256") == digest(state),
        "The ordered deadline view is missing, changed, or already MDP-terminal",
    )
    events_path = output / "budget-events.jsonl"
    _need(final.get("events_sha256") == sha256_file(events_path), "Action-admission journal changed")
    events = [json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    ordered_sequence = ordered.get("last_sequence")
    _need(
        type(ordered_sequence) is int
        and 0 <= ordered_sequence <= len(events)
        and ordered.get("events_sha256")
        == hashlib.sha256(b"".join(events_path.read_bytes().splitlines(keepends=True)[:ordered_sequence])).hexdigest(),
        "The ordered action-journal prefix differs",
    )
    admitted, rejected, last_admitted, last_completed = 0, 0, None, None
    for sequence, event in enumerate(events, 1):
        _need(
            event.get("schema_version") == 1
            and event.get("sequence") == sequence
            and event.get("registration_sha256") == registration["registration_sha256"],
            "Action order/binding differs",
        )
        checked = event.get("checked_ns")
        _need(
            type(checked) is int and registration["started_ns"] <= checked <= final["closed_ns"], "Invalid action clock"
        )
        _need(
            _hex(event.get("action_sha256"), 64)
            and event.get("won_after") is False
            and event.get("done_after") is False
            and (last_completed is None or checked >= last_completed),
            "The ordered action state/clock is unknown or already terminal",
        )
        if event.get("kind") == "admitted":
            completed = event.get("completed_ns")
            _need(
                checked < registration["deadline_ns"]
                and type(completed) is int
                and checked <= completed <= final["closed_ns"]
                and event.get("step") == admitted + 1,
                "An action entered the engine after its deadline",
            )
            _need(sequence <= ordered_sequence, "An action was applied after the ordered budget stop")
            admitted += 1
            last_admitted, last_completed = checked, completed
        elif event.get("kind") == "deadline_rejected":
            _need(
                checked >= registration["deadline_ns"] and event.get("step") == admitted,
                "An early action or changed world was misclassified as budget rejection",
            )
            rejected += 1
        else:
            raise BudgetOutcomeUnknown("An engine action failed or has an unknown admission state")
    _need(
        final.get("admitted_actions") == admitted == engine["steps"]
        and final.get("rejected_after_deadline") == rejected
        and final.get("last_sequence") == len(events)
        and final.get("last_admitted_ns") == last_admitted
        and final.get("last_completed_ns") == last_completed
        and final.get("action_error_count") == 0,
        "Action/engine count or completion evidence differs",
    )
    _need(
        ordered.get("admitted_actions") == admitted
        and ordered.get("last_admitted_ns") == last_admitted
        and ordered.get("last_completed_ns") == last_completed,
        "The world changed after the ordered budget stop",
    )
    return final, ordered


def _verify_backend(output, registration, task, ordered):
    binding = registration["binding"]
    run_root = Path(registration["account"]["owner"]["run_root"])
    directory = run_root / "terminal_bindings" / binding["binding_id"]
    actual_binding = read_private_json(directory / "binding.json")
    stop = read_private_json(directory / "terminal.json")
    _need(
        digest(actual_binding) == binding["binding_sha256"]
        and all(actual_binding.get(key) == binding[key] for key in BINDING_KEYS if key != "binding_sha256"),
        "Private backend binding differs",
    )
    _need(
        all(stop.get(key) == binding[key] for key in BINDING_KEYS if key != "binding_id")
        and stop.get("stop_kind") == "budget_timeout"
        and stop.get("terminal_state") is None
        and stop.get("terminal_state_hash") is None
        and stop.get("reason") == make_budget_stop_reason(ordered)
        and isinstance(stop.get("capability"), str)
        and isinstance(actual_binding.get("capability"), str)
        and secrets.compare_digest(stop["capability"], actual_binding["capability"]),
        "The actual stop is not this budget fence",
    )
    receipt = read_private_json(output / "backend-terminal-drain.json")
    _need(
        task.get("backend_terminal") == receipt and task.get("backend_binding") == binding,
        "Task backend receipt/binding differs",
    )
    _need(
        all(receipt.get(key) == binding[key] for key in BINDING_KEYS)
        and receipt.get("schema_version") == 1
        and receipt.get("stop_kind") == "budget_timeout"
        and receipt.get("status") == "host_stopped"
        and receipt.get("fenced") is True
        and receipt.get("host_stop_requested") is True
        and receipt.get("expected_host_stop") is True
        and receipt.get("backend_stop_confirmed") is True
        and receipt.get("active_requests") == 0
        and receipt.get("backend_error_types") == [],
        "Backend drain is unknown or contains infrastructure errors",
    )
    rows = receipt.get("requests")
    _need(isinstance(rows, list) and bool(rows), "No actual backend request was observed")
    counter = read_private_json(directory / "admission-counter.json")
    _need(
        counter.get("binding_sha256") == binding["binding_sha256"]
        and type(counter.get("next_index")) is int
        and counter["next_index"] == len(rows)
        and {row.get("admission_index") for row in rows} == set(range(len(rows))),
        "Backend admission journal is incomplete",
    )
    _need(
        {path.stem for path in (directory / "requests").glob("*.json")} == {row.get("request_nonce") for row in rows},
        "A backend request disappeared from the drain receipt",
    )
    for row in rows:
        nonce = row.get("request_nonce")
        _need(
            _hex(nonce, 32) and read_private_json(directory / "requests" / (nonce + ".json")) == row,
            "Backend request proof changed",
        )
        _need(
            all(row.get(key) == binding[key] for key in ("binding_sha256", "attempt_id", "gateway_session_id"))
            and row.get("backend_rpc_finished") is True
            and row.get("error_type") is None,
            "An owned backend RPC is unfinished or failed",
        )
        phase = row.get("phase")
        _need(phase in ("completed", "host_refused", "host_tail_archived"), "Unknown backend completion phase")
        if phase == "completed":
            _need(
                _hex(row.get("actual_request_id"), 32) and row.get("raw_stop_reason") == "completed",
                "Unknown raw backend stop cannot become a budget reward",
            )
        else:
            _need(row.get("stop_kind") == "budget_timeout", "The request stopped for another reason")
        if phase == "host_tail_archived":
            _need(row.get("raw_stop_reason") in ("completed", "aborted"), "Unknown stopped-tail reason")
            if row["raw_stop_reason"] == "aborted":
                _need(
                    row.get("abort_acknowledged") is True and row.get("abort_attempts", 0) > 0,
                    "A raw abort lacks its exact owned-request acknowledgment",
                )
            ref = row.get("tail_archive_ref", {})
            path = Path(ref.get("path", "/missing"))
            _need(
                path.resolve() == (directory / "tails" / (nonce + ".json")).resolve(),
                "Aborted tail escaped its bound private request path",
            )
            tail = read_private_json(path)
            tokens, probabilities, prompt = tail.get("token_ids"), tail.get("log_probs"), tail.get("prompt_ids")
            _need(
                ref.get("sha256") == digest(tail)
                and tail.get("archive_only") is True
                and tail.get("behavior_gradient") is False
                and tail.get("model_eos_invented") is False
                and tail.get("binding_sha256") == binding["binding_sha256"]
                and tail.get("actual_request_id") == row.get("actual_request_id")
                and tail.get("attempt_id") == binding["attempt_id"]
                and tail.get("gateway_session_id") == binding["gateway_session_id"]
                and tail.get("stop_kind") == "budget_timeout"
                and tail.get("raw_stop_reason") == row["raw_stop_reason"]
                and ref.get("raw_stop_reason") == row["raw_stop_reason"]
                and ref.get("archive_only") is True
                and isinstance(prompt, list)
                and bool(prompt)
                and isinstance(tokens, list)
                and all(type(token) is int and token >= 0 for token in [*prompt, *tokens])
                and ref.get("token_count") == len(tokens)
                and (
                    not tokens
                    and probabilities is None
                    or isinstance(probabilities, list)
                    and len(probabilities) == len(tokens)
                    and all(type(p) in (int, float) and math.isfinite(p) and p <= 0 for p in probabilities)
                ),
                "Aborted tail provenance differs",
            )
    return receipt


def _verify_cleanup(output, registration, task):
    account = registration["account"]
    closed = task.get("account_isolation", {})
    _need(
        isinstance(closed, dict)
        and all(closed.get(key) == account[key] for key in ACCOUNT_KEYS)
        and closed.get("status") == "attempt_closed"
        and closed.get("backend_drained") is True
        and closed.get("uid_drained") is True,
        "Native process/backend closure is not confirmed",
    )
    receipt_path = output / "native-cleanup.json"
    receipt = read_private_json(receipt_path)
    _need(
        Path(closed.get("cleanup_receipt", "/missing")).resolve() == receipt_path.resolve()
        and receipt.get("status") == "cleaned"
        and receipt.get("account_lease_id") == account["lease_id"]
        and receipt.get("agent_uid") == account["uid"],
        "Cleanup receipt is not bound to this exact native lease",
    )
    sources = receipt.get("sources", [])
    _need(
        isinstance(sources, list) and any(row.get("kind") == "workspace" for row in sources),
        "Native workspace cleanup is incomplete",
    )
    for row in sources:
        _need(
            row.get("removed") is True and not os.path.lexists(row.get("path", "/missing")),
            "A native resource is still retained",
        )
    return receipt


def verify_budget_failure(*, task, engine, output_dir, expected_budget_seconds=900):
    """Return a proven score-zero outcome, or raise without inventing an outcome.

    This does not create/alter TITO, Native finished/error/exit, MDP done, or EOS.
    Consumers still need their normal real-token/behavior-likelihood and paired
    input checks. The verifier intentionally rejects pre-protocol attempts.
    """
    output = Path(output_dir).resolve()
    try:
        _need(
            budget_failure_claimed(task)
            and task.get("budget_exhausted") is True
            and task.get("task_cancelled") is False
            and task.get("native_finished") is False
            and task.get("agent_ok") is False,
            "The Native execution did not end through the declared budget path",
        )
        _need(
            engine.get("won") is False
            and task.get("won") is False
            and task.get("reward") == task.get("accuracy") == engine.get("reward") == 0.0,
            "A budget failure must preserve the private unsuccessful world outcome",
        )
        _need(
            not task.get("account_isolation_error") and not task.get("backend_terminal_error"),
            "An infrastructure/isolation error cannot become budget zero",
        )
        registration = read_private_json(output / "budget-registration.json")
        claimed_hash = registration.get("registration_sha256")
        _need(
            claimed_hash == digest({key: value for key, value in registration.items() if key != "registration_sha256"}),
            "Budget registration digest differs",
        )
        expected = make_registration(
            {
                "schema_version": 1,
                "budget_seconds": registration.get("budget_seconds"),
                "binding": registration.get("binding"),
                "account": registration.get("account"),
            },
            session_id=task.get("environment_session_id"),
            run_root=registration["account"]["owner"]["run_root"],
            output=output,
            server_instance_id=registration.get("server_instance_id"),
            started_ns=registration.get("started_ns"),
            budget_seconds=expected_budget_seconds,
        )
        _need(
            registration == expected and task.get("attempt_id") == registration["binding"]["attempt_id"],
            "Declared budget or attempt differs",
        )
        for name in ("game_id", "game_sha256", "catalog_digest", "split"):
            _need(task.get(name) == engine.get(name), "Task and private world identity differ")
        capture = read_private_json(output / "native-capture-diagnostics.jsonl")
        _need(
            capture.get("timed_out") is True
            and capture.get("cancelled") is False
            and capture.get("drain_timed_out") is False
            and capture.get("terminal_monitor_error") is None
            and capture.get("returned_exit_code") == -1
            and capture.get("declared_budget_seconds") == expected_budget_seconds
            and capture.get("budget_registration_sha256") == claimed_hash
            and capture.get("budget_started_ns") == registration["started_ns"]
            and capture.get("budget_deadline_ns") == registration["deadline_ns"]
            and type(capture.get("native_started_ns")) is int
            and registration["started_ns"] <= capture["native_started_ns"] < registration["deadline_ns"]
            and type(capture.get("pid")) is int
            and capture["pid"] > 0,
            "Actual Native budget timeout is unobserved",
        )
        for line in (output / "codex-events.jsonl").read_text().splitlines():
            if line.strip():
                _need(
                    json.loads(line).get("type") not in ("error", "turn.failed"),
                    "An unexplained Native/transport error precedes budget closure",
                )
        final, ordered = _verify_environment(output, registration, engine)
        backend = _verify_backend(output, registration, task, ordered)
        _need(task.get("backend_stop_confirmed") is True, "Task and backend closure summaries disagree")
        cleanup = _verify_cleanup(output, registration, task)
        return {
            "schema_version": 1,
            "policy_version": BUDGET_POLICY_VERSION,
            "status": "observed_budget_failure",
            "score": 0.0,
            "registration_sha256": claimed_hash,
            "binding": registration["binding"],
            "world_state_sha256": final["world_state_sha256"],
            "backend_receipt_sha256": digest(backend),
            "native_cleanup_confirmed": cleanup["status"] == "cleaned",
            "native_finished": task["native_finished"],
            "native_error_preserved": task.get("agent_error"),
            "model_eos_invented": False,
        }
    except BudgetOutcomeUnknown:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise BudgetOutcomeUnknown(
            "Budget failure proof is missing, malformed, or inconsistent: " + type(error).__name__
        ) from error
