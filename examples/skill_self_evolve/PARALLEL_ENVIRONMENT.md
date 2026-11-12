# Process-sharded ALFWorld service

The original service sends all TextWorld initialization, action and close
operations through one executor thread. This protects TextWorld's shared
TatSu parser, but an initialization wave queues unrelated active games behind
it. Its synchronous journal flush/fsync also runs on the HTTP event loop.

The examples-only `scripts/parallel_environment_server.py` keeps the existing
public state, catalog validation, budget registration and engine-entry
admission protocol. It changes the execution boundary:

* Four independent spawned processes each own a single serial engine shard.
  A session stays on its assigned shard for its complete lifetime. TextWorld
  environments and parser objects never cross process boundaries.
* A separate four-thread I/O executor performs journal, result and proof
  writes. The per-session lock covers engine execution and awaited persistence,
  so another session can respond while one session waits for NAS durability.
* A separate telemetry writer records request timings. It does not share the
  critical I/O executor. Cleanup drains its pending writes and reports errors.

No hashed core source file changes. Because the core fingerprint therefore
does not describe this new behavior, health/config metadata explicitly record
the new source path/hash, policy, worker PIDs, process count, I/O concurrency,
request protocol and stable server instance ID. Launch/driver receipts must
retain that additional identity.

## Timings

`diagnostics/environment-http-timing.jsonl` records each HTTP request with a
unique correlation ID and these fields in milliseconds:

| Field | Scope |
| --- | --- |
| `session_queue_ms` | Waiting for the same session's ordered lock |
| `engine_queue_ms` | Submission until the assigned worker enters the operation |
| `engine_execute_ms` | Actual initialization, action or close in that worker |
| `io_queue_ms` | Waiting for the independent durable I/O executor |
| `io_persist_ms` | File/proof operations, including required flush/fsync |
| `response_ready_ms` | HTTP handler entry until response construction |
| `response_send_ms` | Preparing and writing the response to the transport |
| `total_ms` | Handler entry through response write completion |

These are service-side intervals. Socket arrival before event-loop dispatch is
not directly timestamped; the diagnostic also measures client latency and
event-loop scheduling lag. The `X-Environment-HTTP-ID` response header links
client measurements to the durable timing row. Timing persistence itself runs
after response writing and is not added to the reported response latency.

## Action identities and timeout recovery

Send one UUID hex32 identity per logical action as body `request_id` or the
`X-Request-ID` header. If both are present, they must match. The existing POST
path and state response remain:

```text
POST /session/{session}/step
{"action":"look","request_id":"0123456789abcdef0123456789abcdef"}
```

The same identity/action awaits or returns the original result. Reusing the
identity with a different action returns409. The service owns the pending
action task independently of the HTTP handler, and persists the original
result before acknowledging completion. An unknown engine error is cached as
an infrastructure error; a retry never executes it again or turns it into
reward0.

```text
GET /session/{session}/requests/{request_id}
```

The lookup returns202 while pending;200 with
`{request_id,status:"completed",response_status,result}` when complete; or404
for an unknown identity. The result remains queryable after the world closes.
The client must first require health capability
`action_request_protocol="alfworld-step-request-id-v1"`, bind the stable
`server_instance_id`, and reuse the exact identity. A service restart loses
the in-memory operation registry and is an unknown outcome, not permission to
re-execute. Legacy actions without an identity are accepted once but do not
gain safe automatic retries.

## Verification and measured scope

Run CPU contract tests without a model, GPU, backend or allocated account:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 \
  /root/nas_chaohuang/benchmarks/alfworld/env/bin/python \
  examples/skill_self_evolve/tests/test_parallel_environment_server.py
```

The seven cases cover real four-process concurrency and sticky state;
timeout/query/duplicate behavior; a slow durable write while health and
another session remain responsive; ordered durable acknowledgments; identity
conflicts and missing lookups; a cached unknown engine error; and continued
owned-world/pool cleanup after one world fails to close.

The request-identity recovery contract applies to actions. Creation still uses
the existing distinct output-attempt boundary and does not gain an automatic
retry or restart contract from this extension.

`scripts/benchmark_environment_service.py` runs the frozen one-thread service
and this extension on disposable ports with the same four original ALFWorld
task families and NAS destination. It uses eight warm-up worlds, then creates
32 more worlds while four warm sessions continue taking eight `look` actions
each. Its outputs are explicitly diagnostic, never training or evaluation
examples, and it creates no model/backend/UID. For the recorded comparison
`environment_service_benchmark_1794487930296098141`:

| Measurement | Original one thread | Four processes + I/O |
| --- | ---: | ---: |
| Workload wall time, excluding service startup | 50.09s | 21.53s |
| Mixed action client latency, p95 | 37.09s | 8.07s |
| Mixed action engine queue, p95 | 36.42s | 8.00s |
| Mixed action engine execution, p95 | 79.56ms | 76.84ms |
| Health request latency, p95 | 70.71ms | 0.545ms |
| Event-loop scheduling lag, p95 | 57.72ms | 1.062ms |

Both modes closed40 worlds and admitted40 actions with matching initial
public states and per-game outcome multisets. No HTTP error occurred with the
diagnostic's180-second observation timeout. All five owned benchmark processes
exited; the existing training service was untouched.

This controlled workload identifies the shared engine queue as its dominant
action delay and supports trying four processes. It does not establish the
fraction of historical timeouts caused by each component or whole-rollout
speedup. The new service persists additional result/trace evidence, so its
total durable-I/O timings are not directly comparable to the baseline's
budget-event fsync alone. A sticky shard can still queue an action behind
initializations: the observed maximum action latency was14.27s. Request-ID
recovery remains necessary. Priority scheduling and cache expiry are possible
follow-ups after measuring the resumed workload.
