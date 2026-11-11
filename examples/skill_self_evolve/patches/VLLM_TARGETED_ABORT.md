# Targeted abort with the installed vLLM 0.24 interface

vLLM 0.24 assigns an internal `external-id-randomsuffix` to each request.
`OutputProcessor.request_states` uses that internal ID; `external_req_ids` maps
the external ID to its internal request IDs. The previous verl method looked
up an external ID directly in `request_states`, then used an older direct abort
signature. It could report not-found for a live request.

The single-request server method now checks one live mapping and both state
identities, then calls public `AsyncLLM.abort(external_id, internal=False)`.
That public method owns the final ABORT output and submits the mapped internal
IDs to EngineCore. The check and its synchronous mapping removal have no
intervening await on the server's event loop. Unknown, ambiguous or stale
mappings fail closed without guessing an ID suffix or touching another request.

The method defaults to `reset_prefix_cache=False` and rejects a request to reset
the shared cache. It performs no global abort, cache reset or generation-ID
rewrite. `aborted=True` means the verified abort was submitted successfully;
`ack_scope=submitted_not_gpu_drained` makes that limit explicit. The owner still
waits for its original generation future/drain. CPU tests cannot establish that
a physical GPU kernel has stopped. Postterminal partial IDs, processed logprobs
and raw stop reason remain archive-only, without synthetic model EOS or gradient.

The whole core server file is included in the joint checkpoint source
fingerprint. This is an independent cancellation compatibility repair, not the
cause of the fresh64_v2 GPU3 timeout: its persisted registry has zero targeted
abort attempts on the failed server. The two attempts elsewhere had no ACK and
ended naturally; their original records remain unchanged.

The CPU regression executes the actual installed public `AsyncLLM.abort` and
`OutputProcessor.abort_requests` method bodies with only transport/state fixtures.
Their complete source hashes are pinned in the test; no third-party source is
vendored into this repository. On the server, the default source directory is
`/opt/conda/lib/python3.11/site-packages/vllm/v1/engine`. For an offline workstation
copy, set `SKILL_VLLM_ABORT_SOURCE_DIR` to the independently archived directory.

```bash
uv run --offline --no-project --python 3.10 python -m unittest discover \
  -s tests/experimental/skill_harness -p test_vllm_targeted_abort.py -v
```

These checks allocate no model, Ray actor or GPU. A separate owned-request GPU
diagnostic is needed to verify actual in-flight cancellation and isolation.
