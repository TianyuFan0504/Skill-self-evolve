# Real backend cancellation contract

The one-GPU diagnostic `backend_cancel_contract_20261112` passed with the
actual Qwen3.5-4B vLLM backend and production
`TerminalAwareLLMServerClient`. It exercised the typed host stop implemented
in `308209f1aa180df2ed455e9b0473fe89cc51fc93`.

This is transport evidence. No Native agent, ALFWorld episode, learner,
optimizer, reward or usage event was created. The local `done=false` observer
was a transport fixture. The formal 900-second episode budget was unchanged.

## Observed behavior

- The short admission-zero request completed through the real backend. Its
  first-input proof contains the exact 1,985 archived v6 untreated input token
  IDs; no decoding, normalization or retokenization was performed.
- The target was pending while the owned server reported a running request.
  Its actual request UUID was `cf620e89f254406cb2c874aaf926825a`.
- A typed `task_cancel` fence caused the production watcher to issue one
  UUID-targeted abort. The original generation RPC finished, and the binding
  reported zero active requests and confirmed completion.
- The exact returned tail contains **37 tokens**, aligned finite nonpositive
  behavior log probabilities and raw `stop_reason=aborted`. It remains
  archive-only, with no invented EOS or gradient admission.
- A separate request, `da8af9df258b4ca689687d4341667826`, was pending when the
  target was fenced. It completed normally, unfenced, with zero abort attempts.
- The owned Ray cluster shut down. The driver, server, engine and GPU worker
  processes were absent, GPU 0 had no compute process, temporary scratch was
  removed and there were no unresolved local awaits.

The abort acknowledgment establishes submission under the pinned backend's
contract. Completion of the original RPC and final resource release were
checked separately. A per-request GPU-kernel drain signal was not captured.

## Retained evidence

All paths below are relative to
`/root/nas_chaohuang/skill_training/runs/backend_cancel_contract_20261112`.

| Artifact | SHA256 |
| --- | --- |
| `plan.json` | `27d42363ba644b743716f673c0d6c5827b1b09ba94090d8bd75a75c17407e0c0` |
| `result.json` | `ee557f722618112a0442ef357a33804581f4ace14b009a505b266f0c4d18e525` |
| `target-drain.json` | `734392aa913852271bd1101aa23bbd30da92a42d321577a428f1999ae59b41e2` |
| `unrelated-drain.json` | `4b3e69bac050793f8158e654133389ef35255e6ff4a9292d2f3c623af2d8e7d2` |
| `pre-fence.json` | `ea8cfc1976f78bccd859ca8278063dd9b27a69194dfbd41d5519a283712da1a9` |
| `cleanup.json` | `c010c6a64d7d67caa03696248d9ee0d7817070b4cbd52cc55a3407621fc77c0d` |
| `diagnostics/independent_backend_cancel_contract_1794454168625351515.json` | `2da442821f7e6fae14cf41575c42a1d3cddc0800d9e9c16a1a218cb667b70c51` |

The independent review checked 27 conditions against original inputs, request
journals, raw output, first-input proof and current process/GPU state. The
diagnostic script, resolved config, metrics snapshots and Ray logs remain in
the private run. Its script SHA256 is
`ef0d79fbec6263783bbfa03ced1e33763c70d4ba2d97ca8f1bb6dff812d7d00c`.

## Remaining validation

The failed v6 ALFWorld pair remains failed and censored. This successful
transport diagnostic does not retrospectively close that attempt or release
its quarantined UID. The next fresh pair must still validate real ALFWorld
deadline ordering, safe cleanup and exact skill-only input control. Full
training and initial-versus-final test improvement remain unproven.
