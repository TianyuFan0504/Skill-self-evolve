# ALFWorld joint training currently accepts text trajectories only

UniAgent supports multimodal encoding. Its official serializer emits processed
`multi_modal_inputs` alongside the real sampled token IDs, masks, logprobs and
position IDs. Those inputs can contain image pixels, image/video grids and
modality metadata. A text ID stream with its pixels removed is not the same
model context and cannot be used to compute a valid image-conditioned likelihood.

This joint harness has text-only curator replay and mixed-role packing. It does
not claim general multimodal training support. For ALFWorld, the supported
serializer hook now rejects nonempty trajectory media and checks the official
returned media fields before they can be published to TransferQueue. It does not
strip images, retokenize sampled output, fill pixels or probabilities, or convert
this error into a zero reward. Other UniAgent task families are not globally
disabled by this ALFWorld-specific hook.

Before raising, the serializer writes an atomic, fsynced run-level incident
under the trusted config `artifact_root/protocol_incidents/unsupported_media/`.
The path never comes from a model message. Each lane uses a distinct filename.
The learner checks this sticky run-level failure before preparation and again
immediately before its optimizer RPC. Thus a runner that filters the rejected
case out of TQ cannot silently train on only the surviving text cases. Even an
interrupted incident directory blocks updates; it is never automatically cleared.

The learner has a second check before selecting its text columns. Public
`kv_describe` reports the fields and unfinalized fields for every actual key.
Each media column is fetched on the exact subset of keys that has it; relying on
the common-column result from `kv_batch_get` could otherwise hide a single media
row. Unfinalized, missing or unreadable described columns fail closed. Only
carriers already proved to contain no media may be omitted before the existing text-only mixed
packing, so adding the check does not add a new solver/curator batch schema.

`None`, empty mappings/lists, zero-element tensors and all-empty per-row
NonTensorData/NonTensorStack carriers are valid text cases. The text-only
`mm_token_type_ids` metadata may contain only zeros; nonzero modality flags are
rejected. A nonempty all-black pixel tensor remains media and is rejected.

The CPU regressions use actual Torch/TensorDict types and the installed official
UniAgent serializer body, replacing only the processor boundary with CPU tensor
fixtures. They prove rejection and unchanged text field propagation, not a
successful model or benchmark run. No GPU, Ray actor or optimizer is created.
Run with the existing verified CPU dependency environment:

```bash
python -m unittest discover -s tests/experimental/skill_harness \
  -p test_text_only_media.py -v
```

Tool permissions and per-attempt filesystem isolation are separate boundaries.
Installing `qwen-vl-utils` alone does not make multimodal trajectories supported
by this joint learner. Future multimodal support must preserve the complete
processed media payload, model-specific positions and replay conditioning all
the way into likelihood computation and gradient updates.
