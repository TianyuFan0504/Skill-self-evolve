# Explicit curator abstention

The native curator makes one of two valid model actions. A proposed skill keeps
the existing five-field JSON format. Abstention has exactly this shape:

```json
{"decision":"abstain","reason":"The observed feedback does not establish a reusable prerequisite or procedure."}
```

An abstention reason is nonempty text of at most 2,048 characters, without host
paths or network endpoints. The prompts request English. Unknown fields, missing
reasons and malformed output remain protocol errors; they do not become valid
abstentions. The collector version is `native-codex-curate-v3`. Native structured
output remains disabled; the host validates the returned text.

`collect_curate_candidate` returns `CurateCollection | CurateAbstention`. A
`CurateAbstention` provides `collection_id`, `session_id`, `reason`,
`action_record_path`, `action_record_sha256`, `output_dir` and
`credit_registration_required=False`. Its SHA256 covers the complete immutable
`abstention.json` file, including its integrity envelope.

The action record retains the actual original prompt IDs, authored response IDs,
action mask, processed behavior log probabilities, observed EMA weight versions,
sampling configuration and complete context metadata. `generated.json` retains
the original native capture. The abstention record points to its file hash and
preserves the raw returned JSON text. `complete.json` records `decision=abstain`;
an interrupted acknowledgement can resume from the same capture without another
model call.

Abstention does not fabricate a SkillArtifact or skill reference, stage a
candidate, append a skill replay record, register a counterfactual window, or
create a utility/reward/advantage label. It is durable neutral action evidence,
not an automatically trainable zero-reward sample. The controller must recognize
the typed result and finish the corresponding job without skill admission.

Both valid decisions support an optional `before_commit` callback. The collector
waits for its awaitable after validating the captured action and context, before
the first skill-store/replay mutation or final neutral-action acknowledgement.
The commit following that gate is synchronous and contains no `await`. Model
generation and its private capture may proceed while a checkpoint gate is
closed; shared store/replay commits wait for permission.

The source still requires a nonempty observed environment action sequence.
Consistent zero-action sources are explicitly `source_censored` before curator
sampling. They are not invented empty trajectories or model abstentions.
