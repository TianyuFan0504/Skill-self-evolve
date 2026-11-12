# Direct reads during joint checkpoint restore

The original trainer first selects and fully validates a COMPLETE checkpoint.
Its coordinator's restore method then performs that same full validation again
and copies all of `joint_state` into a new working directory. That copy includes
the large EMA files even though the model, optimizer, EMA and dataloader loaders
already read their original checkpoint paths.

`scripts/direct_checkpoint_restore.py` removes those redundant restore costs
without changing the frozen core or checkpoint format:

1. The original trainer's `latest()` or `validate()` performs one complete
   content-hash, inventory, clock and storage validation.
2. Its coordinator issues a single-use proof bound to the exact verified
   object, checkpoint path, manifest identity, payload and compatibility.
3. Restore consumes that proof instead of hashing the large files again.
   Small manifest and file-metadata/inventory checks guard the source before
   and after full validation and again before any live loader.
4. Only `skill_store`, `replay` and `ledger`, plus the small controller and
   evaluation JSON files, are copied into a fresh writable working root.
   Their semantics and the two small JSON content hashes are checked.
5. The online model/optimizer/RNG, EMA and `data.pt` callbacks receive paths
   in the original checkpoint. There is no copied, linked or unused EMA tree.

The `DirectReadRestoreMixin` replaces only the trainer's coordinator provider
and delegates to the original `_load_checkpoint` method. It preserves the
existing load order, training fence, dataloader restore, parameter clocks and
callback completion requirements. Saving and COMPLETE publication still use
the original implementation and full integrity checks.

The writable state requires independent files because continued training
changes its branch heads, replay records, credit leases and controller state.
It must not modify the historical checkpoint. The source COMPLETE directory
must remain immutable while the loaders read it. Metadata guards do not turn
a changing checkpoint into a supported concurrent-write interface.

Successful restore writes `DIRECT_READ_RESTORE.json` and then the original
`RESTORE_COMPLETE.json` marker in the writable root. The additional receipt
records original actor/EMA/data paths, copied components and bytes, one full
validation pass, zero copied EMA files and the source-guard identity. Failed
or unresolved callbacks do not publish a successful restore marker; no lease
is reset and external model side effects are not rolled back automatically.

## CPU verification

```bash
python examples/skill_self_evolve/tests/test_direct_checkpoint_restore.py -q
```

The bounded real-file suite does not import Torch or Ray or read the live
73 GB checkpoint. It checks one content-hash pass per source file across the
actual frozen trainer method's selection/restore call graph; zero EMA copies;
original callback paths and order; independent mutable-state inodes; unchanged
checkpoint bytes; corruption and stale/forged proof refusal; and failure
without a success marker. A full future training startup remains a separate
integration observation. No startup speedup number is inferred from these CPU
fixtures: the removed large-file reads and copies are counted directly.
