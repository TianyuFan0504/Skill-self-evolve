# Sequential native archive verification

A first-batch v6 episode created 10,746 native filesystem entries. Its completed
112.6 MB gzip archive then spent minutes in verification. The worker remained
CPU-active and repeatedly read compressed data. The original verifier scanned
names to EOF, read the embedded manifest at EOF, then extracted files in the
source dictionary's unsorted order. Gzip backward seeks repeated decompression;
`getmember` also performed a linear lookup for every entry.

`verify_native_archive` now reads members once in physical order through a
forward-only tar stream over gzip. Every expected member must appear exactly
once. Regular file sizes and complete SHA256 values, symlink targets and the
embedded manifest remain verified. Hardlinks reuse only a prior verified
regular-file/hardlink digest; forward and unresolved targets are rejected.
Reading gzip through EOF also validates its CRC/trailer. No package files or
other native contents are excluded.

The remaining archive lifecycle is unchanged: source stability, durable archive
hash/fsync, the pre-removal receipt, a second stability check, exact owned-source
removal and final durable receipt. The standalone verifier changes no source,
receipt, reward, model or live process.

CPU regressions cover 512 reverse-created files with deliberately reversed
manifest-dictionary order, forbidden gzip seeks and member scans, bounded
sequential read counts, hardlinks, missing/duplicate/unexpected members, content
and manifest corruption, gzip CRC/truncation, source changes and failed fsync.
All 17 existing/extended native-files tests passed locally. A separate
verification-only benchmark against the retained large archive must establish
actual runtime behavior; it cannot complete or relabel the original live Task.

The isolated verification-only benchmark subsequently passed on that exact
112,566,857-byte archive and 10,746-entry manifest: all 10,747 archive entries
(including the manifest) and 9,498 file records verified in **2.385 seconds**.
Its measured read-character count was 112,566,963 bytes, approximately one
compressed pass plus reading the process I/O counters. The production source
remained unchanged, and the original worker remained alive with neither its
receipt nor Task result yet published. No live cleanup was substituted.

Evidence:
`/root/nas_chaohuang/skill_training/diagnostics/native_archive_streaming_benchmark_1794464160113415650/result.json`,
SHA256 `c71792f32452d63d1639ae71fa3f717e72fd08f88b447bdd2210d453276281d4`.
The verifier candidate SHA256 was
`a34a0f26858ce6c4162376296bbe135c12d837c583d8a84d508c51536f743e5b`.
