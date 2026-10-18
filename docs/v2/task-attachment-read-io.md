# Exact-file task attachment reads and bounded verification reuse

## Confirmed source issue

`TaskAttachmentService` creates a fresh Store for each operation. The old
Store-local `verified` map therefore disappeared between successive tool read
pages. `readForTurn` called `resolveForTurn`, which streamed a whole-file SHA,
then reopened the pathname to read the page. This both repeated file-sized
verification work for every page and left a hash/read pathname-replacement race.

The separately completed indexed turn/ID projection queries stay unchanged and
were not rerun for this work.

## Implemented boundary

- Service owns one byte-bounded, process-private verification cache shared by
  its short-lived stores. It contains proof metadata only, not raw source
  bytes, file handles, facts, answers or learned neural state. Its default
  metadata accounting envelope is 256KiB. Eviction causes rehashing; it never
  limits attachment size, count or visited content.
- Every operation checks the exact current bound row, public manifest SHA,
  private source-provenance SHA, content SHA, byte count, brain/turn ownership,
  managed directory identity and current registry identity. No all-history
  query or arbitrary path becomes read authority.
- The blob opens once. A cache miss hashes that **actual opened FD** in 64KiB
  windows. The requested page comes from the same FD, not a second pathname
  open. File/path dev, inode, size, nanosecond mtime and ctime must match before,
  during and after hashing/reading. Source/path/registry/owner changes refuse
  the operation before any read result is returned.
- Only a completed real SHA matching the exact expected content, followed by
  successful ownership/file/operation checks, can create a reusable proof.
  Inode identity alone is not cryptographic proof. Public manifests, private
  provenance and native input identities are not rewritten to fit a cache.
- Default cache eligibility is deliberately conservative: supported atomic
  no-follow opens plus known local APFS and noncoarse change timestamps. Unknown,
  network and unqualified filesystems perform a full exact-FD SHA every read.
  No proof survives a process restart or is exported as authority.
- Cancellation propagates through hashing, page reads and the bounded filesystem
  classification probe. Failures close the actual FD and do not publish a new
  proof or a partial page. Hash/page/encoding buffers and proof metadata receive
  physical read-memory admission checks. This is a bounded buffer/cache contract,
  not a measurement or hard guarantee of total process RSS.

Staging remains separate from ingestion. These reads do not create a neural
learning receipt, modify parameters, or claim that an attached file has been
understood. Existing ordinary learning of a visible tool result is unchanged.

## Windows and filesystem limits

Node/libuv does not expose atomic `O_NOFOLLOW` on Windows.
[libuv's filesystem flag documentation](https://docs.libuv.org/en/v1.x/fs.html#c.uv_fs_open)
documents that platform restriction. Windows functionality is retained with
regular-file/symlink rejection, exact opened-FD/path identity checks and a **full
SHA on every read**. It never uses the derived proof cache. This is explicitly
not a claim of an atomic Windows reparse-point/no-follow opener; that requires
a supported native Windows implementation and separate platform acceptance.

The metadata proof assumes the qualified local filesystem reports ordinary
file changes honestly. It is not protection against a privileged or malicious
filesystem that forges all identities/change timestamps, nor a claim that
external handles can never mutate a file. Unsupported identities fail
truthfully. Unknown filesystems do not silently acquire the APFS optimization.

## Focused evidence

Twelve distinct new actual-small-file/SQLite cases passed in two affected runs:

- **10 cases / 519ms**: repeat pages across fresh Service-created stores;
  same-content inode replacement; changed bytes with restored mtime;
  same-byte symlink rejection; hash/read replacement race and FD closure;
  mutation during the page operation; registry recovery/owner separation;
  post-hash private-source tampering; mid-hash cancellation/closure; Windows/
  unknown-FS fallback; and physical-memory refusal/zero-cache/LRU byte bounds.
- **2 additional cases / 66ms**, with the closed ten skipped: actual production
  filesystem classification and cancellation after a cached page read.
  The current Mac's real classifier reported `eligible:true`, `fullHashes:1`,
  `cacheHits:1` across two pages. No classifier was simulated in that case.

The deterministic four-page fixture uses a real 81,920-byte source with a
controlled eligible-filesystem policy. Verification hashed **81,920 rather
than 327,680 bytes** across four reads, returned exact requested page bytes and
left the source SHA unchanged. Other cases use real file/SQLite mutations and
real SHA operations, not a mocked hash result. The Windows-policy case runs on
the current Mac and establishes branch behavior, not real Windows reparse-point,
ACL or packaged-runtime qualification.

The mechanism removes repeated whole-file hashing for an unchanged, qualified
source within one process: file-sized verification once, then bounded page
reads and current authority/identity checks. These are component byte-work
counts, **not a measured elapsed-time speedup, native Load/TPS improvement or a
large-file/end-to-end model result**. No old indexed-query test, model Load,
brain creation/training, capture, corpus, long benchmark, CI or release was run.

Node TypeScript compilation and affected whitespace checks passed. Sources:
`taskAttachmentReadVerifier.ts`, the narrow Store read path, the Service-owned
verifier/optional cancellation seam, and `taskAttachmentReadVerifier.test.ts`.
