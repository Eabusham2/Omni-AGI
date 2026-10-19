# SQLite completion under the local developer runtime

The October 18 composition check exposed a runtime-specific completion wait,
not a teacher-data, metadata-admission, archive-size or full-merge custody
failure. Production custody code and its validations were retained.

## Controlled evidence

- Installed Homebrew Node 26.7.0 on this Mac reproduced the wait in an owned
  raw-SQLite subprocess outside Vitest. A tiny coherent backup took 1200.3 ms,
  completing only when an unrelated 1200 ms keepalive timer fired. The following
  backup took about 1 ms. The fixture contains only one text row and performs no
  model, provider, training or application-load operation.
- The identical subprocess under Electron 43.2.0's bundled Node 24.18.0
  completed the two backups in 1.04 ms and 0.69 ms. The controlled inputs and
  operation order were the same.
- Process sampling during the Node 26 wait showed the main event loop and
  libuv workers idle. Monotonic phase timing placed the wait in native backup
  promise completion; stat and checksum work took roughly 3 ms afterward.
  Untimed buffered Vitest output initially made the wait appear to be in the
  following file read. The timed measurements supersede that interpretation.
- The exact previously failing full-merge teacher composition check passed
  under bundled Node 24.18.0 in 1396 ms, using the unchanged 30000 ms limit. It
  checks real tiny SQLite/media custody, selected metadata admission, dormant
  projections and byte-exact sealed parent history. This is not neural or
  provider-capability proof.

Temporary stream, source-close and event-loop-yield experiments did not fix the
Node 26 wait and were removed. No timeout increase, skipped validation, data
deletion, production-backup replacement or polling workaround was retained.

## Local command

CI and `.nvmrc` now select Node24.18.0, matching the application's bundled
runtime. The earlier Node22 selection is historical evidence, not the current
qualification runtime. To exercise this check with the installed runtime on macOS:

```bash
ELECTRON_RUN_AS_NODE=1 \
  ./node_modules/electron/dist/Electron.app/Contents/MacOS/Electron \
  node_modules/vitest/vitest.mjs run tests/apiTeacherMediaComposition.test.ts \
  -t 'threads selected metadata admission' --maxWorkers=1 --testTimeout=30000
```

`ELECTRON_RUN_AS_NODE` starts a Node process, not the desktop application or a
brain worker. This controlled result does not claim that all Node 26 versions,
all operating systems or untested future runtime versions share this issue.
Current exact-commit CI and native platform package results remain separate
release evidence.

## Windows filename semantics

The Windows run additionally exposed a version-specific API difference:
Node22's incremental `backup()` destination does not interpret SQLite URI
filenames, unlike `DatabaseSync`/`VACUUM INTO`; the supported Node24 implementation
does. The destination flags differ in the official
[Node22.23.3 backup implementation](https://github.com/nodejs/node/blob/v22.23.3/src/node_sqlite.cc#L449)
and [Node24.0.0 implementation](https://github.com/nodejs/node/blob/v24.0.0/src/node_sqlite.cc#L443).
Supplying the constructor's long-path URI to the older backup API is not
a valid cross-version adapter. Backup destinations therefore have their own
version-aware filename boundary. CI is pinned to24.18.0 so that its native
Windows fixtures qualify the actual shipped runtime, including deep paths;
the existing assertions and incremental cancellation behavior are not removed.

The official release provides both architectures on Windows, macOS and Linux:
[Node24.18.0 release checksums](https://nodejs.org/download/release/v24.18.0/SHASUMS256.txt).
Older-runtime limitations remain explicit; no global VFS switch, non-locking
mode, blocking replacement for incremental backup or extra staging copy is
used to claim compatibility.
