# Assembly and native machine-code tools

R025 now has a separate `code.assembly` adapter. It preserves existing
`code.execute` language-run behavior and uses the same cancellable,
output-bounded process runner with `shell: false` and hidden Windows process
windows. It does not load a model, substitute another AI, install a compiler,
or start training.

Source: `assemblyRuntime.ts`, `assemblyPortability.ts`,
`assemblyCapabilityPack.ts`, the narrow ToolExecutor/main composition hooks,
`tools/catalog.json`, and the existing complete workspace archive.

## Offered operations

- `inspect { artifactId? }`: inspect a saved artifact's original receipt and
  current file hashes, or read actually installed toolchain paths/version/
  target information. Default inspection also returns a paged saved-artifact
  list; `after`/`limit` continue the list without limiting stored artifacts.
- `compile { source, syntax?, kind?, toolchain?, architecture?, entryPoint?,
  libraries?, timeoutMs? }`: compile exact UTF-8 source in a fresh managed
  directory. Defaults: GAS syntax, executable output, an installed compatible
  compiler and the host architecture. No source, argv or library operand is
  silently trimmed, coerced or truncated. Unsupported operands fail explicitly.
- `run { artifactId, artifactSha256, arguments?, timeoutMs? }`: separately
  execute a successfully compiled native executable with an exact matching
  binary/source hash and matching recorded host platform/architecture; the OS
  loader remains the authority on whether the actual binary can execute. Raw flat bytes
  and object files cannot be silently treated as native executables.

Compile and Run default to Ask and remain risky operations under Auto. A
changed source or argument invalidates an approval. Off denies dispatch.
Full is an explicit broad host capability, not a concealed permission bypass.
The normal plugin nested-permission and authorized-intent path also applies.

| Syntax | Installed command/output support | Current boundary |
| --- | --- | --- |
| GAS | clang or gcc; object or native executable | Compiler/SDK/linker availability is actual host state. Clang target selection is explicit; unsupported GCC cross-target requests are rejected. |
| NASM | nasm; x86 object, flat binary, or host-x86 executable linked by an installed compiler | Foreign NASM objects may be retained, but a mismatched-host executable link/run is not silently emulated. |
| MASM | Windows x64 ml64.exe object and link.exe executable path | Requires those tools in the app's launch environment. No VS setup script/download is run; Windows qualification is not established by the Mac proof. |

On macOS, an already installed Apple compiler is preferred when present. The
adapter discovers the already installed SDK through `xcrun --show-sdk-path`
and passes its real directory explicitly via `-isysroot`. It does not install
Command Line Tools, Xcode, libraries or emulators. Missing/failed toolchain
availability and compiler/linker diagnostics are returned honestly.

## Exact effects, resources and recovery

Every compilation creates a new private directory under the instance's
`assembly/` area, with an immutable pre-effect source/operand/compiler intent.
There is no overwrite flag and no user-selected output path. Output/source
files must be regular files; unsafe symlinks and changed hashes are rejected.
Receipts retain exact source SHA-256, compiler/linker argv, SDK selection,
bounded stdout/stderr, process exit results, output hashes and useful partial
files. Cancellation/error keeps its typed receipt and partial files; it is
not converted into proof that compilation succeeded. A separate Run intent
binds the exact executable hash and argv before native execution.

Output-string hashes cover the **captured bounded output**, not discarded
bytes after the existing runner's output bound. The `truncated` flag remains
visible. A cancelled runner may not supply partial stdout/stderr, so the
adapter does not invent those bytes; already produced files remain available.

Main-owned app/known-child RSS, selected RAM ceiling, physical/heap headroom
and disk reserve are checked before dispatch and periodically during long
compiler/program work. Resource pressure requests process-group cancellation.
These are cooperative observations, not OS-enforced hard RSS isolation:
very fast allocation spikes or unobserved descendants cannot be guaranteed
safe. Actual platform process/cancellation errors are retained.

Native execution has the current OS user's host powers. This is **not** a
syscall/file/DMA sandbox, a promise that another logical tool's Off setting
physically confines native code, or an admin/protected-memory grant. Compiler
directives may read include files; linked native programs may perform host
operations allowed to their OS user. Fine-grained confinement requires a real
OS sandbox/VM; this adapter does not claim to supply one. User-approved native
code can have irreversible effects; Stop cannot undo them.

## Complete portability

The complete workspace archive preserves assembly source, native/object/flat
bytes, immutable intents, compile/run receipts and other regular output files
as **opaque data**. Bytes that resemble a SQLite header and filenames ending
in `-wal` are not reinterpreted or silently omitted. Importing/forking does
not execute code or grant access. A permitted future Run checks hashes and
host compatibility again; a restored private executable's mode is enabled
only as part of that explicitly permitted Run.

Merge preserves identical proven ID/bytes, unions matching histories, and
retains divergent artifacts under a new local bound ID. Original receipt and
user-output bodies remain byte-exact; the app-owned binding records the local
projection and original receipt hash. Inspection lists local and original
IDs so an ambiguous old reference need not be guessed. Recovery also retains
later native effect/artifact histories as inert branches instead of implying
that restoring neural state undoes those effects.

Interrupted external compiler/program processes are not portable resumable
neural cursors. Source/intent/partial files remain visible as incomplete data;
no process or past effect is automatically replayed on import, merge, restart
or Load.

## Teaching and recorded checks

`native-assembly-tools-1` is a separately versioned ordinary declared
capability pack, installed through existing initial/supplemental teaching
hooks. It explicitly says `executedHostActions: false`, adds no reward model
or hidden persona, and does not modify frozen origin-v3. Source/schema checks
do not prove a future trained brain can author correct assembly.

- Six constructor-free cases passed: actual command discovery, exact source/
  argv, output hashes, failed compiler partials, cancellation, resource/input
  refusal and stale/symlinked artifact rejection.
- Three actual ToolExecutor policy/error cases passed: Auto/Ask exact-source
  binding, Off/revocation, and typed partial receipt retention after dispatch.
- The one harmless native Mac ARM64 case first exposed a real `libSystem`
  linker-path failure. Read-only installed-SDK discovery fixed it. The same
  tiny fixture then compiled and actually printed
  `Omni safe assembly fixture` followed by a newline; that bounded case passed
  in 2.58 seconds. No model, large source corpus or foreign AI was used.
- Four file/declared-pack cases passed: opaque complete export/import,
  source-isolated merge conflict binding/dedup, more artifacts than one page,
  and honest schema-checked capability declarations. Combined constructor/
  portability checks:10/10 passed, 861ms test execution.

Windows/Linux, NASM/MASM linking, hostile native-code confinement and packaged
toolchain availability remain separate qualifications, not established by
the Mac proof or mocked command cases. No CI, commit, push, package or release
was performed by this lane.
