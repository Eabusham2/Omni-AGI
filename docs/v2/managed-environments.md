# Managed sandboxes and virtual machines

Omni AGI Studio exposes two ordinary permissioned chat tools:
`runtime.sandbox` and `runtime.vm`. They create main-owned environment IDs and
durable lifecycle/operation receipts. They do not silently redirect to the
ordinary shell, install a runtime, download an image, run a repository setup
script, adopt an existing user environment, or imply that all host/guest
isolation has been certified.

## Tool lifecycle

Both tools provide `capabilities`, `create`, `list`, `status`, `load`, `run`,
and `stop`.

- `capabilities` reports the adapter executable actually detected, whether the
  exact create/run route is configured, warnings, and `installationPerformed:
  false`.
- `create` commits a unique environment and operation identity before its
  first effect. Reusing one request ID with different operands is rejected;
  repeating the same request returns its retained receipt instead of creating
  another environment.
- `load` rechecks the saved identity and currently installed adapter. It never
  means “use the unrestricted host runner.”
- `run` records the exact program/arguments digest before dispatch, monitors
  selected app RAM and child PIDs cooperatively, retains bounded stdout/stderr,
  and is cancellable. Windows Sandbox is an explicit exception to output
  capture: Microsoft's CLI returns the guest exit code but no guest process
  I/O, so its receipt carries `guestOutput: guest-output-unavailable`; any
  retained stdout/stderr are host-CLI diagnostics, not guest output.
- `stop` cancels the exact in-flight owner, waits for its receipt to settle,
  and then stops a persistent VM when applicable. Stop cannot undo effects that
  already happened.
- A host loss changes unfinished work to `interrupted` and the environment to
  `review-required`; it is never replayed as if nothing happened. Lima status
  can reconcile an actual retained VM, and Windows status queries the exact
  retained Sandbox UUID. The deprecated Darwin sandbox route has
  no durable PID/start-time owner proof after a crash, so it cannot safely
  claim Stop or resume; retain its files for inspection and create a new one.

Unload/app shutdown cancels an in-flight managed operation but does not pretend
that an already loaded external VM stopped. On the next host, `status` queries
the exact private Lima runtime before another use. Permanent brain deletion is
the separate stop-and-remove boundary described below.

Create, Load, Run, and Stop are risky under Auto and therefore use the normal
one-use exact-argument review. Ask and Off retain their ordinary meanings;
Full is still bounded by the installed adapter and OS. A current VM image also
requires the exact bound task attachment, its SHA-256, and attachment access
not being Off.

## Sandbox adapters

On current macOS source, the local adapter is `/usr/bin/sandbox-exec` when it
actually exists. Apple’s local manual marks this command deprecated, and Studio
reports that fact. The generated profile disables network, allows writes only
inside the owned workspace, denies reads of user/volume/network/temp roots
outside that workspace, and starts the child through `env -i`; the host process
environment is not forwarded. Broad App Sandbox certification, future macOS
availability, immunity to every OS/kernel flaw, and absence of every readable
system/application secret outside the denylisted roots are not claimed.

On Linux, Studio uses an actually installed `bwrap` with new namespaces,
network unshared, a clear child environment, read-only system trees, and one
writable `/workspace`. If bubblewrap is absent, the sandbox is unavailable.

On Windows, Studio can use Microsoft's already-installed Windows Sandbox CLI
on a compatible Windows 11 24H2-or-later host. It resolves `wsb.exe` from the
exact Microsoft Windows Sandbox package family under the standard protected
WindowsApps location and requires Windows Authenticode to report a valid
signature; the generic executable finder/PATH cannot authorize it, and an
injected test resolver is not production discovery. Nonstandard OS/package
locations fail closed rather than trusting an environment-provided executable.
A read-only, recognized and untruncated `wsb list --raw` probe must also succeed. Studio does
not enable the optional Windows feature, restart Windows, install an edition,
or bypass virtualization requirements. Unsupported editions (including Home),
older builds, disabled features, and failed runtime probes remain unavailable.

Load chooses a fresh UUID and invokes `wsb start --id` with networking, vGPU,
clipboard, microphone, camera, and printer redirection disabled and Protected
Client enabled. There are no mapped folders, LogonCommand, downloads, host
credential forwarding, or silent RDP connection. Studio refuses Load while a
different current-user Sandbox exists; it never adopts or stops that session.
Run uses the exact owned UUID and `System` only as the *guest* execution
context—it is not host elevation. Stop deletes that disposable guest. A Run or
Load cancellation also stops the whole exact owned Sandbox because the CLI has
no sound per-process cancellation primitive. A later Load starts a clean guest;
no stopped guest filesystem or process state is restored. The current route
passes one command line to an executable already present in the clean guest;
it has no stdin/file-upload channel. Microsoft exposes no guest stdout/stderr.

Studio serializes Start/list/verify and Stop/list/verify across all brains in
its single Electron main process. The operating system remains the final
single-instance arbiter if another current-user program races the CLI. A host
loss while Run/Stop is active marks the UUID as possibly retaining a guest
command; Status will not make it reusable merely because the Sandbox is still
running. The user must Stop that exact UUID and dispose the guest first.

## Lima VM adapter

The VM route is intentionally narrower:

- Lima must already be installed.
- Create requires one current-task local image attachment, exact SHA-256,
  declared `aarch64` or `x86_64`, reviewed CPU/RAM/disk values, and explicit
  `guest-outbound` networking.
- The generated configuration contains only the exact local `file:` image and
  digest, no host mounts, port forwards, container runtime, user provisioning
  or probe scripts, forwarded SSH agent/X11, remote URL, or downloaded template.
  Lima plain mode still performs Lima's documented base guest and SSH setup.
- Lima runs under a fresh installation-private `HOME`/`LIMA_HOME` and an empty
  inherited environment. It cannot read or adopt the user’s normal Lima
  instances/configuration or forward host credentials through this adapter.
- RAM is admitted against the selected host envelope and disk capacity is
  reserved on the actual private runtime volume before Create. The raw external
  VM disk is installation state, not silently placed into a portable brain.

A copied/imported/merged VM record therefore remains dormant evidence with its
external Lima identity removed. Recreate it from a newly reviewed local image.
Sandbox workspace files and exact bounded receipts participate in complete
workspace portability; a copied sandbox is also dormant until explicit Load,
which regenerates its local profile for the new path. Windows Sandbox UUIDs
and live/disposable guest state are never copied; explicit Load creates a new
clean UUID only after current permission and runtime review.
An in-process writer/snapshot lease waits for active environment operations and
blocks new ones through capture. It is not a cross-process filesystem lock or
proof against another OS process that can rewrite the app's private files.

Portable sandbox files and actual captured stdout/stderr receipts are
deliberately unsanitized task data and may contain secrets a program wrote or
printed. Windows Sandbox guest stdout/stderr are never captured, while its
host-CLI diagnostics remain unsanitized. Review a private complete export
accordingly. Dormancy removes live process/adapter identity,
credential-store forwarding and execution authority; it does not claim to
scrub retained content bytes.

Permanent instance deletion first cancels environment operations, refuses to
continue if a managed VM cannot be stopped, and then removes that exact
installation-private VM runtime before deleting the confirmed brain directory.

## Recorded evidence and limits

Focused source tests cover checksummed SQLite state, request idempotence,
interrupted recovery, cancellation/Stop, ToolExecutor Off/Ask/Auto routing,
declared schemas, dormant copy/merge, and a pure injected Lima command path.
The Lima fixture verifies a local `file:` digest, private clean runtime
environment, no remote-template/URL/mount/user-provision/credential operand, and exact
create/start/shell/stop command construction; it does not boot a real VM.

One actual bounded macOS `sandbox-exec` fixture ran `/usr/bin/true`, proved a
synthetic host environment value did not cross, wrote one file inside its owned
workspace, and was denied reading a controlled file outside that workspace.
It ran no user program, model, provider, download, user setup script, or privileged
change.

Seven focused Windows adapter cases use an injected CLI process boundary to
verify hardened configuration, exact UUID selection, refusal to adopt/stop an
unrelated Sandbox, trusted executable provenance, serialized cross-brain Start,
crash-surviving-command and failed-Stop disposal, disposable Stop/reload, command construction,
typed unavailable guest output, and cancellation by whole-owned-Sandbox teardown.
They do not boot Windows Sandbox, verify a particular Windows edition/package,
or prove the undocumented `--raw` JSON shape; parsing therefore binds the
exact UUID and treats unrecognized status as unknown.

Real Lima boot/guest execution, Linux bubblewrap, an actual Windows Sandbox
boot/exec/Stop, hostile sandbox certification, crash behavior of every
external runtime, packaged platform acceptance, CI, signing, and release
remain open.

Primary Windows references: [Windows Sandbox CLI](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-cli),
[supported editions and disposable/single-instance behavior](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/),
[installation prerequisites](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-install),
and [configuration controls](https://learn.microsoft.com/en-us/windows/security/application-security/application-isolation/windows-sandbox/windows-sandbox-configure-using-wsb-file).
