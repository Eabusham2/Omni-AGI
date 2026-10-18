# Instance and host clipboards

Omni AGI Studio exposes two separate text-only tools so a private brain-local
clipboard cannot silently become authority over the computer clipboard:

- `studio.clipboard` accepts only `scope: instance`. It is Auto by default,
  still follows the instance's current tool policy, and never touches the OS
  clipboard.
- `system.clipboard` accepts only `scope: host`. It is Off by default. Host
  Read, Write, and Clear remain risky under Auto and use the ordinary exact
  approval path.

Both tools expose `capabilities`, `read`, `write`, and `clear`. Every request
must state its exact scope. A scope mismatch is rejected before adapter
dispatch; it cannot borrow the other tool's permission.

## Instance clipboard

The instance clipboard stores complete, unsanitized `text/plain` bytes in the
brain's owned `clipboard/` workspace. Writes and clears create immutable
content-addressed versions and durable exact-request receipts. Reusing a
request ID with the same content returns its prior receipt; changing the
content under that ID is rejected.

Reads are bounded UTF-8 pages with a version/revision/hash-bound continuation
cursor. `capabilities` exposes bounded version metadata so inherited versions
can be inspected explicitly. It does not paste historical clipboard content
into a hidden prompt.

Complete exports, copies, recovery snapshots, and full merges stream the
SQLite ledger and exact blobs. Rekeying preserves current content and history.
A full merge chooses the first retained parent clipboard only when the new
target is still empty and retains the other parent versions. The clipboard
workspace contains no host permission, OS handle, session, or live effect.
Destination policy is applied independently: an external import uses safe
defaults and a full merge uses its reviewed destination access; a trusted
local duplicate follows Studio's general preserve-selected-policy contract but
receives its own policy revision history. Content is not sanitized and can
contain private text; review an export before sharing it.

## Host clipboard

The host route currently supports `text/plain` only:

- macOS uses fixed `/usr/bin/osascript` JXA with AppKit's explicit
  `NSPasteboardTypeString`, avoiding `pbcopy`'s RTF/EPS content sniffing.
- Windows uses the fixed system Windows PowerShell executable in STA mode and
  the installed `Get-Clipboard -Raw -TextFormatType UnicodeText` and
  `Set-Clipboard` cmdlets.
- A matching current Linux desktop session uses an already-installed, exact
  root-owned `/usr/bin/wl-copy` plus `/usr/bin/wl-paste`, or falls back to
  exact root-owned `/usr/bin/xclip` for an X11 display.
  Wayland writes use `wl-copy --foreground` with an explicit UTF-8 text MIME;
  reads use `wl-paste --no-newline`. X11 writes use `xclip -quiet -loops 0`;
  `-silent` is deliberately forbidden because xclip forks in silent mode.
  Studio tracks only the foreground child it created.

Studio does not use renderer `navigator.clipboard` or Electron `readText()` for
model tool calls. A host read runs in a killable helper, streams/discards bytes,
keeps only one bounded page in main-process memory, checks a prefix hash for a
continuation, and monitors helper resource use. A continuation proves the
already-read prefix, not that an unseen suffix remained unchanged.

Host Write and Clear persist exact content hashes and an effect intent before
dispatch, then recheck current owner, permission revision, cancellation, and
resource admission immediately before starting the OS adapter. A dispatched
mutation that loses its acknowledgement is retained as `write-uncertain` with
`retrySafe: false`; the same request is never automatically replayed. A
completed request also returns its retained receipt instead of writing twice.
Copied/imported pending effects become nonreplayable historical receipts.

Linux selection writes are not declared complete merely because stdin drained
or a child remained alive. A separate bounded helper reads the newly published
selection while retaining only a byte count and SHA-256 state; byte length and
hash must exactly match the dispatched UTF-8 bytes. A mismatch, lost selection,
cancellation, or timeout is a nonreplayable uncertain effect. Successful new
writes stop only Studio's prior owned holder or holders. Wayland Clear uses the
documented one-shot `wl-copy --clear`; X11 Clear publishes and verifies empty
text through a new owned foreground holder. Natural holder exit, replacement,
resource pressure, or Studio shutdown reconcile that exact child without
killing an unrelated clipboard owner. A completed holder may survive brain
Unload because it is an already-dispatched OS effect, but Unload confers no new
call authority. No PID, approval, or live holder is copied or exported.

Writing or clearing text can replace other formats currently held by the OS
clipboard. Image, rich-text, file-list, and custom clipboard formats are not
claimed in this version. No adapter is installed, downloaded, elevated, or
silently replaced with an unrestricted shell command.

## Evidence boundary

Twelve earlier focused cases use in-memory clipboards and one tiny synthetic
stdout child. Four additional Linux-holder cases use only tiny synthetic
foreground/readback children. Together they cover scope separation,
Off-default host policy, approval routing,
bounded UTF-8 paging, exact instance versions, complete workspace
snapshot/rekey/full-merge, completed-effect deduplication, and uncertain-effect
nonreplay, plus foreground arguments, exact hash-only publication proof,
owned-holder replacement, Unload survival, resource monitoring, and shutdown
cleanup. The earlier twelve were not rerun in the Linux slice. No case inspects,
modifies, or clears the user's real clipboard. Actual host clipboard behavior,
desktop-session rights, installed-package behavior, Linux compositor/X-server
selection ownership, non-text formats, CI, signing, and release remain separate
acceptance gates.

Windows cmdlet semantics are based on Microsoft's
[`Get-Clipboard`](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/get-clipboard?view=powershell-5.1)
and [`Set-Clipboard`](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.management/set-clipboard?view=powershell-5.1)
references. These source references are not an actual host acceptance run.
Linux command choices follow xclip's
[`xclip.c`](https://github.com/astrand/xclip/blob/master/xclip.c) ownership loop
and wl-clipboard's
[`wl-clipboard(1)`](https://github.com/bugaevc/wl-clipboard/blob/master/data/wl-clipboard.1)
foreground, clear, no-newline, and MIME options. Those references and synthetic
children are not an actual Wayland or X11 acceptance run.
