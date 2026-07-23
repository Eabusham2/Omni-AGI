# Security model

Omni AGI Studio can intentionally grant a locally trained model broad authority. Neural freedom and operating-system authority are separate:

- The model is not given a hidden behavioral policy prompt.
- Electron enforces the tool grant selected by the owner.
- Full Authority permits destructive commands and is visibly marked.
- Tool actions, arguments, exit status, and changed paths are journaled.
- Credentials are stored outside neural checkpoints and `.omni` exports.
- Imported `.omni` files are data, never executable installers.
- Imported tensor files must use safe, non-pickle formats.
- Source self-modification occurs in a forked worktree with a visible diff.

Report implementation vulnerabilities privately to the repository owner before publishing details.
