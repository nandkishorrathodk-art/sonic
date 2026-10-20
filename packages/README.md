# Shared packages

- [`avcs/`](avcs/) contains the AVCS visual-perception framework and evaluation
  code.
- [`kernel/`](kernel/) contains the native Rust kernel and its IPC interface.

For AVCS's current SONIC integration and capability boundaries, see
[`../docs/SONIC_AVCS_SYSTEM_FLOW.md`](../docs/SONIC_AVCS_SYSTEM_FLOW.md).

AVCS is part of the root Python workspace so its dependencies and tests use
the shared lockfile. Run the complete Python suite from the repository root.
The Rust kernel remains a Cargo package and can be tested with
`cargo test --locked --manifest-path packages/kernel/Cargo.toml`.
