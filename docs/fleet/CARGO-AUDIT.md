# cargo audit

`cargo audit` against the lock resolved on 2026-11-13 (`Cargo.lock` sha256 `3addbafb869cbd7511ec5064f7248be1b1990389f55e3b01a88a5b3c2de296c5`, gitignored) reports no vulnerabilities. One advisory is unmaintained and has no upgrade.

## Upstream-blocked

| Advisory | Crate | Why it stays |
| --- | --- | --- |
| [RUSTSEC-2024-0436](https://rustsec.org/advisories/RUSTSEC-2024-0436) | `paste` 1.0.15 | Unmaintained since 2024-10-07. The advisory says no fixed upgrade is available. `tokenizers` 0.23.2, the latest stable release, still depends on it (`cargo tree -i paste`). `tokenizers` 1.0.0-rc.4 is a pre-release, not a release, so it is not taken. |

`deny.toml` ignores only this advisory id, with the same reason. It does not ignore a class of advisories.
