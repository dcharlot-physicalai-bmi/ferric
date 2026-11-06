# Actor check, 2026-11-06

Issues and the pull request were not opened. The account that would have written them is not `dcharlot-physicalai-bmi`.

## What was verified

| Check | Result |
|---|---|
| Repository | `dcharlot-physicalai-bmi/ferric` |
| `gh api graphql` `viewer.login` | `cursor[bot]` |
| `gh api user` | HTTP 403, `Resource not accessible by integration` |
| `gh auth status` | logged-in account name `cursor` |
| Installation repositories | only `dcharlot-physicalai-bmi/ferric` |
| Existing issues | `gh issue list --repo dcharlot-physicalai-bmi/ferric --state all --limit 100` returned `[]` |
| Existing pull requests | one merged pull request, author login `dcharlot-physicalai-bmi` (PR 1). None open for this work |
| Cursor run owner | David Charlot, `david@openie.dev` |

`cursor[bot]` cannot be selected into `dcharlot-physicalai-bmi`. The OpenIE address is the run owner, and it is not an acceptable actor for a remote write on this workstream.

## What was not done

- No `gh issue create`.
- No pull request through the sanctioned pull-request tool, because that actor is not confirmed as `dcharlot-physicalai-bmi`.
- No `gh` or REST call as a workaround to open the pull request.
- No push to any repository other than `dcharlot-physicalai-bmi/ferric`. The installation token cannot see another repository.
- The developer tree named in `scripts/speech_joules_browser.sh` was not read and not edited.

Issue bodies for tracks A–F are in `docs/acceptance/issues/`. They are the text to paste once the opener is `dcharlot-physicalai-bmi`. Opening them as `cursor[bot]` or as the OpenIE account would be the wrong identity, so they stay in the branch.
