# Diagram delivery verification

Source snapshot: `282537e43b9693287bb9924a9896aab071f60a26` in `SAVYMINDS/YS_MVP`. Inspected and generated on 2026-09-04. Diagram type: architecture; five complementary views cover components, data flow, candidate search, operator controls and release tooling.

## Accepted outputs

| HTML | Artifact checks | Showcase composition | Desktop containment | Captured themes/sizes | Source references |
|---|---:|---|---:|---:|---:|
| overview.html | 9/9 | pass; 0 errors, 0 warnings | 4/4 | 4 | 19 |
| pipeline.html | 9/9 | pass; 0 errors, 0 warnings | 4/4 | 4 | 8 |
| search.html | 9/9 | pass; 0 errors, 0 warnings | 4/4 | 4 | 8 |
| operations.html | 9/9 | pass; 0 errors, 0 warnings | 4/4 | 4 | 11 |
| release.html | 9/9 | pass; 0 errors, 0 warnings | 4/4 | 4 | 9 |

The five `*.delivery.json` files contain the exact specification and HTML SHA-256 hashes and byte counts. `verification.json` records an independent readback: all specification hashes, HTML hashes and visual-receipt hashes matched. The source-reference check verifies paths and lines against this repository snapshot; source interpretation is documented in `coverage.md` and `ui-operations-evidence.md`.

The index static check passed with 5 diagram links, 25 capability groups, 14 stage entries, no missing local files or anchors, no duplicate IDs and valid embedded JavaScript syntax. `source-link-check.json` verifies 102 pinned source references in the coverage document and 27 in the index against local files and line bounds. Remote GitHub availability for the pinned revision was not established: connector access returned 404 and the local authenticated remote probes did not complete and were stopped. Source references are verified locally; hosted links require repository access and publication of the referenced revision.

`script-check.json` also records successful JavaScript syntax checks for both inline scripts in each of the five diagram HTML files. This static check does not replace interactive browser testing.

## Commands and receipts

Executed from the canonical nested repository using the installed Archify CLI at `C:\Users\yashu\.codex\skills\archify\bin\archify.mjs`:

```powershell
node <archify-cli> validate architecture docs/architecture/archify/<name>.json --quality showcase --repo-root . --json
node <archify-cli> deliver architecture docs/architecture/archify/<name>.json docs/architecture/archify/<name>.html --quality showcase --repo-root . --json
node <archify-cli> visual-check docs/architecture/archify/<name>.html --json
```

Final delivery receipt output for every view:

```json
{"checksPassed":9,"checkCount":9,"compositionProfile":"showcase","compositionStatus":"pass","errors":0,"warnings":0}
```

The desktop checks measured 1440x900, 1600x1000, 1920x1080 and 2048x1320. All had `scrollWidth <= innerWidth` and `scrollHeight <= innerHeight`. Four screenshots per view cover light and dark themes at 1440x900 and 2048x1320. The `*.visual-check.html` contact sheets and `*.visual-check.json` files retain the evidence.

## Visual review and limits

The author inspected each view's 1440x900 light screenshot and 2048x1320 dark screenshot. The reviewed images show contained node labels, clear relationship paths, complete cards, no obscuring line crossings and balanced vertical use at the large viewport. The automated receipts intentionally retain `visualReview: "pending"`; the human-readable inspection record here is separate from those immutable receipts.

Additional browser click testing was attempted, but the browser tool rejected local file URLs under its URL policy. No alternate server, browser surface or raw browser-command workaround was used. Theme/search/export click behavior was therefore not interactively tested in this task. The files contain Archify's bundled viewer controls; the packaged visual checks rendered both themes successfully before that browser-policy rejection.

This is diagram/documentation acceptance. No application source was changed or executed; no Azure jobs, deployment, model promotion or email was sent. Hosted application CI and live release readiness were not validated by this diagram task. The diagrams describe implemented source behavior, and explicitly distinguish registration, retraining, approved drift baselines and deployment.
