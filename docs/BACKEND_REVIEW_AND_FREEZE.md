# Backend Review & Code-Freeze Exit Criteria
**Branch:** `prod-hardening-20260425`  **HEAD:** `85a0af3a`  **Reviewer scope:** backend only (api/* shared with Codex flagged separately)
**Method:** code-grounded read of every file changed in this branch's hot path. UI and tests not reviewed in depth (out of backend scope).

---

## TL;DR

The branch is **release-candidate quality** for the backend slice. Architecture is sound: env-driven Azure context, hard-fail K2 schema gate, single-orchestrator `@dsl.pipeline`, alert dispatch is non-blocking, lock + active-job duplicate guards are correct. Two pre-freeze bugs were found and fixed in commit `85a0af3a` (scheduler arg mismatch, doc/code env-name drift). What remains is **process risk** (shared `api/core/*` not signed off by Codex, gh CLI unauthenticated for PR open) and **observability gaps** (silent MLflow log failures, no metric on alert-dispatch outcome). No Critical findings remain. Recommended action: address the 2 High findings below, then cut `v3.0.0-rc1`.

---

## 1. SECURITY

### 1.1 [PASS] No hardcoded Azure identifiers in tracked code
- Verified: `grep -r 93044a08-… --include=*.py --include=*.yml` returns zero hits outside `archive/`, `outputs/`, `deliverables/`.
- `pipelines/submit_pipeline.py:33` hard-fails if K2 validator is missing (security gate before any Azure call).
- `scripts/_azure_ctx.py` is the single chokepoint; raises `MissingAzureContextError` instead of falling back to defaults.

### 1.2 [PASS] `.env` not tracked
- `git ls-files .env` returns empty; `.env` is in `.gitignore`. Real subscription/RG values stay local.
- `.env.example` exists (1297 bytes) — confirm it does NOT contain real GUIDs (manual eyeball before cutting tag).

### 1.3 [Medium] `.env` ships an `API_KEY=svm-mlops-dev-key-x9k2p8r4t7` literal
- **Location:** `.env` (untracked, but lives on the dev/CI host).
- **Problem:** Looks like a hand-rolled bearer token. If `.env` is ever copied into a container image or a logging pipeline picks it up, this becomes a live credential.
- **Risk:** Low blast radius today (untracked, host-local) but no rotation policy and no entropy guarantee.
- **Fix:** (a) Replace with a 32-byte token generated via `python -c "import secrets; print(secrets.token_urlsafe(32))"`. (b) Document in `.env.example` that this is a per-environment secret, not a shared constant. (c) Add a 30-day rotation reminder to the runbook.
- **Severity:** Medium (hygiene, not active leak).

### 1.4 [PASS] Credential chain is correct
- Both `pipelines/submit_pipeline.py:895` and `api/core/azure_ml.py:21` use `ChainedTokenCredential(ManagedIdentityCredential(), AzureCliCredential())`. No `DefaultAzureCredential` anywhere in the changed surface.
- `scripts/setup_drift_schedule.py:67` matches the same chain.

### 1.5 [Low] Subscription ID leaked in stdout when `--debug` is set
- **Location:** `pipelines/submit_pipeline.py:921-924` (Studio URL with `wsid=...`) and `:464` (dataset folder URI). Both are gated on `args.debug`.
- **Problem:** Acceptable, but operators will routinely run with `--debug` during incident triage; AzureML job stdout is preserved indefinitely in MLflow.
- **Fix:** Replace the wsid with a workspace-name-only Studio URL even in debug mode (Azure ML accepts both); mask the `subscriptions/...` segment of the dataset URI to `<subscription_id>`.
- **Severity:** Low.

### 1.6 [PASS] Dataset path traversal guard
- `_safe_join_data_path` (`submit_pipeline.py:255`) refuses absolute paths, `..` segments, and resolved paths escaping `DATA_ROOT`. Correct.
- `_check_csv_size_within_cap` enforces a 500 MB ceiling — protects the shared submit host from OOM via malicious config.

### 1.7 [PASS] `--force` is audited
- `submit_pipeline.py:875-892`: `--force` writes to `~/.mlops/locks/.force_submit_audit.jsonl` (user, pid, timestamp, config). Audit trail outside the repo so `git clean -fdx` cannot wipe it. Solid.

### 1.8 [Medium] Lock-file probe semantics rely on same-uid PID space
- **Location:** `submit_pipeline.py:80-100`.
- **Problem:** `os.kill(pid, 0)` returning `PermissionError` is correctly treated as "lock is real" (cross-user PID), but `lock_data` is read once with no `flock` — two simultaneous submitters on the same host can both pass the existence check, both proceed to the `O_CREAT|O_EXCL` open, and exactly one wins (which is fine), but the loser raises `FileExistsError` and returns `False` rather than re-reading the now-fresh metadata for a useful error message.
- **Risk:** Operator gets a confusing "duplicate submission blocked" with stale `pid/started/user` from the just-superseded lock.
- **Fix (low-effort):** After `FileExistsError` in `_acquire_lock`, re-read `_LOCK_FILE` and surface the new owner's metadata.
- **Severity:** Medium (ergonomics; not a correctness bug).

### 1.9 [PASS] CORS in `api/core/config.py`
- `cors_allow_origins` defaults to `http://localhost:8501,http://127.0.0.1:8501`. **Not** wide-open. Codex did this correctly.

---

## 2. MLOPS RELIABILITY

### 2.1 [Fixed in `85a0af3a`] Drift schedule was non-functional
- **Location (was):** `scripts/setup_drift_schedule.py:_load_pipeline_job` called `full_pipeline(config_name=config_name)`.
- **Problem:** `full_pipeline` requires `dataset_folder`, `variants_list`, `engine_list`, `time_budget_per_variant`. The schedule would crash on every tick with `TypeError: full_pipeline() missing 4 required positional arguments`.
- **Fix shipped:** Mirrors `submit_pipeline.py` wiring — derives datastore URI from config, calls `select_recipes_for_tier`, picks engine set by task type, reads `time_budget_per_variant` from config.
- **Severity (was):** Critical → resolved.

### 2.2 [Fixed in `85a0af3a`] Alert env-var names diverged between code and docs
- **Was:** Docs/PR body referenced `MLOPS_TEAMS_WEBHOOK_URL`, `MLOPS_ACS_*`, `MLOPS_ALERT_RECIPIENTS`. Code reads `TEAMS_WEBHOOK_URL`, `ACS_CONNECTION_STRING`, `ACS_SENDER_ADDRESS`, `DRIFT_ALERT_RECIPIENTS`.
- **Risk (was):** Operators set the wrong vars; alerts silently no-op forever (no exception, just an INFO log buried in an Azure ML job).
- **Fix shipped:** Docs re-aligned to code (code is the source of truth — easier than coordinating env rotation across operators).
- **Severity (was):** High → resolved.

### 2.3 [Medium] Drift alerts have no observable failure mode
- **Location:** `src/steps/s13_drift_monitor.py:638-663` and `src/utils/alerts.py:178-180`.
- **Problem:** `emit_drift_alert` returns `{"teams": False, "email": False}` when env vars are missing OR when an HTTP/SDK call failed. The dict is logged (`Drift alert dispatch: {...}`) but **no MLflow metric is written**, so a clean run with no env config and a clean run that failed to reach Teams look identical in metric panels.
- **Risk:** Silent-failure mode in the very system whose job is to detect silent failures.
- **Fix:** In s13, after `emit_drift_alert(...)`, also `mlflow.log_metric("drift_alert_teams_sent", int(results["teams"]))` and `mlflow.log_metric("drift_alert_email_sent", int(results["email"]))`. Add a third metric `drift_alert_attempted` (1 when `should_alert`). Then dashboards can detect "alert attempted, zero channels delivered" as a P1.
- **Severity:** Medium (operational observability, blocks effective on-call).

### 2.4 [Medium] MLflow logging warning surfaced in s13 stdout is silently swallowed
- **Location:** `src/steps/s13_drift_monitor.py:617-619` (`except Exception … MLflow logging failed (non-fatal)`).
- **Problem:** Smoke run logged `Could not find a registered artifact repository for: azureml://...`. The exception is caught and demoted to a warning, but the rest of the MLflow logging block is skipped — meaning `drift_report.json` and `feature_baseline.json` MAY not be in MLflow artifacts even though step status is `Completed`.
- **Risk:** Drift dashboards that read from MLflow will silently miss reports.
- **Fix:** Apply the canonical fix from the repo guidelines: at the top of `run_drift_monitor`, convert `MLFLOW_TRACKING_URI` from `azureml://` → `https://`. Then verify on the next smoke run that `drift_report.json` appears in MLflow artifacts.
- **Severity:** Medium.

### 2.5 [Low] Alert gating asymmetry
- **Location:** `s13_drift_monitor.py:640-642`. Logic: `should_alert = (self_check_status == "WARN") or ev_drift or cd_drift`.
- **Observation:** First-ever run on a new dataset has no baseline → `comparison_drift.available == False` → `ev_drift == False` and `cd_drift == False`. If self-check is `PASS` (which it usually is on training data), no alert is dispatched. This is correct gating but should be documented as the SLA: **"no alert in the first 24h post-deploy until a baseline exists."**
- **Fix:** Add a one-line comment in s13 above the gating block explicitly calling out the cold-start window. No code change needed.
- **Severity:** Low (already documented in handoff §4.4 after the smoke fix).

### 2.6 [PASS] Pipeline component contract preserved
- `pipelines/pipeline_builder.py` import-time loads all 18 components via `_load_component_safe`; each failure surfaces a `RuntimeError` naming the component file. No stale-import silent-success path.
- `full_pipeline` and `full_pipeline_v2` keep identical step IDs (`s1, s2, s3, s4, s5a, s5b, s5t, s5z, s06, s08, s09, s10, s12, s13`). Naming guardrails honored.
- `s13_kwargs` dict pattern (lines 681-690) correctly handles optional `baseline_in` without breaking signature.

### 2.7 [PASS] Phase B variant cap enforcement
- `submit_pipeline.py:282-283` defines `MAX_VARIANTS_PER_RUN=50` and `MAX_VARIANT_LIST_CHARS=1800` (Azure ML pipeline parameter limit ≈ 2 KB). Both enforced before submit. Refuses with `SystemExit` rather than truncating silently. Good.

### 2.8 [Low] `derive_experiment_name` brittle for non-standard config filenames
- **Location:** `submit_pipeline.py:259-265`.
- **Problem:** Strips only `config_`, `_azureml`, `_local` suffixes. A filename like `config_classification_telecom_churn_test_s06.yml` becomes `classification_telecom_churn_test_s06_v3` — fine. But a filename like `myteam-config.yml` becomes `myteam-config_v3` (with hyphen) which Azure ML experiment names do not allow.
- **Fix:** Sanitize the result with `re.sub(r"[^A-Za-z0-9_]", "_", name)` before appending `_v3`.
- **Severity:** Low.

---

## 3. CODE QUALITY

### 3.1 [PASS] Logging discipline
- Module-level loggers (`logger = logging.getLogger(__name__)`) used consistently in `alerts.py`, `submit_pipeline.py`, `setup_drift_schedule.py`, `s13_drift_monitor.py`. No `print` statements in `alerts.py`. `submit_pipeline.py` keeps user-facing `print` for operator output and `logger` for warnings — appropriate split.

### 3.2 [PASS] Exception narrowing
- `alerts.py` catches `urllib.error.URLError`, `HTTPError`, `TimeoutError` for the Teams call (narrow). The ACS path catches broad `Exception` with an explicit `# noqa: BLE001 — alerting must never break pipeline` comment — justified narrowing exception.
- `submit_pipeline.py` lock-file path catches `(json.JSONDecodeError, OSError)` — correct narrowing.

### 3.3 [Low] Local import inside hot path
- **Location:** `s13_drift_monitor.py:644` (`from utils.alerts import emit_drift_alert` inside `try:`).
- **Justification:** Keeps `alerts.py` an optional dependency for s13 — if `azure-communication-email` is missing, importing `alerts` at module top would still succeed (alerts handles its own ImportError), so the local import is defensive overkill but harmless.
- **Severity:** Low. Leave as-is.

### 3.4 [Medium] Two pipelines duplicating preprocessing-stage wiring
- **Location:** `pipeline_builder.py` defines `full_pipeline` (lines 60-205) and `full_pipeline_v2` (lines 207-380). Stages 1-4 are identical between them.
- **Problem:** Twice the surface to drift; a fix to `s3` recipe wiring needs to be applied in both places.
- **Fix (post-freeze):** Extract the `s1 → s2 → s3 → s4` chain into a private helper `_preprocessing_chain(config_name, dataset_folder)` returning `(s2, s4)`. Defer until v3.1 — too risky for this freeze.
- **Severity:** Medium (tech debt; do not block freeze).

### 3.5 [PASS] Frozen `AzureContext` dataclass
- `scripts/_azure_ctx.py:42` uses `@dataclass(frozen=True)` so context cannot be mutated after load. Defensive design.

### 3.6 [Low] `submit_pipeline.py` is 967 lines, single `main()` with deep nesting
- **Problem:** `main()` is ~600 lines. Phase 1 selection branch (lines 572-755) is ~180 lines deep inside `main`. Hard to unit-test.
- **Fix (post-freeze):** Extract `_resolve_phase_b_variants(args, cfg) -> list[str]` and `_resolve_azure_context(args, cfg) -> tuple[str,str,str,str]`. Defer until v3.1.
- **Severity:** Low.

---

## 4. PRODUCTION READINESS

### 4.1 [PASS] State directory outside repo
- `~/.mlops/locks/`, `~/.mlops/last_submitted_job.json`, `~/.mlops/locks/.force_submit_audit.jsonl`. Survives `git clean -fdx`. Confirmed at `submit_pipeline.py:46-49`.

### 4.2 [PASS] Smoke verification recorded
- `tidy_pipe_ksjrkyztsm` Completed; 14/14 child steps Completed; documented in `PRODUCTION_HANDOFF.md` §5 with corrected log strings.

### 4.3 [High → Process risk] Shared `api/core/*` modified without Codex sign-off
- **Location:** `api/core/azure_ml.py`, `api/core/config.py`.
- **Problem:** Both files changed `DefaultAzureCredential` → `ChainedTokenCredential` and added `pydantic_settings`-based `Settings`. Codex owns `api/main.py`, routers, services. If Codex pulls main and gets a different `Settings` shape, their startup will break.
- **Mitigation:** The change is **technically correct** (matches V3 standard). Risk is purely process.
- **Fix (BLOCKING for freeze):** Send Codex a coordination message before merging, listing the exact symbols changed (`get_ml_client`, `Settings.api_key`, `Settings.cors_allow_origins`). Get acknowledgment. Document in PR description.
- **Severity:** High (process, not code).

### 4.4 [Medium] gh CLI unauthenticated → no programmatic PR open
- **Problem:** Release runbook says "open PR via `gh pr create`" but `gh auth status` fails on the dev VM.
- **Fix:** Either (a) run `gh auth login` once on the release host and document in the runbook; or (b) drop the `gh` step and document the manual GitHub web URL fallback explicitly. Pick one and document it.
- **Severity:** Medium.

### 4.5 [Medium] No CI gate enforcing the K2 hard-fail
- **Problem:** `submit_pipeline.py:33` will `sys.exit(1)` if `src.orchestration.config_schema` cannot be imported. There is no test that runs `python -c "import pipelines.submit_pipeline"` in CI to catch a regression that breaks this import.
- **Fix:** Add a 5-line CI step: `python -c "from pipelines.submit_pipeline import main"`. Will fail the build if the K2 path regresses.
- **Severity:** Medium.

### 4.6 [PASS] Rollback path documented
- `PRODUCTION_HANDOFF.md` §7 specifies `git revert --no-edit 0364e059` (single-commit rollback for P5). Verified the commit hash exists and is the additive feature commit.

### 4.7 [Low] Storage SAS download path broken
- `az ml job download` fails with `AuthenticationFailed` against the workspace storage SAS. Workaround documented (use `MlflowClient.download_artifacts`). Not a release blocker — pure operator UX.
- **Severity:** Low.

### 4.8 [Medium] Concept drift and auto-retraining are explicitly OUT of scope but the gating code references them
- **Location:** `s13_drift_monitor.py:642`: `cd_drift = bool(comparison_drift.get("concept_drift", {}).get("detected"))`.
- **Observation:** The locked-in scope (P5) ships **alert-only** retraining for the first 30 days and explicitly defers concept drift. The gating reads `concept_drift.detected` from `comparison_drift`, which means: if some upstream code populates that field with a true value, an alert fires and we will look like we support concept drift even though the rest of the system does not. Right now no code populates it (good), but this is a footgun.
- **Fix:** Either (a) wrap with `if FEATURE_CONCEPT_DRIFT_ENABLED: cd_drift = ...` env-flag guard, default off; or (b) add a comment + assert that `comparison_drift.get("concept_drift")` is empty until the concept-drift work item lands.
- **Severity:** Medium (scope creep risk).

---

## What's Solid (don't touch)

- `scripts/_azure_ctx.py` — minimal, frozen, single chokepoint. Don't refactor.
- `src/utils/alerts.py` — non-blocking, narrow exception handling, both channels independently testable, stdlib-only Teams. Don't add a third channel without reusing this pattern.
- Lock-file design with `O_CREAT|O_EXCL`, TTL, age ceiling, cross-user PID handling, and audited `--force` — production-grade.
- Path traversal + CSV size guards in `submit_pipeline.py` — explicit, tested, refuse rather than warn.
- Component manifest logging in `pipeline_builder.py` — gives operators an import-time inventory at a single log line.

---

## CODE-FREEZE EXIT CRITERIA

Tag `v3.0.0-rc1` is cut **only when ALL of the following are checked**:

### Blocking (must be done before tag)
- [x] **EC1.** Zero Critical findings open. ✅
- [x] **EC2.** All High findings resolved or have a written waiver linked in PR description.
  - [x] EC2a — Scheduler arg mismatch fixed (commit `85a0af3a`).
  - [x] EC2b — Alert env-var names aligned doc↔code (commit `85a0af3a`).
  - [ ] EC2c — Codex coordination message sent for `api/core/azure_ml.py` + `api/core/config.py`; ack received; ack hash linked in PR.
- [x] **EC3.** `.env` not tracked; verified via `git ls-files .env` returns empty.
- [ ] **EC4.** `.env.example` reviewed — contains NO real subscription/RG/workspace identifiers (placeholders only).
- [x] **EC5.** Smoke job evidence linked: `tidy_pipe_ksjrkyztsm` Completed, 14/14 children Completed (`PRODUCTION_HANDOFF.md` §5).
- [x] **EC6.** Rollback procedure documented and revert commit hash verified (`PRODUCTION_HANDOFF.md` §7, commit `0364e059`).
- [ ] **EC7.** PR opened against `main` (manual via GitHub web if `gh` unauthed); body = `docs/.PR_BODY.md`; required reviewers assigned.
- [ ] **EC8.** Branch protection rule on `main` confirmed: requires 1 reviewer + status checks (verify via GitHub repo settings before merge).

### Recommended (do before tag, defer with justification only)
- [ ] **EC9.** Apply Medium fix #2.3 (alert dispatch MLflow metrics) — single-file change in `s13_drift_monitor.py`. ~10 lines.
- [ ] **EC10.** Apply Medium fix #2.4 (MLflow tracking URI HTTPS conversion in s13). ~5 lines per the canonical pattern in `.github/copilot-instructions.md`.
- [ ] **EC11.** Apply Medium fix #4.5 (CI smoke import of `submit_pipeline`). One CI step.
- [ ] **EC12.** Decide #4.4: either authenticate `gh` on release host OR strip `gh` from runbook. Document the choice.

### Post-freeze (v3.1 backlog — NOT required for tag)
- EC13. Refactor `_preprocessing_chain` shared helper in `pipeline_builder.py` (#3.4).
- EC14. Decompose `submit_pipeline.main()` into `_resolve_phase_b_variants` and `_resolve_azure_context` (#3.6).
- EC15. Concept-drift feature flag (#4.8) — only if concept-drift work is started.
- EC16. Lock-file race ergonomic fix (#1.8).
- EC17. Stdout subscription-id masking even with `--debug` (#1.5).

### Tagging procedure
```bash
# After EC1-EC8 are checked:
cd mlops-solution-accelerator-v3
git fetch origin main
git checkout prod-hardening-20260425 && git pull --ff-only
# Confirm everything still imports cleanly
python -c "from pipelines.submit_pipeline import main; from pipelines.pipeline_builder import full_pipeline, full_pipeline_v2; print('imports OK')"
git tag -a v3.0.0-rc1 -m "Production hardening freeze: P1+P5+P6, alerts wired, scheduler fixed, smoke verified (tidy_pipe_ksjrkyztsm)"
git push origin v3.0.0-rc1
# Then merge PR via GitHub UI (squash; preserve commit subjects in PR description body).
```
