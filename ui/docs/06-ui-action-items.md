# 6 - UI Fix Action Items

This backlog captures the UI work needed before the Streamlit app should be treated as a production data-scientist control plane. It is ordered by release risk, not by implementation size.

## Release Position

- Current UI status: internal beta / demo usable.
- Production status: not ready until the P0 items are closed.
- Scope: Streamlit UI behavior, UX clarity, browser validation, and UI-facing API contract assumptions.
- Non-goal: do not change V3 pipeline orchestration files from this backlog unless explicitly approved.

## P0 - Ship Blockers

| ID | Action Item | Why It Matters | Acceptance Criteria | Validation |
|---|---|---|---|---|
| UI-P0-01 | Add a Submit preflight panel before launch | Users need to know exactly what Azure ML will run before spending compute. | Submit page shows resolved config, compute, dataset, target, task type, engines, selected Phase B variants, HPO budget, optional drift baseline, and estimated stage plan. | Playwright verifies preflight details render for at least one classification config. |
| UI-P0-02 | Gate production Submit until canonical V3 submission parity is proven | The UI must not imply it launches the canonical production pipeline if the API path can diverge from `pipelines/submit_pipeline.py`. | Submit button is either backed by canonical-equivalent preflight/submit behavior or clearly marked/internal-gated with a warning. | Mocked submit contract test plus one Azure-safe submission validation when approved. |
| UI-P0-03 | Make Live Logs product wording truthful | Current UX can imply real step log streaming even when backend log content is unavailable. | If raw logs are unavailable, page clearly becomes a Studio log launcher/status browser; if logs are available, filters/search/download operate on real content. | Playwright fixture covers both "logs available" and "logs unavailable" states. |
| UI-P0-04 | Add robust UI error states for slow job detail refresh | A single slow Azure ML job-detail call should not stall Focus or show a traceback. | Focus keeps cached status visible, uses bounded request timeouts, and shows a small non-blocking refresh warning. | Current Playwright suites pass; add a mocked timeout test for Focus. |
| UI-P0-05 | Prove Drift Monitor with a real drift artifact fixture | Page routing passes, but production confidence requires a populated PSI report path. | Drift Monitor renders summary KPIs, PSI legend, chart/table, top feature details, and CSV download for a known drift report fixture. | Playwright test asserts non-empty chart/table content from fixture data. |

## P1 - High Priority Product Gaps

| ID | Action Item | Why It Matters | Acceptance Criteria | Validation |
|---|---|---|---|---|
| UI-P1-01 | Upgrade Focus Outputs into a first-class artifact browser | Data scientists need to inspect models, reports, leaderboards, plots, and manifests without leaving the app. | Outputs tab groups artifacts by phase/type, previews JSON/CSV/text/images/HTML where safe, and exposes clear downloads. | Fixture-driven Playwright test opens at least one JSON, one CSV, and one downloadable artifact. |
| UI-P1-02 | Add guided config editing for critical fields | Raw YAML editing is powerful but risky for data scientists. | Configs page exposes guided controls for dataset path, target, task type, compute, Phase B strategy, engines, HPO trials, and drift baseline. YAML remains available as advanced mode. | Unit/UI test verifies guided edits serialize back to valid YAML. |
| UI-P1-03 | Add config diff and rollback affordances | Users can accidentally change production configs with no review surface. | Before save, show old/new diff; after save, record visible timestamp/user/source metadata when available; provide restore from duplicate/export path. | Manual and mocked API tests for update confirmation flow. |
| UI-P1-04 | Add run comparison view | Data scientists need to compare runs across configs, seeds, phases, and datasets. | New or Focus-adjacent comparison UI supports 2-4 selected jobs, champion metrics, phase winners, final metrics, and drift summaries. | Playwright fixture compares two completed jobs. |
| UI-P1-05 | Improve empty states across Focus, Outputs, Drift, and Logs | Empty pages currently look like missing features when artifacts are optional. | Each empty state states why data is absent, which stage creates it, and the next action. | Visual review plus text assertions in Playwright. |

## P2 - Polish and Confidence

| ID | Action Item | Why It Matters | Acceptance Criteria | Validation |
|---|---|---|---|---|
| UI-P2-01 | Remove or suppress Streamlit internal subpage 404 noise where possible | Browser console noise reduces release confidence, even when it is not user-facing. | Confirm whether `_stcore/health` and `_stcore/host-config` relative-path 404s are harmless; document or configure routing to reduce them. | Playwright console report has no unexplained 404s. |
| UI-P2-02 | Add responsive viewport coverage | Data scientists may use laptops, wide monitors, and Azure ML embedded views. | Playwright runs desktop and narrow viewport screenshots for all primary pages. | CI or local script emits per-viewport report artifacts. |
| UI-P2-03 | Add visual regression baselines for core pages | Prevent future layout regressions after Streamlit/theme changes. | Store baseline screenshots for Home, Submit, Focus, Configs, Drift, and Logs under release artifacts. | Screenshot diff threshold agreed and checked in test script. |
| UI-P2-04 | Add UI docs release checklist | The docs should guide final signoff, not only describe pages. | UI docs include a release checklist with environment, tests, known limitations, and owner signoff. | Checklist is updated before each release candidate. |
| UI-P2-05 | Tighten terminology for V3 stages | UI should not claim inactive stages as mandatory production behavior. | Stage labels distinguish active, inferred, optional, and not-wired steps. | Compare UI labels against current pipeline builder behavior and docs. |

## Recommended Work Order

1. UI-P0-04 - lock down Focus timeout and traceback behavior.
2. UI-P0-01 - add Submit preflight visibility.
3. UI-P0-02 - gate or align production Submit.
4. UI-P0-03 - make Live Logs truthful.
5. UI-P0-05 - prove populated Drift Monitor behavior.
6. UI-P1-01 - strengthen Outputs artifact inspection.
7. UI-P1-02 and UI-P1-03 - make Configs safer for data scientists.
8. UI-P1-04 - add run comparison.
9. P2 items - polish, console cleanup, viewport/visual regression coverage.

## Verification Gate

Before calling the UI production-ready, run and archive:

```bash
/anaconda/envs/mlops_pipeline_v2/bin/python -m compileall -q ui scripts/playwright_ui_e2e.py scripts/playwright_data_scientist_journey.py
/anaconda/envs/mlops_pipeline_v2/bin/python -m pip check
/anaconda/envs/mlops_pipeline_v2/bin/python scripts/playwright_ui_e2e.py
/anaconda/envs/mlops_pipeline_v2/bin/python scripts/playwright_data_scientist_journey.py
/anaconda/envs/mlops_pipeline_v2/bin/python -m pytest tests/test_api_security_hardening.py -q
```

The Playwright journey must explicitly state whether Submit was clicked. If no Azure ML job is submitted, the report is a navigation/preflight pass, not a true production submit pass.