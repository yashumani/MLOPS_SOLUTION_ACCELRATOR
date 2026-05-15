"""Metrics table component — sortable model leaderboard with phase grouping."""

import pandas as pd
import streamlit as st


# Canonical phase order + display metadata used for grouping the leaderboard.
PHASE_ORDER = [
    "phase_a", "baseline", "a", "phase_a_baseline",
    "phase_b", "b", "variants",
    "phase_c", "c", "hpo", "optuna", "phase_c_hpo",
    "final", "register",
]

PHASE_DISPLAY = {
    "phase_a": ("🔵 Phase A — Baseline", "Reference scores from PyCaret + FLAML on raw recipe."),
    "baseline": ("🔵 Phase A — Baseline", "Reference scores from PyCaret + FLAML on raw recipe."),
    "a": ("🔵 Phase A — Baseline", "Reference scores from PyCaret + FLAML on raw recipe."),
    "phase_a_baseline": ("🔵 Phase A — Baseline", "Reference scores from PyCaret + FLAML on raw recipe."),
    "phase_b": ("🟣 Phase B — Variant search", "Top variants selected by the recommender (recipes × engines)."),
    "b": ("🟣 Phase B — Variant search", "Top variants selected by the recommender (recipes × engines)."),
    "variants": ("🟣 Phase B — Variant search", "Top variants selected by the recommender (recipes × engines)."),
    "phase_c": ("🟠 Phase C — HPO", "Optuna hyperparameter optimisation on the Phase B champion."),
    "c": ("🟠 Phase C — HPO", "Optuna hyperparameter optimisation on the Phase B champion."),
    "hpo": ("🟠 Phase C — HPO", "Optuna hyperparameter optimisation on the Phase B champion."),
    "optuna": ("🟠 Phase C — HPO", "Optuna hyperparameter optimisation on the Phase B champion."),
    "phase_c_hpo": ("🟠 Phase C — HPO", "Optuna hyperparameter optimisation on the Phase B champion."),
    "final": ("🟢 Final", "Holdout evaluation of the chosen champion."),
    "register": ("🟢 Final", "Holdout evaluation of the chosen champion."),
}


def _phase_key(raw: object) -> str:
    return str(raw or "").strip().lower().replace(" ", "_") or "unknown"


def _phase_label(raw: object) -> tuple[str, str]:
    key = _phase_key(raw)
    return PHASE_DISPLAY.get(key, (f"⚪ {raw or 'Other'}", ""))


def _flatten_model_metric(item: dict) -> dict:
    """Flatten a ModelMetric dict into a single flat row for a DataFrame."""
    if not isinstance(item, dict):
        return {"value": str(item)}

    metrics_dict = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
    flat = {
        "Champion": "yes" if item.get("is_champion") else "",
        "model": item.get("model_name") or item.get("model") or item.get("name") or "—",
        "phase": item.get("phase") or "—",
        "engine": item.get("engine") or "—",
    }
    for k, v in metrics_dict.items():
        if isinstance(v, (int, float)):
            flat[k] = round(float(v), 4)
        else:
            flat[k] = v
    for k, v in item.items():
        if k in {"model_name", "model", "name", "engine", "phase",
                 "metrics", "is_champion"}:
            continue
        if k not in flat:
            flat[k] = v
    return flat


def _arrow_safe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a Streamlit/Arrow-friendly copy for display."""
    safe = df.copy()
    safe.columns = [str(col) for col in safe.columns]
    for col in safe.columns:
        if safe[col].dtype == "object":
            safe[col] = safe[col].map(lambda value: "" if value is None else str(value))
    return safe


def _markdown_table(df: pd.DataFrame, *, max_rows: int = 50) -> str:
    safe = _arrow_safe(df).head(max_rows)
    if safe.empty:
        return ""
    columns = [str(col) for col in safe.columns]
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for _, row in safe.iterrows():
        values = [str(row[col]).replace("|", "\\|").replace("\n", " ") for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    suffix = ""
    if len(df) > max_rows:
        suffix = f"\n\n_Showing first {max_rows} of {len(df)} rows. Download CSV for the full table._"
    return "\n".join([header, separator, *rows]) + suffix


def _to_df(metrics: list[dict]) -> pd.DataFrame:
    rows = [_flatten_model_metric(m) for m in metrics]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    leading = [c for c in ("Champion", "model", "phase", "engine") if c in df.columns]
    remaining = [c for c in df.columns if c not in leading]
    return df[leading + remaining]


def render_metrics_table(metrics: list[dict]):
    """Flat sortable leaderboard (kept for backward compatibility)."""
    if not metrics:
        st.info("No metrics data available.")
        return None
    df = _to_df(metrics)
    if df.empty:
        st.info("No metrics data available.")
        return None
    st.markdown(_markdown_table(df))
    return df


def render_phase_grouped_leaderboard(metrics: list[dict]) -> pd.DataFrame | None:
    """Render the leaderboard grouped by Phase A / B / C inside expanders.

    Each phase expander shows row count, champion model (if any), and the
    sortable metrics table for that phase only. Returns the full DataFrame so
    callers can offer a single CSV download.
    """
    if not metrics:
        st.info("No leaderboard rows yet — waiting for `s05z`/`s06`/`s09` aggregate reports.")
        return None

    df = _to_df(metrics)
    if df.empty:
        st.info("No metrics data available.")
        return None

    # Group key derived from the (possibly messy) phase column.
    df = df.copy()
    df["_phase_key"] = df["phase"].apply(_phase_key)
    seen_keys: list[str] = []
    for k in df["_phase_key"]:
        if k not in seen_keys:
            seen_keys.append(k)
    # Sort by canonical phase order, with unknown phases at the end.
    seen_keys.sort(
        key=lambda k: PHASE_ORDER.index(k) if k in PHASE_ORDER else len(PHASE_ORDER) + 1
    )

    for key in seen_keys:
        sub = df[df["_phase_key"] == key].drop(columns=["_phase_key"])
        label, helptext = _phase_label(key)
        champ = sub[sub["Champion"] == "yes"]
        suffix = f" — {len(sub)} model{'s' if len(sub) != 1 else ''}"
        if not champ.empty:
            cm = champ.iloc[0]
            suffix += f" · 🏆 `{cm['model']}` ({cm['engine']})"
        with st.expander(f"**{label}**{suffix}", expanded=True):
            if helptext:
                st.caption(helptext)
            st.markdown(_markdown_table(sub))

    return df.drop(columns=["_phase_key"])


def render_champion_rationale(metrics: list[dict], summary: dict | None = None) -> None:
    """Compact card explaining who the champion is and which phase delivered it."""
    summary = summary or {}
    champion = next((m for m in metrics if m.get("is_champion")), None)
    if not champion:
        st.caption(
            "No champion flagged yet — the aggregate steps mark the champion "
            "after Phase A / Phase B / Phase C complete."
        )
        return

    name = champion.get("model_name") or champion.get("model") or "—"
    phase_raw = champion.get("phase") or summary.get("champion_phase") or "—"
    phase_label, _ = _phase_label(phase_raw)
    engine = champion.get("engine") or "—"
    score = summary.get("champion_score")
    score_disp = f"{score:.4f}" if isinstance(score, (int, float)) else "—"

    # Per-phase counts for context.
    by_phase: dict[str, int] = {}
    for m in metrics:
        by_phase[_phase_key(m.get("phase"))] = by_phase.get(_phase_key(m.get("phase")), 0) + 1

    parts = []
    if "phase_a" in by_phase or "baseline" in by_phase or "a" in by_phase:
        n = by_phase.get("phase_a", 0) + by_phase.get("baseline", 0) + by_phase.get("a", 0)
        parts.append(f"{n} baseline")
    if any(k in by_phase for k in ("phase_b", "b", "variants")):
        n = sum(by_phase.get(k, 0) for k in ("phase_b", "b", "variants"))
        parts.append(f"{n} variant{'s' if n != 1 else ''}")
    if any(k in by_phase for k in ("phase_c", "c", "hpo", "optuna", "phase_c_hpo")):
        n = sum(by_phase.get(k, 0) for k in ("phase_c", "c", "hpo", "optuna", "phase_c_hpo"))
        parts.append(f"{n} HPO trial{'s' if n != 1 else ''}")
    breakdown = " · ".join(parts) if parts else f"{len(metrics)} candidates"

    with st.container(border=True):
        c1, c2, c3 = st.columns([2, 2, 2])
        c1.metric("Champion model", name)
        c2.metric("From", phase_label.split(" — ")[0])
        c3.metric("Score", score_disp)
        st.caption(
            f"Engine: `{engine}` · Selected from {breakdown}. "
            "Champion is chosen by the aggregate step for each phase; the final "
            "champion is the best across Phase A / B / C."
        )
