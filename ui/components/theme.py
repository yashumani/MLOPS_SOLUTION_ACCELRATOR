"""Savyminds brand theme — polished LIGHT design system.

Single source of truth for colors, typography, cards, buttons, badges and
sidebar styling. Public API is intentionally stable; only the palette
changed (dark → light) so no other UI file needs to be edited.

Public API
----------
- ``inject_theme()``           Inject the CSS into the page
- ``LOGO_HTML``                Sidebar logo block
- ``page_header(title, ...)``  Top-of-page hero
- ``section_label(text)``      Small uppercase section eyebrow
- ``status_badge(status)``     Returns an HTML pill for a job status
"""

import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# Savyminds palette — clean, professional, light
# ──────────────────────────────────────────────────────────────────────────────
BRAND_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
  /* Surfaces (off-white → white → very light grey) */
  --svm-bg:        #F6F8FB;
  --svm-surface:   #FFFFFF;
  --svm-surface-2: #F1F5F9;
  --svm-surface-3: #E9EEF5;

  /* Borders */
  --svm-border:    #E5EAF2;
  --svm-border-2:  #CBD5E1;

  /* Brand */
  --svm-primary:        #2563EB;
  --svm-primary-hover:  #1D4ED8;
  --svm-accent:         #06B6D4;
  --svm-gradient:       linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
  --svm-gradient-soft:  linear-gradient(135deg, rgba(37,99,235,0.08) 0%, rgba(6,182,212,0.08) 100%);

  /* Text — graphite scale */
  --svm-text:    #0F172A;
  --svm-text-2:  #334155;
  --svm-text-3:  #64748B;
  --svm-text-4:  #94A3B8;

  /* Status */
  --svm-success: #10B981;
  --svm-warning: #F59E0B;
  --svm-danger:  #EF4444;
  --svm-info:    #06B6D4;

  /* Soft elevation */
  --svm-shadow-1: 0 1px 2px rgba(15, 23, 42, 0.04), 0 1px 3px rgba(15, 23, 42, 0.06);
  --svm-shadow-2: 0 4px 12px rgba(15, 23, 42, 0.06), 0 2px 4px rgba(15, 23, 42, 0.04);
  --svm-shadow-3: 0 10px 28px rgba(15, 23, 42, 0.08), 0 4px 8px rgba(15, 23, 42, 0.04);
}

/* ── Global ──────────────────────────────────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"], .stApp {
  background: var(--svm-bg) !important;
  color: var(--svm-text) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

[data-testid="stSidebarNav"] { display: none !important; }

header[data-testid="stHeader"] {
  background: transparent !important;
  border-bottom: 1px solid var(--svm-border) !important;
  height: 3px !important;
}
header[data-testid="stHeader"]::before {
  content: ""; display: block; height: 3px;
  background: var(--svm-gradient);
}

.block-container {
  padding-top: 2rem !important;
  padding-bottom: 3rem !important;
  max-width: 1400px !important;
}

/* ── Headings ─────────────────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
  color: var(--svm-text) !important;
  font-weight: 700 !important;
  letter-spacing: -0.01em !important;
}
h1 { font-size: 2rem !important; }
h2 { font-size: 1.5rem !important; }
h3 { font-size: 1.15rem !important; }

p, span, div, label, li { color: var(--svm-text-2); }

a { color: var(--svm-primary) !important; text-decoration: none !important; }
a:hover { color: var(--svm-primary-hover) !important; text-decoration: underline !important; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #FFFFFF !important;
  border-right: 1px solid var(--svm-border) !important;
  box-shadow: 1px 0 2px rgba(15, 23, 42, 0.02);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem !important; }
[data-testid="stSidebar"] * { color: var(--svm-text-2); }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: var(--svm-text) !important; }

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
  background: var(--svm-gradient) !important;
  color: #FFFFFF !important;
  border: none !important;
  border-radius: 8px !important;
  padding: 0.55rem 1.1rem !important;
  font-weight: 600 !important;
  font-size: 0.875rem !important;
  cursor: pointer !important;
  transition: transform 0.12s ease, filter 0.12s ease, box-shadow 0.15s ease !important;
  box-shadow: 0 2px 6px rgba(37, 99, 235, 0.20) !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
.stFormSubmitButton > button:hover {
  transform: translateY(-1px) !important;
  filter: brightness(1.05) !important;
  box-shadow: 0 6px 16px rgba(37, 99, 235, 0.30) !important;
}
.stButton > button[kind="secondary"] {
  background: #FFFFFF !important;
  color: var(--svm-text-2) !important;
  border: 1px solid var(--svm-border-2) !important;
  box-shadow: var(--svm-shadow-1) !important;
}
.stButton > button[kind="secondary"]:hover {
  background: var(--svm-surface-2) !important;
  border-color: var(--svm-primary) !important;
  color: var(--svm-primary) !important;
}

/* ── Page links (sidebar nav rows) ────────────────────────────────────────── */
[data-testid="stPageLink"], [data-testid="stPageLink-NavLink"] {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  padding: 9px 12px !important;
  border-radius: 8px !important;
  color: var(--svm-text-2) !important;
  text-decoration: none !important;
  font-size: 0.9rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  transition: all 0.15s ease !important;
  margin-bottom: 2px !important;
  border: 1px solid transparent !important;
}
[data-testid="stPageLink"]:hover, [data-testid="stPageLink-NavLink"]:hover {
  background: var(--svm-gradient-soft) !important;
  color: var(--svm-primary) !important;
  border-color: rgba(37, 99, 235, 0.20) !important;
  text-decoration: none !important;
}
[data-testid="stPageLink"] *, [data-testid="stPageLink-NavLink"] * {
  cursor: pointer !important;
  color: inherit !important;
}

/* ── Inputs ───────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea textarea,
.stDateInput input {
  background: #FFFFFF !important;
  color: var(--svm-text) !important;
  border: 1px solid var(--svm-border-2) !important;
  border-radius: 8px !important;
  font-size: 0.9rem !important;
  box-shadow: var(--svm-shadow-1) !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
  background: #FFFFFF !important;
  color: var(--svm-text) !important;
  border: 1px solid var(--svm-border-2) !important;
  border-radius: 8px !important;
  box-shadow: var(--svm-shadow-1) !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus,
.stTextArea textarea:focus,
.stDateInput input:focus {
  border-color: var(--svm-primary) !important;
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.15) !important;
  outline: none !important;
}
.stTextInput label, .stSelectbox label, .stNumberInput label,
.stMultiSelect label, .stTextArea label, .stRadio label,
.stCheckbox label, .stDateInput label, .stFileUploader label {
  color: var(--svm-text-3) !important;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.04em !important;
}
.stRadio div[role="radiogroup"] label,
.stCheckbox label span { color: var(--svm-text-2) !important; }

[data-testid="stFileUploader"] section {
  background: var(--svm-surface-2) !important;
  border: 1.5px dashed var(--svm-border-2) !important;
  border-radius: 10px !important;
}

/* ── Cards ────────────────────────────────────────────────────────────────── */
.svm-card {
  background: #FFFFFF !important;
  border: 1px solid var(--svm-border) !important;
  border-radius: 12px !important;
  padding: 1.25rem !important;
  box-shadow: var(--svm-shadow-1) !important;
  transition: border-color 0.15s ease, transform 0.15s ease, box-shadow 0.15s ease !important;
}
.svm-card:hover {
  border-color: rgba(37, 99, 235, 0.45) !important;
  transform: translateY(-2px) !important;
  box-shadow: var(--svm-shadow-3) !important;
}
.svm-card-icon {
  width: 38px; height: 38px; border-radius: 9px;
  display: flex; align-items: center; justify-content: center;
  background: var(--svm-gradient-soft);
  color: var(--svm-primary);
  font-size: 1.15rem;
  margin-bottom: 0.85rem;
}
.svm-card-title {
  color: var(--svm-text) !important;
  font-weight: 600 !important;
  font-size: 0.98rem !important;
  margin: 0 0 0.35rem !important;
}
.svm-card-desc {
  color: var(--svm-text-3) !important;
  font-size: 0.82rem !important;
  margin: 0 !important;
  line-height: 1.4 !important;
}

/* ── Metrics ──────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] {
  background: #FFFFFF !important;
  border: 1px solid var(--svm-border) !important;
  border-radius: 12px !important;
  padding: 1rem 1.25rem !important;
  box-shadow: var(--svm-shadow-1) !important;
  transition: box-shadow 0.15s ease, border-color 0.15s ease !important;
}
[data-testid="stMetric"]:hover {
  border-color: var(--svm-border-2) !important;
  box-shadow: var(--svm-shadow-2) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--svm-text-3) !important;
  font-size: 0.75rem !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.06em !important;
}
[data-testid="stMetricValue"] {
  color: var(--svm-text) !important;
  font-size: 1.875rem !important;
  font-weight: 700 !important;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ── Tabs ─────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--svm-border) !important;
  gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--svm-text-3) !important;
  padding: 0.75rem 1.25rem !important;
  font-weight: 500 !important;
  cursor: pointer !important;
  border-radius: 8px 8px 0 0 !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--svm-primary) !important;
  background: var(--svm-surface-2) !important;
}
.stTabs [aria-selected="true"] {
  color: var(--svm-primary) !important;
  border-bottom: 2px solid var(--svm-primary) !important;
}

/* ── DataFrames / Tables ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"], [data-testid="stTable"] {
  background: #FFFFFF !important;
  border: 1px solid var(--svm-border) !important;
  border-radius: 12px !important;
  overflow: hidden !important;
  box-shadow: var(--svm-shadow-1) !important;
}
[data-testid="stDataFrame"] thead tr th {
  background: var(--svm-surface-2) !important;
  color: var(--svm-text) !important;
  font-weight: 600 !important;
}

/* ── Expander ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
  background: #FFFFFF !important;
  border: 1px solid var(--svm-border) !important;
  border-radius: 10px !important;
  box-shadow: var(--svm-shadow-1) !important;
}
[data-testid="stExpander"] summary {
  color: var(--svm-text) !important;
  font-weight: 600 !important;
  cursor: pointer !important;
  padding: 0.6rem 0.9rem !important;
}
[data-testid="stExpander"] summary:hover { color: var(--svm-primary) !important; }

/* ── Alerts ───────────────────────────────────────────────────────────────── */
.stAlert {
  border-radius: 10px !important;
  border: 1px solid var(--svm-border) !important;
  background: #FFFFFF !important;
  box-shadow: var(--svm-shadow-1) !important;
}
div[data-baseweb="notification"] { color: var(--svm-text-2) !important; }
[data-testid="stAlert"][kind="info"]    { border-left: 3px solid var(--svm-info)    !important; }
[data-testid="stAlert"][kind="success"] { border-left: 3px solid var(--svm-success) !important; }
[data-testid="stAlert"][kind="warning"] { border-left: 3px solid var(--svm-warning) !important; }
[data-testid="stAlert"][kind="error"]   { border-left: 3px solid var(--svm-danger)  !important; }

/* ── Page header ──────────────────────────────────────────────────────────── */
.svm-page-header {
  margin: 0 0 1.75rem;
  padding: 0 0 1.25rem;
  border-bottom: 1px solid var(--svm-border);
}
.svm-page-title {
  margin: 0 !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  color: var(--svm-text) !important;
  letter-spacing: -0.02em !important;
}
.svm-page-subtitle {
  margin: 0.35rem 0 0 !important;
  font-size: 0.95rem !important;
  color: var(--svm-text-3) !important;
}

.svm-section-label {
  font-size: 0.7rem;
  color: var(--svm-text-4);
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-weight: 600;
  margin: 1.5rem 0 0.65rem;
}

/* ── Status pills ─────────────────────────────────────────────────────────── */
.svm-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 10px; border-radius: 999px;
  font-size: 0.75rem; font-weight: 600;
  line-height: 1.2;
}
.svm-pill-running   { background: rgba(6,182,212,0.10);  color: #0E7490; border: 1px solid rgba(6,182,212,0.30); }
.svm-pill-completed { background: rgba(16,185,129,0.10); color: #047857; border: 1px solid rgba(16,185,129,0.30); }
.svm-pill-failed    { background: rgba(239,68,68,0.10);  color: #B91C1C; border: 1px solid rgba(239,68,68,0.30); }
.svm-pill-warning   { background: rgba(245,158,11,0.10); color: #B45309; border: 1px solid rgba(245,158,11,0.30); }
.svm-pill-default   { background: var(--svm-surface-2);  color: var(--svm-text-3); border: 1px solid var(--svm-border-2); }

/* ── Connection badge ─────────────────────────────────────────────────────── */
.svm-conn {
  display: flex; align-items: center; gap: 8px;
  padding: 9px 12px; border-radius: 8px;
  margin-bottom: 1rem; font-size: 0.82rem; font-weight: 600;
  border: 1px solid var(--svm-border);
  background: #FFFFFF;
  box-shadow: var(--svm-shadow-1);
}
.svm-conn-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; flex-shrink: 0; }
.svm-conn-ok      { background: rgba(16,185,129,0.08); border-color: rgba(16,185,129,0.30); color: #047857; }
.svm-conn-ok      .svm-conn-dot { background: #10B981; box-shadow: 0 0 6px rgba(16,185,129,0.6); }
.svm-conn-err     { background: rgba(239,68,68,0.08);  border-color: rgba(239,68,68,0.30);  color: #B91C1C; }
.svm-conn-err     .svm-conn-dot { background: #EF4444; }
.svm-conn-pending { background: rgba(245,158,11,0.08); border-color: rgba(245,158,11,0.30); color: #B45309; }
.svm-conn-pending .svm-conn-dot { background: #F59E0B; animation: svm-pulse 1.5s ease-in-out infinite; }
@keyframes svm-pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ── Logo ─────────────────────────────────────────────────────────────────── */
.svm-logo-row {
  display: flex; align-items: center; gap: 12px;
  padding: 4px 0 18px;
}
.svm-logo-mark {
  width: 40px; height: 40px; border-radius: 10px;
  background: var(--svm-gradient);
  display: flex; align-items: center; justify-content: center;
  color: #FFFFFF; font-weight: 800; font-size: 1.05rem;
  box-shadow: 0 4px 12px rgba(37, 99, 235, 0.30);
  letter-spacing: -0.5px;
}
.svm-logo-text-1 { color: var(--svm-text) !important; font-weight: 700; font-size: 1rem; line-height: 1.1; }
.svm-logo-text-2 { color: var(--svm-text-4) !important; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.12em; margin-top: 2px; }

/* ── Misc ─────────────────────────────────────────────────────────────────── */
iframe[height="0"], iframe[height="0px"] { display: none !important; }

code { color: var(--svm-text) !important; background: var(--svm-surface-2) !important; padding: 1px 6px !important; border-radius: 4px !important; }
pre code { background: transparent !important; padding: 0 !important; }
pre { background: var(--svm-surface-2) !important; border: 1px solid var(--svm-border) !important; border-radius: 10px !important; }

::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-track { background: var(--svm-surface-2); }
::-webkit-scrollbar-thumb { background: var(--svm-border-2); border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: var(--svm-primary); }

button, a, [role="button"], [role="link"], [role="tab"],
.stButton, .stDownloadButton, [data-testid="stPageLink"] { cursor: pointer !important; }
button *, a *, [role="button"] *, [role="link"] * { cursor: pointer !important; }
.svm-card, .svm-card * { cursor: default; }
</style>
"""

LOGO_HTML = """
<div class="svm-logo-row">
  <div class="svm-logo-mark">SM</div>
  <div>
    <div class="svm-logo-text-1">Savyminds</div>
    <div class="svm-logo-text-2">MLOps&nbsp;V3</div>
  </div>
</div>
"""


def inject_theme() -> None:
    """Inject Savyminds CSS. No iframes — page stays fully clickable."""
    st.markdown(BRAND_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "", icon: str = "") -> None:
    display_title = f"{icon} {title}" if icon and icon.isascii() else title
    sub = f'<p class="svm-page-subtitle">{subtitle}</p>' if subtitle else ""
    st.markdown(
        f'<div class="svm-page-header">'
        f'<h1 class="svm-page-title">{display_title}</h1>{sub}</div>',
        unsafe_allow_html=True,
    )


def section_label(text: str) -> None:
    st.markdown(f'<div class="svm-section-label">{text}</div>', unsafe_allow_html=True)


def status_badge(status: str) -> str:
    s = (status or "").lower()
    if s in ("running", "preparing", "starting", "queued"):
        cls = "svm-pill-running"
    elif s in ("completed", "finished", "success"):
        cls = "svm-pill-completed"
    elif s in ("failed", "error", "cancelled", "canceled"):
        cls = "svm-pill-failed"
    elif s in ("warning", "warn"):
        cls = "svm-pill-warning"
    else:
        cls = "svm-pill-default"
    return f'<span class="svm-pill {cls}">{status}</span>'
