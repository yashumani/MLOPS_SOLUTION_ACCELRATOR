#!/usr/bin/env python3
"""Data scientist persona journey test for the Streamlit UI."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, Response, sync_playwright


ERROR_MARKERS = [
    "traceback",
    "modulenotfounderror",
    "attributeerror",
    "typeerror",
    "valueerror",
    "streamlitapi",
    "uncaught exception",
    "failed to load config preview",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a data scientist UI journey with Playwright.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("UI_BASE_URL", "http://127.0.0.1:8501"),
        help="Streamlit base URL.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/playwright-data-scientist",
        help="Directory for screenshots and journey_report.json.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=120_000,
        help="Playwright timeout in milliseconds.",
    )
    parser.add_argument("--headed", action="store_true", help="Run headed for debugging.")
    return parser.parse_args()


def body_text(page: Page) -> str:
    try:
        return page.locator("body").inner_text(timeout=10_000)
    except PlaywrightError:
        return ""


def wait_for_text(page: Page, needles: list[str], timeout_sec: int = 90) -> bool:
    deadline = time.monotonic() + timeout_sec
    lowered = [needle.lower() for needle in needles]
    while time.monotonic() < deadline:
        text = body_text(page).lower()
        if all(needle in text for needle in lowered):
            return True
        page.wait_for_timeout(1_000)
    return False


def visible_errors(page: Page) -> list[str]:
    text = body_text(page).lower()
    markers = [marker for marker in ERROR_MARKERS if marker in text]
    if "loading experiments" in text:
        markers.append("loading experiments still visible")
    if "loading recent jobs" in text:
        markers.append("loading recent jobs still visible")
    return markers


def wait_for_picker_settled(page: Page, timeout_sec: int = 25) -> bool:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        text = body_text(page).lower()
        if "loading experiments" not in text and "loading recent jobs" not in text:
            return True
        page.wait_for_timeout(1_000)
    return False


def screenshot(page: Page, output_dir: Path, filename: str) -> str:
    path = output_dir / filename
    page.screenshot(path=str(path), full_page=True)
    return str(path)


def safe_click_text(page: Page, label: str) -> bool:
    try:
        page.get_by_text(label, exact=False).first.click(timeout=7_000)
        return True
    except PlaywrightError:
        return False


def safe_fill_label(page: Page, label: str, value: str) -> bool:
    try:
        field = page.get_by_label(label, exact=False).first
        field.fill(value, timeout=7_000)
        field.press("Enter", timeout=7_000)
        return True
    except PlaywrightError:
        return False


def goto(page: Page, base_url: str, path: str) -> int | None:
    response = page.goto(f"{base_url.rstrip('/')}{path}", wait_until="domcontentloaded")
    if isinstance(response, Response):
        return response.status
    return None


def record_step(
    page: Page,
    output_dir: Path,
    *,
    name: str,
    http_status: int | None,
    expected: list[str],
    screenshot_name: str,
    actions: list[str] | None = None,
) -> dict[str, Any]:
    ready = wait_for_text(page, expected)
    errors = visible_errors(page)
    return {
        "step": name,
        "http_status": http_status,
        "status": "passed" if ready and not errors else "failed",
        "expected_text_found": ready,
        "visible_error_markers": errors,
        "actions": actions or [],
        "screenshot": screenshot(page, output_dir, screenshot_name),
    }


def main() -> int:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_errors: list[str] = []
    console_errors: list[dict[str, str]] = []
    failed_responses: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "console",
            lambda message: console_errors.append({"type": message.type, "text": message.text})
            if message.type == "error"
            else None,
        )
        page.on(
            "response",
            lambda response: failed_responses.append(
                {"url": response.url, "status": response.status}
            )
            if response.status >= 400 and "_stcore" not in response.url
            else None,
        )

        status = goto(page, base_url, "/")
        steps.append(
            record_step(
                page,
                output_dir,
                name="Open dashboard and orient to active jobs",
                http_status=status,
                expected=["MLOps Intelligence Hub", "Live pipeline activity"],
                screenshot_name="01-home-dashboard.png",
            )
        )

        status = goto(page, base_url, "/Configs")
        actions = []
        if wait_for_text(page, ["Pipeline Configs"]):
            actions.append(f"open_config_help={safe_click_text(page, 'What is a pipeline config?')}")
            actions.append(f"search_telecom={safe_fill_label(page, 'Search by name', 'telecom')}")
        steps.append(
            record_step(
                page,
                output_dir,
                name="Inspect pipeline configs and search for telecom config",
                http_status=status,
                expected=["Pipeline Configs", "pipeline config"],
                screenshot_name="02-configs-search.png",
                actions=actions,
            )
        )

        status = goto(page, base_url, "/Submit_Pipeline")
        actions = []
        if wait_for_text(page, ["Submit Pipeline"]):
            actions.append(f"open_yaml_preview={safe_click_text(page, 'Preview YAML before submit')}")
            actions.append(f"open_advanced_options={safe_click_text(page, 'Advanced Options')}")
            actions.append(
                f"fill_compute_override={safe_fill_label(page, 'Compute target (override)', 'mlopsv2computecluster')}"
            )
            actions.append(
                f"fill_tag_key={safe_fill_label(page, 'Custom tag key', 'persona')}"
            )
            actions.append(
                f"fill_tag_value={safe_fill_label(page, 'Custom tag value', 'data-scientist-e2e')}"
            )
            actions.append("submit_clicked=False; avoided launching Azure ML compute during UI journey")
        steps.append(
            record_step(
                page,
                output_dir,
                name="Prepare a new pipeline submission without launching compute",
                http_status=status,
                expected=["Submit Pipeline", "Launch a new Azure ML V3 pipeline job"],
                screenshot_name="03-submit-preflight.png",
                actions=actions,
            )
        )

        status = goto(page, base_url, "/Focus")
        wait_for_picker_settled(page)
        steps.append(
            record_step(
                page,
                output_dir,
                name="Open Focus cockpit for job investigation",
                http_status=status,
                expected=["Focus", "One job, one cockpit"],
                screenshot_name="04-focus-empty-or-picker.png",
            )
        )

        status = goto(page, base_url, "/Drift_Monitor")
        wait_for_picker_settled(page)
        steps.append(
            record_step(
                page,
                output_dir,
                name="Open Drift Monitor for PSI review",
                http_status=status,
                expected=["Drift Monitor", "Population Stability Index"],
                screenshot_name="05-drift-monitor.png",
            )
        )

        status = goto(page, base_url, "/Live_Logs")
        wait_for_picker_settled(page)
        steps.append(
            record_step(
                page,
                output_dir,
                name="Open Live Logs for step-level diagnostics",
                http_status=status,
                expected=["Live Logs", "Log"],
                screenshot_name="06-live-logs.png",
            )
        )

        browser.close()

    failed_steps = [step for step in steps if step["status"] != "passed"]
    report = {
        "persona": "data_scientist",
        "base_url": base_url,
        "overall_status": "passed" if not failed_steps and not page_errors and not failed_responses else "failed",
        "steps": steps,
        "page_errors": page_errors,
        "console_errors": console_errors,
        "failed_responses": failed_responses,
        "note": "The journey does not click Submit, so no Azure ML job or compute cost is created.",
    }

    report_path = output_dir / "journey_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if report["overall_status"] != "passed" else 0


if __name__ == "__main__":
    sys.exit(main())