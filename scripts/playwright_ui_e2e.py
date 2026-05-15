#!/usr/bin/env python3
"""Playwright smoke test for the Streamlit UI.

This script intentionally exercises the UI through the browser rather than
through API-only checks. It writes screenshots plus a JSON report that can be
attached to release evidence.
"""

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


ROUTES: list[dict[str, Any]] = [
    {
        "name": "Home",
        "path": "/",
        "screenshot": "home.png",
        "needles": ["MLOps Intelligence Hub", "Live pipeline activity"],
    },
    {
        "name": "Submit Pipeline",
        "path": "/Submit_Pipeline",
        "screenshot": "submit-pipeline.png",
        "needles": ["Submit Pipeline", "Advanced Options"],
        "click_text": ["Preview YAML before submit", "Advanced Options"],
    },
    {
        "name": "Focus",
        "path": "/Focus",
        "screenshot": "focus.png",
        "needles": ["Focus", "One job, one cockpit"],
    },
    {
        "name": "Configs",
        "path": "/Configs",
        "screenshot": "configs.png",
        "needles": ["Pipeline Configs", "configuration"],
        "click_text": ["What is a pipeline config?"],
        "fill": {"label": "Search by name", "value": "config"},
    },
    {
        "name": "Drift Monitor",
        "path": "/Drift_Monitor",
        "screenshot": "drift-monitor.png",
        "needles": ["Drift Monitor", "Population Stability Index"],
    },
    {
        "name": "Live Logs",
        "path": "/Live_Logs",
        "screenshot": "live-logs.png",
        "needles": ["Live Logs", "Log"],
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Streamlit UI Playwright smoke tests.")
    parser.add_argument(
        "--base-url",
        default=os.getenv("UI_BASE_URL", "http://127.0.0.1:8501"),
        help="Local or public Streamlit base URL. Defaults to UI_BASE_URL or localhost.",
    )
    parser.add_argument(
        "--public-url",
        default=os.getenv("PUBLIC_UI_URL", ""),
        help="Optional public Azure ML app-proxy URL to verify auth/redirect behavior.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/playwright-e2e",
        help="Directory for screenshots and report.json.",
    )
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=120_000,
        help="Per-page timeout in milliseconds.",
    )
    parser.add_argument(
        "--headed",
        action="store_true",
        help="Run Chromium headed for interactive debugging.",
    )
    return parser.parse_args()


def text_present(page: Page, needles: list[str], timeout_sec: int = 90) -> bool:
    deadline = time.monotonic() + timeout_sec
    lowered_needles = [needle.lower() for needle in needles]
    while time.monotonic() < deadline:
        body_text = page.locator("body").inner_text(timeout=10_000).lower()
        if all(needle in body_text for needle in lowered_needles):
            return True
        page.wait_for_timeout(1_000)
    return False


def safe_click_text(page: Page, label: str) -> bool:
    try:
        locator = page.get_by_text(label, exact=False).first
        locator.click(timeout=5_000)
        return True
    except PlaywrightError:
        return False


def safe_fill(page: Page, label: str, value: str) -> bool:
    try:
        page.get_by_label(label, exact=False).first.fill(value, timeout=5_000)
        return True
    except PlaywrightError:
        try:
            page.get_by_placeholder(label, exact=False).first.fill(value, timeout=5_000)
            return True
        except PlaywrightError:
            return False


def visit_route(page: Page, base_url: str, route: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    page_url = f"{base_url.rstrip('/')}{route['path']}"
    started = time.monotonic()
    result: dict[str, Any] = {
        "page": route["name"],
        "url": page_url,
        "http_status": None,
        "status": "failed",
        "duration_sec": None,
        "notes": [],
        "screenshot": str(output_dir / route["screenshot"]),
    }

    try:
        response = page.goto(page_url, wait_until="domcontentloaded")
        if isinstance(response, Response):
            result["http_status"] = response.status
        page.wait_for_selector("body", state="attached", timeout=30_000)

        content_ready = text_present(page, route["needles"])

        for label in route.get("click_text", []):
            if content_ready:
                result["notes"].append(f"click_{label}={safe_click_text(page, label)}")

        fill_spec = route.get("fill")
        if fill_spec and content_ready:
            filled = safe_fill(page, fill_spec["label"], fill_spec["value"])
            result["notes"].append(f"fill_{fill_spec['label']}={filled}")

        page.screenshot(path=result["screenshot"], full_page=True)

        if result["http_status"] not in (None, 200):
            result["notes"].append(f"unexpected_http_status={result['http_status']}")
        elif not content_ready:
            result["notes"].append("expected_text_missing")
        else:
            result["status"] = "passed"
    except Exception as exc:  # noqa: BLE001 - report browser failures, do not hide them.
        result["notes"].append(f"exception={type(exc).__name__}: {exc}")
        try:
            page.screenshot(path=result["screenshot"], full_page=True)
        except Exception:  # noqa: BLE001
            pass
    finally:
        result["duration_sec"] = round(time.monotonic() - started, 2)

    return result


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    page_errors: list[str] = []
    console_errors: list[dict[str, str]] = []
    failed_responses: list[dict[str, Any]] = []

    report: dict[str, Any] = {
        "base_url": args.base_url.rstrip("/"),
        "pages": [],
        "public_proxy": None,
        "page_errors": page_errors,
        "console_errors": console_errors,
        "failed_responses": failed_responses,
        "screenshots_dir": str(output_dir),
    }

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        context = browser.new_context(viewport={"width": 1440, "height": 1000})
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)

        page.on("pageerror", lambda exc: page_errors.append(str(exc)))
        page.on(
            "console",
            lambda message: console_errors.append({"type": message.type, "text": message.text})
            if message.type in {"error", "warning"}
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

        for route in ROUTES:
            report["pages"].append(visit_route(page, report["base_url"], route, output_dir))

        if args.public_url:
            public_started = time.monotonic()
            public_result: dict[str, Any] = {
                "url": args.public_url,
                "http_status": None,
                "final_url": None,
                "status": "failed",
                "duration_sec": None,
                "screenshot": str(output_dir / "public-proxy-auth.png"),
            }
            try:
                response = page.goto(args.public_url, wait_until="networkidle")
                if isinstance(response, Response):
                    public_result["http_status"] = response.status
                public_result["final_url"] = page.url
                page.screenshot(path=public_result["screenshot"], full_page=True)
                if public_result["http_status"] == 200 and "login.microsoftonline.com" in page.url:
                    public_result["status"] = "passed"
            except Exception as exc:  # noqa: BLE001
                public_result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                public_result["duration_sec"] = round(time.monotonic() - public_started, 2)
                report["public_proxy"] = public_result

        browser.close()

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    failed_pages = [page_result for page_result in report["pages"] if page_result["status"] != "passed"]
    has_failures = bool(failed_pages or page_errors or failed_responses)
    print(json.dumps(report, indent=2))
    return 1 if has_failures else 0


if __name__ == "__main__":
    sys.exit(main())