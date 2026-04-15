"""Evaluate drift results and trigger pipeline re-runs when needed.

The ``PipelineTrigger`` class reads ``DriftResult`` objects, decides
whether to trigger a full retraining pipeline, and logs trigger events
to MLflow.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .drift_checker import DriftResult
from .drift_config import DriftConfig

logger = logging.getLogger(__name__)


class PipelineTrigger:
    """Decide whether drift results warrant a pipeline re-run.

    Parameters
    ----------
    config : DriftConfig
        Drift-detection configuration.
    dry_run : bool
        If ``True``, log what *would* happen but do not trigger the
        pipeline or write MLflow entries.

    Example
    -------
    >>> trigger = PipelineTrigger(cfg, dry_run=True)
    >>> action = trigger.evaluate(drift_results)
    >>> print(action["should_trigger"])
    """

    def __init__(self, config: DriftConfig, dry_run: bool = False) -> None:
        self.config = config
        self.dry_run = dry_run
        self._trigger_log: List[Dict] = []

    # ── Public API ──────────────────────────────────────────────

    def evaluate(self, results: List[DriftResult]) -> Dict:
        """Evaluate drift results and return an action summary.

        Parameters
        ----------
        results : list[DriftResult]
            Results from ``DriftChecker.run_all_checks()``.

        Returns
        -------
        dict
            Keys: ``should_trigger``, ``triggered_by``, ``action``,
            ``dry_run``, ``timestamp``, ``details``.
        """
        triggered_by: List[str] = []
        details: Dict[str, Dict] = {}

        for r in results:
            if r.drift_detected:
                triggered_by.append(r.drift_type)
                details[r.drift_type] = {
                    "score": r.drift_score,
                    "threshold": self.config.get_threshold(r.drift_type),
                    "drifted_columns": r.drifted_columns,
                }

        should_trigger = len(triggered_by) > 0
        action = self.config.actions.on_drift_detected if should_trigger else "none"
        ts = datetime.now(timezone.utc).isoformat()

        summary = {
            "should_trigger": should_trigger,
            "triggered_by": triggered_by,
            "action": action,
            "dry_run": self.dry_run,
            "timestamp": ts,
            "details": details,
        }

        self._trigger_log.append(summary)

        if should_trigger:
            if self.dry_run:
                logger.info(
                    "[DRY-RUN] Would trigger %r due to drift in: %s",
                    action,
                    triggered_by,
                )
            else:
                self._execute_trigger(summary)
                self._log_to_mlflow(summary)
        else:
            logger.info("No drift — no pipeline trigger needed.")

        return summary

    @property
    def trigger_history(self) -> List[Dict]:
        """Return the list of all evaluation summaries so far."""
        return list(self._trigger_log)

    def save_trigger_log(self, output_dir: Optional[str] = None) -> Path:
        """Persist the trigger log to JSON.

        Parameters
        ----------
        output_dir : str or None
            Directory path.  Defaults to ``config.artifact_paths.logs_dir``.

        Returns
        -------
        Path
        """
        out = Path(output_dir or self.config.artifact_paths.logs_dir)
        out.mkdir(parents=True, exist_ok=True)
        ts_slug = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        path = out / f"trigger_log_{ts_slug}.json"

        with open(path, "w") as fh:
            json.dump(self._trigger_log, fh, indent=2, default=str)
        logger.info("Trigger log saved → %s", path)
        return path

    # ── Internal ────────────────────────────────────────────────

    def _execute_trigger(self, summary: Dict) -> None:
        """Trigger the pipeline re-run.

        Currently a placeholder — integration with Azure ML
        ``submit_pipeline.py`` will be added once wired end-to-end.
        """
        logger.warning(
            "PIPELINE TRIGGER: action=%r, triggered_by=%s",
            summary["action"],
            summary["triggered_by"],
        )

    def _log_to_mlflow(self, summary: Dict) -> None:
        """Log trigger event to MLflow (best-effort, non-fatal)."""
        try:
            import mlflow

            mlflow.log_param("drift_trigger_action", summary["action"])
            mlflow.log_param("drift_triggered_by", ",".join(summary["triggered_by"]))
            for dtype, info in summary["details"].items():
                mlflow.log_metric(f"drift_score_{dtype}", info["score"])
            logger.info("Trigger event logged to MLflow.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging failed (non-fatal): %s", exc)
