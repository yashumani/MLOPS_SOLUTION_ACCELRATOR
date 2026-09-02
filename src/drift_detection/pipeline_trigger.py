"""Evaluate drift results and hand drift evidence to the S14 decision gate.

The legacy class name is retained for compatibility. It no longer constructs or
submits child training jobs. S14 owns retrain policy; an external controller may
submit only after consuming S14's explicit ``RetrainDecision`` artifact.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from .drift_config import DriftConfig

if TYPE_CHECKING:
    from .drift_checker import DriftResult

logger = logging.getLogger(__name__)

class PipelineTrigger:
    """Summarize drift evidence without deciding or submitting retraining.

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

    def evaluate(self, results: List["DriftResult"]) -> Dict:
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
            "execution": {
                "status": "not_requested",
                "reason": "no_drift",
            },
        }

        self._trigger_log.append(summary)

        if should_trigger:
            if self.dry_run:
                summary["execution"] = {
                    "status": "dry_run",
                    "reason": "PipelineTrigger was constructed with dry_run=True",
                }
                logger.info(
                    "[DRY-RUN] Would trigger %r due to drift in: %s",
                    action,
                    triggered_by,
                )
            else:
                summary["execution"] = self._execute_trigger(summary)
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

    def _execute_trigger(self, summary: Dict) -> Dict:
        """Delegate drift evidence to S14 without constructing a child job."""
        auto_retrain = self.config.auto_retrain
        if not auto_retrain.enabled:
            logger.warning(
                "PIPELINE TRIGGER DISABLED: action=%r, triggered_by=%s",
                summary["action"],
                summary["triggered_by"],
            )
            return {
                "status": "disabled",
                "reason": "auto_retrain.enabled is false",
            }

        if summary["action"] != "trigger_full_pipeline":
            return {
                "status": "skipped",
                "reason": f"unsupported action: {summary['action']}",
            }

        logger.info(
            "Drift evidence delegated to S14; no pipeline was submitted by PipelineTrigger."
        )
        return {
            "status": "delegated",
            "reason": (
                "S14 must evaluate drift evidence and emit an explicit "
                "RetrainDecision before the external controller may submit"
            ),
            "submitted": False,
            "next_stage": "s14_retrain_decision",
            "submission_owner": "external_controller",
            "required_artifact": "retrain_decision.json",
        }

    def _log_to_mlflow(self, summary: Dict) -> None:
        """Log trigger event to MLflow (best-effort, non-fatal)."""
        try:
            import mlflow

            mlflow.log_param("drift_trigger_action", summary["action"])
            mlflow.log_param("drift_triggered_by", ",".join(summary["triggered_by"]))
            execution = summary.get("execution", {})
            if execution:
                mlflow.log_param("drift_trigger_execution_status", execution.get("status", "unknown"))
            for dtype, info in summary["details"].items():
                mlflow.log_metric(f"drift_score_{dtype}", info["score"])
            logger.info("Trigger event logged to MLflow.")
        except Exception as exc:  # noqa: BLE001
            logger.warning("MLflow logging failed (non-fatal): %s", exc)
