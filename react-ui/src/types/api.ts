export type JobStatusValue =
  | "Completed"
  | "Running"
  | "Preparing"
  | "Queued"
  | "Failed"
  | "Canceled"
  | "CancelRequested"
  | "Finalizing"
  | "NotResponding"
  | string;

export interface StepStatus {
  name: string;
  display_name?: string | null;
  stage_key?: string | null;
  is_inferred?: boolean;
  status: JobStatusValue;
  start_time?: string | null;
  end_time?: string | null;
}

export interface JobStatus {
  job_name: string;
  experiment_name?: string | null;
  display_name?: string | null;
  status: JobStatusValue;
  start_time?: string | null;
  end_time?: string | null;
  studio_url?: string | null;
  tags?: Record<string, string>;
  steps: StepStatus[];
}

export interface JobSummary {
  job_name: string;
  experiment_name?: string | null;
  display_name?: string | null;
  status: JobStatusValue;
  start_time?: string | null;
  studio_url?: string | null;
}

export interface JobListResponse {
  jobs: JobSummary[];
  total: number;
}

export interface ExperimentNode {
  experiment_name: string;
  job_count: number;
  last_activity?: string | null;
  jobs: JobSummary[];
}

export interface ExperimentTreeResponse {
  experiments: ExperimentNode[];
  total_experiments: number;
  total_jobs: number;
}

export interface ModelMetric {
  model_name: string;
  engine?: string | null;
  phase?: string | null;
  metrics: Record<string, number>;
  is_champion: boolean;
}

export interface MetricsResponse {
  job_name: string;
  task_type?: string | null;
  models: ModelMetric[];
}

export interface OutputInfo {
  name: string;
  type?: string | null;
}

export interface OutputListResponse {
  job_name: string;
  outputs: OutputInfo[];
}

export interface OutputFileInfo {
  name: string;
  relative_path: string;
  size_bytes: number;
  kind: string;
}

export interface OutputContentResponse {
  job_name: string;
  output_name: string;
  files: OutputFileInfo[];
  json_content?: unknown;
  text_preview?: string | null;
  csv_preview?: Array<Record<string, unknown>> | null;
  primary_file?: string | null;
  truncated: boolean;
}

export interface PipelineSummaryResponse {
  job_name: string;
  task_type?: string | null;
  status?: string | null;
  champion_phase?: string | null;
  champion_score?: number | null;
  baseline_aggregate?: unknown;
  phaseb_aggregate?: unknown;
  phasec_aggregate?: unknown;
  final_report?: unknown;
  available_outputs: string[];
}

export interface DriftFeature {
  feature: string;
  psi: number;
  drift_detected: boolean;
  severity: string;
}

export interface DriftResponse {
  job_name: string;
  task_type?: string | null;
  dataset_name?: string | null;
  overall_drift_detected: boolean;
  stability_score?: number | null;
  drift_type?: string | null;
  recommended_cadence?: string | null;
  recommended_days?: number | null;
  cadence_rationale?: string | null;
  comparison_available: boolean;
  baseline_status?: string | null;
  baseline_metadata?: Record<string, unknown>;
  auto_retrain_decision?: Record<string, unknown>;
  auto_retrain_trigger?: Record<string, unknown>;
  drifted_columns: string[];
  features: DriftFeature[];
  warnings: string[];
  evidently_report_path?: string | null;
  studio_url?: string | null;
}

export interface HealthResponse {
  status: string;
  azure_ml_connected?: boolean;
  workspace?: string;
  timestamp?: string;
}

export interface ConfigSummary {
  config_name: string;
  task_type?: string | null;
  dataset_name?: string | null;
  target_column?: string | null;
}

export interface ConfigListResponse {
  configs: ConfigSummary[];
  total: number;
}

export interface ConfigDetail extends ConfigSummary {
  content: Record<string, unknown>;
}

export interface ConfigValidationIssue {
  path: string;
  message: string;
  level: string;
}

export interface ConfigValidationResponse {
  valid: boolean;
  errors: ConfigValidationIssue[];
  warnings: ConfigValidationIssue[];
}

export interface ConfigStagePreview {
  stage_id: string;
  label: string;
  enabled: boolean;
  summary?: string | null;
  warnings: string[];
}

export interface ConfigPreviewResponse {
  valid: boolean;
  validation: ConfigValidationResponse;
  config_name?: string | null;
  experiment_name?: string | null;
  task_type?: string | null;
  dataset_name?: string | null;
  target_column?: string | null;
  dataset_uri_preview?: string | null;
  compute_target?: string | null;
  baseline_engines: string[];
  phase_b_engines: string[];
  phase_b_variant_budget?: number | null;
  phase_c_trials?: number | null;
  phase_c_timeout_seconds?: number | null;
  stage_plan: ConfigStagePreview[];
}

export interface SubmitRequest {
  config_name: string;
  compute?: string | null;
  force_rerun: boolean;
  baseline_job?: string | null;
  tags: Record<string, string>;
}

export interface SubmitResponse {
  job_name: string;
  experiment_name: string;
  display_name: string;
  status: string;
  studio_url: string;
}

export interface SubmitStatusResponse {
  request_id?: string;
  status?: string;
  job_name?: string;
  error?: string;
  [key: string]: unknown;
}

export interface LocalOutputFileInfo {
  relative_path: string;
  name: string;
  is_dir: boolean;
  size_bytes?: number | null;
  modified_time?: string | null;
  kind?: string | null;
  depth: number;
}

export interface LocalOutputsResponse {
  root: string;
  files: LocalOutputFileInfo[];
  total: number;
  truncated: boolean;
}

export interface AutoRetrainScheduleRow {
  task_type: string;
  dataset_name: string;
  config_name: string;
  schedule_name: string;
  cadence: string;
  cadence_days: number;
  decision_mode: string;
  promotion_mode: string;
  enabled_expected: boolean;
  live_state: "enabled" | "disabled" | "missing" | "unknown" | "unverified";
  actual_enabled: boolean | null;
  provisioning_status: string | null;
  source: "azure_ml" | "planned_only";
}

export interface AutoRetrainScheduleResponse {
  schedules: AutoRetrainScheduleRow[];
  total: number;
  ledger_path: string;
  latest_records: Array<Record<string, unknown>>;
  azure_checked_at: string | null;
  azure_error: string | null;
}

export interface AutoRetrainDecisionListResponse {
  ledger_path: string;
  records: Array<Record<string, unknown>>;
  total: number;
}

export interface AutoRetrainControllerPlanRequest {
  config_name: string;
  ledger_path?: string | null;
  decision_path: string;
  trigger: string;
  schedule_name?: string | null;
  experiment_name?: string | null;
  display_name?: string | null;
  force_submit: boolean;
  force_reason?: string | null;
}

export interface AutoRetrainControllerPlanResponse {
  config_name: string;
  task_type: string;
  dataset_name: string;
  baseline_uri: string;
  experiment_name: string;
  display_name: string;
  command: string;
  ledger_path: string;
  decision_path: string;
  pending_decision_record: Record<string, unknown>;
}

export interface AutoRetrainBaselineApprovalRequest {
  config_name: string;
  baseline_job_name?: string | null;
  output_baseline_uri?: string | null;
  ledger_path?: string | null;
  schedule_name?: string | null;
  reason: string;
}

export interface AutoRetrainBaselineApprovalResponse {
  status: string;
  ledger_path: string;
  record: Record<string, unknown>;
  baseline_uri: string;
  studio_url?: string | null;
}

export interface BaselineCaptureResponse {
  job_name: string;
  baseline_path?: string | null;
  output_present: boolean;
  status: string;
  studio_url?: string | null;
}

export interface NotificationArtifact {
  name: string;
  path: string;
  size_bytes: number;
  mime_type: string;
  included_in_email: boolean;
}

export interface NotificationEmailResponse {
  job_name: string;
  recipient: string;
  subject: string;
  status: string;
  sent: boolean;
  report_dir: string;
  artifacts: NotificationArtifact[];
  message?: string | null;
  smtp_host?: string | null;
}
