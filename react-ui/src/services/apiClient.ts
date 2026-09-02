import type {
  AutoRetrainBaselineApprovalRequest,
  AutoRetrainBaselineApprovalResponse,
  AutoRetrainControllerPlanRequest,
  AutoRetrainControllerPlanResponse,
  AutoRetrainDecisionListResponse,
  AutoRetrainScheduleResponse,
  BaselineCaptureResponse,
  ConfigDetail,
  ConfigListResponse,
  ConfigPreviewResponse,
  ConfigValidationResponse,
  DriftResponse,
  ExperimentTreeResponse,
  HealthResponse,
  JobListResponse,
  JobStatus,
  LocalOutputsResponse,
  MetricsResponse,
  NotificationEmailResponse,
  OutputContentResponse,
  OutputListResponse,
  PipelineSummaryResponse,
  SubmitRequest,
  SubmitResponse,
  SubmitStatusResponse
} from "../types/api";
import { getRuntimeConfig } from "./runtimeConfig";

export type ApiErrorKind =
  | "unauthorized"
  | "not_found"
  | "metrics_unavailable"
  | "artifact_unavailable"
  | "server_error"
  | "network"
  | "unknown";

export class ApiError extends Error {
  readonly status: number;
  readonly kind: ApiErrorKind;
  readonly detail: string;

  constructor(message: string, status: number, kind: ApiErrorKind, detail = message) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.kind = kind;
    this.detail = detail;
  }
}

function apiUrl(path: string): string {
  const base = getRuntimeConfig().apiBaseUrl.replace(/\/$/, "");
  return `${base}${path}`;
}

function classifyError(path: string, status: number, detail: string): ApiErrorKind {
  const normalizedDetail = detail.toLowerCase();
  if (status === 401 || status === 403) return "unauthorized";
  if (status === 404) {
    if (path.includes("/metrics")) return "metrics_unavailable";
    if (path.includes("/outputs")) return "artifact_unavailable";
    return "not_found";
  }
  if (status >= 500 && path.includes("/metrics")) return "metrics_unavailable";
  if (status >= 500 && normalizedDetail.includes("output")) return "artifact_unavailable";
  if (status >= 500) return "server_error";
  return "unknown";
}

export function toUserMessage(error: unknown): string {
  if (error instanceof ApiError) {
    if (error.kind === "metrics_unavailable") {
      return "Metrics are not available from the API yet. You can retry or review summary and output artifacts instead.";
    }
    if (error.kind === "artifact_unavailable") {
      return "This artifact is not available for preview yet. It may still be uploading, missing, or only available for download.";
    }
    if (error.kind === "unauthorized") {
      return "The API key is missing or invalid. Update the key in Settings and try again.";
    }
    if (error.kind === "not_found") {
      return "The requested job or resource was not found in this Azure ML workspace.";
    }
    return error.detail || error.message;
  }
  if (error instanceof Error) return error.message;
  return "An unexpected error occurred.";
}

async function readJsonResponse<T>(response: Response, path: string): Promise<T> {
  const contentType = response.headers.get("content-type") ?? "";
  const payload = contentType.includes("application/json")
    ? await response.json().catch(() => null)
    : await response.text().catch(() => "");

  if (!response.ok) {
    const detail =
      payload && typeof payload === "object" && "detail" in payload
        ? String((payload as { detail: unknown }).detail)
        : typeof payload === "string"
          ? payload
          : response.statusText;
    throw new ApiError(detail, response.status, classifyError(path, response.status, detail), detail);
  }

  return payload as T;
}

export class MLOpsApiClient {
  private readonly apiKey: string;

  constructor(apiKey: string) {
    this.apiKey = apiKey;
  }

  private async get<T>(path: string): Promise<T> {
    try {
      const response = await fetch(apiUrl(path), {
        headers: {
          "X-API-Key": this.apiKey,
          Accept: "application/json"
        }
      });
      return readJsonResponse<T>(response, path);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError("Unable to reach the API service.", 0, "network");
    }
  }

  private async send<T>(method: "POST" | "PUT" | "DELETE", path: string, body?: unknown): Promise<T> {
    try {
      const response = await fetch(apiUrl(path), {
        method,
        headers: {
          "X-API-Key": this.apiKey,
          Accept: "application/json",
          "Content-Type": "application/json"
        },
        body: body === undefined ? undefined : JSON.stringify(body)
      });
      return readJsonResponse<T>(response, path);
    } catch (error) {
      if (error instanceof ApiError) throw error;
      throw new ApiError("Unable to reach the API service.", 0, "network");
    }
  }

  health(): Promise<HealthResponse> {
    return this.get<HealthResponse>("/api/v1/health");
  }

  configs(): Promise<ConfigListResponse> {
    return this.get<ConfigListResponse>("/api/v1/configs");
  }

  config(configName: string): Promise<ConfigDetail> {
    return this.get<ConfigDetail>(`/api/v1/configs/${encodeURIComponent(configName)}`);
  }

  validateConfig(content: Record<string, unknown>): Promise<ConfigValidationResponse> {
    return this.send<ConfigValidationResponse>("POST", "/api/v1/configs/validate", { content });
  }

  previewConfig(content: Record<string, unknown>, configName?: string): Promise<ConfigPreviewResponse> {
    return this.send<ConfigPreviewResponse>("POST", "/api/v1/configs/preview", { content, config_name: configName });
  }

  createConfig(configName: string, content: Record<string, unknown>): Promise<ConfigDetail> {
    return this.send<ConfigDetail>("POST", `/api/v1/configs/${encodeURIComponent(configName)}`, { content });
  }

  updateConfig(configName: string, content: Record<string, unknown>): Promise<ConfigDetail> {
    return this.send<ConfigDetail>("PUT", `/api/v1/configs/${encodeURIComponent(configName)}`, { content });
  }

  submitPipeline(request: SubmitRequest): Promise<SubmitResponse> {
    return this.send<SubmitResponse>("POST", "/api/v1/pipelines/submit", request);
  }

  submitPipelineAsync(request: SubmitRequest): Promise<SubmitStatusResponse> {
    return this.send<SubmitStatusResponse>("POST", "/api/v1/pipelines/submit/async", request);
  }

  submitStatus(requestId: string): Promise<SubmitStatusResponse> {
    return this.get<SubmitStatusResponse>(`/api/v1/pipelines/submit/status/${encodeURIComponent(requestId)}`);
  }

  experiments(maxResultsPerExperiment = 50): Promise<ExperimentTreeResponse> {
    return this.get<ExperimentTreeResponse>(
      `/api/v1/pipelines/experiments?max_results_per_experiment=${maxResultsPerExperiment}`
    );
  }

  jobs(maxResults = 50): Promise<JobListResponse> {
    return this.get<JobListResponse>(`/api/v1/pipelines/jobs?max_results=${maxResults}`);
  }

  localOutputs(maxDepth = 4, maxFiles = 500): Promise<LocalOutputsResponse> {
    return this.get<LocalOutputsResponse>(`/api/v1/pipelines/local-outputs?max_depth=${maxDepth}&max_files=${maxFiles}`);
  }

  autoRetrainSchedules(limitRecords = 10): Promise<AutoRetrainScheduleResponse> {
    return this.get<AutoRetrainScheduleResponse>(`/api/v1/pipelines/auto-retrain/schedules?limit_records=${limitRecords}`);
  }

  autoRetrainDecisions(limit = 100): Promise<AutoRetrainDecisionListResponse> {
    return this.get<AutoRetrainDecisionListResponse>(`/api/v1/pipelines/auto-retrain/decisions?limit=${limit}`);
  }

  autoRetrainPlan(request: AutoRetrainControllerPlanRequest): Promise<AutoRetrainControllerPlanResponse> {
    return this.send<AutoRetrainControllerPlanResponse>("POST", "/api/v1/pipelines/auto-retrain/controller/plan", request);
  }

  approveAutoRetrainBaseline(request: AutoRetrainBaselineApprovalRequest): Promise<AutoRetrainBaselineApprovalResponse> {
    return this.send<AutoRetrainBaselineApprovalResponse>("POST", "/api/v1/pipelines/auto-retrain/baselines/approve", request);
  }

  captureBaseline(jobName: string): Promise<BaselineCaptureResponse> {
    return this.send<BaselineCaptureResponse>("POST", "/api/v1/pipelines/baseline/capture", { job_name: jobName });
  }

  job(jobName: string): Promise<JobStatus> {
    return this.get<JobStatus>(`/api/v1/pipelines/jobs/${encodeURIComponent(jobName)}`);
  }

  metrics(jobName: string): Promise<MetricsResponse> {
    return this.get<MetricsResponse>(`/api/v1/pipelines/jobs/${encodeURIComponent(jobName)}/metrics`);
  }

  summary(jobName: string): Promise<PipelineSummaryResponse> {
    return this.get<PipelineSummaryResponse>(`/api/v1/pipelines/jobs/${encodeURIComponent(jobName)}/summary`);
  }

  outputs(jobName: string): Promise<OutputListResponse> {
    return this.get<OutputListResponse>(`/api/v1/pipelines/jobs/${encodeURIComponent(jobName)}/outputs`);
  }

  outputContent(jobName: string, outputName: string): Promise<OutputContentResponse> {
    return this.get<OutputContentResponse>(
      `/api/v1/pipelines/jobs/${encodeURIComponent(jobName)}/outputs/${encodeURIComponent(outputName)}/content`
    );
  }

  drift(jobName: string): Promise<DriftResponse> {
    return this.get<DriftResponse>(`/api/v1/pipelines/jobs/${encodeURIComponent(jobName)}/drift`);
  }

  sendNotification(jobName: string, dryRun: boolean): Promise<NotificationEmailResponse> {
    return this.send<NotificationEmailResponse>(
      "POST",
      `/api/v1/pipelines/jobs/${encodeURIComponent(jobName)}/notifications/email`,
      { dry_run: dryRun }
    );
  }
}