import { useQuery } from "@tanstack/react-query";
import { useApi } from "../services/ApiContext";

const terminalStatuses = new Set(["Completed", "Failed", "Canceled", "NotResponding"]);

export function isTerminalStatus(status?: string | null): boolean {
  return Boolean(status && terminalStatuses.has(status));
}

export function useHealthQuery() {
  const { api } = useApi();
  return useQuery({
    queryKey: ["health"],
    queryFn: () => api.health(),
    staleTime: 30_000
  });
}

export function useExperimentsQuery() {
  const { api } = useApi();
  return useQuery({
    queryKey: ["experiments"],
    queryFn: () => api.experiments(75),
    staleTime: 20_000
  });
}

export function useConfigsQuery() {
  const { api } = useApi();
  return useQuery({
    queryKey: ["configs"],
    queryFn: () => api.configs(),
    staleTime: 120_000
  });
}

export function useConfigQuery(configName?: string) {
  const { api } = useApi();
  return useQuery({
    queryKey: ["config", configName],
    queryFn: () => api.config(configName ?? ""),
    enabled: Boolean(configName),
    staleTime: 120_000
  });
}

export function useJobsQuery() {
  const { api } = useApi();
  return useQuery({
    queryKey: ["jobs"],
    queryFn: () => api.jobs(75),
    staleTime: 20_000
  });
}

export function useLocalOutputsQuery() {
  const { api } = useApi();
  return useQuery({
    queryKey: ["local-outputs"],
    queryFn: () => api.localOutputs(5, 750),
    staleTime: 60_000
  });
}

export function useAutoRetrainSchedulesQuery() {
  const { api } = useApi();
  return useQuery({
    queryKey: ["auto-retrain-schedules"],
    queryFn: () => api.autoRetrainSchedules(25),
    staleTime: 60_000
  });
}

export function useAutoRetrainDecisionsQuery() {
  const { api } = useApi();
  return useQuery({
    queryKey: ["auto-retrain-decisions"],
    queryFn: () => api.autoRetrainDecisions(100),
    staleTime: 60_000
  });
}

export function useJobQuery(jobName?: string) {
  const { api } = useApi();
  return useQuery({
    queryKey: ["job", jobName],
    queryFn: () => api.job(jobName ?? ""),
    enabled: Boolean(jobName),
    refetchInterval: (query) => (isTerminalStatus(query.state.data?.status) ? false : 15_000),
    staleTime: (query) => (isTerminalStatus(query.state.data?.status) ? 600_000 : 10_000)
  });
}

export function useMetricsQuery(jobName?: string) {
  const { api } = useApi();
  return useQuery({
    queryKey: ["metrics", jobName],
    queryFn: () => api.metrics(jobName ?? ""),
    enabled: Boolean(jobName),
    retry: 0,
    staleTime: 300_000
  });
}

export function useSummaryQuery(jobName?: string) {
  const { api } = useApi();
  return useQuery({
    queryKey: ["summary", jobName],
    queryFn: () => api.summary(jobName ?? ""),
    enabled: Boolean(jobName),
    retry: 0,
    staleTime: 300_000
  });
}

export function useOutputsQuery(jobName?: string) {
  const { api } = useApi();
  return useQuery({
    queryKey: ["outputs", jobName],
    queryFn: () => api.outputs(jobName ?? ""),
    enabled: Boolean(jobName),
    retry: 0,
    staleTime: 300_000
  });
}

export function useOutputContentQuery(jobName?: string, outputName?: string) {
  const { api } = useApi();
  return useQuery({
    queryKey: ["output-content", jobName, outputName],
    queryFn: () => api.outputContent(jobName ?? "", outputName ?? ""),
    enabled: Boolean(jobName && outputName),
    retry: 0,
    staleTime: 300_000
  });
}

export function useDriftQuery(jobName?: string) {
  const { api } = useApi();
  return useQuery({
    queryKey: ["drift", jobName],
    queryFn: () => api.drift(jobName ?? ""),
    enabled: Boolean(jobName),
    retry: 0,
    staleTime: 300_000
  });
}