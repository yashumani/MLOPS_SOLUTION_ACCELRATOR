export function humanizeKey(key: string): string {
  return key
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return "-";
  if (Math.abs(value) >= 1000) return value.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(value) >= 10) return value.toLocaleString(undefined, { maximumFractionDigits: 2 });
  return value.toLocaleString(undefined, { maximumFractionDigits: 4 });
}

export function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "Yes" : "No";
  if (typeof value === "string") return value;
  if (Array.isArray(value)) return `${value.length} items`;
  if (typeof value === "object") return "Available";
  return String(value);
}

export function statusTone(status?: string | null): "good" | "warn" | "bad" | "info" | "neutral" {
  if (!status) return "neutral";
  const normalized = status.toLowerCase();
  if (normalized.includes("complete") || normalized === "succeeded") return "good";
  if (normalized.includes("fail") || normalized.includes("cancel") || normalized.includes("error")) return "bad";
  if (normalized.includes("running") || normalized.includes("prepar") || normalized.includes("queued")) return "info";
  if (normalized.includes("final")) return "warn";
  return "neutral";
}

export function metricLabel(metric: string): string {
  const labels: Record<string, string> = {
    accuracy: "Accuracy",
    balanced_accuracy: "Balanced Accuracy",
    f1: "F1 Score",
    precision: "Precision",
    recall: "Recall",
    auc: "AUC",
    rmse: "RMSE",
    mae: "MAE",
    r2: "R Squared",
    silhouette: "Silhouette Score",
    calinski_harabasz: "Calinski-Harabasz",
    davies_bouldin: "Davies-Bouldin"
  };
  return labels[metric.toLowerCase()] ?? humanizeKey(metric);
}

export function metricHint(metric: string): string {
  const normalized = metric.toLowerCase();
  if (["rmse", "mae", "mse", "davies_bouldin"].includes(normalized)) return "Lower is better";
  if (["accuracy", "balanced_accuracy", "f1", "precision", "recall", "auc", "r2", "silhouette"].includes(normalized)) {
    return "Higher is better";
  }
  return "Review in context";
}