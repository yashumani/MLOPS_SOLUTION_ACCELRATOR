import { formatValue, humanizeKey } from "./formatters";

export interface SummaryField {
  label: string;
  value: string;
}

export interface ArtifactSummary {
  title: string;
  description: string;
  fields: SummaryField[];
  tableRows: Array<Record<string, unknown>>;
}

const priorityKeys = [
  "status",
  "task_type",
  "dataset_name",
  "champion_phase",
  "champion_score",
  "model_name",
  "algorithm",
  "engine",
  "primary_metric",
  "primary_score",
  "stability_score",
  "overall_drift_detected",
  "review_status",
  "registration_eligible",
  "decision",
  "recommendation"
];

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function firstRecordArray(value: unknown): Array<Record<string, unknown>> {
  if (Array.isArray(value) && value.every(isRecord)) return value.slice(0, 25);
  if (!isRecord(value)) return [];
  for (const nestedValue of Object.values(value)) {
    if (Array.isArray(nestedValue) && nestedValue.every(isRecord)) return nestedValue.slice(0, 25);
  }
  return [];
}

export function summarizeJsonArtifact(outputName: string, content: unknown): ArtifactSummary {
  const title = humanizeKey(outputName);
  if (Array.isArray(content)) {
    return {
      title,
      description: `${content.length} records found. Showing the first records in a table-friendly view.`,
      fields: [{ label: "Records", value: String(content.length) }],
      tableRows: firstRecordArray(content)
    };
  }

  if (!isRecord(content)) {
    return {
      title,
      description: "This artifact contains a single value.",
      fields: [{ label: "Value", value: formatValue(content) }],
      tableRows: []
    };
  }

  const fields: SummaryField[] = [];
  for (const key of priorityKeys) {
    if (key in content) {
      fields.push({ label: humanizeKey(key), value: formatValue(content[key]) });
    }
  }

  for (const [key, value] of Object.entries(content)) {
    if (fields.length >= 10) break;
    if (priorityKeys.includes(key)) continue;
    if (typeof value !== "object" || value === null) {
      fields.push({ label: humanizeKey(key), value: formatValue(value) });
    }
  }

  return {
    title,
    description: "Key details are summarized below. Technical raw JSON is available under Advanced.",
    fields,
    tableRows: firstRecordArray(content)
  };
}