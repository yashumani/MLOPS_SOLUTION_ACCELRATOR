import { statusTone } from "../utils/formatters";

interface StatusBadgeProps {
  status?: string | null;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  const tone = statusTone(status);
  return <span className={`status-badge status-${tone}`}>{status ?? "Unknown"}</span>;
}