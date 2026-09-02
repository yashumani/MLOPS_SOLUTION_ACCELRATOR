import { AlertTriangle, Loader2, RefreshCcw } from "lucide-react";

interface StateBlockProps {
  title: string;
  message?: string;
  kind?: "loading" | "empty" | "error";
  actionLabel?: string;
  onAction?: () => void;
}

export function StateBlock({ title, message, kind = "empty", actionLabel, onAction }: StateBlockProps) {
  const Icon = kind === "loading" ? Loader2 : AlertTriangle;
  return (
    <div className={`state-block state-${kind}`}>
      <Icon className={kind === "loading" ? "spin" : ""} size={20} />
      <div>
        <strong>{title}</strong>
        {message ? <p>{message}</p> : null}
      </div>
      {actionLabel && onAction ? (
        <button className="button secondary" onClick={onAction} type="button">
          <RefreshCcw size={16} />
          {actionLabel}
        </button>
      ) : null}
    </div>
  );
}