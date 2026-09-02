import { useEffect } from "react";
import { useConfigsQuery } from "../hooks/usePipelineQueries";
import { StateBlock } from "./StateBlock";

interface ConfigSelectProps {
  value?: string;
  onChange: (configName: string) => void;
  label?: string;
}

export function ConfigSelect({ value, onChange, label = "Config" }: ConfigSelectProps) {
  const configs = useConfigsQuery();
  const firstConfig = configs.data?.configs[0]?.config_name;

  useEffect(() => {
    if (!value && firstConfig) onChange(firstConfig);
  }, [firstConfig, onChange, value]);

  if (configs.isLoading) return <StateBlock title="Loading configs" kind="loading" />;
  if (configs.isError) return <StateBlock title="Could not load configs" kind="error" message="Check API connectivity and retry." />;

  return (
    <label className="field">
      <span>{label}</span>
      <select value={value ?? ""} onChange={(event) => onChange(event.target.value)}>
        {(configs.data?.configs ?? []).map((config) => (
          <option value={config.config_name} key={config.config_name}>
            {config.config_name} - {config.task_type ?? "task"}
          </option>
        ))}
      </select>
    </label>
  );
}