interface JsonDetailsProps {
  title?: string;
  value: unknown;
}

export function JsonDetails({ title = "Raw details", value }: JsonDetailsProps) {
  return (
    <details className="advanced-json">
      <summary>{title}</summary>
      <pre>{JSON.stringify(value, null, 2)}</pre>
    </details>
  );
}