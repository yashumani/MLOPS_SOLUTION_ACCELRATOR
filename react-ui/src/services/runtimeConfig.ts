export interface RuntimeConfig {
  apiBaseUrl: string;
  uiBaseUrl: string;
  environment: string;
}

declare global {
  interface Window {
    __MLOPS_UI_CONFIG__?: Partial<RuntimeConfig>;
  }
}

export function getRuntimeConfig(): RuntimeConfig {
  const runtime = window.__MLOPS_UI_CONFIG__ ?? {};
  return {
    apiBaseUrl:
      runtime.apiBaseUrl ?? import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000",
    uiBaseUrl: runtime.uiBaseUrl ?? import.meta.env.VITE_UI_BASE_URL ?? window.location.origin,
    environment: runtime.environment ?? import.meta.env.VITE_ENV ?? "dev"
  };
}
