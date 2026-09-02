import { createContext, useContext } from "react";
import { MLOpsApiClient } from "./apiClient";

export interface ApiContextValue {
  api: MLOpsApiClient;
  apiKey: string;
  setApiKey: (apiKey: string) => void;
}

export const ApiContext = createContext<ApiContextValue | null>(null);

export function useApi(): ApiContextValue {
  const value = useContext(ApiContext);
  if (!value) {
    throw new Error("useApi must be used inside ApiContext.Provider");
  }
  return value;
}