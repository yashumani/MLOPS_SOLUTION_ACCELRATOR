import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 8501,
    allowedHosts: ["mlopspipelinev2-8501.eastus2.instances.azureml.ms"],
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: false
      }
    }
  },
  preview: {
    host: "0.0.0.0",
    port: 8501
  },
  test: {
    globals: true,
    environment: "node"
  }
});