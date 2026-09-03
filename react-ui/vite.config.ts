import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import { fileURLToPath } from "node:url";

export default defineConfig({
  plugins: [react()],
  build: {
    rollupOptions: {
      input: { main: fileURLToPath(new URL("./index.html", import.meta.url)), redirect: fileURLToPath(new URL("./redirect.html", import.meta.url)) }
    }
  },
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
