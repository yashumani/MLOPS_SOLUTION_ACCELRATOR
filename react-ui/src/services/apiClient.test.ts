import { afterEach, expect, test, vi } from "vitest";
import { MLOpsApiClient } from "./apiClient";

afterEach(() => vi.unstubAllGlobals());

test("Entra requests use fresh bearer tokens and never the shared key", async () => {
  vi.stubGlobal("window", { __MLOPS_UI_CONFIG__: { apiBaseUrl: "https://api.example.test" }, location: { origin: "https://ui.example.test" } });
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ roles: ["admin"] }), { headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fetch);
  const token = vi.fn().mockResolvedValue("access-token");
  await new MLOpsApiClient(token).identity();
  expect(token).toHaveBeenCalledOnce();
  expect(fetch.mock.calls[0][1].headers).toEqual({ Authorization: "Bearer access-token", Accept: "application/json" });
});

test("user mutations carry a revision and preserve authorization errors", async () => {
  vi.stubGlobal("window", { __MLOPS_UI_CONFIG__: { apiBaseUrl: "https://api.example.test" }, location: { origin: "https://ui.example.test" } });
  const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ detail: "User list changed" }), { status: 409, headers: { "Content-Type": "application/json" } }));
  vi.stubGlobal("fetch", fetch);
  const client = new MLOpsApiClient(async () => "access-token");
  await expect(client.updateUser({ object_id: "user-id", display_name: "Name", role: "viewer", enabled: false }, 3)).rejects.toMatchObject({ status: 409 });
  expect(JSON.parse(fetch.mock.calls[0][1].body)).toEqual({ display_name: "Name", role: "viewer", enabled: false, expected_revision: 3 });
});
