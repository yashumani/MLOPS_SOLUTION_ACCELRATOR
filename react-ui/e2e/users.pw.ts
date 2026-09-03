import { expect, test, type Page } from "@playwright/test";

const owner = "b03e4295-9fce-4b3b-b6ba-e7e750e639ef";
const other = "40000000-0000-4000-8000-000000000004";
const tenant = "3bc05bc3-19d1-4d30-89c5-134f4b278b11";
type User = { object_id: string; display_name: string; role: string; enabled: boolean };

async function setup(page: Page, role = "admin", conflict = false) {
  let revision = 0;
  const users: User[] = [{ object_id: owner, display_name: "yashu.savyminds@gmail.com", role, enabled: true }];
  const mutations: unknown[] = [];
  // Mock the identity provider only in the test browser, never in shipped code.
  await page.route("**/src/services/entraSession.ts*", (route) => route.fulfill({ contentType: "text/javascript", body: "export async function createEntraSession() { return { login: async () => {}, token: async () => 'browser-test-token', clear: async () => {} }; }" }));
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const json = (body: unknown, status = 200) => route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body), headers: { "Access-Control-Allow-Origin": "*" } });
    if (request.method() === "OPTIONS") return route.fulfill({ status: 204, headers: { "Access-Control-Allow-Origin": "*", "Access-Control-Allow-Headers": "Authorization,Content-Type", "Access-Control-Allow-Methods": "GET,POST,PUT" } });
    if (path === "/api/v1/auth/config") return json({ mode: "entra", tenant_id: tenant, client_id: "test-client", scope: "test-scope", redirect_uri: "http://127.0.0.1:8510/redirect.html" });
    expect(request.headers().authorization).toBe("Bearer browser-test-token");
    expect(request.headers()["x-api-key"]).toBeUndefined();
    if (path === "/api/v1/auth/me") return json({ mode: "entra", tenant_id: tenant, object_id: owner, roles: [users[0].role] });
    if (path === "/api/v1/health") return json({ status: "ok", workspace: "mlops-accelerator" });
    if (path.startsWith("/api/v1/users")) {
      if (role !== "admin") return json({ detail: "Administrator access is required" }, 403);
      if (request.method() !== "GET") {
        const body = request.postDataJSON();
        mutations.push(body);
        if (conflict || body.expected_revision !== revision) return json({ detail: "User list changed; refresh before saving" }, 409);
        if (request.method() === "POST") users.push({ object_id: body.object_id, display_name: body.display_name, role: body.role, enabled: body.enabled });
        else Object.assign(users.find((user) => path.endsWith(user.object_id))!, { display_name: body.display_name, role: body.role, enabled: body.enabled });
        revision++;
      }
      return json({ tenant_id: tenant, revision, users }, request.method() === "POST" ? 201 : 200);
    }
    return json({ jobs: [], configs: [], total: 0 });
  });
  await page.goto("/users");
  await page.getByRole("button", { name: "Sign in with Microsoft" }).click();
  return mutations;
}

test("sole owner, add admin, revoke access, and clear session", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 960 });
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  const mutations = await setup(page);
  await expect(page.getByRole("row")).toHaveCount(2);
  await expect(page.getByText("You", { exact: true })).toBeVisible();
  await page.getByRole("button", { name: "Edit yashu.savyminds@gmail.com" }).click();
  await expect(page.getByRole("combobox", { name: "Role", exact: true })).toBeDisabled();
  await expect(page.getByLabel("Access enabled")).toBeDisabled();
  await page.getByRole("button", { name: "Cancel", exact: true }).click();
  await page.screenshot({ path: "test-results/users-desktop.png", fullPage: true });
  await page.getByRole("button", { name: "Add user", exact: true }).click();
  await page.getByLabel("Display name").fill("Second admin");
  await page.getByLabel("Entra object ID").fill(other);
  await page.getByRole("combobox", { name: "Role", exact: true }).selectOption("admin");
  await page.getByRole("button", { name: "Save access" }).click();
  await expect(page.getByText("Second admin", { exact: true })).toBeVisible();
  expect(mutations).toHaveLength(1);
  await page.getByRole("button", { name: "Edit Second admin" }).click();
  await page.getByLabel("Access enabled").uncheck();
  await page.getByRole("button", { name: "Save access" }).click();
  await expect(page.getByText("Revoked", { exact: true })).toBeVisible();
  await page.getByRole("link", { name: "Settings", exact: true }).click();
  await page.getByRole("button", { name: "Sign out", exact: true }).click();
  await expect(page.getByRole("button", { name: "Sign in with Microsoft" })).toBeVisible();
  await expect(page.getByRole("link", { name: "Users", exact: true })).toHaveCount(0);
  expect(await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }))).not.toContain("browser-test-token");
  expect(errors).toEqual([]);
});

test("mobile admin form fits and concurrent edit requires refresh", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await setup(page, "admin", true);
  await page.getByRole("heading", { name: "Users", exact: true }).scrollIntoViewIfNeeded();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.screenshot({ path: "test-results/users-mobile.png", fullPage: true });
  await page.getByRole("button", { name: "Add user", exact: true }).click();
  await page.getByLabel("Display name").fill("Another user");
  await page.getByLabel("Entra object ID").fill(other);
  await page.screenshot({ path: "test-results/users-mobile-dialog.png" });
  const box = await page.getByRole("dialog").boundingBox();
  expect(box!.x).toBeGreaterThanOrEqual(0);
  expect(box!.x + box!.width).toBeLessThanOrEqual(390);
  await page.getByRole("button", { name: "Save access" }).click();
  await expect(page.getByRole("alert")).toContainText("User list changed");
  await expect(page.getByRole("button", { name: "Save access" })).toBeDisabled();
  await page.getByRole("button", { name: "Close and refresh" }).click();
  await expect(page.getByRole("dialog")).toHaveCount(0);
});

test("non-admin cannot open Users page", async ({ page }) => {
  await setup(page, "operator");
  await expect(page).toHaveURL(/\/settings$/);
  await expect(page.getByRole("link", { name: "Users", exact: true })).toHaveCount(0);
  await expect(page.getByRole("button", { name: "Add user", exact: true })).toHaveCount(0);
});
