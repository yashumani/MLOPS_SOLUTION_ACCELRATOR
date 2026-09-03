import { BrowserCacheLocation, PublicClientApplication, type AccountInfo } from "@azure/msal-browser";

export interface EntraConfig {
  mode: "entra";
  tenant_id: string;
  client_id: string;
  scope: string;
  redirect_uri: string;
}

export async function createEntraSession(config: EntraConfig) {
  const redirect = new URL(config.redirect_uri);
  if (redirect.origin !== window.location.origin || !redirect.pathname.endsWith("/redirect.html")) {
    throw new Error("The registered sign-in redirect must be this website's redirect.html page.");
  }
  const app = new PublicClientApplication({
    auth: { clientId: config.client_id, authority: `https://login.microsoftonline.com/${config.tenant_id}`, redirectUri: config.redirect_uri },
    cache: { cacheLocation: BrowserCacheLocation.MemoryStorage }
  });
  await app.initialize();
  let account: AccountInfo | null = null;
  return {
    async login() {
      const result = await app.loginPopup({ scopes: [config.scope], prompt: "select_account" });
      account = result.account;
      if (!account) throw new Error("Sign-in returned no account.");
    },
    async token() {
      if (!account) throw new Error("Sign-in is required.");
      const result = await app.acquireTokenSilent({ scopes: [config.scope], account });
      return result.accessToken;
    },
    async clear() {
      account = null;
      await app.clearCache();
    }
  };
}
