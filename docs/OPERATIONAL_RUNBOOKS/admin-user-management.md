# Admin User Management

## Release State

Implemented on the development branch. Not enabled on the shared website yet.
The initial-user decision is complete; app-registration and hosting configuration
remain prerequisites for live sign-in. This does not complete the 15-scenario
Azure release qualification.

## Initial Administrator

- Account: `yashu.savyminds@gmail.com`.
- Tenant: `3bc05bc3-19d1-4d30-89c5-134f4b278b11`.
- Entra object ID: `b03e4295-9fce-4b3b-b6ba-e7e750e639ef`.
- Role: `admin`.

Verified with `az account show` and `az ad signed-in-user show` on 2026-09-03.
`configs/access/initial_admin.json` contains this account and no other users.
Object IDs are identifiers, not credentials. Authentication never relies on a
mutable email or display name.

## Website Workflow

After Microsoft sign-in, administrators have a **Users** navigation entry.
**Add user** accepts the user's existing object ID in this Entra tenant, a
display name, a role, and whether access is enabled. To grant another admin,
choose **Admin** and save. Edit an existing row to change its role or revoke
access by clearing **Access enabled**. There is no public sign-up or automatic
access based on compute-instance membership.

Roles are application roles only:

| Role | Permissions |
| --- | --- |
| Admin | Operator actions plus user management |
| Operator | Existing operational API actions, subject to deployment safeguards |
| Viewer | Read-only operational API requests |

User management does not create Entra accounts, send directory invitations,
grant Azure RBAC, grant Entra directory administrator privileges, or authorize
production model promotion. New directory users must be provisioned separately.
The API does not require tenant-wide Microsoft Graph user-read permissions.

## Server Configuration

Use the existing multi-user settings, with:

```text
API_DEPLOYMENT_PROFILE=multi_user
API_CONFIG_MUTATION_ENABLED=false
API_RELOAD=false
API_ENTRA_TENANT_ID=3bc05bc3-19d1-4d30-89c5-134f4b278b11
API_USER_ALLOWLIST_PATH=<absolute canonical repo>/configs/access/initial_admin.json
MLOPS_OPERATIONAL_STATE_DB=<absolute local persistent disk>/mlops-state.sqlite3
```

Supply the real API and SPA application IDs in `API_ENTRA_API_CLIENT_ID`,
`API_ENTRA_SPA_CLIENT_ID`, and `API_ENTRA_ALLOWED_CLIENT_IDS`. Configure a v2
delegated API access token and scope `access_as_user`; consent must be provided
through the approved Entra process. Set `API_ENTRA_REDIRECT_URI` to the deployed
UI's exact HTTPS `/redirect.html` URL, register it as an SPA redirect, and include
the UI origin in `CORS_ALLOW_ORIGINS`. No client secret belongs in the browser.

The build emits a dedicated `redirect.html`. It must not receive a
`Cross-Origin-Opener-Policy` header. It must remain publicly reachable without
the application login gate and be served as that file, not an index fallback.
MSAL uses popup sign-in and an in-memory cache; the API receives access tokens,
not ID tokens. See Microsoft's [redirect bridge setup](https://learn.microsoft.com/en-us/entra/msal/javascript/browser/redirect-bridge)
and [MSAL initialization](https://learn.microsoft.com/en-us/entra/msal/javascript/browser/initialization).

The API/controller state model supports one host with local persistent disk and
multiple worker processes. Do not place SQLite on Azure Files, SMB, NFS, or a
multi-host shared mount. Set all durable-state variables in the service process
environment and verify the controller and API share the exact same database,
ledger identity, and canonical-submit lock directory. Confirm disk persistence
and filesystem ownership before enabling the shared service.

## Persistence And Safety

Startup seeds the single admin only when no user directory exists. Later starts
preserve the database, including revoked users; editing the bootstrap file does
not silently recreate or elevate accounts. A mismatched bootstrap identity or
tenant fails startup. Do not delete the database as a way to reset user access.

Every authenticated request reloads access from the database. Admin mutations
recheck the acting admin inside the write transaction, so a revoked admin cannot
use an old in-memory principal. User changes and their actor/before/after audit
event commit together. Each edit includes the observed directory revision;
concurrent edits return HTTP 409 and require a refresh. Both backend and UI
protect the last active admin from demotion or removal. Non-admin access is
denied by the API even when calling the route directly.

Use `scripts/migrate_operational_state.py` for legacy request/ledger state only
after stopping writers. Preserve backups of the operational database, including
the access directory and audit records. Use SQLite's backup API or stop all
writers for an approved consistent backup; do not copy only the live main file
while ignoring its WAL. Backup/retention and emergency recovery ownership must
be assigned before production activation.

## Verification

Local backend command:

```text
python -m pytest -q -m "not integration and not requires_azure_ml and not slow"
634 passed, 6 warnings
```

React lint, seven unit tests, and the production build passed. Three browser
tests passed in headless Edge, covering sole-admin protection, adding an admin,
revocation, sign-out, non-admin denial, mobile layout, and stale-edit handling.
Browser tests mock the identity provider and API; they are not live Entra proof.
The API tests independently exercise signed-token validation and actual SQLite
authorization. Hosted CI now runs the browser tests using Chromium.

Before live acceptance, prove real owner sign-in, verify that only the owner is
seeded, and use a separately approved test identity to check add/revoke behavior.
No second live user has been created by this implementation.
