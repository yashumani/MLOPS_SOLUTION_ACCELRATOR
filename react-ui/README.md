# MLOps V3 React UI

This is the React replacement track for the Streamlit UI. It is intentionally
kept beside `ui/` until feature parity is accepted.

## Run Locally On The Azure ML Compute Instance

Start the FastAPI backend from the repo root:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Start the React UI:

```bash
cd react-ui
npm install
npm run dev
```

Open the Azure ML application proxy URL:

```text
https://mlopspipelinev2-8501.eastus2.instances.azureml.ms/
```

The app prompts for the API key if `runtime-config.js` does not provide one.
For production, use a proper auth or backend-for-frontend layer rather than
embedding a static API key in the browser bundle.

For Azure ML application proxy access, `public/runtime-config.js` leaves
`apiBaseUrl` empty. Browser API calls stay on the UI origin (`/api/...`) and the
Vite dev server proxies those calls to FastAPI on `127.0.0.1:8000`.

## Scope

- UI-only implementation.
- FastAPI remains the API surface.
- Azure ML pipeline code, component contracts, and step scripts are unchanged.

## First Milestone

- Route-driven Focus cockpit at `/focus/:jobName`.
- Change Job opens a searchable picker and updates the URL.
- Leaderboard errors are shown as retryable user states.
- Outputs are rendered as client-friendly cards and tables, with raw JSON hidden
  under Advanced details.