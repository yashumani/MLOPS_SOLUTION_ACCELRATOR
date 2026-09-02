import { Navigate, Route, Routes } from "react-router-dom";
import { ApiKeyGate } from "./components/ApiKeyGate";
import { Layout } from "./components/Layout";
import { AutoRetrain } from "./pages/AutoRetrain";
import { Configs } from "./pages/Configs";
import { Drift } from "./pages/Drift";
import { Focus } from "./pages/Focus";
import { Home } from "./pages/Home";
import { Logs } from "./pages/Logs";
import { Notifications } from "./pages/Notifications";
import { Reports } from "./pages/Reports";
import { Settings } from "./pages/Settings";
import { Submit } from "./pages/Submit";

export function App() {
  return (
    <ApiKeyGate>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Home />} />
          <Route path="focus" element={<Focus />} />
          <Route path="focus/:jobName" element={<Focus />} />
          <Route path="submit" element={<Submit />} />
          <Route path="configs" element={<Configs />} />
          <Route path="auto-retrain" element={<AutoRetrain />} />
          <Route path="drift" element={<Drift />} />
          <Route path="reports" element={<Reports />} />
          <Route path="logs" element={<Logs />} />
          <Route path="notifications" element={<Notifications />} />
          <Route path="settings" element={<Settings />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </ApiKeyGate>
  );
}