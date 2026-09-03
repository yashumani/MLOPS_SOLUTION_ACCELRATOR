import { FormEvent, useEffect, useRef, useState } from "react";
import { Navigate } from "react-router-dom";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { Pencil, RefreshCw, Save, UserPlus, X } from "lucide-react";
import { useApi } from "../services/ApiContext";
import { ApiError, toUserMessage, type UserDirectory, type UserRole, type WorkspaceUser } from "../services/apiClient";

const roleLabels: Record<UserRole, string> = { admin: "Admin", operator: "Operator", viewer: "Viewer" };

function UserEditor({ user, directory, close }: { user: WorkspaceUser | null; directory: UserDirectory; close: () => void }) {
  const { api, refreshIdentity } = useApi();
  const queryClient = useQueryClient();
  const dialog = useRef<HTMLDialogElement>(null);
  const [draft, setDraft] = useState<WorkspaceUser>(user ?? { object_id: "", display_name: "", role: "viewer", enabled: true });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [conflict, setConflict] = useState(false);
  const isLastAdmin = !!user && user.enabled && user.role === "admin" && directory.users.filter((item) => item.enabled && item.role === "admin").length === 1;

  useEffect(() => {
    const element = dialog.current!;
    element.showModal();
    return () => { element.close(); };
  }, []);

  async function save(event: FormEvent) {
    event.preventDefault();
    if (busy || conflict) return;
    setBusy(true);
    setError("");
    try {
      const result = user ? await api.updateUser(draft, directory.revision) : await api.createUser(draft, directory.revision);
      queryClient.setQueryData(["workspace-users"], result);
      await refreshIdentity();
      close();
    } catch (failure) {
      setConflict(failure instanceof ApiError && failure.status === 409);
      setError(toUserMessage(failure));
    } finally { setBusy(false); }
  }

  return <dialog ref={dialog} className="user-dialog" aria-labelledby="user-editor-title" onCancel={(event) => { event.preventDefault(); if (!busy) close(); }}>
    <form onSubmit={save} className="form-panel">
      <div className="modal-header">
        <h2 id="user-editor-title">{user ? "Edit access" : "Add user"}</h2>
        <button type="button" className="icon-button" aria-label="Close" title="Close" onClick={close} disabled={busy}><X size={18} /></button>
      </div>
      <label className="field"><span>Display name</span><input autoFocus value={draft.display_name} maxLength={160} required onChange={(event) => setDraft({ ...draft, display_name: event.target.value })} /></label>
      <label className="field"><span>Entra object ID</span><input value={draft.object_id} disabled={!!user} required pattern="[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}" onChange={(event) => setDraft({ ...draft, object_id: event.target.value.trim() })} /></label>
      <label className="field"><span>Role</span><select value={draft.role} disabled={isLastAdmin} onChange={(event) => setDraft({ ...draft, role: event.target.value as UserRole })}>
        {Object.entries(roleLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select></label>
      <label className="user-enabled"><input type="checkbox" checked={draft.enabled} disabled={isLastAdmin} onChange={(event) => setDraft({ ...draft, enabled: event.target.checked })} />Access enabled</label>
      {error && <p role="alert" className="user-error">{error}</p>}
      <div className="button-row">
        <button className="button primary" type="submit" disabled={busy || conflict}><Save size={17} />{busy ? "Saving..." : "Save access"}</button>
        <button className="button secondary" type="button" disabled={busy} onClick={() => { if (conflict) void queryClient.invalidateQueries({ queryKey: ["workspace-users"] }); close(); }}>{conflict ? "Close and refresh" : "Cancel"}</button>
      </div>
    </form>
  </dialog>;
}

export function Users() {
  const { api, identity } = useApi();
  const admin = identity.mode === "entra" && identity.roles.includes("admin");
  const query = useQuery({ queryKey: ["workspace-users"], queryFn: () => api.users(), enabled: admin, retry: false });
  const [editor, setEditor] = useState<{ user: WorkspaceUser | null } | null>(null);
  if (!admin) return <Navigate to="/settings" replace />;
  return <main className="page users-page">
    <section className="page-heading">
      <div><p className="eyebrow">Administration</p><h1>Users</h1></div>
      <div className="button-row">
        <button type="button" className="icon-button" title="Refresh users" aria-label="Refresh users" disabled={query.isFetching} onClick={() => void query.refetch()}><RefreshCw size={18} /></button>
        <button type="button" className="button primary" disabled={!query.data || query.isError} onClick={() => setEditor({ user: null })}><UserPlus size={18} />Add user</button>
      </div>
    </section>
    {query.isPending && <p role="status">Loading users...</p>}
    {query.error && <p role="alert" className="user-error">{toUserMessage(query.error)}</p>}
    {query.data && <div className="table-wrap"><table className="users-table">
      <thead><tr><th>User</th><th>Role</th><th>Access</th><th><span className="sr-only">Actions</span></th></tr></thead>
      <tbody>{query.data.users.map((user) => <tr key={user.object_id}>
        <td className="user-identity"><strong>{user.display_name}</strong>{user.object_id === identity.object_id && <span className="user-self">You</span>}<small>{user.object_id}</small></td>
        <td>{roleLabels[user.role]}</td><td>{user.enabled ? "Active" : "Revoked"}</td>
        <td><button type="button" className="icon-button" aria-label={`Edit ${user.display_name}`} title={`Edit ${user.display_name}`} disabled={query.isError} onClick={() => setEditor({ user })}><Pencil size={17} /></button></td>
      </tr>)}</tbody>
    </table></div>}
    {editor && query.data && <UserEditor user={editor.user} directory={query.data} close={() => setEditor(null)} />}
  </main>;
}
