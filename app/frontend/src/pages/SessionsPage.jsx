import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

async function retrySession(sessionId, setSessions, token, setError) {
  const res = await fetch(`${API_URL}/sessions/${sessionId}/retry`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });

  if (res.ok) {
    const updated = await res.json();
    setSessions((prev) => prev.map((s) => (s.id === sessionId ? updated : s)));
    return;
  }

  const body = await res.json().catch(() => ({}));
  setError(body.detail || "Failed to retry session.");
}

async function deleteSession(sessionId, setSessions, token, setError) {
  if (!window.confirm("Delete this session? This cannot be undone.")) return;

  const res = await fetch(`${API_URL}/sessions/${sessionId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });

  if (res.status === 204) {
    setSessions((prev) => prev.filter((s) => s.id !== sessionId));
    return;
  }

  if (res.status === 409) {
    setError("Session is currently processing. Please wait until it completes.");
    return;
  }

  const body = await res.json().catch(() => ({}));
  setError(body.detail || "Failed to delete session.");
}

export default function SessionsPage() {
  const { token, logout } = useAuth();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const fetchSessions = useCallback(async () => {
    setError("");
    try {
      const res = await fetch(`${API_URL}/sessions/`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      if (res.status === 401) {
        await logout();
        navigate("/login", { replace: true });
        return;
      }

      const body = await res.json().catch(() => []);
      if (!res.ok) {
        setError(body?.detail || "Failed to load sessions.");
        return;
      }

      setSessions(Array.isArray(body) ? body : []);
    } catch (_err) {
      setError("Failed to load sessions.");
    } finally {
      setLoading(false);
    }
  }, [token, logout, navigate]);

  useEffect(() => {
    fetchSessions();
  }, [fetchSessions]);

  // Poll every 5 s while any session is queued or processing
  useEffect(() => {
    const hasInflight = sessions.some(
      (s) => s.status === "queued" || s.status === "processing"
    );
    if (!hasInflight) return;
    const timer = setInterval(fetchSessions, 5000);
    return () => clearInterval(timer);
  }, [sessions, fetchSessions]);

  if (loading) return <div>Loading sessions...</div>;

  return (
    <div style={{ padding: "2rem" }}>
      <h2>My Sessions</h2>
      {error ? (
        <div
          style={{
            marginBottom: "1rem",
            padding: "0.8rem 1rem",
            borderRadius: "10px",
            border: "1px solid rgba(239,68,68,0.3)",
            background: "rgba(239,68,68,0.12)",
            color: "#ff8a8a",
          }}
        >
          {error}
        </div>
      ) : null}

      {sessions.length === 0 ? (
        <p>No sessions found.</p>
      ) : (
        <table
          style={{
            width: "100%",
            borderCollapse: "collapse",
            border: "1px solid #1e1e2e",
          }}
        >
          <thead>
            <tr style={{ background: "#13131a", borderBottom: "1px solid #1e1e2e" }}>
              <th style={{ padding: "0.75rem", textAlign: "left" }}>Session</th>
              <th style={{ padding: "0.75rem", textAlign: "left" }}>File</th>
              <th style={{ padding: "0.75rem", textAlign: "left" }}>Status</th>
              <th style={{ padding: "0.75rem", textAlign: "left" }}>Created</th>
              <th style={{ padding: "0.75rem", textAlign: "left" }}>Action</th>
            </tr>
          </thead>
          <tbody>
            {sessions.map((session) => (
              <tr key={session.id} style={{ borderBottom: "1px solid #1e1e2e" }}>
                <td style={{ padding: "0.75rem" }}>#{session.id}</td>
                <td style={{ padding: "0.75rem" }}>{session.original_filename}</td>
                <td style={{ padding: "0.75rem" }}>{session.status}</td>
                <td style={{ padding: "0.75rem" }}>
                  {new Date(session.created_at).toLocaleString()}
                </td>
                <td style={{ padding: "0.75rem", display: "flex", gap: "0.5rem", alignItems: "center" }}>
                  {session.status === "failed" && (
                    <button
                      onClick={() => retrySession(session.id, setSessions, token, setError)}
                      style={{
                        padding: "0.5rem 0.8rem",
                        borderRadius: "8px",
                        border: "1px solid #1f4a5f",
                        color: "#7dd3fc",
                        background: "#0c2233",
                        cursor: "pointer",
                      }}
                    >
                      Retry
                    </button>
                  )}
                  <button
                    onClick={() => deleteSession(session.id, setSessions, token, setError)}
                    style={{
                      padding: "0.5rem 0.8rem",
                      borderRadius: "8px",
                      border: "1px solid #5f1f1f",
                      color: "#ffb1b1",
                      background: "#2a1111",
                      cursor: "pointer",
                    }}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
