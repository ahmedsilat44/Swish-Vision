import { useEffect, useState, useCallback } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";
import Tooltip from "../components/Tooltip";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

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

function StatusBadge({ status }) {
  const style = {
    display: "inline-block",
    padding: "2px 10px",
    borderRadius: "12px",
    fontSize: "0.78rem",
    fontWeight: 600,
    textTransform: "capitalize",
    ...badgeColors[status] || badgeColors.default,
  };
  return <span style={style}>{status}</span>;
}

const badgeColors = {
  queued:     { background: "#e5e7eb", color: "#374151" },
  processing: { background: "#fef3c7", color: "#92400e" },
  completed:  { background: "#dcfce7", color: "#15803d" },
  failed:     { background: "#fee2e2", color: "#b91c1c" },
  default:    { background: "#e5e7eb", color: "#374151" },
};

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

  useEffect(() => {
    const hasInflight = sessions.some(
      (s) => s.status === "queued" || s.status === "processing"
    );
    if (!hasInflight) return;
    const timer = setInterval(fetchSessions, 5000);
    return () => clearInterval(timer);
  }, [sessions, fetchSessions]);

  if (loading) return <div style={s.page}>Loading sessions…</div>;

  return (
    <div style={s.page}>
      <h2 style={s.heading}>Session History</h2>

      {error && <div style={s.errorBanner}>{error}</div>}

      {sessions.length === 0 ? (
        <div style={s.emptyState}>
          <p style={s.emptyText}>You haven't uploaded any videos yet.</p>
          <Link to="/upload" style={s.uploadLink}>Upload Your First Video</Link>
        </div>
      ) : (
        <div style={s.tableWrap}>
          <table style={s.table}>
            <thead>
              <tr style={s.headerRow}>
                <th style={s.th}>Date</th>
                <th style={s.th}>Video</th>
                <th style={s.th}>Shots</th>
                <th style={s.th}>Made %</th>
                <th style={s.th}>
                  Consistency
                  <Tooltip
                    text="A score from 0–100 measuring how uniform your shooting form is across all detected shots. Higher is more consistent."
                    label="What is Consistency Score?"
                  />
                </th>
                <th style={s.th}>Status</th>
                <th style={s.th}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {sessions.map((session) => {
                const totalShots =
                  session.total_shots != null
                    ? session.total_shots
                    : (session.shots_made ?? 0) + (session.shots_missed ?? 0);
                const madePercent =
                  session.shot_percentage != null
                    ? session.shot_percentage.toFixed(1) + "%"
                    : "—";
                const isProcessing = session.status === "processing";

                return (
                  <tr key={session.id} style={s.row}>
                    <td style={s.td}>
                      {new Date(session.created_at).toLocaleDateString()}
                    </td>
                    <td style={s.td}>
                      <Link to={`/results/${session.id}`} style={s.fileLink}>
                        {session.original_filename}
                      </Link>
                    </td>
                    <td style={s.td}>{totalShots || "—"}</td>
                    <td style={s.td}>{madePercent}</td>
                    <td style={s.td}>—</td>
                    <td style={s.td}>
                      <StatusBadge status={session.status} />
                    </td>
                    <td style={s.td}>
                      <button
                        disabled={isProcessing}
                        title={isProcessing ? "Cannot delete while processing" : "Delete session"}
                        onClick={() =>
                          deleteSession(session.id, setSessions, token, setError)
                        }
                        style={{
                          ...s.deleteBtn,
                          ...(isProcessing ? s.deleteBtnDisabled : {}),
                        }}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

const s = {
  page: {
    maxWidth: 960,
    margin: "0 auto",
    padding: "2rem 1.5rem",
    fontFamily: "'DM Sans', system-ui, sans-serif",
  },
  heading: {
    fontSize: "1.5rem",
    fontWeight: 700,
    color: "#1a1a2e",
    marginBottom: "1.25rem",
  },
  errorBanner: {
    marginBottom: "1rem",
    padding: "0.8rem 1rem",
    borderRadius: "10px",
    border: "1px solid rgba(239,68,68,0.3)",
    background: "rgba(239,68,68,0.08)",
    color: "#dc2626",
    fontSize: "0.9rem",
  },
  emptyState: {
    textAlign: "center",
    padding: "3rem 1rem",
    background: "#fff",
    borderRadius: 12,
    boxShadow: "0 2px 12px rgba(0,0,0,0.07)",
  },
  emptyText: {
    color: "#555",
    marginBottom: "1rem",
    fontSize: "1rem",
  },
  uploadLink: {
    display: "inline-block",
    padding: "0.6rem 1.5rem",
    background: "linear-gradient(135deg, #ff6400, #ff9a00)",
    color: "#fff",
    borderRadius: 8,
    fontWeight: 700,
    textDecoration: "none",
    fontSize: "0.9rem",
  },
  tableWrap: {
    overflowX: "auto",
    borderRadius: 12,
    boxShadow: "0 2px 12px rgba(0,0,0,0.07)",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
    background: "#fff",
    fontSize: "0.9rem",
  },
  headerRow: {
    background: "#1a1a2e",
  },
  th: {
    padding: "0.75rem 1rem",
    textAlign: "left",
    color: "#fff",
    fontWeight: 600,
    fontSize: "0.82rem",
    letterSpacing: "0.4px",
    whiteSpace: "nowrap",
  },
  row: {
    borderBottom: "1px solid #f0f0f0",
  },
  td: {
    padding: "0.75rem 1rem",
    color: "#333",
    verticalAlign: "middle",
  },
  fileLink: {
    color: "#3b82f6",
    textDecoration: "none",
    fontWeight: 500,
  },
  deleteBtn: {
    padding: "0.35rem 0.8rem",
    borderRadius: 6,
    border: "1px solid #fecaca",
    background: "#fee2e2",
    color: "#b91c1c",
    cursor: "pointer",
    fontSize: "0.8rem",
    fontFamily: "inherit",
  },
  deleteBtnDisabled: {
    opacity: 0.45,
    cursor: "not-allowed",
    background: "#f3f4f6",
    border: "1px solid #e5e7eb",
    color: "#9ca3af",
  },
};
