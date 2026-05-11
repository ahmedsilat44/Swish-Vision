import React, { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

export function ShotsTable() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [sessions, setSessions] = useState([]);
  const [selectedSessionId, setSelectedSessionId] = useState("");
  const [shots, setShotsData] = useState([]);
  const [shotSummary, setShotSummary] = useState({ makes: 0, totalShots: 0 });
  const [loadingSessions, setLoadingSessions] = useState(true);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchSessions = async () => {
      try {
        const response = await fetch(`${API_URL}/sessions/`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        const data = await response.json();

        if (response.status === 401) {
          navigate("/login", { replace: true });
          return;
        }

        if (!response.ok) throw new Error(data?.detail || "Failed to load sessions");

        setSessions(data);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoadingSessions(false);
      }
    };

    fetchSessions();
  }, [token, navigate]);

  useEffect(() => {
    if (!selectedSessionId) {
      setShotsData([]);
      setShotSummary({ makes: 0, totalShots: 0 });
      setLoading(false);
      return;
    }

    const fetchShots = async () => {
      setLoading(true);
      try {
        const response = await fetch(`${API_URL}/sessions/${selectedSessionId}/shots`, {
          headers: { Authorization: `Bearer ${token}` },
        });

        if (response.status === 401) {
          navigate("/login", { replace: true });
          return;
        }
        
        if (!response.ok) throw new Error("Failed to fetch shots");
        
        const data = await response.json();
        setShotsData(data.shots);
        setShotSummary({
          makes: Number(data.makes) || 0,
          totalShots: Number(data.total_shots) || 0,
        });
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchShots();
  }, [selectedSessionId, token, navigate]);

  if (loadingSessions) return <div style={{ minHeight: "calc(100vh - 56px)", background: "#0a0a0f", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center" }}>Loading sessions...</div>;
  if (error) return <div style={{ minHeight: "calc(100vh - 56px)", background: "#0a0a0f", color: "#fff", display: "flex", alignItems: "center", justifyContent: "center" }}>Error: {error}</div>;

  const makes = shotSummary.makes;
  const totalShots = shotSummary.totalShots;
  const percentage = totalShots ? ((makes / totalShots) * 100).toFixed(1) : "0.0";

  return (
    <div style={{
      minHeight: "calc(100vh - 56px)",
      background: "#0a0a0f",
      color: "#fff",
      padding: "2rem",
      fontFamily: "'DM Sans', sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
    }}>
      <h2 style={{ color: "#fff", marginBottom: "1.5rem" }}>Shot Results</h2>
      <div
        style={{
          marginBottom: "1rem",
          background: "#13131a",
          border: "1px solid #1e1e2e",
          borderRadius: "14px",
          padding: "1rem 1.25rem",
          width: "100%",
          maxWidth: "760px",
        }}
      >
        <p style={{ color: "#888", fontSize: "0.72rem", letterSpacing: "1.5px", textTransform: "uppercase", margin: "0 0 0.6rem" }}>
          Select Session
        </p>
        <select
          value={selectedSessionId}
          onChange={(e) => setSelectedSessionId(e.target.value)}
          disabled={!sessions.length}
          style={{
            width: "100%",
            padding: "0.8rem 1rem",
            background: "#0d0d14",
            border: "1px solid #1e1e2e",
            borderRadius: "10px",
            color: "#fff",
            fontSize: "0.95rem",
            outline: "none",
            colorScheme: "dark",
          }}
        >
          <option value="">Choose a session</option>
          {sessions.map((session) => (
            <option key={session.id} value={session.id}>
              #{session.id} - {session.original_filename} ({session.status})
            </option>
          ))}
        </select>
        <p style={{ color: "#666", fontSize: "0.72rem", marginTop: "0.6rem", marginBottom: 0 }}>
          The table will appear after you pick a session.
        </p>
      </div>

      {!selectedSessionId ? null : loading ? (
        <div style={{ color: "#aaa", fontSize: "0.95rem" }}>Loading shots...</div>
      ) : (
        <div style={{ width: "100%", maxWidth: "760px" }}>
          <div style={{ marginBottom: "0.75rem", color: "#888", fontSize: "0.9rem" }}>
            Session ID: {selectedSessionId}
          </div>
          <div style={{ marginBottom: "1rem", color: "#888" }}>
            {makes}/{totalShots} made ({percentage}%)
          </div>

          <table style={{
            width: "100%",
            borderCollapse: "collapse",
            border: "1px solid #1e1e2e",
            background: "#13131a",
          }}>
            <thead>
              <tr style={{ background: "#0d0d14", borderBottom: "1px solid #1e1e2e" }}>
                <th style={{ padding: "0.75rem", textAlign: "left", color: "#fff", fontWeight: 600 }}>Shot #</th>
                <th style={{ padding: "0.75rem", textAlign: "left", color: "#fff", fontWeight: 600 }}>Outcome</th>
                <th style={{ padding: "0.75rem", textAlign: "left", color: "#fff", fontWeight: 600 }}>SEW Angle (°)</th>
                <th style={{ padding: "0.75rem", textAlign: "left", color: "#fff", fontWeight: 600 }}>ESH Angle (°)</th>
              </tr>
            </thead>
            <tbody>
              {shots.map((shot) => (
                <tr key={shot.shot_number} style={{ borderBottom: "1px solid #1e1e2e" }}>
                  <td style={{ padding: "0.75rem", color: "#ccc" }}>{shot.shot_number}</td>
                  <td
                    style={{
                      padding: "0.75rem",
                      color: (shot.outcome || shot.result) === "made" || shot.result === "make" ? "#4ade80" : "#ff6b6b",
                    }}
                  >
                    {(shot.outcome || (shot.result === "make" ? "made" : shot.result === "miss" ? "missed" : "missed")).toUpperCase()}
                  </td>
                  <td style={{ padding: "0.75rem", color: "#ccc" }}>
                    {shot.sew_angle != null ? `${shot.sew_angle.toFixed(1)}` : "—"}
                  </td>
                  <td style={{ padding: "0.75rem", color: "#ccc" }}>
                    {shot.esh_angle != null ? `${shot.esh_angle.toFixed(1)}` : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}