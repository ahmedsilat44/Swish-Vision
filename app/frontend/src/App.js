import { useState } from "react";
import { LoginPage, LogoutButton } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import { ShotsTable } from "./pages/ShotsTable";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

// ─── PLACEHOLDER DASHBOARD ────────────────────────────────────────────────────
function Dashboard({
  onLogout,
  token,
  onOpenShotsTable,
  actionStatus,
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(token).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#0a0a0f",
        color: "#fff",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        fontFamily: "'DM Sans', sans-serif",
        gap: "1rem",
        padding: "2rem",
      }}
    >
      <div
        style={{
          width: "48px",
          height: "48px",
          background: "linear-gradient(135deg, #ff6400, #ff9a00)",
          borderRadius: "14px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "24px",
          marginBottom: "0.5rem",
        }}
      >
        🏀
      </div>
      <h1 style={{ fontSize: "1.8rem", fontWeight: "700", margin: 0, letterSpacing: "-0.5px" }}>
        Swish Vision
      </h1>
      <p style={{ color: "#555", margin: 0 }}>Dashboard coming soon</p>

      {/* ── Bearer Token Box (for Postman testing) ── */}
      <div
        style={{
          marginTop: "1.5rem",
          background: "#13131a",
          border: "1px solid #1e1e2e",
          borderRadius: "14px",
          padding: "1.25rem 1.5rem",
          width: "100%",
          maxWidth: "640px",
        }}
      >
        <p style={{ color: "#888", fontSize: "0.72rem", letterSpacing: "1.5px", textTransform: "uppercase", margin: "0 0 0.6rem" }}>
          Bearer Token (Postman)
        </p>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start" }}>
          <code
            style={{
              flex: 1,
              background: "#0d0d14",
              border: "1px solid #1e1e2e",
              borderRadius: "8px",
              padding: "0.6rem 0.8rem",
              fontSize: "0.72rem",
              color: "#ff9a00",
              wordBreak: "break-all",
              lineHeight: "1.6",
              display: "block",
            }}
          >
            {token}
          </code>
          <button
            onClick={handleCopy}
            style={{
              flexShrink: 0,
              padding: "0.6rem 1rem",
              background: copied ? "#22c55e" : "#1e1e2e",
              border: "1px solid " + (copied ? "#22c55e" : "#333"),
              borderRadius: "8px",
              color: copied ? "#fff" : "#aaa",
              fontSize: "0.8rem",
              cursor: "pointer",
              transition: "all 0.2s",
              whiteSpace: "nowrap",
            }}
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <p style={{ color: "#444", fontSize: "0.72rem", marginTop: "0.6rem", marginBottom: 0 }}>
          In Postman → Authorization tab → Bearer Token → paste above
        </p>
      </div>

      <div
        style={{
          width: "100%",
          maxWidth: "640px",
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: "0.75rem",
        }}
      >
        <button
          onClick={onOpenShotsTable}
          style={{
            padding: "0.85rem 1rem",
            background: "linear-gradient(135deg, #ff6400, #ff9a00)",
            border: "none",
            borderRadius: "10px",
            color: "#fff",
            fontWeight: 700,
            cursor: "pointer",
          }}
        >
          Open Shots Table
        </button>
      </div>

      {actionStatus ? (
        <p style={{ margin: 0, color: "#aaa", fontSize: "0.9rem", maxWidth: "640px", textAlign: "center" }}>
          {actionStatus}
        </p>
      ) : null}

      <LogoutButton onLogout={onLogout} />
    </div>
  );
}

// ─── APP ──────────────────────────────────────────────────────────────────────
function App() {
  // Lazy initialiser reads localStorage once on mount — keeps user logged in on refresh
  const [token, setToken] = useState(() => localStorage.getItem("access_token"));
  const [page, setPage] = useState("login"); // "login" | "register"
  const [dashboardView, setDashboardView] = useState("dashboard"); // "dashboard" | "shots"
  const [actionStatus, setActionStatus] = useState("");

  const handleAuthError = () => {
    localStorage.removeItem("access_token");
    setToken(null);
    setPage("login");
    setDashboardView("dashboard");
    setActionStatus("Session expired. Please log in again.");
  };

  const handleLoginSuccess = (data) => {
    setToken(data.access_token);
  };

  const handleRegisterSuccess = (data) => {
    if (data?.access_token) {
      setToken(data.access_token);
    } else {
      // Registration succeeded but no token returned — send to login
      setPage("login");
    }
  };

  const handleLogout = () => {
    setToken(null);
    setPage("login");
    setDashboardView("dashboard");
    setActionStatus("");
  };

  const handleOpenShotsTable = async () => {
    setActionStatus("");
    setDashboardView("shots");
  };

  // ── Protected: token present ─────────────────────────────────────────────────
  if (token) {
    if (dashboardView === "shots") {
      return (
        <div
          style={{
            minHeight: "100vh",
            background: "#0a0a0f",
            color: "#fff",
            fontFamily: "'DM Sans', sans-serif",
            padding: "1rem",
          }}
        >
          <button
            onClick={() => setDashboardView("dashboard")}
            style={{
              marginBottom: "1rem",
              padding: "0.65rem 0.9rem",
              background: "#1e1e2e",
              border: "1px solid #333",
              borderRadius: "8px",
              color: "#fff",
              cursor: "pointer",
            }}
          >
            Back to Dashboard
          </button>
          <ShotsTable token={token} onAuthError={handleAuthError} />
        </div>
      );
    }

    return (
      <Dashboard
        onLogout={handleLogout}
        token={token}
        onOpenShotsTable={handleOpenShotsTable}
        actionStatus={actionStatus}
      />
    );
  }

  // ── Public: register ─────────────────────────────────────────────────────────
  if (page === "register") {
    return (
      <RegisterPage
        onRegisterSuccess={handleRegisterSuccess}
        onGoToLogin={() => setPage("login")}
      />
    );
  }

  // ── Default: login (unauthenticated users always land here) ──────────────────
  return (
    <LoginPage
      onLoginSuccess={handleLoginSuccess}
      onGoToRegister={() => setPage("register")}
    />
  );
}

export default App;
