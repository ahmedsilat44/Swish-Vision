import { Link, useNavigate, useLocation } from "react-router-dom";
import { useAuth } from "./AuthContext";

const NAV_LINKS = [
  { to: "/sessions", label: "Sessions" },
  { to: "/shots", label: "Shots" },
  { to: "/upload", label: "Upload" },
];

export default function NavBar() {
  const { token, logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  if (!token) return null;

  const handleLogout = async () => {
    await logout();
    navigate("/login", { replace: true });
  };

  return (
    <nav
      style={{
        background: "#13131a",
        borderBottom: "1px solid #1e1e2e",
        padding: "0 2rem",
        height: "56px",
        display: "flex",
        alignItems: "center",
        gap: "2rem",
        fontFamily: "'DM Sans', sans-serif",
        position: "sticky",
        top: 0,
        zIndex: 100,
      }}
    >
      {/* Brand */}
      <Link
        to="/dashboard"
        style={{ display: "flex", alignItems: "center", gap: "8px", textDecoration: "none", flexShrink: 0 }}
      >
        <span
          style={{
            width: "28px",
            height: "28px",
            background: "linear-gradient(135deg, #ff6400, #ff9a00)",
            borderRadius: "8px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            fontSize: "14px",
          }}
        >
          🏀
        </span>
        <span style={{ fontSize: "1rem", fontWeight: 700, color: "#fff", letterSpacing: "-0.3px" }}>
          Swish Vision
        </span>
      </Link>

      <div style={{ flex: 1 }} />

      {/* Nav links */}
      {NAV_LINKS.map(({ to, label }) => {
        const active = location.pathname === to;
        return (
          <Link
            key={to}
            to={to}
            style={{
              color: active ? "#fff" : "#666",
              fontSize: "0.875rem",
              fontWeight: 600,
              textDecoration: "none",
              borderBottom: active ? "2px solid #ff6400" : "2px solid transparent",
              paddingBottom: "2px",
              transition: "color 0.15s",
            }}
          >
            {label}
          </Link>
        );
      })}

      {/* Logout */}
      <button
        onClick={handleLogout}
        style={{
          padding: "0.45rem 1.1rem",
          background: "transparent",
          border: "1px solid #333",
          borderRadius: "8px",
          color: "#666",
          fontSize: "0.8rem",
          cursor: "pointer",
          fontFamily: "inherit",
          transition: "all 0.15s",
        }}
        onMouseEnter={(e) => { e.currentTarget.style.borderColor = "#ff6400"; e.currentTarget.style.color = "#ff6400"; }}
        onMouseLeave={(e) => { e.currentTarget.style.borderColor = "#333"; e.currentTarget.style.color = "#666"; }}
      >
        Log Out
      </button>
    </nav>
  );
}
