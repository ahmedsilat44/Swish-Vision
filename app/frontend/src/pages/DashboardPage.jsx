import { useState } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../AuthContext";

export default function DashboardPage() {
  const { token } = useAuth();
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
        minHeight: "calc(100vh - 56px)",
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
      {/* Logo */}
      <div
        style={{
          width: "52px",
          height: "52px",
          background: "linear-gradient(135deg, #ff6400, #ff9a00)",
          borderRadius: "14px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "26px",
          marginBottom: "0.25rem",
        }}
      >
        🏀
      </div>

      <h1 style={{ fontSize: "1.8rem", fontWeight: 700, margin: 0, letterSpacing: "-0.5px" }}>
        Swish Vision
      </h1>
      <p style={{ color: "#555", margin: 0, fontSize: "0.9rem" }}>
        Use the navigation above or the shortcuts below.
      </p>

      {/* Bearer Token Box */}
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
        <p
          style={{
            color: "#555",
            fontSize: "0.7rem",
            letterSpacing: "1.5px",
            textTransform: "uppercase",
            margin: "0 0 0.6rem",
          }}
        >
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
              fontSize: "0.7rem",
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
              border: `1px solid ${copied ? "#22c55e" : "#333"}`,
              borderRadius: "8px",
              color: copied ? "#fff" : "#aaa",
              fontSize: "0.8rem",
              cursor: "pointer",
              transition: "all 0.2s",
              whiteSpace: "nowrap",
              fontFamily: "inherit",
            }}
          >
            {copied ? "✓ Copied" : "Copy"}
          </button>
        </div>
        <p style={{ color: "#333", fontSize: "0.7rem", marginTop: "0.6rem", marginBottom: 0 }}>
          Postman → Authorization → Bearer Token → paste above
        </p>
      </div>

      {/* Quick-nav buttons */}
      <div
        style={{
          width: "100%",
          maxWidth: "640px",
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: "0.75rem",
        }}
      >
        <Link to="/shots" style={{ textDecoration: "none" }}>
          <button
            style={{
              width: "100%",
              padding: "0.9rem 1rem",
              background: "linear-gradient(135deg, #ff6400, #ff9a00)",
              border: "none",
              borderRadius: "10px",
              color: "#fff",
              fontWeight: 700,
              cursor: "pointer",
              fontSize: "0.95rem",
              fontFamily: "inherit",
            }}
          >
            Shots
          </button>
        </Link>
        <Link to="/sessions" style={{ textDecoration: "none" }}>
          <button
            style={{
              width: "100%",
              padding: "0.9rem 1rem",
              background: "#1e1e2e",
              border: "1px solid #333",
              borderRadius: "10px",
              color: "#fff",
              fontWeight: 700,
              cursor: "pointer",
              fontSize: "0.95rem",
              fontFamily: "inherit",
            }}
          >
            Sessions
          </button>
        </Link>
        <Link to="/upload" style={{ textDecoration: "none" }}>
          <button
            style={{
              width: "100%",
              padding: "0.9rem 1rem",
              background: "#1e1e2e",
              border: "1px solid #333",
              borderRadius: "10px",
              color: "#fff",
              fontWeight: 700,
              cursor: "pointer",
              fontSize: "0.95rem",
              fontFamily: "inherit",
            }}
          >
            Upload
          </button>
        </Link>
      </div>
    </div>
  );
}
