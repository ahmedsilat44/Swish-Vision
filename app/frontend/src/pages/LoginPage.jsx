import { useState } from "react";

// ─── CONFIG ───────────────────────────────────────────────────────────────────
// Replace with your actual API base URL (from .env: REACT_APP_API_URL)
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

// ─── STYLES ───────────────────────────────────────────────────────────────────
const styles = {
  page: {
    minHeight: "100vh",
    background: "#0a0a0f",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontFamily: "'DM Sans', sans-serif",
    padding: "1rem",
  },
  card: {
    background: "#13131a",
    border: "1px solid #1e1e2e",
    borderRadius: "20px",
    padding: "2.5rem",
    width: "100%",
    maxWidth: "420px",
    boxShadow: "0 0 60px rgba(255, 100, 0, 0.08)",
  },
  logo: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    marginBottom: "2rem",
  },
  logoIcon: {
    width: "38px",
    height: "38px",
    background: "linear-gradient(135deg, #ff6400, #ff9a00)",
    borderRadius: "10px",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    fontSize: "18px",
  },
  logoText: {
    fontSize: "1.2rem",
    fontWeight: "700",
    color: "#fff",
    letterSpacing: "-0.3px",
  },
  logoSub: {
    fontSize: "0.7rem",
    color: "#555",
    letterSpacing: "2px",
    textTransform: "uppercase",
    display: "block",
  },
  heading: {
    fontSize: "1.6rem",
    fontWeight: "700",
    color: "#fff",
    marginBottom: "0.4rem",
    letterSpacing: "-0.5px",
  },
  subheading: {
    fontSize: "0.875rem",
    color: "#555",
    marginBottom: "2rem",
  },
  field: {
    marginBottom: "1.2rem",
  },
  label: {
    display: "block",
    fontSize: "0.78rem",
    color: "#888",
    marginBottom: "6px",
    letterSpacing: "0.5px",
    textTransform: "uppercase",
  },
  input: {
    width: "100%",
    padding: "0.75rem 1rem",
    background: "#0d0d14",
    border: "1px solid #1e1e2e",
    borderRadius: "10px",
    color: "#fff",
    fontSize: "0.95rem",
    outline: "none",
    boxSizing: "border-box",
    transition: "border-color 0.2s",
  },
  inputError: {
    borderColor: "#ff4444",
  },
  errorText: {
    color: "#ff4444",
    fontSize: "0.78rem",
    marginTop: "5px",
  },
  alertError: {
    background: "rgba(255, 68, 68, 0.1)",
    border: "1px solid rgba(255, 68, 68, 0.3)",
    borderRadius: "10px",
    padding: "0.75rem 1rem",
    color: "#ff6666",
    fontSize: "0.875rem",
    marginBottom: "1.2rem",
  },
  button: {
    width: "100%",
    padding: "0.85rem",
    background: "linear-gradient(135deg, #ff6400, #ff9a00)",
    border: "none",
    borderRadius: "10px",
    color: "#fff",
    fontWeight: "700",
    fontSize: "0.95rem",
    cursor: "pointer",
    letterSpacing: "0.3px",
    transition: "opacity 0.2s, transform 0.1s",
    marginTop: "0.5rem",
  },
  buttonDisabled: {
    opacity: 0.6,
    cursor: "not-allowed",
  },
  footer: {
    textAlign: "center",
    marginTop: "1.5rem",
    fontSize: "0.85rem",
    color: "#555",
  },
  link: {
    color: "#ff6400",
    textDecoration: "none",
    fontWeight: "600",
  },
};

// ─── LOGIN PAGE ───────────────────────────────────────────────────────────────
export function LoginPage({ onLoginSuccess, onGoToRegister }) {
  const [form, setForm] = useState({ email: "", password: "" });
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState("");
  const [loading, setLoading] = useState(false);

  const validate = () => {
    const e = {};
    if (!form.email.trim()) e.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(form.email)) e.email = "Enter a valid email";
    if (!form.password) e.password = "Password is required";
    return e;
  };

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setErrors({ ...errors, [e.target.name]: "" });
    setApiError("");
  };

  const handleSubmit = async () => {
    const e = validate();
    if (Object.keys(e).length) return setErrors(e);

    setLoading(true);
    setApiError("");

    try {
      const res = await fetch(`${API_URL}/auth/login`, {

        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: form.email, password: form.password }),
      });

      const data = await res.json();

      if (!res.ok) {
        // Handle specific backend error messages
        if (res.status === 401) setApiError("Invalid email or password.");
        else if (res.status === 403) setApiError("Your account has been suspended.");
        else if (res.status === 429) setApiError("Too many attempts. Please wait a moment.");
        else setApiError(data?.detail || "Something went wrong. Please try again.");
        return;
      }

      // Store JWT token
      localStorage.setItem("access_token", data.access_token);
      if (onLoginSuccess) onLoginSuccess(data);

    } catch (err) {
      setApiError("Cannot connect to server. Check your connection.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        {/* Logo */}
        <div style={styles.logo}>
          <div style={styles.logoIcon}>🏀</div>
          <div>
            <span style={styles.logoText}>Swish Vision</span>
            <span style={styles.logoSub}>Shot Analytics</span>
          </div>
        </div>

        <h1 style={styles.heading}>Welcome back</h1>
        <p style={styles.subheading}>Sign in to your account to continue</p>

        {/* API Error Banner */}
        {apiError && <div style={styles.alertError}>⚠ {apiError}</div>}

        {/* Email */}
        <div style={styles.field}>
          <label htmlFor="email" style={styles.label}>Email</label>
          <input
            id="email"

            style={{ ...styles.input, ...(errors.email ? styles.inputError : {}) }}
            type="email"
            name="email"
            placeholder="you@example.com"
            value={form.email}
            onChange={handleChange}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
          {errors.email && <p style={styles.errorText}>{errors.email}</p>}
        </div>

        {/* Password */}
        <div style={styles.field}>

          <label htmlFor="password" style={styles.label}>Password</label>
          <input
            id="password"
            style={{ ...styles.input, ...(errors.password ? styles.inputError : {}) }}
            type="password"
            name="password"
            placeholder="••••••••"
            value={form.password}
            onChange={handleChange}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
          {errors.password && <p style={styles.errorText}>{errors.password}</p>}
        </div>

        {/* Submit */}
        <button
          style={{ ...styles.button, ...(loading ? styles.buttonDisabled : {}) }}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "Signing in..." : "Sign In"}
        </button>

        <div style={styles.footer}>
          Don't have an account?{" "}
          <button
            type="button"
            style={{
              ...styles.link,
              background: "none",
              border: "none",
              padding: 0,
              cursor: "pointer",
              font: "inherit",
            }}
            onClick={() => { if (onGoToRegister) onGoToRegister(); }}
          >
            Register
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── LOGOUT BUTTON ────────────────────────────────────────────────────────────
// Drop this anywhere in your app (navbar, dashboard, etc.)
export function LogoutButton({ onLogout }) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogout = async () => {
    setLoading(true);
    setError("");

    try {
      const token = localStorage.getItem("access_token");

      // Optional: tell the backend to invalidate the token
      if (token) {
        await fetch(`${API_URL}/auth/logout`, {

          method: "POST",
          headers: {
            "Authorization": `Bearer ${token}`,
            "Content-Type": "application/json",
          },
        });
        // We don't block on the response — clear locally regardless
      }
    } catch (err) {
      // Non-critical: still log out locally even if server call fails
      setError("Server error, but you've been logged out locally.");
    } finally {
      localStorage.removeItem("access_token");
      setLoading(false);
      if (onLogout) onLogout();
    }
  };

  return (
    <div>
      {error && <p style={{ color: "#ff4444", fontSize: "0.8rem", marginBottom: "6px" }}>{error}</p>}
      <button
        style={{
          padding: "0.6rem 1.2rem",
          background: "transparent",
          border: "1px solid #333",
          borderRadius: "8px",
          color: "#888",
          fontSize: "0.875rem",
          cursor: loading ? "not-allowed" : "pointer",
          transition: "all 0.2s",
        }}
        onClick={handleLogout}
        disabled={loading}
        onMouseEnter={(e) => { e.target.style.borderColor = "#ff6400"; e.target.style.color = "#ff6400"; }}
        onMouseLeave={(e) => { e.target.style.borderColor = "#333"; e.target.style.color = "#888"; }}
      >
        {loading ? "Logging out..." : "Log Out"}
      </button>
    </div>
  );
}

// ─── DEFAULT EXPORT (for routing) ─────────────────────────────────────────────
export default LoginPage;
