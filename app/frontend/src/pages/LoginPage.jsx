import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";

// ─── CONFIG ───────────────────────────────────────────────────────────────────
// Replace with your actual API base URL (from .env: REACT_APP_API_URL)
const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

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
export function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState("login"); // login | reset
  const [form, setForm] = useState({ email: "", password: "", newPassword: "", confirmNewPassword: "" });
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState("");
  const [apiSuccess, setApiSuccess] = useState("");
  const [loading, setLoading] = useState(false);

  const validate = () => {
    const e = {};
    if (!form.email.trim()) e.email = "Email is required";
    else if (!/\S+@\S+\.\S+/.test(form.email)) e.email = "Enter a valid email";
    if (!form.password) e.password = "Password is required";
    if (mode === "reset") {
      if (!form.newPassword) e.newPassword = "New password is required";
      else if (!(form.newPassword.length >= 8 && /[A-Za-z]/.test(form.newPassword) && /\d/.test(form.newPassword))) {
        e.newPassword = "Password must be at least 8 chars with letters and numbers";
      }
      if (!form.confirmNewPassword) e.confirmNewPassword = "Please confirm your new password";
      else if (form.newPassword !== form.confirmNewPassword) e.confirmNewPassword = "Passwords do not match";
      if (form.newPassword && form.password && form.newPassword === form.password) {
        e.newPassword = "New password must be different from current password";
      }
    }
    return e;
  };

  const handleChange = (e) => {
    setForm({ ...form, [e.target.name]: e.target.value });
    setErrors({ ...errors, [e.target.name]: "" });
    setApiError("");
    setApiSuccess("");
  };

  const handleSubmit = async () => {
    const e = validate();
    if (Object.keys(e).length) return setErrors(e);

    const normalizedEmail = form.email.trim().toLowerCase();

    setLoading(true);
    setApiError("");
    setApiSuccess("");

    try {
      if (mode === "login") {
        const res = await fetch(`${API_URL}/auth/login`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ email: normalizedEmail, password: form.password }),
        });

        const data = await res.json();

        if (!res.ok) {
          if (res.status === 401) setApiError("Invalid email or password.");
          else if (res.status === 403) setApiError("Your account has been suspended.");
          else if (res.status === 429) setApiError("Too many attempts. Please wait a moment.");
          else setApiError(data?.detail || "Something went wrong. Please try again.");
          return;
        }

        login(data.access_token);
        navigate("/dashboard", { replace: true });
        return;
      }

      const res = await fetch(`${API_URL}/auth/reset-password`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email: normalizedEmail,
          current_password: form.password,
          new_password: form.newPassword,
        }),
      });
      const data = await res.json().catch(() => ({}));

      if (!res.ok) {
        if (res.status === 401) setApiError("Current email/password is incorrect.");
        else setApiError(data?.detail || "Could not update password. Please try again.");
        return;
      }

      setApiSuccess(data?.message || "Password updated. Sign in with your new password.");
      setMode("login");
      setForm((prev) => ({
        ...prev,
        password: "",
        newPassword: "",
        confirmNewPassword: "",
      }));

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

        <h1 style={styles.heading}>{mode === "login" ? "Welcome back" : "Change password"}</h1>
        <p style={styles.subheading}>
          {mode === "login"
            ? "Sign in to your account to continue"
            : "Use your current login details, then set a new password"}
        </p>

        {/* API Error Banner */}
        {apiError && <div style={styles.alertError}>⚠ {apiError}</div>}
        {apiSuccess ? (
          <div
            style={{
              background: "rgba(34, 197, 94, 0.12)",
              border: "1px solid rgba(34, 197, 94, 0.35)",
              borderRadius: "10px",
              padding: "0.75rem 1rem",
              color: "#86efac",
              fontSize: "0.875rem",
              marginBottom: "1.2rem",
            }}
          >
            {apiSuccess}
          </div>
        ) : null}

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

        {mode === "reset" ? (
          <>
            <div style={styles.field}>
              <label htmlFor="newPassword" style={styles.label}>New Password</label>
              <input
                id="newPassword"
                style={{ ...styles.input, ...(errors.newPassword ? styles.inputError : {}) }}
                type="password"
                name="newPassword"
                placeholder="New password"
                value={form.newPassword}
                onChange={handleChange}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              />
              {errors.newPassword && <p style={styles.errorText}>{errors.newPassword}</p>}
            </div>

            <div style={styles.field}>
              <label htmlFor="confirmNewPassword" style={styles.label}>Confirm New Password</label>
              <input
                id="confirmNewPassword"
                style={{ ...styles.input, ...(errors.confirmNewPassword ? styles.inputError : {}) }}
                type="password"
                name="confirmNewPassword"
                placeholder="Confirm new password"
                value={form.confirmNewPassword}
                onChange={handleChange}
                onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
              />
              {errors.confirmNewPassword && <p style={styles.errorText}>{errors.confirmNewPassword}</p>}
            </div>
          </>
        ) : null}

        {/* Submit */}
        <button
          style={{ ...styles.button, ...(loading ? styles.buttonDisabled : {}) }}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? (mode === "login" ? "Signing in..." : "Updating password...") : (mode === "login" ? "Sign In" : "Update Password")}
        </button>

        <div style={{ ...styles.footer, marginTop: "0.75rem" }}>
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
            onClick={() => {
              setMode(mode === "login" ? "reset" : "login");
              setErrors({});
              setApiError("");
              setApiSuccess("");
              setForm((prev) => ({
                ...prev,
                password: "",
                newPassword: "",
                confirmNewPassword: "",
              }));
            }}
          >
            {mode === "login" ? "Forgot password? Change it here" : "Back to sign in"}
          </button>
        </div>

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
            onClick={() => navigate("/register")}
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
  const { logout } = useAuth();
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogout = async () => {
    setLoading(true);
    setError("");
    await logout();
    setLoading(false);
    if (onLogout) onLogout();
    navigate("/login", { replace: true });
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
