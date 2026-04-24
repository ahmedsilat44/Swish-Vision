import { useState } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

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
    maxWidth: "440px",
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
  row: {
    display: "flex",
    gap: "1rem",
  },
  field: {
    marginBottom: "1.1rem",
    flex: 1,
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
  inputSuccess: {
    borderColor: "#22c55e",
  },
  errorText: {
    color: "#ff4444",
    fontSize: "0.75rem",
    marginTop: "4px",
  },
  successText: {
    color: "#22c55e",
    fontSize: "0.75rem",
    marginTop: "4px",
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
  strengthBar: {
    display: "flex",
    gap: "4px",
    marginTop: "6px",
  },
  strengthSegment: (active, color) => ({
    height: "3px",
    flex: 1,
    borderRadius: "2px",
    background: active ? color : "#1e1e2e",
    transition: "background 0.3s",
  }),
  strengthLabel: (color) => ({
    fontSize: "0.72rem",
    color: color,
    marginTop: "4px",
  }),
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
    background: "none",
    border: "none",
    padding: 0,
    font: "inherit",
    cursor: "pointer",
  },
};

// Password strength checker
function getStrength(password) {
  let score = 0;
  if (password.length >= 8) score++;
  if (/[A-Z]/.test(password)) score++;
  if (/[0-9]/.test(password)) score++;
  if (/[^A-Za-z0-9]/.test(password)) score++;
  const labels = ["", "Weak", "Fair", "Good", "Strong"];
  const colors = ["", "#ef4444", "#f97316", "#eab308", "#22c55e"];
  return { score, label: labels[score] || "", color: colors[score] || "#1e1e2e" };
}

export function RegisterPage({ onRegisterSuccess, onGoToLogin }) {
  const [form, setForm] = useState({
    firstName: "",
    lastName: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState({});
  const [apiError, setApiError] = useState("");
  const [loading, setLoading] = useState(false);
  const [touched, setTouched] = useState({});

  const strength = getStrength(form.password);

  const validateField = (name, value, allForm) => {
    switch (name) {
      case "firstName":
        if (!value.trim()) return "First name is required";
        if (value.trim().length < 2) return "At least 2 characters";
        return "";
      case "lastName":
        if (!value.trim()) return "Last name is required";
        if (value.trim().length < 2) return "At least 2 characters";
        return "";
      case "email":
        if (!value.trim()) return "Email is required";
        if (!/\S+@\S+\.\S+/.test(value)) return "Enter a valid email";
        return "";
      case "password":
        if (!value) return "Password is required";
        if (value.length < 8) return "At least 8 characters";
        if (!/[0-9]/.test(value)) return "Must contain at least one number";
        return "";
      case "confirmPassword":
        if (!value) return "Please confirm your password";
        if (value !== (allForm || form).password) return "Passwords do not match";
        return "";
      default:
        return "";
    }
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    const updatedForm = { ...form, [name]: value };
    setForm(updatedForm);
    setApiError("");
    if (touched[name]) {
      setErrors((prev) => ({
        ...prev,
        [name]: validateField(name, value, updatedForm),
        // Re-validate confirmPassword live if password changes
        ...(name === "password" && touched.confirmPassword
          ? { confirmPassword: validateField("confirmPassword", updatedForm.confirmPassword, updatedForm) }
          : {}),
      }));
    }
  };

  const handleBlur = (e) => {
    const { name, value } = e.target;
    setTouched((prev) => ({ ...prev, [name]: true }));
    setErrors((prev) => ({ ...prev, [name]: validateField(name, value) }));
  };

  const validateAll = () => {
    const e = {};
    Object.keys(form).forEach((key) => {
      const err = validateField(key, form[key]);
      if (err) e[key] = err;
    });
    return e;
  };

  const handleSubmit = async () => {
    const e = validateAll();
    setTouched({ firstName: true, lastName: true, email: true, password: true, confirmPassword: true });
    if (Object.keys(e).length) return setErrors(e);

    setLoading(true);
    setApiError("");

    try {
      const res = await fetch(`${API_URL}/auth/register`, {

        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: `${form.firstName.trim()} ${form.lastName.trim()}`.trim(),
          email: form.email.trim().toLowerCase(),
          password: form.password,
        }),
      });

      const data = await res.json();

      if (!res.ok) {
        if (res.status === 409) setApiError("An account with this email already exists.");
        else if (res.status === 429) setApiError("Too many attempts. Please wait a moment.");
        else setApiError(data?.detail || "Something went wrong. Please try again.");
        return;
      }

      if (data?.access_token) {
        localStorage.setItem("access_token", data.access_token);
      }
      if (onRegisterSuccess) onRegisterSuccess(data);

    } catch (err) {
      setApiError("Cannot connect to server. Check your connection.");
    } finally {
      setLoading(false);
    }
  };

  const getInputStyle = (name) => ({
    ...styles.input,
    ...(errors[name] ? styles.inputError : {}),
    ...(touched[name] && !errors[name] && form[name] ? styles.inputSuccess : {}),
  });

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

        <h1 style={styles.heading}>Create account</h1>
        <p style={styles.subheading}>Start analyzing your shooting form today</p>

        {apiError && <div style={styles.alertError}>⚠ {apiError}</div>}

        {/* Name Row */}
        <div style={styles.row}>
          <div style={styles.field}>
            <label htmlFor="firstName" style={styles.label}>First Name</label>
            <input
              id="firstName"

              style={getInputStyle("firstName")}
              type="text"
              name="firstName"
              placeholder="Sameer"
              value={form.firstName}
              onChange={handleChange}
              onBlur={handleBlur}
            />
            {errors.firstName && <p style={styles.errorText}>{errors.firstName}</p>}
          </div>
          <div style={styles.field}>
            <label htmlFor="lastName" style={styles.label}>Last Name</label>
            <input
              id="lastName"
              style={getInputStyle("lastName")}
              type="text"
              name="lastName"
              placeholder="Hassan"
              value={form.lastName}
              onChange={handleChange}
              onBlur={handleBlur}
            />
            {errors.lastName && <p style={styles.errorText}>{errors.lastName}</p>}
          </div>
        </div>

        {/* Email */}
        <div style={styles.field}>
          <label htmlFor="email" style={styles.label}>Email</label>
          <input
            id="email"
            style={getInputStyle("email")}
            type="email"
            name="email"
            placeholder="you@example.com"
            value={form.email}
            onChange={handleChange}
            onBlur={handleBlur}
          />
          {errors.email && <p style={styles.errorText}>{errors.email}</p>}
        </div>

        {/* Password */}
        <div style={styles.field}>
          <label htmlFor="password" style={styles.label}>Password</label>
          <input
            id="password"
            style={getInputStyle("password")}
            type="password"
            name="password"
            placeholder="••••••••"
            value={form.password}
            onChange={handleChange}
            onBlur={handleBlur}
          />
          {/* Strength bar — only show once user starts typing */}
          {form.password && (
            <>
              <div style={styles.strengthBar}>
                {[1, 2, 3, 4].map((i) => (
                  <div key={i} style={styles.strengthSegment(i <= strength.score, strength.color)} />
                ))}
              </div>
              <p style={styles.strengthLabel(strength.color)}>{strength.label}</p>
            </>
          )}
          {errors.password && <p style={styles.errorText}>{errors.password}</p>}
        </div>

        {/* Confirm Password */}
        <div style={styles.field}>
          <label htmlFor="confirmPassword" style={styles.label}>Confirm Password</label>
          <input
            id="confirmPassword"
            style={getInputStyle("confirmPassword")}
            type="password"
            name="confirmPassword"
            placeholder="••••••••"
            value={form.confirmPassword}
            onChange={handleChange}
            onBlur={handleBlur}
            onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          />
          {errors.confirmPassword && <p style={styles.errorText}>{errors.confirmPassword}</p>}
          {touched.confirmPassword && !errors.confirmPassword && form.confirmPassword && (
            <p style={styles.successText}>✓ Passwords match</p>
          )}
        </div>

        <button
          style={{ ...styles.button, ...(loading ? styles.buttonDisabled : {}) }}
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? "Creating account..." : "Create Account"}
        </button>

        <div style={styles.footer}>
          Already have an account?{" "}
          <button
            type="button"
            style={styles.link}
            onClick={() => { if (onGoToLogin) onGoToLogin(); }}
          >
            Sign in
          </button>
        </div>
      </div>
    </div>
  );
}

export default RegisterPage;
