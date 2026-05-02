import { createContext, useContext, useState, useCallback } from "react";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000/api";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setToken] = useState(() => localStorage.getItem("access_token"));

  const login = useCallback((accessToken) => {
    localStorage.setItem("access_token", accessToken);
    setToken(accessToken);
  }, []);

  const logout = useCallback(async () => {
    try {
      const t = localStorage.getItem("access_token");
      if (t) {
        await fetch(`${API_URL}/auth/logout`, {
          method: "POST",
          headers: {
            Authorization: `Bearer ${t}`,
            "Content-Type": "application/json",
          },
        });
      }
    } catch (_) {
      // Non-critical: clear locally regardless
    } finally {
      localStorage.removeItem("access_token");
      setToken(null);
    }
  }, []);

  return (
    <AuthContext.Provider value={{ token, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
