import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./AuthContext";
import NavBar from "./NavBar";
import { LoginPage, LogoutButton } from "./pages/LoginPage";
import RegisterPage from "./pages/RegisterPage";
import DashboardPage from "./pages/DashboardPage";
import { ShotsTable } from "./pages/ShotsTable";
import SessionsPage from "./pages/SessionsPage";
import UploadPage from "./pages/UploadPage";
import SessionDetailPage from "./pages/SessionDetailPage";

function ProtectedRoute({ children }) {
  const { token } = useAuth();
  return token ? children : <Navigate to="/login" replace />;
}

function AppShell() {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/login" element={<LoginPage />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <DashboardPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/sessions"
          element={
            <ProtectedRoute>
              <SessionsPage />
            </ProtectedRoute>
          }
        />
        <Route
          path="/shots"
          element={
            <ProtectedRoute>
              <ShotsTable />
            </ProtectedRoute>
          }
        />
        <Route
          path="/upload"
          element={
            <ProtectedRoute>
              <UploadPage />
            </ProtectedRoute>
          }
        />
        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </>
  );
}

function App() {
  // Lazy initialiser reads localStorage once on mount — keeps user logged in on refresh
  const [token, setToken] = useState(() => localStorage.getItem("access_token"));
  const [page, setPage] = useState("login"); // "login" | "register"

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
  };

  // ── Protected routes: token present ──────────────────────────────────────────
  if (token) {
    return (
      <Routes>
        <Route path="/" element={<Dashboard onLogout={handleLogout} token={token} />} />
        <Route path="/upload" element={<UploadPage />} />
        <Route path="/sessions/:sessionId" element={<SessionDetailPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
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
    <BrowserRouter>
      <AuthProvider>
        <AppShell />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
