import { useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, Outlet } from "react-router-dom";
import { LoginPage, LogoutButton } from "./pages/LoginPage";
import { RegisterPage } from "./pages/RegisterPage";
import UploadPage from "./pages/UploadPage";
import ResultsPage from "./pages/ResultsPage";

function AppLayout({ onLogout }) {
  return (
    <div>
      <nav style={navStyle}>
        <span style={navBrand}>🏀 Swish Vision</span>
        <LogoutButton onLogout={onLogout} />
      </nav>
      <Outlet />
    </div>
  );
}

const navStyle = {
  display: "flex",
  alignItems: "center",
  justifyContent: "space-between",
  padding: "0.75rem 1.5rem",
  background: "#0a0a0f",
  borderBottom: "1px solid #1e1e2e",
};

const navBrand = {
  color: "#fff",
  fontWeight: "700",
  fontSize: "1rem",
  fontFamily: "'DM Sans', sans-serif",
};

function App() {
  const [token, setToken] = useState(() => localStorage.getItem("access_token"));
  const [authPage, setAuthPage] = useState("login"); // "login" | "register"

  const handleLoginSuccess = (data) => {
    setToken(data.access_token);
  };

  const handleRegisterSuccess = (data) => {
    if (data?.access_token) {
      setToken(data.access_token);
    } else {
      setAuthPage("login");
    }
  };

  const handleLogout = () => {
    setToken(null);
    setAuthPage("login");
  };

  if (token) {
    return (
      <BrowserRouter>
        <Routes>
          <Route element={<AppLayout onLogout={handleLogout} />}>
            <Route path="/upload" element={<UploadPage />} />
            <Route path="/sessions/:id" element={<ResultsPage />} />
            <Route path="*" element={<Navigate to="/upload" replace />} />
          </Route>
        </Routes>
      </BrowserRouter>
    );
  }

  if (authPage === "register") {
    return (
      <RegisterPage
        onRegisterSuccess={handleRegisterSuccess}
        onGoToLogin={() => setAuthPage("login")}
      />
    );
  }

  return (
    <LoginPage
      onLoginSuccess={handleLoginSuccess}
      onGoToRegister={() => setAuthPage("register")}
    />
  );
}

export default App;
