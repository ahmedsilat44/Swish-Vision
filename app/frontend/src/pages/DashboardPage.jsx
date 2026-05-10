import { useState, useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { useAuth } from "../AuthContext";

const API_URL = (process.env.REACT_APP_API_URL || 'http://localhost:8000').replace(/\/+$/, '');

function apiFetch(path) {
  const token = localStorage.getItem('access_token');
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  const normalizedBase =
    API_URL.endsWith('/api') && normalizedPath.startsWith('/api')
      ? API_URL.slice(0, -4)
      : API_URL;

  return fetch(`${normalizedBase}${normalizedPath}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export default function DashboardPage() {
  const { token } = useAuth();
  const [copied, setCopied] = useState(false);
  const [trendData, setTrendData] = useState([]);
  const [trendDataLoaded, setTrendDataLoaded] = useState(false);
  const chartRef = useRef(null);

  useEffect(() => {
    async function loadTrendData() {
      try {
        const res = await apiFetch('/api/dashboard/trends');
        if (res.ok) {
          const data = await res.json();
          setTrendData(data);
        }
      } catch (err) {
        console.error('Failed to load trend data:', err);
      } finally {
        setTrendDataLoaded(true);
      }
    }

    loadTrendData();
  }, []);

  useEffect(() => {
    if (trendDataLoaded && trendData.length >= 2 && chartRef.current) {
      renderTrendChart(trendData, chartRef);
    }
  }, [trendDataLoaded, trendData]);
  const [summary, setSummary] = useState(null);

  useEffect(() => {
    if (!token) return;
    apiFetch(`/api/dashboard/summary`, {
      headers: { Authorization: `Bearer ${token}` },
    })
      .then((r) => r.ok ? r.json() : null)
      .then((data) => setSummary(data))
      .catch(() => {});
  }, [token]);

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

      {/* Stats cards */}
      {summary && (
        <div
          style={{
            width: "100%",
            maxWidth: "640px",
            display: "grid",
            gridTemplateColumns: "repeat(2, 1fr)",
            gap: "0.75rem",
            marginTop: "0.5rem",
          }}
        >
          <StatCard
            label="Shot %"
            value={summary.shot_percentage != null ? `${summary.shot_percentage.toFixed(1)}%` : "—"}
          />
          <StatCard
            label="Total Shots"
            value={summary.total_shots ?? "—"}
          />
        </div>
      )}

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

      {/* Trend Chart Section */}
      {trendDataLoaded && (
        <div
          style={{
            width: "100%",
            maxWidth: "640px",
            marginTop: "2rem",
            background: "#13131a",
            border: "1px solid #1e1e2e",
            borderRadius: "14px",
            padding: "1.25rem 1.5rem",
          }}
        >
          <h2 style={{ fontSize: "1rem", fontWeight: 700, margin: "0 0 1rem", color: "#fff" }}>
            Shot % Over Time
          </h2>
          {trendData.length >= 2 ? (
            <div style={{ position: "relative", height: 300, width: "100%" }}>
              <canvas id="trend-chart" ref={chartRef} />
            </div>
          ) : (
            <div id="trend-empty" style={{ textAlign: "center", padding: "2rem 0" }}>
              <p style={{ color: "#aaa", margin: "0 0 1rem", fontSize: "0.95rem" }}>
                {trendData.length === 0
                  ? "Complete your first session to see your progress over time."
                  : "Complete more sessions to see your trend over time."}
              </p>
              <Link to="/upload" style={{ textDecoration: "none" }}>
                <button
                  style={{
                    padding: "0.75rem 1.5rem",
                    background: "linear-gradient(135deg, #ff6400, #ff9a00)",
                    border: "none",
                    borderRadius: "8px",
                    color: "#fff",
                    fontWeight: 700,
                    cursor: "pointer",
                    fontSize: "0.9rem",
                    fontFamily: "inherit",
                  }}
                >
                  Upload a Video
                </button>
              </Link>
            </div>
          )}
        </div>
      )}

      {/* Quick-nav buttons */}
      <div
        style={{
          width: "100%",
          maxWidth: "640px",
          display: "grid",
          gridTemplateColumns: "1fr 1fr 1fr",
          gap: "0.75rem",
          marginTop: "1.5rem",
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

function renderTrendChart(trendData, chartRef) {
  if (!trendData || trendData.length < 2 || !chartRef.current) {
    return;
  }

  // Calculate shot percentage for each trend point
  const labels = trendData.map((point) => {
    const date = new Date(point.created_at);
    return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  });

  const data = trendData.map((point) => {
    const percentage = point.total_shots > 0
      ? (point.makes / point.total_shots) * 100
      : 0;
    return percentage;
  });

  const ctx = chartRef.current.getContext('2d');

  // Destroy previous chart instance if it exists
  if (window.trend_chart_instance) {
    window.trend_chart_instance.destroy();
  }

  window.trend_chart_instance = new window.Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Shot %',
          data,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37, 99, 235, 0.08)',
          fill: true,
          borderWidth: 2.5,
          tension: 0.3,
          pointRadius: 4,
          pointBackgroundColor: '#2563eb',
          pointBorderColor: '#fff',
          pointBorderWidth: 1.5,
          pointHoverRadius: 5,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: 12,
          titleFont: { size: 12, weight: 'bold' },
          bodyFont: { size: 11 },
          displayColors: false,
          callbacks: {
            label: function (context) {
              return `${context.parsed.y.toFixed(1)}%`;
            },
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Date',
            font: { size: 12, weight: 600 },
            color: '#9ca3af',
          },
          ticks: {
            font: { size: 11 },
            color: '#6b7280',
          },
          grid: {
            color: 'rgba(107, 114, 128, 0.1)',
            drawBorder: false,
          },
        },
        y: {
          title: {
            display: true,
            text: 'Shot Percentage (%)',
            font: { size: 12, weight: 600 },
            color: '#9ca3af',
          },
          min: 0,
          max: 100,
          ticks: {
            font: { size: 11 },
            color: '#6b7280',
            stepSize: 20,
            callback: function (value) {
              return value + '%';
            },
          },
          grid: {
            color: 'rgba(107, 114, 128, 0.1)',
            drawBorder: false,
          },
        },
      },
    },
  });
function StatCard({ label, value }) {
  return (
    <div
      style={{
        background: "#13131a",
        border: "1px solid #1e1e2e",
        borderRadius: "12px",
        padding: "1rem",
        textAlign: "center",
      }}
    >
      <p
        style={{
          color: "#aaa",
          fontSize: "0.7rem",
          letterSpacing: "1px",
          textTransform: "uppercase",
          margin: "0 0 0.4rem",
        }}
      >
        {label}
      </p>
      <p style={{ color: "#ff9a00", fontSize: "1.4rem", fontWeight: 700, margin: 0 }}>
        {value}
      </p>
    </div>
  );
}
