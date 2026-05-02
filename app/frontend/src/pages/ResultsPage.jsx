import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from 'chart.js';
import { Doughnut } from 'react-chartjs-2';

ChartJS.register(ArcElement, Tooltip, Legend);

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

export default function ResultsPage() {
  const { id: sessionId } = useParams();
  const navigate = useNavigate();

  // 'loading' | 'processing' | 'queued' | 'failed' | 'completed'
  const [pageState, setPageState] = useState('loading');
  const [errorMsg, setErrorMsg] = useState('');
  const [report, setReport] = useState(null);
  const [shots, setShots] = useState([]);
  const pollTimer = useRef(null);

  useEffect(() => {
    loadSession();
    return () => clearTimeout(pollTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  async function loadSession() {
    if (!sessionId) {
      setPageState('failed');
      setErrorMsg('Invalid session ID.');
      return;
    }

    try {
      const res = await apiFetch(`/api/sessions/${sessionId}`);
      if (!res.ok) {
        setPageState('failed');
        setErrorMsg('Session not found or access denied.');
        return;
      }
      const session = await res.json();

      if (session.status === 'processing' || session.status === 'queued') {
        setPageState(session.status);
        pollTimer.current = setTimeout(loadSession, 5000);
        return;
      }
      if (session.status === 'failed') {
        setPageState('failed');
        setErrorMsg(session.error_message || 'An unexpected error occurred during processing.');
        return;
      }
      if (session.status === 'completed') {
        await loadAnalytics();
        return;
      }

      setPageState('failed');
      setErrorMsg(
        session.status
          ? `Unexpected session status: ${session.status}.`
          : 'Unexpected session status received.'
      );
    } catch {
      setPageState('failed');
      setErrorMsg('Network error — could not load session.');
    }
  }

  async function loadAnalytics() {
    try {
      const [reportRes, shotsRes] = await Promise.all([
        apiFetch(`/api/sessions/${sessionId}/report`),
        apiFetch(`/api/sessions/${sessionId}/shots`),
      ]);
      if (!reportRes.ok || !shotsRes.ok) {
        setPageState('failed');
        setErrorMsg('Failed to load analytics data.');
        return;
      }
      const reportData = await reportRes.json();
      const shotsData = await shotsRes.json();
      setReport(reportData);
      setShots(shotsData.shots);
      setPageState('completed');
    } catch {
      setPageState('failed');
      setErrorMsg('Failed to load analytics data.');
    }
  }

  if (pageState === 'loading') {
    return <CenteredCard><p style={s.muted}>Loading session…</p></CenteredCard>;
  }

  if (pageState === 'processing' || pageState === 'queued') {
    return (
      <CenteredCard>
        <div style={s.spinner} />
        <p style={s.processingText}>
          Your video is being analysed. This page will update automatically.
        </p>
      </CenteredCard>
    );
  }

  if (pageState === 'failed') {
    return (
      <CenteredCard>
        <div style={s.failedBanner}>
          <strong>Processing failed</strong>
          <p style={s.errorDetail}>{errorMsg}</p>
        </div>
        <button style={s.backBtn} onClick={() => navigate('/upload')}>
          ← Upload another video
        </button>
      </CenteredCard>
    );
  }

  // completed
  const shotPercentage = Number.isFinite(report?.shot_percentage) ? report.shot_percentage : 0;
  const shotsMade = Number(report?.shots_made) || 0;
  const shotsMissed = Number(report?.shots_missed) || 0;
  const totalShots = Number(report?.total_shots) || shots.length;

  const doughnutData = {
    labels: ['Made', 'Missed'],
    datasets: [
      {
        data: [shotsMade, shotsMissed],
        backgroundColor: ['#22c55e', '#ef4444'],
        borderWidth: 0,
      },
    ],
  };

  return (
    <div style={s.page}>
      {/* Hero stat */}
      <div style={s.heroCard}>
        <p style={s.heroLabel}>Shot Percentage</p>
        <p style={s.heroStat}>{shotPercentage.toFixed(1)}%</p>
        <p style={s.heroSub}>
          {shotsMade} made · {shotsMissed} missed · {totalShots} total
        </p>
      </div>

      {/* Doughnut chart */}
      <div style={s.card}>
        <h3 style={s.cardTitle}>Made / Missed</h3>
        <div style={s.chartWrap}>
          <Doughnut
            data={doughnutData}
            options={{ cutout: '70%', plugins: { legend: { position: 'bottom' } } }}
          />
        </div>
      </div>

      {/* Per-shot table */}
      <div style={s.card}>
        <h3 style={s.cardTitle}>Shot Details</h3>
        {shots.length === 0 ? (
          <p style={s.muted}>No shot data available.</p>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={s.table}>
              <thead>
                <tr>
                  <th style={s.th}>#</th>
                  <th style={s.th}>Outcome</th>
                  <th style={s.th}>Release Angle</th>
                  <th style={s.th}>Elbow Angle</th>
                </tr>
              </thead>
              <tbody>
                {shots.map((shot) => {
                  const outcome = shot.outcome || (shot.result === 'make' ? 'made' : shot.result === 'miss' ? 'missed' : 'missed');
                  return (
                    <tr key={shot.shot_number}>
                      <td style={s.td}>{shot.shot_number}</td>
                      <td style={{ ...s.td, color: outcome === 'made' ? '#22c55e' : '#ef4444', fontWeight: 600 }}>
                        {outcome}
                      </td>
                      <td style={s.td}>{shot.release_angle != null ? `${shot.release_angle.toFixed(1)}°` : '—'}</td>
                      <td style={s.td}>{shot.elbow_angle_at_release != null ? `${shot.elbow_angle_at_release.toFixed(1)}°` : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <button style={s.backBtn} onClick={() => navigate('/upload')}>
        ← Upload another video
      </button>
    </div>
  );
}

function CenteredCard({ children }) {
  return (
    <div style={s.centered}>
      <div style={{ ...s.card, textAlign: 'center', minWidth: 320, maxWidth: 480 }}>{children}</div>
    </div>
  );
}

const s = {
  page: {
    maxWidth: 760,
    margin: '0 auto',
    padding: '2rem 1rem',
    display: 'flex',
    flexDirection: 'column',
    gap: '1.5rem',
    fontFamily: "'DM Sans', system-ui, sans-serif",
  },
  centered: {
    minHeight: '70vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    background: '#fff',
    borderRadius: 12,
    padding: '1.5rem',
    boxShadow: '0 2px 12px rgba(0,0,0,0.07)',
  },
  heroCard: {
    background: '#1a1a2e',
    color: '#fff',
    borderRadius: 12,
    padding: '2rem',
    textAlign: 'center',
  },
  heroLabel: {
    margin: '0 0 0.25rem',
    fontSize: '0.85rem',
    textTransform: 'uppercase',
    letterSpacing: '1.5px',
    color: '#888',
  },
  heroStat: {
    margin: '0 0 0.5rem',
    fontSize: '3.5rem',
    fontWeight: 800,
    color: '#ff9a00',
    lineHeight: 1,
  },
  heroSub: {
    margin: 0,
    color: '#aaa',
    fontSize: '0.9rem',
  },
  chartWrap: {
    maxWidth: 280,
    margin: '0 auto',
  },
  cardTitle: {
    margin: '0 0 1rem',
    fontSize: '1rem',
    fontWeight: 700,
    color: '#1a1a2e',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    fontSize: '0.9rem',
  },
  th: {
    padding: '0.6rem 0.75rem',
    textAlign: 'left',
    background: '#f8f9fa',
    color: '#555',
    fontWeight: 600,
    borderBottom: '2px solid #e9ecef',
    whiteSpace: 'nowrap',
  },
  td: {
    padding: '0.6rem 0.75rem',
    borderBottom: '1px solid #f0f0f0',
    color: '#333',
  },
  spinner: {
    width: 48,
    height: 48,
    border: '5px solid #e5e7eb',
    borderTopColor: '#3b82f6',
    borderRadius: '50%',
    animation: 'spin 0.8s linear infinite',
    margin: '0 auto 1.25rem',
  },
  processingText: {
    color: '#555',
    fontSize: '0.95rem',
    margin: 0,
  },
  failedBanner: {
    background: '#fef2f2',
    border: '1px solid #fecaca',
    borderRadius: 8,
    padding: '1rem 1.25rem',
    marginBottom: '1rem',
    color: '#b91c1c',
    textAlign: 'left',
  },
  errorDetail: {
    margin: '0.5rem 0 0',
    fontSize: '0.875rem',
    color: '#dc2626',
  },
  backBtn: {
    background: 'none',
    border: '1px solid #d1d5db',
    borderRadius: 8,
    padding: '0.6rem 1.25rem',
    cursor: 'pointer',
    color: '#555',
    fontSize: '0.875rem',
    marginTop: '0.75rem',
    alignSelf: 'flex-start',
  },
  muted: {
    color: '#888',
    margin: 0,
  },
};

if (typeof document !== 'undefined' && !document.getElementById('swish-spin-style')) {
  const style = document.createElement('style');
  style.id = 'swish-spin-style';
  style.textContent = '@keyframes spin { to { transform: rotate(360deg); } }';
  document.head.appendChild(style);
}
