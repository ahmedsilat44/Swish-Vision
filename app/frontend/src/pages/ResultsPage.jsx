import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function apiFetch(path) {
  const token = localStorage.getItem('access_token');
  return fetch(`${API_URL}${path}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export default function ResultsPage() {
  const { id: sessionId } = useParams();
  const navigate = useNavigate();

  // 'loading' | 'processing' | 'queued' | 'failed' | 'completed'
  const [pageState, setPageState] = useState('loading');
  const [errorMsg, setErrorMsg] = useState('');
  const pollTimer = useRef(null);

  useEffect(() => {
    loadSession();
    return () => clearTimeout(pollTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  async function loadSession() {
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
        setPageState('completed');
      }
    } catch {
      setPageState('failed');
      setErrorMsg('Network error — could not load session.');
    }
  }

  if (pageState === 'loading') {
    return (
      <CenteredCard>
        <p style={s.muted}>Loading session…</p>
      </CenteredCard>
    );
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

  // completed placeholder — analytics will be added in subsequent tasks
  return (
    <CenteredCard>
      <p style={{ color: '#22c55e', fontWeight: 700, fontSize: '1.1rem', margin: '0 0 1rem' }}>
        Analysis complete!
      </p>
      <p style={s.muted}>Analytics are loading…</p>
      <button style={s.backBtn} onClick={() => navigate('/upload')}>
        ← Upload another video
      </button>
    </CenteredCard>
  );
}

function CenteredCard({ children }) {
  return (
    <div style={s.centered}>
      <div style={s.card}>{children}</div>
    </div>
  );
}

const s = {
  centered: {
    minHeight: '70vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
  },
  card: {
    background: '#fff',
    borderRadius: 12,
    padding: '2.5rem',
    boxShadow: '0 4px 20px rgba(0,0,0,0.08)',
    textAlign: 'center',
    minWidth: 320,
    maxWidth: 480,
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
