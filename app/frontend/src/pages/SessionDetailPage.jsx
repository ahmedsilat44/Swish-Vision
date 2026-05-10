import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import Tooltip from '../components/Tooltip';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export default function SessionDetailPage() {
  const { sessionId } = useParams();
  const navigate = useNavigate();
  const [session, setSession] = useState(null);
  const [shots, setShots] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    const token = localStorage.getItem('access_token');
    if (!token) {
      navigate('/login');
      return;
    }

    const headers = { Authorization: `Bearer ${token}` };

    Promise.all([
      fetch(`${API_URL}/api/sessions/${sessionId}`, { headers }),
      fetch(`${API_URL}/api/sessions/${sessionId}/angles`, { headers }),
    ])
      .then(async ([sRes, aRes]) => {
        if (!sRes.ok) throw new Error('Session not found or access denied');
        if (!aRes.ok) throw new Error('Could not load angle data');
        const [sessionData, angleData] = await Promise.all([sRes.json(), aRes.json()]);
        setSession(sessionData);
        setShots(angleData.frames || []);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [sessionId, navigate]);

  function fmt(val) {
    if (val === null || val === undefined) return '—';
    return `${Number(val).toFixed(1)}°`;
  }

  if (loading) return <p style={styles.msg}>Loading…</p>;
  if (error) return <p style={{ ...styles.msg, color: '#e74c3c' }}>{error}</p>;

  return (
    <div style={styles.page}>
      <button style={styles.back} onClick={() => navigate('/')}>
        ← Dashboard
      </button>

      <h2 style={styles.title}>
        {session?.original_filename ?? `Session ${sessionId}`}
        <span style={{ ...styles.badge, ...statusColor(session?.status) }}>
          {session?.status}
        </span>
      </h2>

      {shots.length === 0 ? (
        <p className="info-msg">
          ⚠️ Pose data unavailable for this session. The player may have been out of frame
          or occluded during shot attempts.
        </p>
      ) : (
        <div style={styles.tableWrap}>
          <table style={styles.table}>
            <thead>
              <tr>
                <th style={styles.th}>Shot</th>
                <th style={styles.th}>Frame</th>
                <th style={styles.th}>
                  Elbow Angle
                  <Tooltip
                    text="The angle at your shooting elbow at the point of release. Consistent values across shots indicate repeatable form."
                    label="What is Elbow Angle?"
                  />
                </th>
                <th style={styles.th}>Shoulder Angle</th>
                <th style={styles.th}>Knee Angle</th>
                <th style={styles.th}>Outcome</th>
              </tr>
            </thead>
            <tbody>
              {shots.map((shot) => (
                <tr key={shot.frame_number} style={styles.row}>
                  <td style={styles.td}>{shot.shot_number}</td>
                  <td style={styles.td}>{shot.frame_number}</td>
                  <td style={styles.td}>{fmt(shot.elbow_angle)}</td>
                  <td style={styles.td}>{fmt(shot.shoulder_angle)}</td>
                  <td style={styles.td}>{fmt(shot.knee_angle)}</td>
                  <td style={styles.td}>
                    {shot.outcome ? (
                      <span style={shot.outcome === 'make' ? styles.make : styles.miss}>
                        {shot.outcome}
                      </span>
                    ) : (
                      '—'
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function statusColor(status) {
  switch (status) {
    case 'completed': return { background: '#d4edda', color: '#155724' };
    case 'failed':    return { background: '#f8d7da', color: '#721c24' };
    case 'processing':return { background: '#fff3cd', color: '#856404' };
    default:          return { background: '#e2e3e5', color: '#383d41' };
  }
}

const styles = {
  page: {
    minHeight: '100vh',
    backgroundColor: '#f0f2f5',
    padding: '32px 24px',
    fontFamily: 'sans-serif',
  },
  back: {
    background: 'none',
    border: 'none',
    color: '#4a90e2',
    fontSize: '14px',
    cursor: 'pointer',
    padding: 0,
    marginBottom: '16px',
    textDecoration: 'underline',
  },
  title: {
    fontSize: '22px',
    fontWeight: '700',
    color: '#1a1a2e',
    marginBottom: '24px',
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
  },
  badge: {
    fontSize: '12px',
    fontWeight: '600',
    padding: '3px 10px',
    borderRadius: '12px',
    textTransform: 'capitalize',
  },
  msg: {
    textAlign: 'center',
    marginTop: '60px',
    color: '#555',
    fontSize: '16px',
  },
  tableWrap: {
    overflowX: 'auto',
    borderRadius: '10px',
    boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
  },
  table: {
    width: '100%',
    borderCollapse: 'collapse',
    backgroundColor: '#fff',
    borderRadius: '10px',
    overflow: 'hidden',
  },
  th: {
    background: '#1a1a2e',
    color: '#fff',
    padding: '12px 16px',
    textAlign: 'left',
    fontSize: '13px',
    fontWeight: '600',
    letterSpacing: '0.5px',
  },
  td: {
    padding: '12px 16px',
    fontSize: '14px',
    color: '#333',
    borderBottom: '1px solid #f0f0f0',
  },
  row: { transition: 'background 0.15s' },
  make: {
    background: '#d4edda',
    color: '#155724',
    borderRadius: '6px',
    padding: '2px 8px',
    fontWeight: '600',
    fontSize: '12px',
  },
  miss: {
    background: '#f8d7da',
    color: '#721c24',
    borderRadius: '6px',
    padding: '2px 8px',
    fontWeight: '600',
    fontSize: '12px',
  },
};
