import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { Chart as ChartJS, ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, LineController } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';
import { Doughnut } from 'react-chartjs-2';
import './ResultsPage.css';
import MetricTooltip from '../components/Tooltip';

ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, PointElement, LineElement, LineController, annotationPlugin);

const API_URL = (process.env.REACT_APP_API_URL || 'http://localhost:8000/api').replace(/\/+$/, '');
const API_BASE = API_URL.endsWith('/api') ? API_URL.slice(0, -4) : API_URL;

function apiFetch(path) {
  const token = localStorage.getItem('access_token');
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;

  return fetch(`${API_BASE}${normalizedPath}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
}

export default function ResultsPage() {
  const { sessionId } = useParams();
  const navigate = useNavigate();

  // 'loading' | 'processing' | 'queued' | 'failed' | 'completed'
  const [pageState, setPageState] = useState('loading');
  const [errorMsg, setErrorMsg] = useState('');
  const [report, setReport] = useState(null);
  const [shots, setShots] = useState([]);
  const [angleFrames, setAngleFrames] = useState([]);
  const [analysisReady, setAnalysisReady] = useState(false);
  const [videoSrc, setVideoSrc] = useState(null);
  const chartRef = useRef(null);
  const chartInstance = useRef(null);
  const pollTimer = useRef(null);

  useEffect(() => {
    if (pageState !== 'completed') return;
    apiFetch(`/api/sessions/${sessionId}/video_token`)
      .then(r => r.ok ? r.json() : Promise.reject())
      .then(data => {
        setVideoSrc(`${API_BASE}/api/sessions/${sessionId}/output_video?video_token=${encodeURIComponent(data.video_token)}`);
      })
      .catch(() => {
        // Fallback: if token endpoint fails the video section will remain empty
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [pageState, sessionId]);

  useEffect(() => {
    if (pageState !== 'completed' || !report) {
      setAnalysisReady(false);
      return undefined;
    }

    const animationFrame = window.requestAnimationFrame(() => setAnalysisReady(true));
    return () => window.cancelAnimationFrame(animationFrame);
  }, [pageState, report]);

  useEffect(() => {
    if (chartInstance.current) {
      chartInstance.current.destroy();
      chartInstance.current = null;
    }

    if (angleFrames.length > 0 && chartRef.current) {
      renderElbowAngleChart(angleFrames, chartRef, chartInstance);
    }

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
        chartInstance.current = null;
      }
    };
  }, [angleFrames]);

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
      const status = session.status;
      const inProgressStatuses = new Set(['queued', 'processing', 'pending', 'uploading']);

      if (inProgressStatuses.has(status)) {
        clearTimeout(pollTimer.current);
        setPageState(status === 'queued' ? 'queued' : 'processing');
        setErrorMsg('');
        pollTimer.current = setTimeout(loadSession, 5000);
        return;
      }
      if (status === 'failed') {
        setPageState('failed');
        setErrorMsg(session.error_message || 'An unexpected error occurred during processing.');
        return;
      }
      if (status === 'completed') {
        await loadAnalytics();
        return;
      }

      // Continue polling for unknown non-terminal statuses instead of getting stuck on loading.
      clearTimeout(pollTimer.current);
      setPageState('processing');
      setErrorMsg(
        status
          ? `Unknown session status "${status}". Continuing to check for updates...`
          : 'Unknown session status received. Continuing to check for updates...'
      );
      pollTimer.current = setTimeout(loadSession, 5000);
    } catch {
      setPageState('failed');
      setErrorMsg('Network error — could not load session.');
    }
  }

  async function loadAnalytics() {
    try {
      const [reportRes, shotsRes, anglesRes] = await Promise.all([
        apiFetch(`/api/sessions/${sessionId}/report`),
        apiFetch(`/api/sessions/${sessionId}/shots`),
        apiFetch(`/api/sessions/${sessionId}/angles`),
      ]);
      if (!reportRes.ok || !shotsRes.ok) {
        setPageState('failed');
        setErrorMsg('Failed to load analytics data.');
        return;
      }
      const reportData = await reportRes.json();
      const shotsData = await shotsRes.json();
      const anglesData = anglesRes.ok ? await anglesRes.json() : { frames: [] };
      setReport(reportData);
      setShots(shotsData.shots);
      setAngleFrames(anglesData.frames || []);
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
    const processingMessage =
      errorMsg || 'Your video is being analysed. This page will update automatically.';

    return (
      <CenteredCard>
        <div style={s.spinner} />
        <p style={s.processingText}>{processingMessage}</p>
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
  const shotPercentage = getNullableNumber(report?.shot_percentage);
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
      {renderFormAnalysis(report, analysisReady, angleFrames, chartRef, sessionId)}

      {/* Hero stat */}
      <div style={s.heroCard}>
        <p style={s.heroLabel}>Shot Percentage</p>
        <p style={s.heroStat}>{shotPercentage != null ? `${shotPercentage.toFixed(1)}%` : 'N/A'}</p>
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
                  <th style={s.th}>
                    SEW Angle
                    <MetricTooltip
                      text="Shoulder-Elbow-Wrist angle at release. Indicates your arm form and shooting consistency. Ideal range is 65–75°."
                      label="What is SEW Angle?"
                    />
                  </th>
                  <th style={s.th}>
                    ESH Angle
                    <MetricTooltip
                      text="Elbow-Shoulder-Hip angle at release. Indicates your shooting arc. Ideal range is 120–135°."
                      label="What is ESH Angle?"
                    />
                  </th>
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
                      <td style={s.td}>{shot.sew_angle != null ? `${shot.sew_angle.toFixed(1)}°` : '—'}</td>
                      <td style={s.td}>{shot.esh_angle != null ? `${shot.esh_angle.toFixed(1)}°` : '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Annotated video player */}
      <div style={s.card}>
        <h3 style={s.cardTitle}>Annotated Video</h3>
        {videoSrc ? (
          <video key={videoSrc} controls width="100%" preload="metadata" style={s.video}>
            <source src={videoSrc} type="video/mp4" />
            Your browser does not support HTML5 video.
          </video>
        ) : (
          <p style={s.muted}>Preparing video…</p>
        )}
      </div>

      <button style={s.backBtn} onClick={() => navigate('/upload')}>
        ← Upload another video
      </button>
    </div>
  );
}

function renderFormAnalysis(report, animate, angleFrames, chartRef, sessionId) {
  const consistencyScore = getNullableNumber(report?.shot_percentage);
  const avgSewAngle = getNullableNumber(report?.avg_sew_angle);
  const avgEshAngle = getNullableNumber(report?.avg_esh_angle);
  const feedbackText = normalizeFeedbackText(report?.feedback_text);
  const consistencyLabel = getConsistencyLabel(consistencyScore);
  const sewTone = getAngleTone(avgSewAngle, 65, 75);
  const eshTone = getAngleTone(avgEshAngle, 120, 135);

  return (
    <div style={s.card}>
      <h3 style={s.cardTitle}>Form Analysis</h3>
      <div style={s.formAnalysisGrid}>
        <section style={{ ...s.analysisBlock, ...s.consistencyBlockFull }}>
          <div style={s.analysisHeader}>
            <span style={s.analysisLabel}>Consistency Score</span>
            <span style={s.analysisMeta}>{consistencyLabel}</span>
          </div>
          <div style={s.scoreRow}>
            <div style={s.scoreValue}>{consistencyScore != null ? consistencyScore.toFixed(0) : 'N/A'}</div>
            <div style={s.scoreBarWrap} aria-label="Consistency score bar">
              <div style={s.scoreBarTrack}>
                <div style={s.scoreBarFillContainer}>
                  <div
                    style={{
                      ...s.scoreBarFill,
                      width: consistencyScore != null && animate ? `${Math.max(0, Math.min(100, consistencyScore))}%` : '0%',
                    }}
                  />
                </div>
              </div>
            </div>
          </div>
          <p style={s.analysisHint}>Excellent (≥80), Good (≥60), Fair (≥40), Needs Work (&lt;40)</p>
        </section>

        <section style={s.analysisBlock}>
          <div style={s.analysisHeader}>
            <span style={s.analysisLabel}>Average SEW Angle</span>
            <span style={{ ...s.analysisMeta, color: sewTone.color }}>{sewTone.label}</span>
          </div>
          <div style={s.angleValueRow}>
            <div style={{ ...s.angleValue, color: sewTone.color }}>
              {avgSewAngle != null ? `${avgSewAngle.toFixed(1)}°` : 'N/A'}
            </div>
            <div style={s.angleBandWrap} aria-label="SEW angle reference band">
              <div style={s.angleScale}>
                <div
                  style={{
                    ...s.angleIdealBand,
                    left: `${(65 / 180) * 100}%`,
                    width: `${((75 - 65) / 180) * 100}%`,
                  }}
                />
                <div style={s.angleMarkerRail} />
                {avgSewAngle != null ? (
                  <div
                    style={{
                      ...s.angleMarker,
                      left: `${Math.max(0, Math.min(100, (avgSewAngle / 180) * 100))}%`,
                      opacity: animate ? 1 : 0,
                    }}
                  />
                ) : null}
              </div>
              <div style={s.angleScaleLabels}>
                <span>0°</span>
                <span>Ideal: 65–75°</span>
                <span>180°</span>
              </div>
            </div>
          </div>
        </section>

        <section style={s.analysisBlock}>
          <div style={s.analysisHeader}>
            <span style={s.analysisLabel}>Average ESH Angle</span>
            <span style={{ ...s.analysisMeta, color: eshTone.color }}>{eshTone.label}</span>
          </div>
          <div style={s.angleValueRow}>
            <div style={{ ...s.angleValue, color: eshTone.color }}>
              {avgEshAngle != null ? `${avgEshAngle.toFixed(1)}°` : 'N/A'}
            </div>
            <div style={s.angleBandWrap} aria-label="ESH angle reference band">
              <div style={s.angleScale}>
                <div
                  style={{
                    ...s.angleIdealBand,
                    left: `${(120 / 180) * 100}%`,
                    width: `${((135 - 120) / 180) * 100}%`,
                  }}
                />
                <div style={s.angleMarkerRail} />
                {avgEshAngle != null ? (
                  <div
                    style={{
                      ...s.angleMarker,
                      left: `${Math.max(0, Math.min(100, (avgEshAngle / 180) * 100))}%`,
                      opacity: animate ? 1 : 0,
                    }}
                  />
                ) : null}
              </div>
              <div style={s.angleScaleLabels}>
                <span>0°</span>
                <span>Ideal: 120–135°</span>
                <span>180°</span>
              </div>
            </div>
          </div>
        </section>

        <section style={{ ...s.analysisBlock, ...s.feedbackBlock }}>
          <div style={s.analysisHeader}>
            <span style={s.analysisLabel}>Feedback</span>
          </div>
          <p style={s.feedbackText}>{feedbackText}</p>
        </section>

        {angleFrames.length > 0 && (
          <section style={{ ...s.analysisBlock, ...s.chartBlock }}>
            <div style={s.analysisHeader}>
              <span style={s.analysisLabel}>Angles by Shot</span>
            </div>
            <div style={s.chartContainer}>
              <canvas id="angles-chart" ref={chartRef} width="400" height="200" />
            </div>
          </section>
        )}
      </div>
    </div>
  );
}

function getNullableNumber(value) {
  return Number.isFinite(value) ? value : null;
}

function normalizeFeedbackText(value) {
  if (typeof value !== 'string') {
    return 'No feedback available for this session.';
  }

  const text = value.trim();
  return text ? text : 'No feedback available for this session.';
}

function getConsistencyLabel(score) {
  if (score == null) return 'N/A';
  if (score >= 80) return 'Excellent';
  if (score >= 60) return 'Good';
  if (score >= 40) return 'Fair';
  return 'Needs Work';
}

function getAngleTone(angle, minThreshold, maxThreshold) {
  if (angle == null) {
    return { label: 'N/A', color: '#6b7280' };
  }

  if (angle >= minThreshold && angle <= maxThreshold) {
    return { label: 'Ideal', color: '#16a34a' };
  }

  if (angle >= minThreshold - 5 && angle <= maxThreshold + 5) {
    return { label: 'Near ideal', color: '#d97706' };
  }

  return { label: 'Outside range', color: '#dc2626' };
}

function renderElbowAngleChart(angleFrames, chartRef, chartInstance) {
  if (!angleFrames || angleFrames.length === 0 || !chartRef.current) {
    return;
  }

  // Group frames by shot_number
  const shotMap = {};
  angleFrames.forEach((frame) => {
    const shotNum = frame.shot_number;
    if (!shotMap[shotNum]) {
      shotMap[shotNum] = [];
    }
    shotMap[shotNum].push(frame);
  });

  // Get shot numbers in order
  const shotNumbers = Object.keys(shotMap)
    .map(Number)
    .sort((a, b) => a - b);

  // Calculate average SEW and ESH for each shot
  const sewData = [];
  const eshData = [];

  shotNumbers.forEach((shotNum) => {
    const frames = shotMap[shotNum];
    
    // Average SEW angle (elbow_angle)
    const sewValues = frames.filter(f => f.elbow_angle != null).map(f => f.elbow_angle);
    const sewAvg = sewValues.length > 0 ? sewValues.reduce((a, b) => a + b, 0) / sewValues.length : null;
    sewData.push(sewAvg);

    // Average ESH angle (shoulder_angle)
    const eshValues = frames.filter(f => f.shoulder_angle != null).map(f => f.shoulder_angle);
    const eshAvg = eshValues.length > 0 ? eshValues.reduce((a, b) => a + b, 0) / eshValues.length : null;
    eshData.push(eshAvg);
  });

  // Create datasets: one for SEW, one for ESH
  const datasets = [
    {
      label: 'SEW Angle (65–75°)',
      data: sewData,
      borderColor: '#3b82f6',
      backgroundColor: '#3b82f608',
      borderWidth: 2.5,
      tension: 0.3,
      spanGaps: true,
      pointRadius: 4,
      pointBackgroundColor: '#3b82f6',
      pointBorderColor: '#fff',
      pointBorderWidth: 1.5,
    },
    {
      label: 'ESH Angle (120–135°)',
      data: eshData,
      borderColor: '#f59e0b',
      backgroundColor: '#f59e0b08',
      borderWidth: 2.5,
      tension: 0.3,
      spanGaps: true,
      pointRadius: 4,
      pointBackgroundColor: '#f59e0b',
      pointBorderColor: '#fff',
      pointBorderWidth: 1.5,
    },
  ];

  // Generate x-axis labels (shot numbers)
  const xLabels = shotNumbers.map((n) => `Shot ${n}`);

  const ctx = chartRef.current.getContext('2d');
  
  // Destroy previous chart instance if it exists
  if (chartInstance.current) {
    chartInstance.current.destroy();
  }

  chartInstance.current = new ChartJS(ctx, {
    type: 'line',
    data: {
      labels: xLabels,
      datasets,
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            usePointStyle: true,
            padding: 15,
            font: { size: 12, weight: 500 },
            color: '#374151',
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Shot Number',
            font: { size: 13, weight: 600 },
            color: '#111827',
          },
          ticks: {
            font: { size: 11 },
            color: '#6b7280',
          },
          grid: {
            color: 'rgba(107, 114, 128, 0.1)',
          },
        },
        y: {
          title: {
            display: true,
            text: 'Angle (°)',
            font: { size: 13, weight: 600 },
            color: '#111827',
          },
          min: 50,
          max: 180,
          ticks: {
            font: { size: 11 },
            color: '#6b7280',
            stepSize: 20,
          },
          grid: {
            color: 'rgba(107, 114, 128, 0.1)',
          },
        },
      },
    },
  });
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
  formAnalysisGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
    gap: '1rem',
  },
  analysisBlock: {
    border: '1px solid #e5e7eb',
    borderRadius: 12,
    padding: '1rem 1.1rem',
    background: 'linear-gradient(180deg, #ffffff 0%, #fbfcfe 100%)',
  },
  consistencyBlockFull: {
    gridColumn: '1 / -1',
  },
  analysisHeader: {
    display: 'flex',
    alignItems: 'baseline',
    justifyContent: 'space-between',
    gap: '0.75rem',
    marginBottom: '0.85rem',
  },
  analysisLabel: {
    fontSize: '0.95rem',
    fontWeight: 700,
    color: '#111827',
  },
  analysisMeta: {
    fontSize: '0.8rem',
    fontWeight: 700,
    color: '#6b7280',
    textTransform: 'uppercase',
    letterSpacing: '0.08em',
  },
  scoreRow: {
    display: 'grid',
    gridTemplateColumns: 'auto 1fr',
    gap: '0.9rem',
    alignItems: 'center',
  },
  scoreValue: {
    fontSize: '2.1rem',
    fontWeight: 800,
    lineHeight: 1,
    color: '#111827',
    minWidth: 88,
  },
  scoreBarWrap: {
    width: '100%',
  },
  scoreBarTrack: {
    position: 'relative',
    width: '100%',
    height: 16,
    borderRadius: 999,
    overflow: 'hidden',
    background: '#e5e7eb',
    boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.08)',
  },
  scoreBarFillContainer: {
    position: 'absolute',
    inset: 0,
  },
  scoreBarFill: {
    height: '100%',
    borderRadius: 999,
    background: 'linear-gradient(90deg, #ef4444 0%, #f59e0b 50%, #22c55e 100%)',
    transition: 'width 800ms cubic-bezier(0.22, 1, 0.36, 1)',
  },
  analysisHint: {
    margin: '0.75rem 0 0',
    fontSize: '0.82rem',
    color: '#6b7280',
  },
  angleValueRow: {
    display: 'grid',
    gap: '0.85rem',
  },
  angleValue: {
    fontSize: '2rem',
    fontWeight: 800,
    lineHeight: 1,
  },
  angleBandWrap: {
    display: 'grid',
    gap: '0.45rem',
  },
  angleScale: {
    position: 'relative',
    height: 18,
    borderRadius: 999,
    background: 'linear-gradient(90deg, #fee2e2 0%, #fef3c7 35%, #dcfce7 100%)',
    overflow: 'hidden',
  },
  angleIdealBand: {
    position: 'absolute',
    top: 2,
    bottom: 2,
    borderRadius: 999,
    background: 'rgba(34, 197, 94, 0.28)',
    border: '1px solid rgba(34, 197, 94, 0.55)',
  },
  angleMarkerRail: {
    position: 'absolute',
    inset: '50% 0 auto 0',
    height: 2,
    transform: 'translateY(-50%)',
    background: 'rgba(17, 24, 39, 0.12)',
  },
  angleMarker: {
    position: 'absolute',
    top: '50%',
    width: 14,
    height: 14,
    borderRadius: '50%',
    transform: 'translate(-50%, -50%)',
    background: '#111827',
    boxShadow: '0 0 0 4px rgba(17, 24, 39, 0.14)',
    transition: 'left 800ms cubic-bezier(0.22, 1, 0.36, 1), opacity 250ms ease',
  },
  angleScaleLabels: {
    display: 'flex',
    justifyContent: 'space-between',
    gap: '0.75rem',
    fontSize: '0.78rem',
    color: '#6b7280',
  },
  feedbackBlock: {
    gridColumn: '1 / -1',
  },
  feedbackText: {
    margin: 0,
    color: '#374151',
    fontSize: '0.95rem',
    lineHeight: 1.6,
    whiteSpace: 'pre-wrap',
  },
  chartBlock: {
    gridColumn: '1 / -1',
  },
  chartContainer: {
    position: 'relative',
    width: '100%',
    minHeight: 300,
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
  video: {
    borderRadius: 8,
    background: '#000',
    display: 'block',
  },
  muted: {
    color: '#888',
    margin: 0,
  },
};
