import { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../AuthContext';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api';

export default function UploadPage() {
  const { token } = useAuth();
  const navigate = useNavigate();
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('idle'); // idle | uploading | queued | failed
  const inputRef = useRef();
  const xhrRef = useRef(null);

  useEffect(() => {
    return () => { xhrRef.current?.abort(); };
  }, []);

  const ALLOWED_EXTS = ['.mp4', '.avi', '.mov', '.mkv'];
  const MAX_BYTES = 500 * 1024 * 1024;

  function handleFile(f) {
    if (!f) return;
    const dotIndex = f.name.lastIndexOf('.');
    if (dotIndex === -1 || dotIndex === f.name.length - 1) {
      setError('Unsupported file type. Allowed: .mp4, .avi, .mov, .mkv');
      setFile(null);
      return;
    }
    const ext = f.name.slice(dotIndex).toLowerCase();
    if (!ALLOWED_EXTS.includes(ext)) {
      setError('Unsupported file type. Allowed: .mp4, .avi, .mov, .mkv');
      setFile(null);
      return;
    }
    if (f.size > MAX_BYTES) {
      setError('File exceeds 500 MB limit');
      setFile(null);
      return;
    }
    setError('');
    setFile(f);
  }

  function handleDrop(e) {
    e.preventDefault();
    handleFile(e.dataTransfer.files[0]);
  }

  function handleDragOver(e) {
    e.preventDefault();
  }

  function upload() {
    setProgress(0);
    setError('');

    if (!token) {
      setError('You must be logged in to upload');
      setStatus('failed');
      return;
    }

    const xhr = new XMLHttpRequest();
    xhrRef.current = xhr;
    const fd = new FormData();
    fd.append('file', file);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status === 201) {
        JSON.parse(xhr.responseText);
        setStatus('queued');
        setProgress(100);
      } else if (xhr.status === 409) {
        setError('Cannot upload while another session is processing');
        setStatus('failed');
      } else {
        try {
          const data = JSON.parse(xhr.responseText);
          setError(data.detail || 'Upload failed');
        } catch {
          setError('Upload failed');
        }
        setStatus('failed');
      }
    };

    xhr.onerror = () => {
      setError('Network error during upload');
      setStatus('failed');
    };

    xhr.onabort = () => {};

    xhr.open('POST', `${API_URL}/sessions/upload`);
    xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    setStatus('uploading');
    xhr.send(fd);
  }

  return (
    <div style={styles.page}>
      <div style={styles.card}>
        <h2 style={styles.title}>Upload Practice Video</h2>

        {/* Drop Zone */}
        <div
          style={styles.dropZone}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => inputRef.current.click()}
        >
          {file ? (
            <p style={styles.fileName}>📹 {file.name}</p>
          ) : (
            <>
              <p style={styles.dropText}>Drag & drop video here</p>
              <p style={styles.dropSub}>or click to browse</p>
              <p style={styles.dropSub}>MP4, AVI, MOV, MKV · max 500 MB</p>
            </>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".mp4,.avi,.mov,.mkv"
            style={{ display: 'none' }}
            onChange={(e) => handleFile(e.target.files[0])}
          />
        </div>

        {/* Error */}
        {error && <p style={styles.error}>{error}</p>}

        {/* Progress Bar */}
        {status === 'uploading' && (
          <div style={styles.progressContainer}>
            <div style={{ ...styles.progressBar, width: `${progress}%` }} />
            <p style={styles.progressText}>{progress}%</p>
          </div>
        )}

        {/* Success */}
        {status === 'queued' && (
          <div style={styles.success}>
            <p>✅ Video queued for analysis!</p>
            <button style={styles.linkBtn} onClick={() => navigate('/sessions')}>
              View Sessions →
            </button>
          </div>
        )}

        {/* Upload Button */}
        {status !== 'queued' && (
          <button
            style={{
              ...styles.uploadBtn,
              opacity: !file || status === 'uploading' ? 0.5 : 1,
              cursor: !file || status === 'uploading' ? 'not-allowed' : 'pointer',
            }}
            onClick={upload}
            disabled={!file || status === 'uploading'}
          >
            {status === 'uploading' ? 'Uploading...' : 'Upload Video'}
          </button>
        )}
      </div>
    </div>
  );
}

const styles = {
  page: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#0a0a0f',
    padding: '20px',
    fontFamily: "'DM Sans', sans-serif",
  },
  card: {
    backgroundColor: '#13131a',
    border: '1px solid #1e1e2e',
    borderRadius: '12px',
    padding: '40px',
    width: '100%',
    maxWidth: '520px',
    boxShadow: 'none',
  },
  title: {
    marginBottom: '24px',
    fontSize: '22px',
    fontWeight: '700',
    color: '#fff',
  },
  dropZone: {
    border: '2px dashed #60a5fa',
    borderRadius: '10px',
    padding: '40px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    marginBottom: '16px',
    backgroundColor: 'rgba(96, 165, 250, 0.05)',
    transition: 'background 0.2s',
  },
  dropText: { fontSize: '16px', color: '#fff', margin: '0 0 6px' },
  dropSub: { fontSize: '13px', color: '#aaa', margin: '2px 0' },
  fileName: { fontSize: '15px', color: '#60a5fa', fontWeight: '600' },
  error: { color: '#f87171', fontSize: '14px', marginBottom: '12px' },
  progressContainer: {
    backgroundColor: '#1e1e2e',
    borderRadius: '8px',
    overflow: 'hidden',
    marginBottom: '16px',
    position: 'relative',
    height: '24px',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#60a5fa',
    transition: 'width 0.3s ease',
  },
  progressText: {
    position: 'absolute',
    top: '3px',
    width: '100%',
    textAlign: 'center',
    fontSize: '13px',
    color: '#fff',
    margin: 0,
  },
  success: { textAlign: 'center', marginBottom: '16px', color: '#4ade80' },
  linkBtn: {
    background: 'none',
    border: 'none',
    color: '#60a5fa',
    cursor: 'pointer',
    fontSize: '14px',
    textDecoration: 'underline',
  },
  uploadBtn: {
    width: '100%',
    padding: '14px',
    backgroundColor: 'linear-gradient(135deg, #ff6400, #ff9a00)',
    background: 'linear-gradient(135deg, #ff6400, #ff9a00)',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer',
  },
};
