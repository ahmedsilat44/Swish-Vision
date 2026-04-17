import { useState, useRef } from 'react';
import { useNavigate } from 'react-router-dom';

export default function UploadPage() {
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('idle'); // idle | uploading | queued | failed
  const inputRef = useRef();
  const navigate = useNavigate();

  const ALLOWED_EXTS = ['.mp4', '.avi', '.mov', '.mkv'];
  const MAX_BYTES = 500 * 1024 * 1024;

  function handleFile(f) {
    if (!f) return;
    const ext = f.name.slice(f.name.lastIndexOf('.')).toLowerCase();
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
    const token = localStorage.getItem('token');
    if (!token) {
      setError('You must be logged in to upload');
      setStatus('failed');
      return;
    }

    const xhr = new XMLHttpRequest();
    const fd = new FormData();
    fd.append('file', file);

    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) {
        setProgress(Math.round((e.loaded / e.total) * 100));
      }
    };

    xhr.onload = () => {
      if (xhr.status === 201) {
        const data = JSON.parse(xhr.responseText);
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

    xhr.open('POST', '/api/sessions/upload');
    xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.send(fd);
    setStatus('uploading');
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
    backgroundColor: '#f0f2f5',
    padding: '20px',
  },
  card: {
    backgroundColor: '#fff',
    borderRadius: '12px',
    padding: '40px',
    width: '100%',
    maxWidth: '520px',
    boxShadow: '0 4px 20px rgba(0,0,0,0.1)',
  },
  title: {
    marginBottom: '24px',
    fontSize: '22px',
    fontWeight: '700',
    color: '#1a1a2e',
  },
  dropZone: {
    border: '2px dashed #4a90e2',
    borderRadius: '10px',
    padding: '40px 20px',
    textAlign: 'center',
    cursor: 'pointer',
    marginBottom: '16px',
    backgroundColor: '#f8fbff',
    transition: 'background 0.2s',
  },
  dropText: { fontSize: '16px', color: '#333', margin: '0 0 6px' },
  dropSub: { fontSize: '13px', color: '#888', margin: '2px 0' },
  fileName: { fontSize: '15px', color: '#4a90e2', fontWeight: '600' },
  error: { color: '#e74c3c', fontSize: '14px', marginBottom: '12px' },
  progressContainer: {
    backgroundColor: '#eee',
    borderRadius: '8px',
    overflow: 'hidden',
    marginBottom: '16px',
    position: 'relative',
    height: '24px',
  },
  progressBar: {
    height: '100%',
    backgroundColor: '#4a90e2',
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
  success: { textAlign: 'center', marginBottom: '16px', color: '#27ae60' },
  linkBtn: {
    background: 'none',
    border: 'none',
    color: '#4a90e2',
    cursor: 'pointer',
    fontSize: '14px',
    textDecoration: 'underline',
  },
  uploadBtn: {
    width: '100%',
    padding: '14px',
    backgroundColor: '#4a90e2',
    color: '#fff',
    border: 'none',
    borderRadius: '8px',
    fontSize: '16px',
    fontWeight: '600',
    cursor: 'pointer',
  },
};
