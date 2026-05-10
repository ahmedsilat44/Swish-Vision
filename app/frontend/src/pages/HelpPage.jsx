import { Link } from "react-router-dom";

const CARD_STYLE = {
  background: "#13131a",
  border: "1px solid #1e1e2e",
  borderRadius: "16px",
  padding: "1.5rem",
  width: "100%",
  maxWidth: "720px",
};

const SECTION_TITLE = {
  fontSize: "1rem",
  fontWeight: 700,
  color: "#fff",
  margin: "0 0 1rem",
  display: "flex",
  alignItems: "center",
  gap: "0.5rem",
};

const STEP_NUM = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  width: "24px",
  height: "24px",
  borderRadius: "50%",
  background: "linear-gradient(135deg, #ff6400, #ff9a00)",
  fontSize: "0.75rem",
  fontWeight: 700,
  flexShrink: 0,
  color: "#fff",
};

function Step({ num, title, desc }) {
  return (
    <div style={{ display: "flex", gap: "0.85rem", alignItems: "flex-start" }}>
      <span style={STEP_NUM}>{num}</span>
      <div>
        <p style={{ margin: "0 0 2px", fontWeight: 600, fontSize: "0.9rem", color: "#fff" }}>{title}</p>
        <p style={{ margin: 0, fontSize: "0.82rem", color: "#666", lineHeight: 1.6 }}>{desc}</p>
      </div>
    </div>
  );
}

function Tip({ icon, text }) {
  return (
    <div style={{ display: "flex", gap: "0.75rem", alignItems: "flex-start", padding: "0.75rem 1rem", borderRadius: "10px", background: "#0d0d14", border: "1px solid #1e1e2e" }}>
      <span style={{ fontSize: "1.1rem", flexShrink: 0 }}>{icon}</span>
      <p style={{ margin: 0, fontSize: "0.83rem", color: "#aaa", lineHeight: 1.6 }}>{text}</p>
    </div>
  );
}

function FAQ({ q, a }) {
  return (
    <div style={{ borderBottom: "1px solid #1e1e2e", paddingBottom: "1rem" }}>
      <p style={{ margin: "0 0 4px", fontWeight: 600, fontSize: "0.875rem", color: "#fff" }}>{q}</p>
      <p style={{ margin: 0, fontSize: "0.82rem", color: "#666", lineHeight: 1.6 }}>{a}</p>
    </div>
  );
}

export default function HelpPage() {
  return (
    <div
      style={{
        minHeight: "calc(100vh - 56px)",
        background: "#0a0a0f",
        color: "#fff",
        fontFamily: "'DM Sans', sans-serif",
        padding: "2.5rem 2rem",
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        gap: "1.5rem",
      }}
    >
      <div style={{ width: "100%", maxWidth: "720px" }}>
        <h1 style={{ fontSize: "1.6rem", fontWeight: 700, margin: "0 0 0.25rem", letterSpacing: "-0.5px" }}>
          Help & Recording Guidelines
        </h1>
        <p style={{ color: "#555", margin: 0, fontSize: "0.9rem" }}>
          Get the most accurate analysis from Swish Vision.
        </p>
      </div>

      <div style={CARD_STYLE}>
        <h2 style={SECTION_TITLE}>⚡ How It Works</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <Step num="1" title="Record your session" desc="Film yourself shooting from a fixed camera position. Keep the basket and your full body in frame." />
          <Step num="2" title="Upload the video" desc="Go to Upload, choose your video file (MP4, MOV, AVI, or MKV — up to 500 MB), and submit." />
          <Step num="3" title="Wait for analysis" desc="Swish Vision detects each shot attempt, tracks the ball and your pose, and computes release angles and outcomes. Processing time depends on video length." />
          <Step num="4" title="Review your results" desc="Open the session from the Sessions list to see shot-by-shot results, shooting percentage, angle charts, and AI feedback." />
        </div>
      </div>

      <div style={CARD_STYLE}>
        <h2 style={SECTION_TITLE}>📷 Camera Setup</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
          <Tip icon="📐" text="Position the camera at roughly basket height (about 3 m / 10 ft) or slightly below, angled so the full arc of the shot is visible." />
          <Tip icon="↔️" text="Shoot from the side (90° to the shooter). A side-on angle gives the most accurate elbow, shoulder, and release-angle measurements." />
          <Tip icon="🏀" text="Keep the hoop, backboard, and your full body — head to toe — in frame for every shot attempt." />
          <Tip icon="📏" text="Stand 3–6 m (10–20 ft) from the camera. Too close clips your body; too far reduces pose-detection accuracy." />
          <Tip icon="🔒" text="Use a tripod or fixed mount. Camera shake degrades ball and pose tracking significantly." />
        </div>
      </div>

      <div style={CARD_STYLE}>
        <h2 style={SECTION_TITLE}>💡 Lighting & Environment</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
          <Tip icon="☀️" text="Shoot in good, even lighting. Outdoor sessions work best in overcast daylight or shade — harsh direct sunlight causes blown-out highlights that confuse the detector." />
          <Tip icon="🌑" text="Avoid shooting into the sun or a bright window behind you. Backlit silhouettes reduce pose-tracking accuracy." />
          <Tip icon="🔆" text="For indoor gyms, ensure overhead lights are on. Dim gyms produce noise that makes ball detection harder." />
          <Tip icon="👕" text="Wear clothing that contrasts with the background. A dark shirt against a light wall (or vice versa) helps pose estimation." />
        </div>
      </div>

      <div style={CARD_STYLE}>
        <h2 style={SECTION_TITLE}>🎬 Video Quality</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.65rem" }}>
          <Tip icon="📹" text="Recommended resolution: 1080p (1920×1080) or 720p. Higher resolutions increase processing time with minimal accuracy gain." />
          <Tip icon="⏱️" text="Frame rate: 30 fps is ideal. 60 fps is supported but produces larger files. Avoid very low frame rates (< 24 fps)." />
          <Tip icon="✂️" text="Trim dead time (walking to retrieve the ball, long pauses) before uploading. Shorter videos process faster." />
          <Tip icon="🎯" text="Each shot attempt should be clearly separated. Wait 1–2 seconds between shots so the tracker can reset." />
        </div>
      </div>

      <div style={CARD_STYLE}>
        <h2 style={SECTION_TITLE}>📊 Understanding Your Results</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
          <div>
            <p style={{ margin: "0 0 2px", fontWeight: 600, fontSize: "0.875rem", color: "#fff" }}>Shot Percentage</p>
            <p style={{ margin: 0, fontSize: "0.82rem", color: "#666", lineHeight: 1.6 }}>Makes ÷ total attempts × 100. A "Pending" badge means the session is still processing or the report hasn't been generated yet.</p>
          </div>
          <div>
            <p style={{ margin: "0 0 2px", fontWeight: 600, fontSize: "0.875rem", color: "#fff" }}>Release Angle</p>
            <p style={{ margin: 0, fontSize: "0.82rem", color: "#666", lineHeight: 1.6 }}>The angle of your shooting arm at ball release, measured at the shoulder. Research suggests an optimal release angle of 45–55° produces the largest target window on the rim.</p>
          </div>
          <div>
            <p style={{ margin: "0 0 2px", fontWeight: 600, fontSize: "0.875rem", color: "#fff" }}>Elbow Angle</p>
            <p style={{ margin: 0, fontSize: "0.82rem", color: "#666", lineHeight: 1.6 }}>Elbow flexion at the moment of release. A fully extended elbow (close to 180°) at follow-through indicates good mechanics.</p>
          </div>
          <div>
            <p style={{ margin: "0 0 2px", fontWeight: 600, fontSize: "0.875rem", color: "#fff" }}>Session Statuses</p>
            <p style={{ margin: 0, fontSize: "0.82rem", color: "#666", lineHeight: 1.6 }}>
              <strong style={{ color: "#818cf8" }}>queued</strong> — waiting for a worker. &nbsp;
              <strong style={{ color: "#fbbf24" }}>processing</strong> — actively analyzing. &nbsp;
              <strong style={{ color: "#4ade80" }}>completed</strong> — results ready. &nbsp;
              <strong style={{ color: "#f87171" }}>failed</strong> — use the Retry button on the Sessions page.
            </p>
          </div>
        </div>
      </div>

      <div style={CARD_STYLE}>
        <h2 style={SECTION_TITLE}>❓ FAQ</h2>
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          <FAQ q="How long does processing take?" a="Typically 1–3 minutes for a 2-minute 1080p video, depending on server load. Check the session status badge — it updates automatically." />
          <FAQ q="My session shows 'failed'. What should I do?" a="Click Retry on the Sessions page. If it fails again, check that your video is a supported format (MP4, MOV, AVI, MKV) and that the ball and hoop are clearly visible." />
          <FAQ q="Shot percentage shows 'Pending' even though the session is completed." a="This can happen if no shots were detected (e.g., the ball or hoop was out of frame). Review your camera setup and re-upload." />
          <FAQ q="Can I upload multiple videos?" a="Yes — each upload creates a separate session. You can view all sessions on the Sessions page." />
          <FAQ q="Is my data private?" a="Videos and results are linked to your account only. No other users can see your sessions." />
        </div>
      </div>

      <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", justifyContent: "center" }}>
        <Link to="/upload" style={{ display: "inline-block", background: "linear-gradient(135deg, #ff6400, #ff9a00)", color: "#fff", padding: "12px 28px", borderRadius: "10px", fontSize: "0.95rem", fontWeight: 700, textDecoration: "none" }}>
          🏀 Upload a Video
        </Link>
        <Link to="/sessions" style={{ display: "inline-block", background: "#1e1e2e", color: "#fff", padding: "12px 28px", borderRadius: "10px", fontSize: "0.95rem", fontWeight: 700, textDecoration: "none", border: "1px solid #333" }}>
          View Sessions
        </Link>
      </div>
    </div>
  );
}