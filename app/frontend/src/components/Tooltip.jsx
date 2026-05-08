export default function Tooltip({ text, label }) {
  return (
    <span className="tooltip-wrap" tabIndex={0}>
      <span className="tooltip-icon" aria-label={label || 'What is this metric?'}>
        &#9432;
      </span>
      <span className="tooltip-text" role="tooltip">
        {text}
      </span>
    </span>
  );
}
