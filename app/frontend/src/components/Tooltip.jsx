import { useId } from "react";

export default function Tooltip({ text, label }) {
  const tooltipId = useId();
  const accessibleName = label || "What is this metric?";

  return (
    <span className="tooltip-wrap">
      <button
        type="button"
        className="tooltip-trigger"
        aria-label={accessibleName}
        aria-describedby={tooltipId}
      >
        <span className="tooltip-icon" aria-hidden="true">
          &#9432;
        </span>
      </button>
      <span id={tooltipId} className="tooltip-text" role="tooltip">
        {text}
      </span>
    </span>
  );
}
