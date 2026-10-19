import type { BBox, OCRSpan } from "../api/types";

interface Props {
  width: number;
  height: number;
  imageUrl?: string | null;
  spans: OCRSpan[];
  /** Draw every OCR box (faint), not only the highlighted evidence. */
  showSpans?: boolean;
  /** Evidence regions of the selected field. */
  highlights?: BBox[];
  /** OCR span ids to emphasise (e.g. hovered in the OCR text list). */
  activeSpanIds?: string[];
  onSpanClick?: (span: OCRSpan) => void;
}

/**
 * Receipt image with OCR boxes, in *original-image* coordinates.
 *
 * The SVG viewBox equals the image size, so the browser scales image and boxes
 * together: they stay aligned at any rendered width. The backend serves the image
 * already turned upright (EXIF), the same frame the OCR boxes use.
 */
export function EvidenceCanvas({
  width,
  height,
  imageUrl,
  spans,
  showSpans = false,
  highlights = [],
  activeSpanIds = [],
  onSpanClick,
}: Props) {
  const active = new Set(activeSpanIds);
  const stroke = Math.max(2, Math.round(width / 300));
  const pad = stroke * 2;
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", display: "block", background: "#fafafa" }}
      role="img"
      aria-label="Receipt with OCR regions"
    >
      {imageUrl && <image href={imageUrl} x={0} y={0} width={width} height={height} />}
      {spans.map((s) => {
        const on = active.has(s.id);
        if (!showSpans && !on) return null;
        return (
          <rect
            key={s.id}
            x={s.bbox[0]}
            y={s.bbox[1]}
            width={s.bbox[2] - s.bbox[0]}
            height={s.bbox[3] - s.bbox[1]}
            fill={on ? "rgba(22, 119, 255, 0.18)" : "rgba(22, 119, 255, 0.04)"}
            stroke={on ? "#1677ff" : "#91caff"}
            strokeWidth={on ? stroke * 1.5 : stroke}
            style={{ cursor: onSpanClick ? "pointer" : undefined }}
            onClick={onSpanClick ? () => onSpanClick(s) : undefined}
          >
            <title>{s.text}</title>
          </rect>
        );
      })}
      {highlights.map((b, i) => (
        <rect
          key={i}
          x={b[0] - pad}
          y={b[1] - pad}
          width={b[2] - b[0] + 2 * pad}
          height={b[3] - b[1] + 2 * pad}
          fill="rgba(250, 173, 20, 0.2)"
          stroke="#fa8c16"
          strokeWidth={stroke * 2}
          rx={stroke}
          pointerEvents="none"
        />
      ))}
    </svg>
  );
}
