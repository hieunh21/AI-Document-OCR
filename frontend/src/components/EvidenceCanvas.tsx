import type { BBox, OCRSpan } from "../api/types";

interface Props {
  width: number;
  height: number;
  spans: OCRSpan[];
  highlight?: BBox | null;
  imageUrl?: string | null;
}

/**
 * Draws OCR boxes over the receipt in *original-image* coordinates.
 * The SVG viewBox equals the original size, so boxes stay aligned at any
 * rendered size (the browser scales image and overlay together).
 * Phase 0 has no image yet: the boxes are drawn on a blank page.
 */
export function EvidenceCanvas({ width, height, spans, highlight, imageUrl }: Props) {
  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      style={{ width: "100%", height: "auto", background: "#fafafa", border: "1px solid #eee" }}
      role="img"
      aria-label="Receipt with OCR regions"
    >
      {imageUrl && <image href={imageUrl} x={0} y={0} width={width} height={height} />}
      {spans.map((s) => (
        <g key={s.id}>
          <rect
            x={s.bbox[0]}
            y={s.bbox[1]}
            width={s.bbox[2] - s.bbox[0]}
            height={s.bbox[3] - s.bbox[1]}
            fill="none"
            stroke="#91caff"
            strokeWidth={2}
          />
          {!imageUrl && (
            <text x={s.bbox[0] + 4} y={s.bbox[3] - 8} fontSize={22} fill="#595959">
              {s.text}
            </text>
          )}
        </g>
      ))}
      {highlight && (
        <rect
          x={highlight[0] - 6}
          y={highlight[1] - 6}
          width={highlight[2] - highlight[0] + 12}
          height={highlight[3] - highlight[1] + 12}
          fill="rgba(250, 173, 20, 0.15)"
          stroke="#fa8c16"
          strokeWidth={4}
        />
      )}
    </svg>
  );
}
