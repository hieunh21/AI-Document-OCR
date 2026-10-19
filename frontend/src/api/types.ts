// Mirrors ai/schemas.py and backend/app/schemas/documents.py — keep in sync.

export type FieldName = "seller" | "seller_address" | "timestamp" | "total_cost";
export const FIELD_NAMES: FieldName[] = ["seller", "seller_address", "timestamp", "total_cost"];

export type DocumentStatus =
  | "uploaded"
  | "processing"
  | "needs_review"
  | "ready"
  | "approved"
  | "failed";

export type BBox = [number, number, number, number]; // x_min, y_min, x_max, y_max (original pixels)

export interface OCRSpan {
  id: string;
  text: string;
  bbox: BBox;
  polygon?: [number, number][] | null;
  confidence?: number | null;
}

export interface OCRResult {
  image_width: number;
  image_height: number;
  spans: OCRSpan[];
  model_version?: string | null;
  latency_ms?: number | null;
}

export interface Evidence {
  page: number;
  bbox: BBox;
  text?: string | null;
  source_id?: string | null;
}

export interface FieldEvidence {
  span_ids: string[];
  bbox: BBox | null;
  regions: Evidence[];
}

export interface ReceiptFields {
  seller: string | null;
  seller_address: string | null;
  timestamp: string | null;
  total_cost: number | null;
}

export type RawFields = Record<FieldName, string | null>;

export interface ExtractionWarning {
  code: string;
  field: FieldName | null;
  message: string;
}

export interface ReceiptExtraction {
  document_type: "receipt";
  model_version: string;
  fields: ReceiptFields;
  raw_fields: RawFields;
  evidence: Partial<Record<FieldName, FieldEvidence>>;
  warnings: ExtractionWarning[];
}

export interface DocumentSummary {
  id: string;
  original_name: string;
  status: DocumentStatus;
  created_at: string;
  updated_at: string | null;
  seller: string | null;
  seller_address: string | null;
  timestamp: string | null;
  total_cost: number | null;
}

export type FieldSource = "ai" | "human";

export interface FieldCorrection {
  id: number;
  field_name: FieldName;
  old_value: string | number | null;
  new_value: string | number | null;
  reviewer: string | null;
  extraction_id: number | null;
  created_at: string;
}

export interface DocumentEvent {
  action: "uploaded" | "processed" | "failed" | "corrected" | "approved" | string;
  actor: string | null;
  detail: Record<string, unknown> | null;
  created_at: string;
}

export interface QualityReport {
  score: number | null;
  level: string;
  warnings: string[];
}

export interface DocumentDetail extends DocumentSummary {
  image_url: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
  image_width: number | null;
  image_height: number | null;
  error: string | null;
  approved_at: string | null;
  approved_by: string | null;
  /** Current values: latest AI extraction with later human corrections applied. */
  fields: ReceiptFields;
  field_sources: Partial<Record<FieldName, FieldSource>>;
  /** Warnings recomputed on the current values. */
  warnings: ExtractionWarning[];
  extraction_id: number | null;
  /** Latest AI output, unchanged by corrections. */
  extraction: ReceiptExtraction | null;
  ocr: OCRResult | null;
  quality: QualityReport | null;
  pipeline_version: string | null;
  model_version: string | null;
  processing_time_ms: number | null;
  timings_ms: Record<string, number> | null;
  corrections: FieldCorrection[];
  events: DocumentEvent[];
}

export interface DocumentList {
  items: DocumentSummary[];
  total: number;
  page: number;
  page_size: number;
}

export interface DocumentFilters {
  status?: DocumentStatus;
  seller?: string;
  created_from?: string; // YYYY-MM-DD
  created_to?: string;
  receipt_from?: string;
  receipt_to?: string;
}

export interface DocumentStats {
  total: number;
  by_status: Record<DocumentStatus, number>;
  approved_total_vnd: number;
}

/** Body of PATCH /documents/{id}/fields: only the keys present are changed. */
export interface FieldsUpdate {
  seller?: string | null;
  seller_address?: string | null;
  timestamp?: string | null;
  total_cost?: number | string | null;
  reviewer?: string | null;
}

export interface Health {
  status: "ok" | "degraded";
  version: string;
  env: string;
  database: string;
  limits: { max_upload_mb: number };
  pipeline: { name: string; version: string; loaded: boolean };
}
