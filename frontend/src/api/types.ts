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
  seller: string | null;
  timestamp: string | null;
  total_cost: number | null;
}

export interface DocumentDetail extends DocumentSummary {
  image_width: number | null;
  image_height: number | null;
  ocr: OCRResult | null;
  extraction: ReceiptExtraction | null;
  error: string | null;
  processing_time_ms: number | null;
}

export interface DocumentList {
  items: DocumentSummary[];
  total: number;
}

export interface Health {
  status: string;
  version: string;
  env: string;
}
