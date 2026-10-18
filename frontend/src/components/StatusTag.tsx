import { Tag } from "antd";
import type { DocumentStatus } from "../api/types";

const STATUS: Record<DocumentStatus, { color: string; label: string }> = {
  uploaded: { color: "default", label: "Uploaded" },
  processing: { color: "processing", label: "Processing" },
  needs_review: { color: "warning", label: "Needs review" },
  ready: { color: "cyan", label: "Ready" },
  approved: { color: "success", label: "Approved" },
  failed: { color: "error", label: "Failed" },
};

export function StatusTag({ status }: { status: DocumentStatus }) {
  const s = STATUS[status];
  return <Tag color={s.color}>{s.label}</Tag>;
}

export const STATUS_OPTIONS = (Object.keys(STATUS) as DocumentStatus[]).map((value) => ({
  value,
  label: STATUS[value].label,
}));
