import { ApiError } from "./api/client";
import type { FieldName } from "./api/types";

export const FIELD_LABELS: Record<FieldName, string> = {
  seller: "Seller",
  seller_address: "Seller address",
  timestamp: "Timestamp",
  total_cost: "Total cost",
};

export function formatVnd(v: number | null | undefined): string {
  return v == null ? "—" : `${v.toLocaleString("vi-VN")} ₫`;
}

/** Receipt time "2020-04-30T10:41:57" -> "30/04/2020 10:41:57" (no timezone on receipts). */
export function formatReceiptTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const m = /^(\d{4})-(\d{2})-(\d{2})(?:T(\d{2}:\d{2}(?::\d{2})?))?$/.exec(iso);
  if (!m) return iso;
  return `${m[3]}/${m[2]}/${m[1]}${m[4] ? ` ${m[4]}` : ""}`;
}

/** Server time (UTC ISO) in the viewer's local time. */
export function formatDateTime(iso: string | null | undefined): string {
  return iso ? new Date(iso).toLocaleString("vi-VN") : "—";
}

export function formatFieldValue(field: FieldName, v: string | number | null | undefined): string {
  if (v == null || v === "") return "—";
  if (field === "total_cost") return formatVnd(Number(v));
  if (field === "timestamp") return formatReceiptTime(String(v));
  return String(v);
}

export function formatBytes(n: number): string {
  return n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}

/** Message for an error from the API client, with the request id for support. */
export function errorText(e: unknown): string {
  if (e instanceof ApiError) {
    return e.requestId ? `${e.message} (request ${e.requestId})` : e.message;
  }
  return e instanceof Error ? e.message : String(e);
}

const REVIEWER_KEY = "vietdocai.reviewer";

export function loadReviewer(): string {
  try {
    return localStorage.getItem(REVIEWER_KEY) ?? "";
  } catch {
    return "";
  }
}

export function saveReviewer(name: string): void {
  try {
    localStorage.setItem(REVIEWER_KEY, name);
  } catch {
    /* storage unavailable (private mode): the name is just not remembered */
  }
}
