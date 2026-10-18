export const FIELD_LABELS = {
  seller: "Seller",
  seller_address: "Seller address",
  timestamp: "Timestamp",
  total_cost: "Total cost",
} as const;

export function formatVnd(v: number | null | undefined): string {
  return v == null ? "—" : `${v.toLocaleString("vi-VN")} ₫`;
}
