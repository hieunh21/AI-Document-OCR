import type { DocumentDetail, DocumentList, DocumentStatus, Health } from "./types";

const API_PREFIX = "/api/v1";

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(path, { headers: { Accept: "application/json" } });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(res.status, String(detail));
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => getJson<Health>("/health"),
  listDocuments: (status?: DocumentStatus) =>
    getJson<DocumentList>(
      `${API_PREFIX}/documents${status ? `?status=${encodeURIComponent(status)}` : ""}`,
    ),
  getDocument: (id: string) =>
    getJson<DocumentDetail>(`${API_PREFIX}/documents/${encodeURIComponent(id)}`),
};
