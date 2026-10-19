import type {
  DocumentDetail,
  DocumentFilters,
  DocumentList,
  DocumentStats,
  FieldsUpdate,
  Health,
} from "./types";

const API_PREFIX = "/api/v1";
const DOCS = `${API_PREFIX}/documents`;

/** Error body of every backend error: ``{detail, code, request_id}``. */
export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
    public code?: string,
    public requestId?: string,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<{ data: T; status: number }> {
  let res: Response;
  try {
    res = await fetch(path, { ...init, headers: { Accept: "application/json", ...init?.headers } });
  } catch {
    throw new ApiError(0, "Cannot reach the API server. Is the backend running?", "network_error");
  }
  if (!res.ok) {
    let detail: string = res.statusText;
    let code: string | undefined;
    let requestId: string | undefined;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? detail);
      code = body.code;
      requestId = body.request_id;
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(res.status, detail, code, requestId);
  }
  return { data: (await res.json()) as T, status: res.status };
}

const get = async <T>(path: string) => (await request<T>(path)).data;

function sendJson<T>(method: string, path: string, body?: unknown) {
  return request<T>(path, {
    method,
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }).then((r) => r.data);
}

function query(params: Record<string, string | number | undefined | null>): string {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== "") q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

export const api = {
  health: () => get<Health>("/health"),

  listDocuments: (filters: DocumentFilters = {}, page = 1, pageSize = 20) =>
    get<DocumentList>(`${DOCS}${query({ ...filters, page, page_size: pageSize })}`),

  stats: () => get<DocumentStats>(`${DOCS}/stats`),

  getDocument: (id: string) => get<DocumentDetail>(`${DOCS}/${encodeURIComponent(id)}`),

  /** ``created`` is false when the same file was uploaded before (existing document). */
  uploadDocument: async (file: File) => {
    const form = new FormData();
    form.append("file", file, file.name);
    const { data, status } = await request<DocumentDetail>(DOCS, { method: "POST", body: form });
    return { doc: data, created: status === 201 };
  },

  processDocument: (id: string) =>
    sendJson<DocumentDetail>("POST", `${DOCS}/${encodeURIComponent(id)}/process`),

  updateFields: (id: string, update: FieldsUpdate) =>
    sendJson<DocumentDetail>("PATCH", `${DOCS}/${encodeURIComponent(id)}/fields`, update),

  approveDocument: (id: string, reviewer?: string | null) =>
    sendJson<DocumentDetail>("POST", `${DOCS}/${encodeURIComponent(id)}/approve`, {
      reviewer: reviewer || null,
    }),

  /** Plain links: the browser downloads them (Content-Disposition: attachment). */
  exportUrl: (id: string, format: "json" | "csv") =>
    `${DOCS}/${encodeURIComponent(id)}/export${query({ format })}`,
  bulkExportUrl: (filters: DocumentFilters, format: "json" | "csv") =>
    `${DOCS}/export${query({ ...filters, format })}`,
};
