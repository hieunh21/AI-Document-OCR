import {
  CheckCircleOutlined,
  DownloadOutlined,
  EditOutlined,
  EyeOutlined,
  LockOutlined,
  ReloadOutlined,
  RobotOutlined,
  SaveOutlined,
  UserOutlined,
} from "@ant-design/icons";
import {
  Alert,
  App,
  Button,
  Card,
  Col,
  Descriptions,
  Dropdown,
  Empty,
  Input,
  InputNumber,
  List,
  Result,
  Row,
  Space,
  Spin,
  Switch,
  Table,
  Tabs,
  Tag,
  Timeline,
  Tooltip,
  Typography,
} from "antd";
import { useCallback, useEffect, useMemo, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import {
  FIELD_NAMES,
  type BBox,
  type DocumentDetail,
  type DocumentEvent,
  type FieldName,
  type FieldsUpdate,
} from "../api/types";
import { EvidenceCanvas } from "../components/EvidenceCanvas";
import { StatusTag } from "../components/StatusTag";
import {
  FIELD_LABELS,
  errorText,
  formatDateTime,
  formatFieldValue,
  formatReceiptTime,
  loadReviewer,
  saveReviewer,
} from "../format";

interface Draft {
  seller: string;
  seller_address: string;
  timestamp: string; // as typed: "30/04/2020 10:41" or ISO; the backend normalizes
  total_cost: number | null;
}

function toDraft(doc: DocumentDetail): Draft {
  const f = doc.fields;
  return {
    seller: f.seller ?? "",
    seller_address: f.seller_address ?? "",
    timestamp: f.timestamp ? formatReceiptTime(f.timestamp) : "",
    total_cost: f.total_cost,
  };
}

function changedFields(draft: Draft, base: Draft): FieldName[] {
  return FIELD_NAMES.filter((f) => {
    const a = draft[f];
    const b = base[f];
    return typeof a === "string" ? a.trim() !== String(b ?? "").trim() : a !== b;
  });
}

function toUpdate(draft: Draft, fields: FieldName[], reviewer: string): FieldsUpdate {
  const update: FieldsUpdate = { reviewer: reviewer.trim() || null };
  for (const f of fields) {
    const v = draft[f];
    (update as Record<FieldName, string | number | null>)[f] =
      typeof v === "string" ? v.trim() || null : v;
  }
  return update;
}

const EVENT_LABEL: Record<string, { color: string; text: string }> = {
  uploaded: { color: "gray", text: "Uploaded" },
  processed: { color: "blue", text: "Extracted by AI" },
  failed: { color: "red", text: "Processing failed" },
  corrected: { color: "purple", text: "Corrected" },
  approved: { color: "green", text: "Approved" },
};

function eventText(e: DocumentEvent): string {
  const d = e.detail ?? {};
  if (e.action === "processed") return `${d.pipeline_version ?? ""} · ${d.processing_time_ms ?? "?"} ms → ${d.status ?? ""}`;
  if (e.action === "corrected") return `Fields: ${((d.fields as string[]) ?? []).map((f) => FIELD_LABELS[f as FieldName] ?? f).join(", ")}`;
  if (e.action === "failed") return String(d.message ?? "");
  if (e.action === "approved") {
    const open = (d.open_warnings as string[]) ?? [];
    return open.length ? `With open warnings: ${open.join(", ")}` : "No open warnings";
  }
  return "";
}

export function ReviewPage() {
  const { id = "" } = useParams();
  const { message, modal } = App.useApp();
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [selected, setSelected] = useState<FieldName | null>(null);
  const [activeSpans, setActiveSpans] = useState<string[]>([]);
  const [showSpans, setShowSpans] = useState(false);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [reviewer, setReviewer] = useState(loadReviewer);
  const [busy, setBusy] = useState<"save" | "approve" | "process" | null>(null);

  const apply = useCallback((d: DocumentDetail) => {
    setDoc(d);
    setDraft(toDraft(d));
  }, []);

  const load = useCallback(() => {
    setLoadError(null);
    return api
      .getDocument(id)
      .then(apply)
      .catch((e) => setLoadError(errorText(e)));
  }, [id, apply]);

  useEffect(() => {
    setDoc(null);
    setSelected(null);
    load();
  }, [load]);

  // Another tab or request may be processing this document: poll until it settles.
  useEffect(() => {
    if (doc?.status !== "processing") return;
    const t = setInterval(load, 2000);
    return () => clearInterval(t);
  }, [doc?.status, load]);

  const base = useMemo(() => (doc ? toDraft(doc) : null), [doc]);
  const dirty = draft && base ? changedFields(draft, base) : [];

  if (loadError && !doc) {
    return (
      <Result
        status="warning"
        title="Document not available"
        subTitle={loadError}
        extra={<Link to="/history">Back to history</Link>}
      />
    );
  }
  if (!doc || !draft) return <Spin size="large" style={{ display: "block", margin: "80px auto" }} />;

  const ex = doc.extraction;
  const processed = doc.extraction_id != null && ["ready", "needs_review", "approved"].includes(doc.status);
  const locked = doc.status === "approved";
  const editable = processed && !locked;
  const highlights: BBox[] = selected && ex?.evidence[selected] ? ex.evidence[selected]!.regions.map((r) => r.bbox) : [];
  const fieldWarnings = (f: FieldName) => doc.warnings.filter((w) => w.field === f);
  const generalWarnings = doc.warnings.filter((w) => !w.field);

  function rememberReviewer(name: string) {
    setReviewer(name);
    saveReviewer(name);
  }

  async function run<T>(kind: "save" | "approve" | "process", fn: () => Promise<T>): Promise<T | undefined> {
    setBusy(kind);
    try {
      return await fn();
    } catch (e) {
      message.error(errorText(e));
      await load(); // show the state the server actually has
      return undefined;
    } finally {
      setBusy(null);
    }
  }

  async function save(): Promise<DocumentDetail | undefined> {
    if (!dirty.length) return doc ?? undefined;
    const d = await run("save", () => api.updateFields(id, toUpdate(draft!, dirty, reviewer)));
    if (d) {
      apply(d);
      message.success(`Saved ${dirty.length} correction${dirty.length > 1 ? "s" : ""}.`);
    }
    return d;
  }

  function approve() {
    const pending = dirty.length;
    const open = doc!.warnings;
    modal.confirm({
      title: "Approve this receipt?",
      icon: <CheckCircleOutlined style={{ color: "#52c41a" }} />,
      content: (
        <Space direction="vertical">
          {pending > 0 && <Typography.Text>Your {pending} unsaved change(s) will be saved first.</Typography.Text>}
          {open.length > 0 && (
            <Alert
              type="warning"
              showIcon
              message="Open warnings"
              description={
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {open.map((w) => (
                    <li key={`${w.code}-${w.field}`}>{w.message}</li>
                  ))}
                </ul>
              }
            />
          )}
          <Typography.Text type="secondary">
            Approved values are locked: they can no longer be edited or re-processed.
          </Typography.Text>
        </Space>
      ),
      okText: "Approve",
      onOk: async () => {
        if (pending > 0 && !(await save())) return;
        const d = await run("approve", () => api.approveDocument(id, reviewer));
        if (d) {
          apply(d);
          message.success("Approved.");
        }
      },
    });
  }

  function process() {
    const go = async () => {
      setDoc({ ...doc!, status: "processing" });
      const d = await run("process", () => api.processDocument(id));
      if (d) {
        apply(d);
        message.success("Extraction finished.");
      }
    };
    if (!processed) return go();
    modal.confirm({
      title: "Run the extraction again?",
      content: "The current values are replaced by the new AI result. Earlier corrections stay in the history.",
      okText: "Re-process",
      onOk: go,
    });
  }

  const statusBanner = (() => {
    if (doc.status === "uploaded")
      return (
        <Alert
          type="info"
          showIcon
          message="This image has not been processed yet."
          action={<Button type="primary" loading={busy === "process"} onClick={process}>Extract now</Button>}
        />
      );
    if (doc.status === "processing")
      return <Alert type="info" showIcon icon={<Spin size="small" />} message="Extracting… the page refreshes by itself." />;
    if (doc.status === "failed")
      return (
        <Alert
          type="error"
          showIcon
          message="Processing failed"
          description={doc.error}
          action={<Button danger loading={busy === "process"} onClick={process}>Retry</Button>}
        />
      );
    if (locked)
      return (
        <Alert
          type="success"
          showIcon
          icon={<LockOutlined />}
          message={`Approved${doc.approved_by ? ` by ${doc.approved_by}` : ""} · ${formatDateTime(doc.approved_at)}`}
          description="The values are locked. Export them below or from History."
        />
      );
    return null;
  })();

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Card size="small">
        <Space style={{ width: "100%", justifyContent: "space-between" }} wrap>
          <Space wrap>
            <Typography.Title level={5} style={{ margin: 0 }}>
              {doc.original_name}
            </Typography.Title>
            <StatusTag status={doc.status} />
          </Space>
          <Space wrap>
            {processed && !locked && (
              <Button icon={<ReloadOutlined />} loading={busy === "process"} onClick={process}>
                Re-process
              </Button>
            )}
            <Dropdown
              disabled={!processed}
              menu={{
                items: [
                  { key: "json", label: <a href={api.exportUrl(id, "json")}>JSON (with evidence and history)</a> },
                  { key: "csv", label: <a href={api.exportUrl(id, "csv")}>CSV (one row)</a> },
                ],
              }}
            >
              <Button icon={<DownloadOutlined />}>Export</Button>
            </Dropdown>
          </Space>
        </Space>
        <Descriptions
          size="small"
          style={{ marginTop: 12 }}
          column={{ xs: 1, sm: 2, lg: 4 }}
          items={[
            { key: "up", label: "Uploaded", children: formatDateTime(doc.created_at) },
            { key: "pipe", label: "Pipeline", children: doc.pipeline_version ?? "—" },
            {
              key: "time",
              label: "Processing",
              children: doc.processing_time_ms != null ? `${Math.round(doc.processing_time_ms)} ms` : "—",
            },
            {
              key: "q",
              label: "Image quality",
              children: doc.quality ? (
                <Tag color={doc.quality.level === "good" ? "green" : doc.quality.level === "poor" ? "orange" : "default"}>
                  {doc.quality.level}
                  {doc.quality.score != null ? ` · ${doc.quality.score.toFixed(2)}` : ""}
                </Tag>
              ) : (
                "—"
              ),
            },
          ]}
        />
      </Card>

      {statusBanner}
      {generalWarnings.map((w) => (
        <Alert key={w.code} type="warning" showIcon message={w.message} />
      ))}

      <Row gutter={[16, 16]}>
        <Col xs={24} lg={10}>
          <Card
            size="small"
            title="Receipt"
            style={{ position: "sticky", top: 16 }}
            extra={
              <Space size={6}>
                <Typography.Text type="secondary">All OCR boxes</Typography.Text>
                <Switch size="small" checked={showSpans} onChange={setShowSpans} disabled={!doc.ocr} />
              </Space>
            }
          >
            {doc.image_width && doc.image_height ? (
              <EvidenceCanvas
                width={doc.image_width}
                height={doc.image_height}
                imageUrl={doc.image_url}
                spans={doc.ocr?.spans ?? []}
                showSpans={showSpans}
                highlights={highlights}
                activeSpanIds={activeSpans}
                onSpanClick={(s) => setActiveSpans([s.id])}
              />
            ) : (
              <Empty description="No image" />
            )}
            <Typography.Paragraph type="secondary" style={{ margin: "8px 0 0" }}>
              {selected
                ? highlights.length
                  ? `Orange: where the AI read "${FIELD_LABELS[selected]}".`
                  : `The AI found no evidence for "${FIELD_LABELS[selected]}".`
                : "Select a field to see where its value was read."}
            </Typography.Paragraph>
          </Card>
        </Col>

        <Col xs={24} lg={14}>
          <Card
            size="small"
            title="Extracted fields"
            extra={
              editable && (
                <Space wrap>
                  <Input
                    size="small"
                    prefix={<UserOutlined />}
                    placeholder="Reviewer name"
                    value={reviewer}
                    onChange={(e) => rememberReviewer(e.target.value)}
                    style={{ width: 160 }}
                    maxLength={128}
                  />
                  <Button
                    icon={<SaveOutlined />}
                    disabled={!dirty.length}
                    loading={busy === "save"}
                    onClick={save}
                  >
                    Save correction{dirty.length > 1 ? "s" : ""}
                  </Button>
                  <Button type="primary" icon={<CheckCircleOutlined />} loading={busy === "approve"} onClick={approve}>
                    Approve
                  </Button>
                </Space>
              )
            }
          >
            {!processed ? (
              <Empty description="No extraction yet" />
            ) : (
              <Space direction="vertical" size={12} style={{ width: "100%" }}>
                {FIELD_NAMES.map((f) => {
                  const human = doc.field_sources[f] === "human";
                  const changed = dirty.includes(f);
                  const warnings = fieldWarnings(f);
                  const raw = ex?.raw_fields[f];
                  const hasEvidence = Boolean(ex?.evidence[f]?.regions.length);
                  const lastFix = [...doc.corrections].reverse().find((c) => c.field_name === f && c.extraction_id === doc.extraction_id);
                  return (
                    <div
                      key={f}
                      onClick={() => setSelected(f)}
                      style={{
                        padding: 12,
                        borderRadius: 8,
                        border: `1px solid ${selected === f ? "#fa8c16" : "#f0f0f0"}`,
                        background: selected === f ? "#fff7e6" : undefined,
                        cursor: "pointer",
                      }}
                    >
                      <Space style={{ width: "100%", justifyContent: "space-between", marginBottom: 6 }} wrap>
                        <Space size={6}>
                          <Typography.Text strong>{FIELD_LABELS[f]}</Typography.Text>
                          {changed ? (
                            <Tag color="gold" icon={<EditOutlined />}>unsaved</Tag>
                          ) : human ? (
                            <Tooltip title={lastFix ? `${lastFix.reviewer ?? "Someone"} · ${formatDateTime(lastFix.created_at)}` : undefined}>
                              <Tag color="purple" icon={<UserOutlined />}>edited by a person</Tag>
                            </Tooltip>
                          ) : (
                            <Tag color="blue" icon={<RobotOutlined />}>AI</Tag>
                          )}
                        </Space>
                        <Button
                          size="small"
                          type="text"
                          icon={<EyeOutlined />}
                          disabled={!hasEvidence}
                          onClick={(e) => {
                            e.stopPropagation();
                            setSelected(f);
                          }}
                        >
                          {hasEvidence ? "Evidence" : "No evidence"}
                        </Button>
                      </Space>
                      {f === "total_cost" ? (
                        <InputNumber<string | number>
                          style={{ width: "100%" }}
                          min={0}
                          addonAfter="₫"
                          placeholder="not on the receipt"
                          disabled={!editable}
                          value={draft.total_cost}
                          formatter={(v) => (v === undefined || v === "" ? "" : Number(v).toLocaleString("vi-VN"))}
                          parser={(v) => (v ?? "").replace(/\D/g, "")}
                          onFocus={() => setSelected(f)}
                          onChange={(v) =>
                            setDraft({ ...draft, total_cost: v === null || v === "" ? null : Number(v) })
                          }
                        />
                      ) : f === "seller_address" ? (
                        <Input.TextArea
                          autoSize={{ minRows: 1, maxRows: 3 }}
                          placeholder="not on the receipt"
                          disabled={!editable}
                          value={draft.seller_address}
                          onFocus={() => setSelected(f)}
                          onChange={(e) => setDraft({ ...draft, seller_address: e.target.value })}
                        />
                      ) : (
                        <Input
                          placeholder={f === "timestamp" ? "dd/mm/yyyy hh:mm, or empty if not printed" : "not on the receipt"}
                          disabled={!editable}
                          value={draft[f]}
                          onFocus={() => setSelected(f)}
                          onChange={(e) => setDraft({ ...draft, [f]: e.target.value })}
                        />
                      )}
                      <div style={{ marginTop: 6 }}>
                        {raw && (
                          <Typography.Text type="secondary" style={{ display: "block", fontSize: 12 }}>
                            Read on the receipt: “{raw}”
                          </Typography.Text>
                        )}
                        {human && ex && (
                          <Typography.Text type="secondary" style={{ display: "block", fontSize: 12 }}>
                            AI value: {formatFieldValue(f, ex.fields[f])}
                          </Typography.Text>
                        )}
                        {warnings.map((w) => (
                          <Typography.Text key={w.code} type="warning" style={{ display: "block", fontSize: 12 }}>
                            ⚠ {w.message}
                          </Typography.Text>
                        ))}
                      </div>
                    </div>
                  );
                })}
                {editable && (
                  <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                    Values come from the AI and are not verified until a person approves them. Leave a field
                    empty when it is not printed on the receipt.
                  </Typography.Text>
                )}
              </Space>
            )}
          </Card>

          <Card size="small" style={{ marginTop: 16 }}>
            <Tabs
              size="small"
              items={[
                {
                  key: "ocr",
                  label: `OCR text (${doc.ocr?.spans.length ?? 0})`,
                  children: doc.ocr ? (
                    <List
                      size="small"
                      style={{ maxHeight: 360, overflow: "auto" }}
                      dataSource={doc.ocr.spans}
                      renderItem={(s) => (
                        <List.Item
                          onMouseEnter={() => setActiveSpans([s.id])}
                          onMouseLeave={() => setActiveSpans([])}
                          style={{ cursor: "default", background: activeSpans.includes(s.id) ? "#e6f4ff" : undefined }}
                          extra={
                            s.confidence != null && (
                              <Typography.Text type={s.confidence < 0.8 ? "warning" : "secondary"} style={{ fontSize: 12 }}>
                                {(s.confidence * 100).toFixed(0)}%
                              </Typography.Text>
                            )
                          }
                        >
                          {s.text}
                        </List.Item>
                      )}
                    />
                  ) : (
                    <Empty description="No OCR result" />
                  ),
                },
                {
                  key: "history",
                  label: `Corrections (${doc.corrections.length})`,
                  children: (
                    <Table
                      size="small"
                      rowKey="id"
                      pagination={false}
                      dataSource={[...doc.corrections].reverse()}
                      locale={{ emptyText: "No corrections" }}
                      columns={[
                        { title: "Field", dataIndex: "field_name", render: (f: FieldName) => FIELD_LABELS[f] },
                        {
                          title: "Before",
                          dataIndex: "old_value",
                          render: (v, c) => <Typography.Text delete>{formatFieldValue(c.field_name, v)}</Typography.Text>,
                        },
                        { title: "After", dataIndex: "new_value", render: (v, c) => formatFieldValue(c.field_name, v) },
                        { title: "By", dataIndex: "reviewer", render: (v) => v ?? "—" },
                        { title: "When", dataIndex: "created_at", render: formatDateTime },
                        {
                          title: "",
                          key: "old",
                          render: (_, c) => c.extraction_id !== doc.extraction_id && <Tag>earlier run</Tag>,
                        },
                      ]}
                    />
                  ),
                },
                {
                  key: "activity",
                  label: "Activity",
                  children: (
                    <Timeline
                      style={{ marginTop: 8 }}
                      items={[...doc.events].reverse().map((e) => ({
                        color: EVENT_LABEL[e.action]?.color ?? "gray",
                        children: (
                          <>
                            <Typography.Text strong>{EVENT_LABEL[e.action]?.text ?? e.action}</Typography.Text>
                            {e.actor && <Typography.Text> · {e.actor}</Typography.Text>}
                            <Typography.Text type="secondary"> · {formatDateTime(e.created_at)}</Typography.Text>
                            <div>
                              <Typography.Text type="secondary" style={{ fontSize: 12 }}>
                                {eventText(e)}
                              </Typography.Text>
                            </div>
                          </>
                        ),
                      }))}
                    />
                  ),
                },
                {
                  key: "tech",
                  label: "Technical",
                  children: (
                    <Descriptions
                      size="small"
                      column={1}
                      items={[
                        { key: "m", label: "Model", children: doc.model_version ?? "—" },
                        {
                          key: "t",
                          label: "Timings",
                          children: doc.timings_ms
                            ? Object.entries(doc.timings_ms)
                                .map(([k, v]) => `${k} ${Math.round(v)} ms`)
                                .join(" · ")
                            : "—",
                        },
                        { key: "s", label: "Image", children: `${doc.image_width}×${doc.image_height} px · ${doc.content_type}` },
                        { key: "h", label: "SHA-256", children: <Typography.Text code>{doc.sha256.slice(0, 16)}…</Typography.Text> },
                      ]}
                    />
                  ),
                },
              ]}
            />
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
