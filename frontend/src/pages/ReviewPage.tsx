import { Alert, Button, Card, Col, Descriptions, Empty, Result, Row, Space, Spin, Table, Tag, Typography } from "antd";
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api/client";
import { FIELD_NAMES, type DocumentDetail, type FieldName } from "../api/types";
import { EvidenceCanvas } from "../components/EvidenceCanvas";
import { StatusTag } from "../components/StatusTag";
import { FIELD_LABELS, formatVnd } from "../format";

export function ReviewPage() {
  const { id = "" } = useParams();
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<FieldName | null>(null);

  useEffect(() => {
    setDoc(null);
    setError(null);
    api.getDocument(id).then(setDoc).catch((e: Error) => setError(e.message));
  }, [id]);

  if (error) return <Result status="warning" title="Document not available" subTitle={error} extra={<Link to="/history">Back to history</Link>} />;
  if (!doc) return <Spin size="large" style={{ display: "block", margin: "80px auto" }} />;

  const ex = doc.extraction;
  const highlight = selected && ex?.evidence[selected]?.bbox;

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Descriptions
        bordered
        size="small"
        column={{ xs: 1, md: 4 }}
        items={[
          { key: "file", label: "File", children: doc.original_name },
          { key: "status", label: "Status", children: <StatusTag status={doc.status} /> },
          { key: "model", label: "Model", children: ex?.model_version ?? "—" },
          { key: "time", label: "Processing", children: doc.processing_time_ms != null ? `${doc.processing_time_ms} ms` : "—" },
        ]}
      />
      {doc.error && <Alert type="error" showIcon message="Processing failed" description={doc.error} />}
      <Row gutter={16}>
        <Col xs={24} lg={10}>
          <Card title="Receipt" size="small">
            {doc.ocr && doc.image_width && doc.image_height ? (
              <EvidenceCanvas width={doc.image_width} height={doc.image_height} spans={doc.ocr.spans} highlight={highlight} />
            ) : (
              <Empty description="No OCR result" />
            )}
          </Card>
        </Col>
        <Col xs={24} lg={14}>
          <Card
            title="Extracted fields"
            size="small"
            extra={
              <Space>
                <Button disabled title="Phase 5">Save correction</Button>
                <Button type="primary" disabled title="Phase 5">Approve</Button>
              </Space>
            }
          >
            {ex ? (
              <Table
                size="small"
                rowKey="field"
                pagination={false}
                onRow={(r) => ({ onClick: () => setSelected(r.field), style: { cursor: "pointer" } })}
                rowClassName={(r) => (r.field === selected ? "ant-table-row-selected" : "")}
                dataSource={FIELD_NAMES.map((field) => ({
                  field,
                  value: ex.fields[field],
                  raw: ex.raw_fields[field],
                  warnings: ex.warnings.filter((w) => w.field === field),
                  hasEvidence: Boolean(ex.evidence[field]?.bbox),
                }))}
                columns={[
                  { title: "Field", dataIndex: "field", render: (f: FieldName) => FIELD_LABELS[f] },
                  {
                    title: "Value (normalized)",
                    dataIndex: "value",
                    render: (v, r) =>
                      v == null ? <Tag>missing</Tag> : r.field === "total_cost" ? formatVnd(v as number) : String(v),
                  },
                  { title: "Raw text", dataIndex: "raw", render: (v) => <Typography.Text type="secondary">{v ?? "—"}</Typography.Text> },
                  {
                    title: "",
                    key: "flags",
                    render: (_, r) => (
                      <Space size={4}>
                        {r.hasEvidence && <Tag color="orange">evidence</Tag>}
                        {r.warnings.map((w) => (
                          <Tag key={w.code} color="warning" title={w.message}>{w.code}</Tag>
                        ))}
                      </Space>
                    ),
                  },
                ]}
              />
            ) : (
              <Empty description="No extraction" />
            )}
            <Typography.Paragraph type="secondary" style={{ marginTop: 12, marginBottom: 0 }}>
              Click a field to highlight its evidence on the receipt. Values come from the AI and are not verified until a person approves them.
            </Typography.Paragraph>
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
