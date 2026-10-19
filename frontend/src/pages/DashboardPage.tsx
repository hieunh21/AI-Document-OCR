import { Alert, Button, Card, Col, List, Progress, Row, Space, Statistic, Typography } from "antd";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api/client";
import type { DocumentStats, DocumentStatus, DocumentSummary } from "../api/types";
import { STATUS_OPTIONS, StatusTag } from "../components/StatusTag";
import { errorText, formatDateTime, formatVnd } from "../format";

export function DashboardPage() {
  const navigate = useNavigate();
  const [stats, setStats] = useState<DocumentStats | null>(null);
  const [toReview, setToReview] = useState<DocumentSummary[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.stats(), api.listDocuments({ status: "needs_review" }, 1, 5)])
      .then(([s, r]) => {
        setStats(s);
        setToReview(r.items);
      })
      .catch((e) => setError(errorText(e)));
  }, []);

  if (error) return <Alert type="error" showIcon message="Could not load the dashboard" description={error} />;
  const loading = stats === null;
  const s = stats ?? { total: 0, by_status: {} as Record<DocumentStatus, number>, approved_total_vnd: 0 };
  const n = (k: DocumentStatus) => s.by_status[k] ?? 0;
  const processed = n("ready") + n("needs_review") + n("approved");
  const go = (status: DocumentStatus) => navigate(`/history?status=${status}`);

  return (
    <Space direction="vertical" size="middle" style={{ width: "100%" }}>
      <Row gutter={[16, 16]}>
        <Col xs={24} sm={12} lg={6}>
          <Card loading={loading} hoverable onClick={() => navigate("/history")}>
            <Statistic title="Documents" value={s.total} />
            <Typography.Text type="secondary">{processed} processed</Typography.Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card loading={loading} hoverable onClick={() => go("needs_review")}>
            <Statistic title="Need review" value={n("needs_review")} valueStyle={{ color: n("needs_review") ? "#d48806" : undefined }} />
            <Typography.Text type="secondary">{n("ready")} ready, no warnings</Typography.Text>
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card loading={loading} hoverable onClick={() => go("approved")}>
            <Statistic title="Approved" value={n("approved")} valueStyle={{ color: "#389e0d" }} />
            <Progress percent={processed ? Math.round((100 * n("approved")) / processed) : 0} size="small" />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card loading={loading}>
            {/* Only human-approved records count towards money totals. */}
            <Statistic title="Total of approved receipts" value={formatVnd(s.approved_total_vnd)} />
            <Typography.Text type="secondary">AI-only values are not counted</Typography.Text>
          </Card>
        </Col>
      </Row>
      <Row gutter={[16, 16]}>
        <Col xs={24} lg={14}>
          <Card
            size="small"
            title="Waiting for review"
            extra={<Link to="/history?status=needs_review">All</Link>}
          >
            <List
              loading={loading}
              dataSource={toReview}
              locale={{ emptyText: "Nothing to review" }}
              renderItem={(d) => (
                <List.Item actions={[<Link key="open" to={`/documents/${d.id}`}>Review</Link>]}>
                  <List.Item.Meta
                    title={<Link to={`/documents/${d.id}`}>{d.seller ?? d.original_name}</Link>}
                    description={`${formatVnd(d.total_cost)} · uploaded ${formatDateTime(d.created_at)}`}
                  />
                </List.Item>
              )}
            />
          </Card>
        </Col>
        <Col xs={24} lg={10}>
          <Card size="small" title="By status">
            <List
              loading={loading}
              dataSource={STATUS_OPTIONS}
              renderItem={(o) => (
                <List.Item onClick={() => go(o.value)} style={{ cursor: "pointer" }}>
                  <StatusTag status={o.value} />
                  <Typography.Text strong>{n(o.value)}</Typography.Text>
                </List.Item>
              )}
            />
            <Button block type="primary" style={{ marginTop: 12 }} onClick={() => navigate("/upload")}>
              Upload a receipt
            </Button>
          </Card>
        </Col>
      </Row>
    </Space>
  );
}
