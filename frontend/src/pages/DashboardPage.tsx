import { Card, Col, Row, Statistic } from "antd";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DocumentSummary } from "../api/types";

export function DashboardPage() {
  const [items, setItems] = useState<DocumentSummary[] | null>(null);

  useEffect(() => {
    api.listDocuments().then((r) => setItems(r.items)).catch(() => setItems([]));
  }, []);

  const loading = items === null;
  const docs = items ?? [];
  const approved = docs.filter((d) => d.status === "approved");
  return (
    <Row gutter={[16, 16]}>
      <Col xs={24} md={8}>
        <Card loading={loading}>
          <Statistic title="Processed documents" value={docs.length} />
        </Card>
      </Col>
      <Col xs={24} md={8}>
        <Card loading={loading}>
          <Statistic title="Need review" value={docs.filter((d) => d.status === "needs_review").length} />
        </Card>
      </Col>
      <Col xs={24} md={8}>
        <Card loading={loading}>
          {/* Only human-approved records count towards money totals. */}
          <Statistic
            title="Total of approved receipts (₫)"
            value={approved.reduce((s, d) => s + (d.total_cost ?? 0), 0)}
            groupSeparator="."
          />
        </Card>
      </Col>
    </Row>
  );
}
