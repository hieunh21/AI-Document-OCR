import { Alert, Card, Select, Space, Table, Typography } from "antd";
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import type { DocumentStatus, DocumentSummary } from "../api/types";
import { STATUS_OPTIONS, StatusTag } from "../components/StatusTag";
import { formatVnd } from "../format";

export function HistoryPage() {
  const [status, setStatus] = useState<DocumentStatus | undefined>();
  const [items, setItems] = useState<DocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    api
      .listDocuments(status)
      .then((r) => setItems(r.items))
      .catch((e: Error) => setError(e.message))
      .finally(() => setLoading(false));
  }, [status]);

  return (
    <Card
      title="History"
      extra={
        <Space>
          <Typography.Text type="secondary">Status</Typography.Text>
          <Select
            allowClear
            placeholder="All"
            style={{ width: 160 }}
            options={STATUS_OPTIONS}
            value={status}
            onChange={setStatus}
          />
        </Space>
      }
    >
      {error && <Alert type="error" showIcon message="Could not load documents" description={error} />}
      <Table<DocumentSummary>
        rowKey="id"
        loading={loading}
        dataSource={items}
        pagination={{ pageSize: 20 }}
        columns={[
          {
            title: "File",
            dataIndex: "original_name",
            render: (name: string, d) => <Link to={`/documents/${d.id}`}>{name}</Link>,
          },
          { title: "Status", dataIndex: "status", render: (s: DocumentStatus) => <StatusTag status={s} /> },
          { title: "Seller", dataIndex: "seller", render: (v) => v ?? "—" },
          { title: "Receipt time", dataIndex: "timestamp", render: (v) => v ?? "—" },
          { title: "Total", dataIndex: "total_cost", align: "right", render: (v) => formatVnd(v) },
          {
            title: "Uploaded",
            dataIndex: "created_at",
            render: (v: string) => new Date(v).toLocaleString("vi-VN"),
          },
        ]}
      />
    </Card>
  );
}
