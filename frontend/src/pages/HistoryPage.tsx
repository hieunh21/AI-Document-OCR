import { DownloadOutlined, ReloadOutlined, UploadOutlined } from "@ant-design/icons";
import { Alert, Button, Card, DatePicker, Dropdown, Input, Select, Space, Table, Typography } from "antd";
import type { Dayjs } from "dayjs";
import { useCallback, useEffect, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { api } from "../api/client";
import type { DocumentFilters, DocumentStatus, DocumentSummary } from "../api/types";
import { STATUS_OPTIONS, StatusTag } from "../components/StatusTag";
import { errorText, formatDateTime, formatReceiptTime, formatVnd } from "../format";

type Range = [Dayjs | null, Dayjs | null] | null;
const day = (d: Dayjs | null | undefined) => (d ? d.format("YYYY-MM-DD") : undefined);

export function HistoryPage() {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const [seller, setSeller] = useState(params.get("seller") ?? "");
  const [uploaded, setUploaded] = useState<Range>(null);
  const [receipt, setReceipt] = useState<Range>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);
  const [items, setItems] = useState<DocumentSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const status = (params.get("status") as DocumentStatus | null) ?? undefined;
  const filters: DocumentFilters = {
    status,
    seller: params.get("seller") ?? undefined,
    created_from: day(uploaded?.[0]),
    created_to: day(uploaded?.[1]),
    receipt_from: day(receipt?.[0]),
    receipt_to: day(receipt?.[1]),
  };
  const key = JSON.stringify(filters);

  const load = useCallback(() => {
    setLoading(true);
    setError(null);
    api
      .listDocuments(JSON.parse(key) as DocumentFilters, page, pageSize)
      .then((r) => {
        setItems(r.items);
        setTotal(r.total);
      })
      .catch((e) => setError(errorText(e)))
      .finally(() => setLoading(false));
  }, [key, page, pageSize]);

  useEffect(load, [load]);
  useEffect(() => setPage(1), [key]); // a new filter starts at page 1

  function setParam(name: string, value: string | undefined) {
    const next = new URLSearchParams(params);
    if (value) next.set(name, value);
    else next.delete(name);
    setParams(next, { replace: true });
  }

  const exportItems = [
    { key: "csv", label: <a href={api.bulkExportUrl(filters, "csv")}>CSV (opens in Excel)</a> },
    { key: "json", label: <a href={api.bulkExportUrl(filters, "json")}>JSON (with evidence and history)</a> },
  ];

  return (
    <Card
      title="History"
      extra={
        <Space wrap>
          <Button icon={<ReloadOutlined />} onClick={load}>
            Refresh
          </Button>
          <Dropdown menu={{ items: exportItems }} disabled={!total}>
            <Button icon={<DownloadOutlined />}>
              Export {status ? `${STATUS_OPTIONS.find((o) => o.value === status)?.label.toLowerCase()} ` : ""}({total})
            </Button>
          </Dropdown>
          <Button type="primary" icon={<UploadOutlined />} onClick={() => navigate("/upload")}>
            Upload
          </Button>
        </Space>
      }
    >
      <Space wrap style={{ marginBottom: 16 }}>
        <Select
          allowClear
          placeholder="Any status"
          style={{ width: 160 }}
          options={STATUS_OPTIONS}
          value={status}
          onChange={(v) => setParam("status", v)}
        />
        <Input.Search
          allowClear
          placeholder="Seller contains…"
          style={{ width: 220 }}
          value={seller}
          onChange={(e) => setSeller(e.target.value)}
          onSearch={(v) => setParam("seller", v.trim() || undefined)}
        />
        <Space size={4}>
          <Typography.Text type="secondary">Uploaded</Typography.Text>
          <DatePicker.RangePicker value={uploaded} onChange={setUploaded} format="DD/MM/YYYY" allowEmpty={[true, true]} />
        </Space>
        <Space size={4}>
          <Typography.Text type="secondary">Receipt date</Typography.Text>
          <DatePicker.RangePicker value={receipt} onChange={setReceipt} format="DD/MM/YYYY" allowEmpty={[true, true]} />
        </Space>
      </Space>
      {status === "approved" && (
        <Typography.Paragraph type="secondary">
          Approved records were checked by a person; their values can be exported as final.
        </Typography.Paragraph>
      )}
      {error && <Alert type="error" showIcon message="Could not load documents" description={error} style={{ marginBottom: 16 }} />}
      <Table<DocumentSummary>
        rowKey="id"
        size="middle"
        loading={loading}
        dataSource={items}
        scroll={{ x: 900 }}
        onRow={(d) => ({ onClick: () => navigate(`/documents/${d.id}`), style: { cursor: "pointer" } })}
        pagination={{
          current: page,
          pageSize,
          total,
          showSizeChanger: true,
          pageSizeOptions: [10, 20, 50, 100],
          showTotal: (t) => `${t} documents`,
          onChange: (p, s) => {
            setPage(s === pageSize ? p : 1);
            setPageSize(s);
          },
        }}
        locale={{ emptyText: total === 0 && !status ? "No documents yet: upload a receipt to start." : "No matching documents" }}
        columns={[
          {
            title: "File",
            dataIndex: "original_name",
            ellipsis: true,
            render: (name: string, d) => (
              <Link to={`/documents/${d.id}`} onClick={(e) => e.stopPropagation()}>
                {name}
              </Link>
            ),
          },
          { title: "Status", dataIndex: "status", width: 130, render: (s: DocumentStatus) => <StatusTag status={s} /> },
          { title: "Seller", dataIndex: "seller", ellipsis: true, render: (v) => v ?? "—" },
          { title: "Receipt time", dataIndex: "timestamp", width: 170, render: formatReceiptTime },
          { title: "Total", dataIndex: "total_cost", width: 130, align: "right", render: (v) => formatVnd(v) },
          { title: "Uploaded", dataIndex: "created_at", width: 180, render: formatDateTime },
        ]}
      />
    </Card>
  );
}
