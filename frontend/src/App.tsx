import { Layout, Menu, Typography } from "antd";
import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { HealthBadge } from "./components/HealthBadge";
import { DashboardPage } from "./pages/DashboardPage";
import { HistoryPage } from "./pages/HistoryPage";
import { ReviewPage } from "./pages/ReviewPage";
import { UploadPage } from "./pages/UploadPage";

const NAV = [
  { key: "/dashboard", label: <Link to="/dashboard">Dashboard</Link> },
  { key: "/upload", label: <Link to="/upload">Upload</Link> },
  { key: "/history", label: <Link to="/history">History</Link> },
];

export default function App() {
  const { pathname } = useLocation();
  // A document page belongs to History in the menu.
  const selected = NAV.find((n) => pathname.startsWith(n.key))?.key ?? "/history";

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Layout.Header style={{ display: "flex", alignItems: "center", gap: 24 }}>
        <Link to="/dashboard">
          <Typography.Title level={4} style={{ color: "#fff", margin: 0, whiteSpace: "nowrap" }}>
            VietDocAI
          </Typography.Title>
        </Link>
        <Menu theme="dark" mode="horizontal" selectedKeys={[selected]} items={NAV} style={{ flex: 1, minWidth: 0 }} />
        <HealthBadge />
      </Layout.Header>
      <Layout.Content style={{ padding: 24, maxWidth: 1400, width: "100%", margin: "0 auto" }}>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/upload" element={<UploadPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/documents/:id" element={<ReviewPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="*" element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </Layout.Content>
    </Layout>
  );
}
