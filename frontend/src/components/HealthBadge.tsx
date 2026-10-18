import { Badge, Tooltip } from "antd";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Health } from "../api/types";

export function HealthBadge() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e: Error) => setError(e.message));
  }, []);

  if (error) {
    return (
      <Tooltip title={error}>
        <Badge status="error" text={<span style={{ color: "#fff" }}>API offline</span>} />
      </Tooltip>
    );
  }
  if (!health) return <Badge status="processing" text={<span style={{ color: "#fff" }}>API…</span>} />;
  return (
    <Badge
      status="success"
      text={<span style={{ color: "#fff" }}>{`API ${health.version} · ${health.env}`}</span>}
    />
  );
}
