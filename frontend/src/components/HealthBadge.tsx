import { Badge, Tooltip } from "antd";
import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { Health } from "../api/types";
import { errorText } from "../format";

const white = { color: "#fff" };

export function HealthBadge() {
  const [health, setHealth] = useState<Health | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api
      .health()
      .then(setHealth)
      .catch((e) => setError(errorText(e)));
  }, []);

  if (error) {
    return (
      <Tooltip title={error}>
        <Badge status="error" text={<span style={white}>API offline</span>} />
      </Tooltip>
    );
  }
  if (!health) return <Badge status="processing" text={<span style={white}>API…</span>} />;
  const ok = health.status === "ok";
  return (
    <Tooltip
      title={`Database: ${health.database} · pipeline ${health.pipeline.name} (${
        health.pipeline.loaded ? "models loaded" : "models load on first use"
      })`}
    >
      <Badge
        status={ok ? "success" : "warning"}
        text={
          <span style={white}>
            {ok ? `${health.pipeline.version} · API ${health.version}` : "Database unavailable"}
          </span>
        }
      />
    </Tooltip>
  );
}
