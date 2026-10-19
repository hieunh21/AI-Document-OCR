import { CloudUploadOutlined, DeleteOutlined, InboxOutlined } from "@ant-design/icons";
import { Alert, App, Button, Card, Col, Image, Row, Space, Steps, Typography, Upload } from "antd";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ApiError, api } from "../api/client";
import { errorText, formatBytes } from "../format";

const ACCEPT = ["image/jpeg", "image/png", "image/webp"];
const DEFAULT_MAX_MB = 10;

/** Receipts drawn by scripts/make_demo_receipts.py (invented content). */
const DEMO_RECEIPTS = [
  { file: "demo_coffee.jpg", label: "Coffee shop" },
  { file: "demo_minimart.jpg", label: "Minimart" },
  { file: "demo_tilted.jpg", label: "Tilted photo" },
];

type Phase = "idle" | "uploading" | "processing";

export function UploadPage() {
  const navigate = useNavigate();
  const { message } = App.useApp();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [error, setError] = useState<string | null>(null);
  const [maxMb, setMaxMb] = useState(DEFAULT_MAX_MB);

  useEffect(() => {
    api
      .health()
      .then((h) => setMaxMb(h.limits.max_upload_mb))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!file) return setPreview(null);
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  function choose(f: File): boolean {
    setError(null);
    if (!ACCEPT.includes(f.type)) {
      setError(`"${f.name}" is not a JPG, PNG or WebP image.`);
    } else if (f.size > maxMb * 1024 * 1024) {
      setError(`"${f.name}" is ${formatBytes(f.size)}; the limit is ${maxMb} MB.`);
    } else if (f.size === 0) {
      setError(`"${f.name}" is empty.`);
    } else {
      setFile(f);
    }
    return false; // never let antd upload by itself
  }

  async function chooseDemo(name: string) {
    try {
      const res = await fetch(`/demo/${name}`);
      if (!res.ok) throw new Error(`Demo image ${name} is missing (${res.status}).`);
      choose(new File([await res.blob()], name, { type: "image/jpeg" }));
    } catch (e) {
      setError(errorText(e));
    }
  }

  async function submit() {
    if (!file) return;
    setError(null);
    setPhase("uploading");
    let docId: string | null = null;
    try {
      const { doc, created } = await api.uploadDocument(file);
      docId = doc.id;
      if (!created) {
        message.info("This image was uploaded before: opening the existing document.");
        navigate(`/documents/${doc.id}`);
        return;
      }
      setPhase("processing");
      await api.processDocument(doc.id);
      message.success("Extraction finished. Please review the fields.");
      navigate(`/documents/${doc.id}`);
    } catch (e) {
      if (docId) {
        // The upload worked but extraction failed: the review page shows the error and a retry.
        message.error(`Extraction failed: ${errorText(e)}`);
        navigate(`/documents/${docId}`);
      } else {
        setError(errorText(e));
        if (e instanceof ApiError && e.code === "network_error") setFile(null);
      }
    } finally {
      setPhase("idle");
    }
  }

  const busy = phase !== "idle";
  const current = phase === "processing" ? 1 : 0;

  return (
    <Card title="Upload receipt">
      <Space direction="vertical" size="large" style={{ width: "100%" }}>
        <Steps
          size="small"
          current={current}
          items={[
            { title: "Upload", description: "JPG / PNG / WebP" },
            { title: "Extract", description: "OCR + 4 fields", status: phase === "processing" ? "process" : undefined },
            { title: "Review", description: "Check, edit, approve" },
          ]}
        />
        {error && <Alert type="error" showIcon message="Cannot upload this file" description={error} closable onClose={() => setError(null)} />}
        <Row gutter={24}>
          <Col xs={24} md={12}>
            <Upload.Dragger
              accept={ACCEPT.join(",")}
              multiple={false}
              showUploadList={false}
              beforeUpload={choose}
              disabled={busy}
            >
              <p className="ant-upload-drag-icon">
                <InboxOutlined />
              </p>
              <p className="ant-upload-text">Drop a receipt photo here, or click to choose</p>
              <p className="ant-upload-hint">One receipt per image, up to {maxMb} MB.</p>
            </Upload.Dragger>
            <Space wrap style={{ marginTop: 16 }}>
              <Typography.Text type="secondary">No receipt at hand? Try a demo:</Typography.Text>
              {DEMO_RECEIPTS.map((d) => (
                <Button key={d.file} size="small" disabled={busy} onClick={() => chooseDemo(d.file)}>
                  {d.label}
                </Button>
              ))}
            </Space>
          </Col>
          <Col xs={24} md={12}>
            {file && preview ? (
              <Space direction="vertical" style={{ width: "100%" }}>
                <div style={{ textAlign: "center", background: "#fafafa", border: "1px solid #f0f0f0", padding: 8 }}>
                  <Image src={preview} alt={file.name} style={{ maxHeight: 420, objectFit: "contain" }} />
                </div>
                <Typography.Text>
                  {file.name} · {formatBytes(file.size)}
                </Typography.Text>
                <Space>
                  <Button
                    type="primary"
                    icon={<CloudUploadOutlined />}
                    loading={busy}
                    onClick={submit}
                  >
                    {phase === "uploading" ? "Uploading…" : phase === "processing" ? "Extracting…" : "Upload & extract"}
                  </Button>
                  <Button icon={<DeleteOutlined />} disabled={busy} onClick={() => setFile(null)}>
                    Remove
                  </Button>
                </Space>
                {phase === "processing" && (
                  <Typography.Text type="secondary">
                    The first image after a server start also loads the models (about 10 s); later ones take about a second.
                  </Typography.Text>
                )}
              </Space>
            ) : (
              <Alert
                type="info"
                showIcon
                message="How it works"
                description="The image is read by PP-OCRv6 (fine-tuned for Vietnamese), then the seller, address, time and total are extracted with the text regions they came from. Nothing is final until a person approves it."
              />
            )}
          </Col>
        </Row>
      </Space>
    </Card>
  );
}
