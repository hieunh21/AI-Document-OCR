import { InboxOutlined } from "@ant-design/icons";
import { Alert, Card, Space, Upload } from "antd";

export function UploadPage() {
  return (
    <Card title="Upload receipt">
      <Space direction="vertical" size="middle" style={{ width: "100%" }}>
        <Alert
          type="info"
          showIcon
          message="Upload is wired up in Phase 5"
          description="The backend currently serves mock documents only. Open History to browse them."
        />
        <Upload.Dragger disabled accept="image/jpeg,image/png" multiple={false}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">Drop a JPG/PNG receipt here</p>
          <p className="ant-upload-hint">One image per receipt, up to 10 MB.</p>
        </Upload.Dragger>
      </Space>
    </Card>
  );
}
