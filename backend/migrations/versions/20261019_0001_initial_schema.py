"""initial schema: documents, extractions, field_corrections, document_events

Revision ID: 0001
Revises:
Create Date: 2026-10-19
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

JSON = sa.JSON().with_variant(postgresql.JSONB(astext_type=sa.Text()), "postgresql")
BIGID = sa.BigInteger().with_variant(sa.Integer(), "sqlite")
TS = sa.DateTime(timezone=True)


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.String(32), primary_key=True),
        sa.Column("storage_key", sa.String(255), nullable=False),
        sa.Column("original_name", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(64), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False),
        sa.Column("image_width", sa.Integer()),
        sa.Column("image_height", sa.Integer()),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error", sa.Text()),
        sa.Column("seller", sa.String(512)),
        sa.Column("seller_address", sa.Text()),
        sa.Column("receipt_timestamp", sa.String(19)),
        sa.Column("total_cost", sa.BigInteger()),
        sa.Column("created_at", TS, nullable=False),
        sa.Column("updated_at", TS, nullable=False),
        sa.Column("approved_at", TS),
        sa.Column("approved_by", sa.String(128)),
    )
    op.create_index("ix_documents_created_at", "documents", ["created_at"])
    op.create_index("ix_documents_sha256", "documents", ["sha256"])
    op.create_index("ix_documents_status", "documents", ["status"])
    op.create_index("ix_documents_status_created", "documents", ["status", "created_at"])

    op.create_table(
        "extractions",
        sa.Column("id", BIGID, primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("pipeline_name", sa.String(32), nullable=False),
        sa.Column("pipeline_version", sa.String(32), nullable=False),
        sa.Column("model_version", sa.String(255), nullable=False),
        sa.Column("raw_json", JSON, nullable=False),
        sa.Column("normalized_json", JSON, nullable=False),
        sa.Column("processing_time_ms", sa.Float(), nullable=False),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_extractions_document_id", "extractions", ["document_id"])

    op.create_table(
        "field_corrections",
        sa.Column("id", BIGID, primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("extraction_id", BIGID, sa.ForeignKey("extractions.id", ondelete="SET NULL")),
        sa.Column("field_name", sa.String(32), nullable=False),
        sa.Column("old_value", JSON),
        sa.Column("new_value", JSON),
        sa.Column("reviewer", sa.String(128)),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_field_corrections_document_id", "field_corrections", ["document_id"])

    op.create_table(
        "document_events",
        sa.Column("id", BIGID, primary_key=True, autoincrement=True),
        sa.Column(
            "document_id",
            sa.String(32),
            sa.ForeignKey("documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("action", sa.String(32), nullable=False),
        sa.Column("actor", sa.String(128)),
        sa.Column("detail", JSON),
        sa.Column("created_at", TS, nullable=False),
    )
    op.create_index("ix_document_events_document_id", "document_events", ["document_id"])


def downgrade() -> None:
    op.drop_table("document_events")
    op.drop_table("field_corrections")
    op.drop_table("extractions")
    op.drop_table("documents")
