"""API test fixtures: SQLite instead of PostgreSQL, a temp storage dir and the mock
pipeline (``ai.mock``), so tests need neither a database server nor a GPU."""

from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw
from sqlalchemy.orm import sessionmaker

from ai.mock import MockKIE, MockOCR
from ai.pipeline import ReceiptPipeline
from app.api import deps
from app.core.config import Settings, get_settings
from app.db import Base, get_db, make_engine
from app.main import app
from app.services.processing import PipelineInfo, PipelineRunner
from app.services.storage import Storage


class FlakyFactory:
    """Pipeline factory whose pipeline can be told to fail."""

    def __init__(self) -> None:
        self.fail = False

    def __call__(self) -> ReceiptPipeline:
        factory = self

        class _Pipeline(ReceiptPipeline):
            def run(self, image):
                if factory.fail:
                    raise RuntimeError("GPU exploded")
                return super().run(image)

        return _Pipeline(MockOCR(), MockKIE())


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        storage_dir=str(tmp_path / "storage"),
        uploads_per_minute=1000,
        api_key=None,
        _env_file=None,
    )


@pytest.fixture
def engine(settings):
    eng = make_engine(settings.database_url)
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture
def pipeline_factory() -> FlakyFactory:
    return FlakyFactory()


@pytest.fixture
def client(settings, engine, pipeline_factory):
    sm = sessionmaker(bind=engine, expire_on_commit=False)
    runner = PipelineRunner(PipelineInfo("default", "pipeline-test"), pipeline_factory)

    def _db():
        with sm() as s:
            yield s

    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_db] = _db
    app.dependency_overrides[deps.get_storage] = lambda: Storage(settings.storage_path)
    app.dependency_overrides[deps.get_pipeline_runner] = lambda: runner
    deps.upload_limiter.reset()
    yield TestClient(app)
    app.dependency_overrides.clear()


def receipt_image(fmt: str = "JPEG", size=(600, 900), text: str = "TONG TIEN 150.000") -> bytes:
    im = Image.new("RGB", size, "white")
    ImageDraw.Draw(im).text((40, 40), text, fill="black")
    buf = io.BytesIO()
    im.save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def image():
    """``image(fmt="JPEG", size=(600, 900), text=...)`` -> encoded bytes."""
    return receipt_image


@pytest.fixture
def upload(client):
    def _upload(data: bytes | None = None, name="receipt.jpg", process=False, **kw):
        data = receipt_image() if data is None else data
        return client.post(
            "/api/v1/documents",
            files={"file": (name, data, "image/jpeg")},
            params={"process": str(process).lower()},
            **kw,
        )

    return _upload
