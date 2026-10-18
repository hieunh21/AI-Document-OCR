from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from ai.schemas import OCRSpan, ReceiptExtraction


class KIEProvider(ABC):
    """Key information extraction for the four MC-OCR fields.

    Evidence returned in the extraction must reference ids of ``ocr_spans``;
    a provider must not invent coordinates.
    """

    name: str = "kie"
    version: str = "0"

    @abstractmethod
    def extract(self, image: Image.Image | None, ocr_spans: list[OCRSpan]) -> ReceiptExtraction:
        raise NotImplementedError
