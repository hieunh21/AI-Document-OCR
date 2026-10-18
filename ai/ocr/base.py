from __future__ import annotations

from abc import ABC, abstractmethod

from PIL import Image

from ai.schemas import OCRResult


class OCRProvider(ABC):
    """Text detection + recognition.

    Implementations must return boxes in the coordinate system of the image
    they were given; the pipeline is responsible for mapping back to the
    original upload if it resized/rotated the image beforehand.
    """

    name: str = "ocr"
    version: str = "0"

    @abstractmethod
    def predict(self, image: Image.Image) -> OCRResult:
        raise NotImplementedError
