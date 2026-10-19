"""KIE with a large language model over OCR text (Phase 3B candidate).

The model never sees the image and never produces coordinates. It receives the
OCR lines, each tagged with its span id, and returns for every field the text
it read plus the ids of the spans it used. Our code then

* drops ids that do not exist (evidence always points at real OCR boxes),
* normalizes timestamp / total with the same parsers as the rule baseline,
* turns an empty answer into ``None`` (absent field), never a guess.

Backends are pluggable (``LLMBackend``); ``GeminiBackend`` uses the official
``google-genai`` SDK and accepts a custom base URL (proxy/reseller endpoints).
Responses are cached on disk by request hash, so re-running an evaluation is
free and reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from ai.kie.base import KIEProvider
from ai.kie.gazetteer import Gazetteer
from ai.kie.rule_based import RuleBasedKIE
from ai.kie.validate import validate_extraction
from ai.layout import group_lines
from ai.normalization.address import AdminUnits
from ai.normalization.datetime_vi import parse_timestamp
from ai.normalization.text import normalize_text
from ai.schemas import (
    FIELD_NAMES,
    Evidence,
    ExtractionWarning,
    FieldEvidence,
    OCRSpan,
    RawFields,
    ReceiptExtraction,
    ReceiptFields,
)

CACHE_DIR = Path("outputs/llm_cache")
PROMPT_VERSION = "p1"

SYSTEM_PROMPT = """You extract key information from the OCR text of a Vietnamese retail receipt.

Input: OCR lines in reading order. Every text fragment is prefixed with its id in
square brackets, e.g. "[s012] Tổng tiền:". Fragments on the same printed line are
on the same input line. OCR makes mistakes, especially in Vietnamese diacritics.

Return exactly four fields. For each field give:
- "text": the value as printed on the receipt. You may fix obvious OCR character
  errors (wrong or missing Vietnamese diacritics, O/0 confusions) when you are
  confident; never add information that is not on the receipt. Use "" when the
  field is not on the receipt or cannot be read.
- "span_ids": the ids of every fragment the value comes from ([] when text is "").

Fields:
- seller: the store / brand name printed at the top (e.g. "VinCommerce",
  "MINIMART ANAN", "NHÀ SÁCH GD-TC CẨM PHẢ"). Not the legal company line, not the
  receipt title ("HÓA ĐƠN BÁN HÀNG"), not a phone number.
- seller_address: the store address lines, joined in reading order with a space.
  Include store-code lines such as "VM+ QNH 112 Thanh Niên". Keep prefixes like
  "ĐC:". Exclude phone, fax, tax code (MST), website and email lines.
- timestamp: the date and time of the transaction as printed (e.g.
  "Ngày bán: 15/08/2020 11:41"). If check-in and check-out times are printed,
  use the first one. Not expiry or voucher dates.
- total_cost: the final amount the customer has to pay, digits as printed
  (e.g. "74,000"). Prefer "tổng tiền phải thanh toán" / "tổng thanh toán" /
  "khách phải trả" over "tổng cộng" or "cộng tiền hàng" when they differ.
  Never the cash given ("tiền khách đưa"), the change ("tiền thừa"), discounts,
  VAT, or item counts.
"""

_FIELD_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "span_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["text", "span_ids"],
}
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {f: _FIELD_SCHEMA for f in FIELD_NAMES},
    "required": list(FIELD_NAMES),
}


class LLMResponseError(ValueError):
    """The model answer is not a JSON object of the expected shape."""


def parse_json_object(text: str) -> dict:
    """JSON object from a model answer, tolerating markdown code fences.

    Some endpoints (proxies, OpenAI-compatible gateways) ignore the JSON response
    mode and wrap the object in ```json … ```; we must not depend on it.
    """
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    try:
        obj = json.loads(t)
    except json.JSONDecodeError:
        start, end = t.find("{"), t.rfind("}")
        if start < 0 or end <= start:
            raise LLMResponseError(f"no JSON object in model answer: {text[:200]!r}") from None
        try:
            obj = json.loads(t[start : end + 1])
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"invalid JSON in model answer: {e}") from None
    if not isinstance(obj, dict):
        raise LLMResponseError(f"expected a JSON object, got {type(obj).__name__}")
    return obj


@dataclass
class LLMUsage:
    input_tokens: int = 0
    output_tokens: int = 0  # includes "thinking" tokens when the model reports them
    latency_ms: float = 0.0
    cached: bool = False


class LLMBackend(ABC):
    name: str
    model: str

    @abstractmethod
    def complete_json(self, system: str, user: str, schema: dict) -> tuple[dict, LLMUsage]:
        """Return the parsed JSON object and token usage."""


class GeminiBackend(LLMBackend):
    name = "gemini"

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        thinking_budget: int | None = None,
        timeout_ms: int = 60_000,
    ) -> None:
        from google import genai
        from google.genai import types

        from ai.config import load_dotenv

        load_dotenv()
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
        base_url = base_url or os.environ.get("GEMINI_BASE_URL") or None
        if thinking_budget is None and os.environ.get("GEMINI_THINKING_BUDGET"):
            thinking_budget = int(os.environ["GEMINI_THINKING_BUDGET"])
        self.thinking_budget = thinking_budget
        self._types = types
        self._client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                base_url=base_url,
                timeout=timeout_ms,
                retry_options=types.HttpRetryOptions(attempts=4),
            ),
        )

    def complete_json(self, system: str, user: str, schema: dict) -> tuple[dict, LLMUsage]:
        t = self._types
        cfg = dict(
            system_instruction=system,
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0.0,
        )
        if self.thinking_budget is not None:
            cfg["thinking_config"] = t.ThinkingConfig(thinking_budget=self.thinking_budget)
        t0 = time.perf_counter()
        resp = self._client.models.generate_content(
            model=self.model, contents=user, config=t.GenerateContentConfig(**cfg)
        )
        latency = (time.perf_counter() - t0) * 1000
        um = resp.usage_metadata
        usage = LLMUsage(
            input_tokens=(um.prompt_token_count or 0) if um else 0,
            output_tokens=((um.candidates_token_count or 0) + (um.thoughts_token_count or 0))
            if um
            else 0,
            latency_ms=latency,
        )
        return parse_json_object(resp.text or ""), usage


@dataclass
class LLMKIE(KIEProvider):
    backend: LLMBackend
    cache_dir: Path | None = CACHE_DIR
    gazetteer: Gazetteer | None = None  # optional post-normalization of seller/address
    admin: AdminUnits | None = None  # official place names: address accent repair
    # Re-query the same prompt without reusing cached answers (e.g. to check whether the
    # model behind a proxy changed). Part of the cache key; the request itself is unchanged.
    run_tag: str = ""
    name: str = "llm"
    last_usage: LLMUsage = field(default_factory=LLMUsage)

    @property
    def request_version(self) -> str:
        """Identity of the model request (cache key). Post-processing is not part of it."""
        tb = getattr(self.backend, "thinking_budget", None)
        think = "" if tb is None else f"-think{tb}"
        rerun = f"-rerun{self.run_tag}" if self.run_tag else ""
        return f"{self.backend.name}:{self.backend.model}{think}-{PROMPT_VERSION}{rerun}"

    @property
    def version(self) -> str:  # type: ignore[override]
        return (
            self.request_version
            + ("+gaz" if self.gazetteer else "")
            + ("+adm" if self.admin else "")
        )

    @staticmethod
    def render(spans: list[OCRSpan]) -> str:
        lines = group_lines(spans, lambda s: s.bbox)
        return "\n".join(" ".join(f"[{s.id}] {s.text}" for s in line) for line in lines)

    def _call(self, user: str) -> tuple[dict, LLMUsage]:
        key = hashlib.sha256(
            json.dumps([self.request_version, SYSTEM_PROMPT, user], ensure_ascii=False).encode()
        ).hexdigest()
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            return data["response"], LLMUsage(**data["usage"], cached=True)
        response, usage = self.backend.complete_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": self.request_version,
                "response": response,
                "usage": {k: v for k, v in usage.__dict__.items() if k != "cached"},
            }
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return response, usage

    def extract(self, image: Image.Image | None, ocr_spans: list[OCRSpan]) -> ReceiptExtraction:
        extraction, self.last_usage = self.extract_with_usage(ocr_spans)
        return extraction

    def extract_with_usage(self, ocr_spans: list[OCRSpan]) -> tuple[ReceiptExtraction, LLMUsage]:
        """Thread-safe variant returning the token usage of this call."""
        spans = [s for s in ocr_spans if s.text.strip()]
        by_id = {s.id: s for s in spans}
        warnings: list[ExtractionWarning] = []
        if not spans:
            return ReceiptExtraction(
                model_version=f"{self.name}-{self.version}",
                warnings=[ExtractionWarning(code="no_text_detected", message="No OCR text.")],
            ), LLMUsage()
        response, usage = self._call(self.render(spans))

        fields, raw, evidence = {}, {}, {}
        for f in FIELD_NAMES:
            item = response.get(f)
            if isinstance(item, str):  # model flattened the object: {"seller": "ABC"}
                item = {"text": item, "span_ids": []}
            if not isinstance(item, dict):
                item = {}
            text = normalize_text(str(item.get("text") or ""))
            raw_ids = item.get("span_ids")
            ids = [i for i in raw_ids if isinstance(i, str)] if isinstance(raw_ids, list) else []
            used = [by_id[i] for i in ids if i in by_id]
            if len(used) < len(ids):
                warnings.append(
                    ExtractionWarning(
                        code="llm_unknown_span",
                        field=f,
                        message=f"LLM cited span ids that do not exist: "
                        f"{sorted(set(ids) - set(by_id))}",
                    )
                )
            if not text:
                continue
            value = self._normalize(f, text, used)
            if value is None:
                warnings.append(
                    ExtractionWarning(
                        code="unparsable_value",
                        field=f,
                        message=f"Could not normalize {f!r} from {text!r}.",
                    )
                )
                continue
            fields[f], raw[f] = value, text
            if used:
                evidence[f] = FieldEvidence(
                    regions=[Evidence(bbox=s.bbox, text=s.text, source_id=s.id) for s in used]
                )
            else:
                warnings.append(
                    ExtractionWarning(
                        code="no_evidence", field=f, message="Value has no supporting OCR span."
                    )
                )
        if self.gazetteer and "seller" in fields:  # canonical spelling of known stores
            fields["seller"] = self.gazetteer.canonical_seller(fields["seller"]) or fields["seller"]
        if "seller_address" in fields:
            addr = fields["seller_address"]
            canon = (
                self.gazetteer.match_address(addr, fields.get("seller")) if self.gazetteer else None
            )
            fields["seller_address"] = canon or (self.admin.correct(addr) if self.admin else addr)
        return validate_extraction(
            ReceiptExtraction(
                model_version=f"{self.name}-{self.version}",
                fields=ReceiptFields(**fields),
                raw_fields=RawFields(**raw),
                evidence=evidence,
                warnings=warnings,
            )
        ), usage

    @staticmethod
    def _normalize(field_name: str, text: str, used: list[OCRSpan]):
        if field_name in ("seller", "seller_address"):
            return text
        if field_name == "timestamp":
            for candidate in (text, " ".join(s.text for s in used)):
                parsed = parse_timestamp(candidate)
                if parsed and parsed.iso:
                    return parsed.iso
            return None
        # total_cost
        for candidate in (text, " ".join(s.text for s in used)):
            amounts = RuleBasedKIE._amounts(candidate)
            if amounts:
                return amounts[-1]
        return None
