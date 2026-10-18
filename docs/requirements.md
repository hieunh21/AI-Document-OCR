# Requirements

## Problem

Read a photo of a Vietnamese retail receipt and return four fields — `seller`,
`seller_address`, `timestamp`, `total_cost` — with the place on the image each value
came from, so a person can check and correct them quickly.

Scope and non-goals: see `VietDocAI_MC_OCR_Phase_Roadmap.md` §0.1 (no line items, no
VAT breakdown, no PO/GR matching, no RAG/chatbot, no automatic accounting).

## User stories

### US-1 Upload a receipt
*As a* bookkeeper, *I want to* upload a photo of a receipt *so that* its key fields are
extracted without typing.

Acceptance:
- JPG/PNG up to the configured size (default 10 MB) is accepted; other types and larger
  files are rejected with a clear message.
- After processing, the document shows status `needs_review` (any field missing or any
  warning) or `ready`; on failure `failed` with a reason.
- An unreadable or text-less image never crashes the app; it ends in `failed` or with
  all fields `null` and a warning.

### US-2 Review and correct the fields
*As a* bookkeeper, *I want to* see the receipt next to the four extracted fields and
click a field to see where it was read *so that* I can verify and fix it.

Acceptance:
- The image is shown with the evidence box of the selected field, correctly aligned at
  any window size.
- Each field shows its normalized value, the raw text and warnings; missing fields are
  shown as missing, not as empty strings.
- I can edit any field; the old and new values are stored. *Save correction* and
  *Approve* are separate actions; only *Approve* sets status `approved`.
- The UI marks which values came from the AI and which a person changed.

### US-3 Find past receipts and export
*As a* bookkeeper, *I want to* list processed receipts and export approved ones *so
that* I can use them in other tools.

Acceptance:
- History lists documents with status, seller, receipt time, total and upload time;
  filter by status (later: date range, seller), paginated.
- Approved documents export to JSON and CSV (Excel optional) with exactly the approved
  values.
- History stays available when a new upload fails.

## Document statuses

| Status | Meaning |
|---|---|
| `uploaded` | File stored, not processed yet |
| `processing` | Pipeline running |
| `needs_review` | Finished; at least one field missing or a warning raised |
| `ready` | Finished; all four fields present and no warning — still **not** verified |
| `approved` | A person reviewed and approved the values |
| `failed` | Processing failed (unreadable file, no text, inference error) |

Defined in `ai/schemas.py::DocumentStatus`.

## Non-functional

- Every extraction carries `model_version`; offline benchmarks use the same pipeline object.
- Evidence boxes reference real OCR spans; a model never invents coordinates.
- Missing/unreadable field → `null`, never a guess.
- Structured logs with `request_id` and per-stage latency.
- No secrets, dataset images or labels in the repository (`docs/data_usage.md`).
